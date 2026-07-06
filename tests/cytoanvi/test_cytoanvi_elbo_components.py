"""Unit tests for CytoANVAE.loss() — ELBO components, nan-mask contract, CE ratio, n_labels guard.

These tests drive the module-level `loss()` forward pass directly (not through `model.train()`),
making it possible to assert fine-grained properties of each ELBO term independently.

Design notes
------------
- `reconstruction_loss` and `kl_local` in `LossOutput` are *dicts* with a single key each;
  the tensor lives at ``lo.reconstruction_loss["reconstruction_loss"]`` and
  ``lo.kl_local["kl_local"]`` respectively.
- `CytoANVAE.loss()` calls `encoder_z2_z1` internally (stochastic z2 sample), so two calls
  with identical inputs can return slightly different per-cell values.  Fix this by resetting
  the RNG with ``torch.manual_seed`` before each call — determinism is required for the
  nan-mask equality assertion.
- The nan-mask key in the tensors dict is ``"nan_layer"``
  (``CYTOVI_REGISTRY_KEYS.PROTEIN_NAN_MASK``).
"""

from __future__ import annotations

import pytest
import torch

from conftest import (
    BATCH_KEY,
    LABELS_KEY,
    SAMPLE_KEY,
    SCALED_LAYER_KEY,
    UNLABELED,
    make_adata,
)

from cytoanvi import CytoANVI
from scvi.external.cytovi._constants import CYTOVI_REGISTRY_KEYS

_NAN_LAYER_KEY = CYTOVI_REGISTRY_KEYS.PROTEIN_NAN_MASK  # "nan_layer"


def _setup_model(adata):
    """Setup and return an *untrained* CytoANVI model."""
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    return CytoANVI(adata, n_latent=10)


def _one_batch(model, n_cells: int = 16):
    """Return a single minibatch of tensors from the labeled subset."""
    indices = model._labeled_indices[:n_cells]
    dl = model._make_data_loader(adata=model.adata, indices=indices)
    return next(iter(dl))


def _forward(module, tensors):
    """Run inference + generative passes; returns (inference_outputs, generative_outputs)."""
    inf = module.inference(**module._get_inference_input(tensors))
    gen = module.generative(**module._get_generative_input(tensors, inf))
    return inf, gen


# ---------------------------------------------------------------------------
# C1-a: ELBO returns finite scalar loss and finite per-cell components
# ---------------------------------------------------------------------------


def test_elbo_returns_finite_loss():
    """loss(), reconstruction_loss, and kl_local are all finite scalars/tensors."""
    adata = make_adata(n_genes=20, batch_size=64)
    model = _setup_model(adata)
    tensors = _one_batch(model)

    module = model.module
    module.eval()
    with torch.no_grad():
        inf, gen = _forward(module, tensors)
        lo = module.loss(tensors, inf, gen)

    # scalar loss must be finite
    assert torch.isfinite(lo.loss), f"loss is not finite: {lo.loss}"

    # reconstruction_loss — dict with one key holding a per-cell tensor
    reconst = lo.reconstruction_loss["reconstruction_loss"]
    assert torch.all(torch.isfinite(reconst)), "reconstruction_loss contains non-finite values"

    # kl_local — dict with one key holding a per-cell tensor
    kl = lo.kl_local["kl_local"]
    assert torch.all(torch.isfinite(kl)), "kl_local contains non-finite values"

    # Sanity: scalar loss is roughly the mean of reconst + kl (kl_weight=1, no CE)
    expected = torch.mean(reconst + kl)
    torch.testing.assert_close(lo.loss, expected, atol=1e-5, rtol=0.0)


# ---------------------------------------------------------------------------
# C1-b: nan-mask contract — masked columns do NOT contribute to reconst loss
# ---------------------------------------------------------------------------


def test_nan_mask_excludes_missing_markers():
    """Cells/markers with nan_mask=0 must NOT contribute to reconstruction loss.

    Strategy
    --------
    1. Run *one* forward pass (inference + generative) and fix those outputs.
    2. Call loss() twice with the *same* fixed inf/gen but different tensors:
       - ``tensors_masked``:  original x + nan_mask zeroing column ``col``
       - ``tensors_pert``:    x with column ``col`` perturbed by +5, same mask
    3. Because ``px`` (from gen) is fixed and ``nan_mask[:, col] = 0`` zeros the
       log-prob contribution, ``reconstruction_loss`` must be identical.
    4. Positive control: the *same* perturbation WITHOUT a mask must change the loss.

    ``torch.manual_seed`` is reset identically before both calls to neutralise the
    stochastic z2 sample inside loss().
    """
    adata = make_adata(n_genes=20, batch_size=64)
    model = _setup_model(adata)
    tensors = _one_batch(model)

    module = model.module
    module.eval()

    col = 3  # an interior column to mask/perturb

    # Pre-compute fixed inference + generative outputs
    with torch.no_grad():
        inf, gen = _forward(module, tensors)

    # Build masked tensors
    nan_mask = torch.ones_like(tensors["X"])
    nan_mask[:, col] = 0.0

    tensors_masked = {**tensors, _NAN_LAYER_KEY: nan_mask}

    tensors_pert = {**tensors_masked, "X": tensors["X"].clone()}
    tensors_pert["X"][:, col] += 5.0  # perturb the masked column

    # ---- Negative test: masked column change must NOT affect reconst_loss ----
    with torch.no_grad():
        torch.manual_seed(42)
        lo_orig = module.loss(tensors_masked, inf, gen)
        torch.manual_seed(42)  # same seed → same z2 sample
        lo_pert = module.loss(tensors_pert, inf, gen)

    orig_reconst = lo_orig.reconstruction_loss["reconstruction_loss"]
    pert_reconst = lo_pert.reconstruction_loss["reconstruction_loss"]
    torch.testing.assert_close(
        orig_reconst,
        pert_reconst,
        atol=1e-6,
        rtol=0.0,
        msg="reconstruction_loss changed despite masked column — nan_mask contract violated",
    )

    # ---- Positive control: unmasked perturbation MUST change reconst_loss ----
    tensors_unmasked_orig = dict(tensors)  # no nan_layer key
    tensors_unmasked_pert = {**tensors, "X": tensors["X"].clone()}
    tensors_unmasked_pert["X"][:, col] += 5.0

    with torch.no_grad():
        torch.manual_seed(42)
        lo_um_orig = module.loss(tensors_unmasked_orig, inf, gen)
        torch.manual_seed(42)
        lo_um_pert = module.loss(tensors_unmasked_pert, inf, gen)

    assert not torch.allclose(
        lo_um_orig.reconstruction_loss["reconstruction_loss"],
        lo_um_pert.reconstruction_loss["reconstruction_loss"],
        atol=1e-6,
    ), "Positive control failed: unmasked column perturbation did not change reconstruction_loss"


# ---------------------------------------------------------------------------
# C1-c: classification_ratio controls the CE contribution
# ---------------------------------------------------------------------------


def test_classification_ratio_controls_ce_contribution():
    """With ratio=0 CE is not added to loss; with ratio=1 it is.

    The relationship ``loss(ratio=1) == loss(ratio=0) + 1 * ce_loss`` holds
    when the RNG is seeded identically before each call so that the
    classification_loss() internal encoder draw is reproducible.
    """
    adata = make_adata(n_genes=20, batch_size=64)
    model = _setup_model(adata)
    tensors = _one_batch(model)

    module = model.module
    module.eval()

    with torch.no_grad():
        inf, gen = _forward(module, tensors)

        # ratio=0 → CE term is computed but NOT added to loss
        torch.manual_seed(0)
        lo0 = module.loss(tensors, inf, gen, classification_ratio=0.0, labelled_tensors=tensors)

        # ratio=1 → CE term IS added to loss (same seed → same CE draw)
        torch.manual_seed(0)
        lo1 = module.loss(tensors, inf, gen, classification_ratio=1.0, labelled_tensors=tensors)

    ce = lo0.classification_loss
    assert ce is not None, "classification_loss should not be None when labelled_tensors provided"
    assert torch.isfinite(ce), f"classification_loss is not finite: {ce}"

    # ratio=0 → CE NOT added
    torch.testing.assert_close(
        lo1.loss,
        lo0.loss + ce,
        atol=1e-5,
        rtol=0.0,
        msg="loss(ratio=1) != loss(ratio=0) + ce_loss",
    )

    # The CE values from both calls must be the same (same seed)
    torch.testing.assert_close(
        lo0.classification_loss, lo1.classification_loss, atol=1e-6, rtol=0.0
    )


# ---------------------------------------------------------------------------
# C1-d: n_labels == 0 guard raises ValueError cleanly
# ---------------------------------------------------------------------------


def test_cytoanvae_n_labels_zero_raises():
    """Constructing CytoANVAE with n_labels < 1 must raise a descriptive ValueError."""
    from cytoanvi._module import CytoANVAE

    with pytest.raises(ValueError, match="n_labels >= 1"):
        CytoANVAE(n_input=10, n_labels=0)


# ---------------------------------------------------------------------------
# M3: training descends (slow — requires ~20 epochs)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_training_elbo_descends():
    """Train CytoANVI for 20 epochs and assert elbo_train falls from epoch 1 to last epoch."""
    adata = make_adata(n_genes=20, n_batches=2, n_labels=5, batch_size=64)
    model = _setup_model(adata)
    model.train(max_epochs=20, accelerator="cpu")

    elbo = model.history_["elbo_train"]
    first = float(elbo.iloc[0, 0])
    last = float(elbo.iloc[-1, 0])
    assert last < first, (
        f"ELBO did not descend over 20 epochs: epoch-1 = {first:.4f}, epoch-20 = {last:.4f}"
    )
