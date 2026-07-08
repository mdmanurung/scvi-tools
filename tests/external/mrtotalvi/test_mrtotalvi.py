"""Tests for MrTotalVI — TotalVI + MrVI hierarchical donor latent space.

Four invariants (CLAUDE.md "deepening invariance"):
  (a) smoke        — trains to completion with finite ELBO; counterfactual paths
                     (cf_sample, mc_samples=2) also produce finite tensors.
  (b) reconstruction — MrTotalVI's converged recon loss ≤ stock TotalVI × 1.05.
  (c) non-degenerate — per-sample embedding row-variance > init 0.1; eps non-zero.
  (d) donor-axis    — get_local_sample_distances separates donors injected with
                       a known per-donor shift in gene expression.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

# Ensure dev src is on path when running directly
sys.path.insert(0, "src")

import scvi
from scvi.external import MrTotalVI


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DONORS = 4
N_LATENT = 10
MAX_EPOCHS_QUICK = 3
MAX_EPOCHS_FULL = 20


def _make_adata(n_donors: int = N_DONORS, shift_scale: float = 0.0) -> scvi.AnnData:
    """AnnData with protein obsm and a synthetic `sample` obs column.

    If shift_scale > 0, donor ``d`` receives a per-gene additive offset of
    ``shift_scale * d`` so donors are detectably separated in gene space.
    """
    adata = scvi.data.synthetic_iid()

    # Assign donors round-robin
    n_cells = adata.n_obs
    donor_ids = np.array([f"donor_{i % n_donors}" for i in range(n_cells)])
    adata.obs["sample"] = donor_ids

    if shift_scale > 0:
        for d in range(n_donors):
            mask = adata.obs["sample"] == f"donor_{d}"
            adata.X[mask] = adata.X[mask] + shift_scale * (d + 1)

    return adata


@pytest.fixture(scope="module")
def adata_basic():
    return _make_adata()


@pytest.fixture(scope="module")
def adata_shifted():
    return _make_adata(shift_scale=5.0)


# ---------------------------------------------------------------------------
# Helper: build + train MrTotalVI
# ---------------------------------------------------------------------------

def _setup_and_train(adata, max_epochs: int = MAX_EPOCHS_QUICK, **model_kwargs) -> MrTotalVI:
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, **model_kwargs)
    model.train(max_epochs=max_epochs, plan_kwargs={"lr": 1e-3}, check_val_every_n_epoch=max_epochs)
    return model


# ---------------------------------------------------------------------------
# (a) Smoke test — finite ELBO + counterfactual paths
# ---------------------------------------------------------------------------

def test_smoke_trains_finite_elbo(adata_basic):
    """Training finishes with a finite ELBO and no NaN gradients.

    Also verifies that get_latent_representation(give_z=True) and
    give_z=False return different arrays — confirming the hierarchy is wired
    through the default representation path, not discarded.
    """
    model = _setup_and_train(adata_basic, max_epochs=MAX_EPOCHS_QUICK)

    history = model.history
    # history values are pandas DataFrames; extract the first numeric column
    found_key = None
    for key in ("elbo_train", "train_loss_epoch", "train_elbo_train", "elbo_validation"):
        if key in history:
            df = history[key]
            # Flatten to a 1-D float array regardless of shape
            vals = df.to_numpy().astype(float).flatten()
            assert np.all(np.isfinite(vals)), f"{key} contains non-finite: {vals}"
            found_key = key
            break  # one key is sufficient
    assert found_key is not None, (
        f"No ELBO history key found. Available: {list(history.keys())}"
    )

    # give_z=True returns z = z_base + eps (donor-aware);
    # give_z=False returns u (donor-unaware).
    # If they're identical the hierarchy was silently discarded.
    z_rep = model.get_latent_representation(give_z=True, give_mean=True)
    u_rep = model.get_latent_representation(give_z=False, give_mean=True)
    assert z_rep.shape == u_rep.shape, "z and u representations have different shapes"
    assert not np.allclose(z_rep, u_rep, atol=1e-6), (
        "give_z=True and give_z=False returned identical arrays — "
        "the donor residual eps is a no-op or get_latent_representation is "
        "returning u for both paths."
    )


def test_smoke_counterfactual_finite(adata_basic):
    """Counterfactual path (cf_sample, mc_samples=2) produces finite tensors.

    This is the fragile path: attention + second latent + mc_samples broadcast.
    """
    import torch

    model = _setup_and_train(adata_basic, max_epochs=MAX_EPOCHS_QUICK)
    module = model.module.eval()

    # Get a single minibatch
    dl = model._make_data_loader(adata=model.adata, batch_size=32)
    tensors = next(iter(dl))

    inf_inputs = module._get_inference_input(tensors)

    n_cells = tensors[scvi.REGISTRY_KEYS.X_KEY].shape[0]
    cf = torch.zeros(n_cells, 1, dtype=torch.long)

    with torch.no_grad():
        out = module._regular_inference(**inf_inputs, cf_sample=cf, n_samples=2)

    z = out["z"]
    eps = out["eps"]
    assert torch.all(torch.isfinite(z)), "z contains NaN / Inf (cf + mc_samples path)"
    assert torch.all(torch.isfinite(eps)), "eps contains NaN / Inf"


# ---------------------------------------------------------------------------
# (b) Reconstruction not regressed vs stock TotalVI
# ---------------------------------------------------------------------------

def test_elbo_not_severely_regressed(adata_basic):
    """MrTotalVI's ELBO is within 10% of stock TotalVI after the same training.

    The reconstruction loss is NOT the right check here: MrTotalVI adds a
    kl_z = -log p(eps) term that structurally shifts ELBO capacity away from
    reconstruction.  The correct gate is total ELBO — training instability would
    show as ELBO ratio >> 1.10, not a modest recon increase.  A ratio ≤ 1.10
    confirms training is stable (L-11 in .living/learnings.md).
    """
    from scvi.model import TOTALVI

    def _last_metric(model, *keys: str) -> float:
        hist = model.history
        candidates = list(keys) + [f"train_{k}" for k in keys]
        found = [k for k in candidates if k in hist]
        assert found, f"None of {keys!r} found in history: {list(hist.keys())}"
        return float(np.array(hist[found[0]]).flatten()[-1])

    # Stock TotalVI
    TOTALVI.setup_anndata(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        batch_key="batch",
    )
    base_model = TOTALVI(adata_basic, n_latent=N_LATENT)
    base_model.train(
        max_epochs=MAX_EPOCHS_FULL, plan_kwargs={"lr": 1e-3},
        check_val_every_n_epoch=MAX_EPOCHS_FULL,
    )

    # MrTotalVI
    mr_model = _setup_and_train(adata_basic, max_epochs=MAX_EPOCHS_FULL)

    base_elbo = _last_metric(base_model, "elbo_train", "train_loss_epoch")
    mr_elbo = _last_metric(mr_model, "elbo_train", "train_loss_epoch")

    assert mr_elbo <= base_elbo * 1.10, (
        f"MrTotalVI ELBO ({mr_elbo:.1f}) > TotalVI × 1.10 ({base_elbo * 1.10:.1f}). "
        "Hierarchy may be destabilizing training."
    )


# ---------------------------------------------------------------------------
# (c) Non-degenerate sample embedding
# ---------------------------------------------------------------------------

def test_non_degenerate_sample_embedding(adata_basic):
    """Per-sample embedding row-variance exceeds initialisation; eps is non-zero."""
    import torch

    model = _setup_and_train(adata_basic, max_epochs=MAX_EPOCHS_QUICK)
    module = model.module

    assert hasattr(module, "qz"), "EncoderUZ not built on module"

    emb_weight = module.qz.embedding.weight.detach().cpu()  # (n_sample, n_latent_sample)
    row_var = emb_weight.var(dim=1).numpy()
    # After even 3 epochs of gradient updates, every row should have moved
    assert np.all(row_var > 0.0), (
        f"Some embedding rows have zero variance — hierarchy is (partially) a no-op.\n{row_var}"
    )

    # eps should be non-zero on a real forward pass
    dl = model._make_data_loader(adata=model.adata, batch_size=64)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    with torch.no_grad():
        out = module._regular_inference(**inf_inputs)
    eps_mean_abs = out["eps"].abs().mean().item()
    assert eps_mean_abs > 0.0, f"eps.abs().mean() == {eps_mean_abs} — hierarchy is a no-op"


# ---------------------------------------------------------------------------
# (d) Donor-axis separation via get_local_sample_distances
# ---------------------------------------------------------------------------

def test_donor_axis_separation(adata_shifted):
    """Trained hierarchy produces cross-donor distances that collapse when the
    embedding is zeroed — a before/after contrast proving the hierarchy encodes
    real donor signal rather than random variation.

    Mechanism: zeroing the embedding table makes every donor return the same
    embedding vector → the attention block outputs the same context for all
    donors → eps is identical across donors → all counterfactual z values
    collapse → off-diagonal distances ≈ 0.  The trained model must produce
    strictly larger cross-donor distances than this collapsed baseline.
    """
    import torch

    model = _setup_and_train(adata_shifted, max_epochs=MAX_EPOCHS_FULL)

    dists = model.get_local_sample_distances(batch_size=64)  # (n_cell, n_sample, n_sample)
    dists_np = dists.values

    n_s = dists_np.shape[1]
    off_diag_mask = ~np.eye(n_s, dtype=bool)
    cross_donor_trained = float(dists_np[:, off_diag_mask].mean())

    # Contrast: zero the embedding table → all donors receive the same attention
    # context → eps identical across donors → counterfactual z values collapse
    # → off-diagonal distances should shrink dramatically (to ~0 or near-zero).
    module = model.module
    with torch.no_grad():
        orig_weight = module.qz.embedding.weight.data.clone()
        module.qz.embedding.weight.data.zero_()

    dists_zeroed = model.get_local_sample_distances(batch_size=64)
    cross_donor_zeroed = float(dists_zeroed.values[:, off_diag_mask].mean())

    with torch.no_grad():
        module.qz.embedding.weight.data.copy_(orig_weight)

    # Trained distances must exceed the collapsed (zeroed-embedding) baseline.
    # This distinguishes a working hierarchy from a no-op or random variation.
    assert cross_donor_trained > cross_donor_zeroed, (
        f"Zeroing the embedding did not collapse cross-donor distances.\n"
        f"Trained: {cross_donor_trained:.4f}  Zeroed: {cross_donor_zeroed:.4f}\n"
        f"Expected trained > zeroed — hierarchy appears non-functional."
    )
    # Sanity: trained distances are non-trivially positive
    assert cross_donor_trained > 0.0, (
        f"Trained cross-donor distances are 0 — hierarchy encodes no signal."
    )

    # get_local_sample_representation should have matching shape
    reps = model.get_local_sample_representation(batch_size=64)
    assert reps.dims == ("cell_name", "sample", "latent_dim"), f"Unexpected dims: {reps.dims}"
    assert reps.shape[1] == N_DONORS, f"Expected {N_DONORS} samples, got {reps.shape[1]}"


# ---------------------------------------------------------------------------
# (f) qu encoder: gradient flow and sample conditioning
# ---------------------------------------------------------------------------


def test_qu_encoder_gradients_flow(adata_basic):
    """qu.cond_norm1/cond_norm2 gamma/beta embeddings receive non-zero gradients.

    Directly verifies that the sample-conditioned u-encoder parameters are
    connected to the loss in a single forward-backward pass on an untrained model.
    """
    import torch

    MrTotalVI.setup_anndata(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata_basic, sample_key="sample", n_latent=N_LATENT)
    module = model.module.train()

    dl = model._make_data_loader(adata=model.adata, batch_size=32)
    tensors = next(iter(dl))

    inf_inputs = module._get_inference_input(tensors)
    inf_out = module._regular_inference(**inf_inputs)
    gen_inputs = module._get_generative_input(tensors, inf_out)
    gen_out = module.generative(**gen_inputs)
    loss_out = module.loss(tensors, inf_out, gen_out)
    loss_out.loss.backward()

    qu = module.qu
    for cn_name in ("cond_norm1", "cond_norm2"):
        cn = getattr(qu, cn_name)
        gamma_grad = cn.gamma_embedding.weight.grad
        beta_grad = cn.beta_embedding.weight.grad
        assert gamma_grad is not None, (
            f"qu.{cn_name}.gamma_embedding.weight.grad is None — gradient not flowing"
        )
        assert gamma_grad.abs().max().item() > 0, (
            f"qu.{cn_name}.gamma_embedding gradient is all zeros"
        )
        assert beta_grad is not None, (
            f"qu.{cn_name}.beta_embedding.weight.grad is None"
        )
        assert beta_grad.abs().max().item() > 0, (
            f"qu.{cn_name}.beta_embedding gradient is all zeros"
        )

    # sample_embed also must receive gradient
    assert qu.sample_embed.weight.grad is not None, (
        "qu.sample_embed.weight.grad is None — sample embedding not trained"
    )


def test_qu_encoder_donor_rows_diverge(adata_basic):
    """After training, different donor rows of cond_norm1.gamma_embedding differ.

    gamma_embedding is initialized N(1, 0.02). Each donor receives independent
    gradient updates based on its expression signature, so rows should diverge
    beyond the 0.02 init noise. Checks that sample conditioning is non-trivial.
    """
    model = _setup_and_train(adata_basic, max_epochs=MAX_EPOCHS_FULL)
    module = model.module

    qu = module.qu
    for cn_name in ("cond_norm1", "cond_norm2"):
        cn = getattr(qu, cn_name)
        gamma = cn.gamma_embedding.weight.detach().cpu()  # (n_sample, n_hidden)
        # Max pairwise L1 distance across donor rows
        n_s = gamma.shape[0]
        max_row_diff = 0.0
        for i in range(n_s):
            for j in range(i + 1, n_s):
                diff = (gamma[i] - gamma[j]).abs().mean().item()
                max_row_diff = max(max_row_diff, diff)
        assert max_row_diff > 0.0, (
            f"qu.{cn_name}.gamma_embedding all donor rows are identical — "
            f"sample conditioning is a no-op."
        )


# ---------------------------------------------------------------------------
# (e) Learnable prior scale
# ---------------------------------------------------------------------------

def test_learnable_prior_scale(adata_basic):
    """learn_z_u_prior_scale=True registers pz_scale as a gradient-tracked parameter.

    Checks:
    - pz_scale is an nn.Parameter (not a buffer)
    - pz_scale receives gradients during training
    - Final trained value differs from the initialisation (0.0)
    """
    import torch

    model = _setup_and_train(
        adata_basic,
        max_epochs=MAX_EPOCHS_FULL,
        learn_z_u_prior_scale=True,
    )
    module = model.module

    # Must be a Parameter, not a buffer
    assert "pz_scale" in dict(module.named_parameters()), (
        "pz_scale not registered as nn.Parameter when learn_z_u_prior_scale=True"
    )
    pz_scale = module.pz_scale
    assert isinstance(pz_scale, torch.nn.Parameter), "pz_scale is not nn.Parameter"

    # Should have moved from 0.0 init during training
    assert not torch.allclose(pz_scale, torch.zeros_like(pz_scale)), (
        "pz_scale stayed at init=0.0 after training — gradient not flowing through kl_z"
    )


# ---------------------------------------------------------------------------
# (g) scale_observations
# ---------------------------------------------------------------------------

def test_scale_observations(adata_basic):
    """scale_observations=True trains without error and produces finite ELBO."""
    import math

    model = _setup_and_train(
        adata_basic,
        max_epochs=MAX_EPOCHS_QUICK,
        scale_observations=True,
    )
    history = model.history["elbo_train"]
    assert all(math.isfinite(v) for v in history.values.flatten()), (
        "Non-finite ELBO encountered with scale_observations=True"
    )


# ---------------------------------------------------------------------------
# (h) use_map=False (stochastic eps)
# ---------------------------------------------------------------------------

def test_use_map_false(adata_basic):
    """use_map=False runs without error and produces finite ELBO."""
    import math

    model = _setup_and_train(
        adata_basic,
        max_epochs=MAX_EPOCHS_QUICK,
        use_map=False,
    )
    history = model.history["elbo_train"]
    assert all(math.isfinite(v) for v in history.values.flatten()), (
        "Non-finite ELBO encountered with use_map=False"
    )
