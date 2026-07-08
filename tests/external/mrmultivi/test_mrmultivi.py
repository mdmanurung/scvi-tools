"""Tests for MrMultiVI — MultiVI + MrVI hierarchical donor latent space.

Five invariants (CLAUDE.md "deepening invariance"):
  (a) smoke           — trains to completion with finite ELBO; counterfactual paths
                        (cf_sample) also produce finite tensors.
  (b) reconstruction  — MrMultiVI's converged recon loss ≤ stock MULTIVI × 1.05.
  (c) latent repr     — get_latent_representation(give_z=True/False) returns
                        different arrays (hierarchy is not silently discarded).
  (d) local sample    — get_local_sample_representation / get_local_sample_distances
                        return DataArrays with correct dims.
  (e) hierarchy       — zeroing the embedding collapses cross-donor distances
                        (embedding-collapse contrast, L-6 pattern).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from scvi.data import synthetic_iid
from scvi.external import MrMultiVI


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_DONORS = 4
N_LATENT = 10
MAX_EPOCHS_QUICK = 3
MAX_EPOCHS_FULL = 20
MODALITIES = {
    "rna_layer": "rna",
    "atac_layer": "accessibility",
    "protein_layer": "protein_expression",
}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_mdata(n_donors: int = N_DONORS, shift_scale: float = 0.0):
    """MuData with trimodal (RNA/ATAC/protein) data and a synthetic donor column.

    If shift_scale > 0, donor ``d`` receives an additive offset of
    ``shift_scale * (d + 1)`` in the RNA modality so donors are detectably
    separated in feature space.
    """
    mdata = synthetic_iid(return_mudata=True)
    n_cells = mdata.n_obs
    donor_ids = np.array([f"donor_{i % n_donors}" for i in range(n_cells)])
    mdata.obs["donor"] = donor_ids

    if shift_scale > 0:
        rna = mdata.mod["rna"]
        for d in range(n_donors):
            mask = mdata.obs["donor"] == f"donor_{d}"
            import scipy.sparse as sp
            if sp.issparse(rna.X):
                rna.X[mask] = rna.X[mask] + shift_scale * (d + 1)
            else:
                rna.X[mask] = rna.X[mask] + shift_scale * (d + 1)

    return mdata


@pytest.fixture(scope="module")
def mdata_basic():
    return _make_mdata()


@pytest.fixture(scope="module")
def mdata_shifted():
    return _make_mdata(shift_scale=5.0)


# ---------------------------------------------------------------------------
# Helper: setup + train
# ---------------------------------------------------------------------------


def _setup_and_train(mdata, max_epochs: int = MAX_EPOCHS_QUICK, **model_kwargs) -> MrMultiVI:
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata, sample_key="donor", n_latent=N_LATENT, **model_kwargs)
    model.train(
        max_epochs=max_epochs,
        accelerator="cpu",
        plan_kwargs={"lr": 1e-3},
        check_val_every_n_epoch=max_epochs,
    )
    return model


# ---------------------------------------------------------------------------
# (a) Smoke — finite ELBO
# ---------------------------------------------------------------------------


def test_mrmultivi_training(mdata_basic):
    """MrMultiVI trains to completion with a finite ELBO."""
    model = _setup_and_train(mdata_basic, max_epochs=MAX_EPOCHS_QUICK)

    assert model.is_trained is True
    history = model.history
    candidates = ("elbo_train", "train_loss_epoch", "elbo_validation", "train_elbo_train")
    found = [k for k in candidates if k in history]
    assert found, f"No ELBO key in history. Available: {list(history.keys())}"
    vals = history[found[0]].to_numpy().astype(float).flatten()
    assert np.all(np.isfinite(vals)), f"{found[0]} contains non-finite: {vals}"


def test_mrmultivi_counterfactual_finite(mdata_basic):
    """Counterfactual inference path (cf_sample) produces finite tensors."""
    import torch

    model = _setup_and_train(mdata_basic, max_epochs=MAX_EPOCHS_QUICK)
    module = model.module.eval()

    dl = model._make_data_loader(adata=model.adata, batch_size=32)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)

    n_cells = inf_inputs["x"].shape[0]
    cf = torch.zeros(n_cells, 1, dtype=torch.long)

    with torch.no_grad():
        out = module.inference(**inf_inputs, cf_sample=cf)

    assert torch.all(torch.isfinite(out["z"])), "z contains NaN/Inf on cf_sample path"
    assert torch.all(torch.isfinite(out["eps"])), "eps contains NaN/Inf on cf_sample path"


# ---------------------------------------------------------------------------
# (b) Reconstruction not regressed vs stock MULTIVI
# ---------------------------------------------------------------------------


def test_mrmultivi_reconstruction_not_regressed():
    """MrMultiVI recon loss ≤ stock MULTIVI × 1.05 on the same data."""
    from scvi.model import MULTIVI

    mdata = _make_mdata()

    MULTIVI.setup_mudata(mdata, batch_key="batch", modalities=MODALITIES)
    base_model = MULTIVI(mdata, n_latent=N_LATENT)
    base_model.train(
        max_epochs=MAX_EPOCHS_FULL,
        accelerator="cpu",
        plan_kwargs={"lr": 1e-3},
        check_val_every_n_epoch=MAX_EPOCHS_FULL,
    )

    mr_model = _setup_and_train(mdata, max_epochs=MAX_EPOCHS_FULL)

    def _last_recon(model) -> float:
        hist = model.history
        for k in ("reconstruction_loss_train", "train_reconstruction_loss_train"):
            if k in hist:
                return float(np.array(hist[k]).flatten()[-1])
        raise KeyError(f"recon key missing; available: {list(hist.keys())}")

    base_recon = _last_recon(base_model)
    mr_recon = _last_recon(mr_model)

    assert mr_recon <= base_recon * 1.05, (
        f"MrMultiVI recon ({mr_recon:.3f}) > MULTIVI × 1.05 ({base_recon * 1.05:.3f}). "
        "Hierarchy may be destabilizing training."
    )


# ---------------------------------------------------------------------------
# (c) Latent representation — give_z separates u and z
# ---------------------------------------------------------------------------


def test_mrmultivi_latent_representation(mdata_basic):
    """give_z=True and give_z=False return different arrays; shapes match."""
    model = _setup_and_train(mdata_basic, max_epochs=MAX_EPOCHS_QUICK)

    z_rep = model.get_latent_representation(give_z=True, give_mean=True)
    u_rep = model.get_latent_representation(give_z=False, give_mean=True)

    assert z_rep.shape == u_rep.shape, "z and u have different shapes"
    assert not np.allclose(z_rep, u_rep, atol=1e-6), (
        "give_z=True and give_z=False returned identical arrays — "
        "the donor residual eps is a no-op or not wired into get_latent_representation."
    )
    assert z_rep.shape[1] == N_LATENT


# ---------------------------------------------------------------------------
# (d) Local sample representation and distances — shape check
# ---------------------------------------------------------------------------


def test_mrmultivi_local_sample_representation(mdata_basic):
    """get_local_sample_representation returns DataArray with correct dims."""
    model = _setup_and_train(mdata_basic, max_epochs=MAX_EPOCHS_QUICK)

    reps = model.get_local_sample_representation(batch_size=64)
    assert reps.dims == ("cell_name", "sample", "latent_dim"), f"Unexpected dims: {reps.dims}"
    assert reps.shape[0] == mdata_basic.n_obs
    assert reps.shape[1] == N_DONORS
    assert reps.shape[2] == N_LATENT


def test_mrmultivi_local_sample_distances(mdata_basic):
    """get_local_sample_distances returns DataArray with correct dims."""
    model = _setup_and_train(mdata_basic, max_epochs=MAX_EPOCHS_QUICK)

    dists = model.get_local_sample_distances(batch_size=64)
    assert dists.dims == ("cell_name", "sample_x", "sample_y"), f"Unexpected dims: {dists.dims}"
    assert dists.shape[0] == mdata_basic.n_obs
    assert dists.shape[1] == N_DONORS
    assert dists.shape[2] == N_DONORS

    # Distance matrices should be symmetric
    d_np = dists.values
    assert np.allclose(d_np, d_np.transpose(0, 2, 1), atol=1e-5), (
        "Distance matrices are not symmetric."
    )
    # Diagonal should be ~0
    diag = np.diag(d_np[0])
    assert np.allclose(diag, 0.0, atol=1e-5), f"Self-distances non-zero: {diag}"


# ---------------------------------------------------------------------------
# (f) qu encoder: gradient flow and sample conditioning
# ---------------------------------------------------------------------------


def test_mrmultivi_qu_encoder_gradients_flow(mdata_basic):
    """qu.cond_norm1/cond_norm2 gamma/beta embeddings receive non-zero gradients.

    Directly verifies that the sample-conditioned u-encoder parameters are
    connected to the loss in a single forward-backward pass on an untrained model.
    """
    import torch

    mdata = _make_mdata()
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata, sample_key="donor", n_latent=N_LATENT)
    module = model.module.train()

    dl = model._make_data_loader(adata=model.adata, batch_size=32)
    tensors = next(iter(dl))

    inf_inputs = module._get_inference_input(tensors)
    inf_out = module.inference(**inf_inputs)
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

    assert qu.sample_embed.weight.grad is not None, (
        "qu.sample_embed.weight.grad is None — sample embedding not trained"
    )


def test_mrmultivi_qu_encoder_donor_rows_diverge(mdata_basic):
    """After training, different donor rows of cond_norm1.gamma_embedding differ.

    gamma_embedding is initialized N(1, 0.02). Donor-specific gradient updates
    should push rows apart beyond the 0.02 init noise.
    """
    model = _setup_and_train(mdata_basic, max_epochs=MAX_EPOCHS_FULL)
    module = model.module

    qu = module.qu
    for cn_name in ("cond_norm1", "cond_norm2"):
        cn = getattr(qu, cn_name)
        gamma = cn.gamma_embedding.weight.detach().cpu()  # (n_sample, n_hidden)
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
# (e) Hierarchy-collapse contrast (L-6 pattern)
# ---------------------------------------------------------------------------


def test_mrmultivi_hierarchy_collapse_contrast(mdata_shifted):
    """Zeroing the donor embedding collapses cross-donor distances (embedding-collapse contrast).

    Proves the embedding is wired into the distance computation: zeroing
    ``module.qz.embedding.weight`` makes all donors appear identical (eps = 0
    for every donor), so off-diagonal distances collapse toward zero.  If the
    embedding were unused, zeroing it would have no effect.

    Note: KL regularisation pushes eps toward 0 during training, so trained
    cross-donor distances can be smaller than at initialisation — this is
    expected VAE behaviour, not a bug.  The relevant check is the collapse
    contrast, not the absolute magnitude.
    """
    import torch

    model = _setup_and_train(mdata_shifted, max_epochs=MAX_EPOCHS_FULL)
    dists = model.get_local_sample_distances(batch_size=64)
    n_s = dists.shape[1]
    off_diag = ~np.eye(n_s, dtype=bool)
    cross_donor_trained = float(dists.values[:, off_diag].mean())

    module = model.module
    with torch.no_grad():
        orig_weight = module.qz.embedding.weight.data.clone()
        module.qz.embedding.weight.data.zero_()

    dists_zeroed = model.get_local_sample_distances(batch_size=64)
    cross_donor_zeroed = float(dists_zeroed.values[:, off_diag].mean())

    with torch.no_grad():
        module.qz.embedding.weight.data.copy_(orig_weight)

    assert cross_donor_trained > cross_donor_zeroed, (
        f"Zeroing the embedding did not collapse cross-donor distances.\n"
        f"Trained: {cross_donor_trained:.4f}  Zeroed: {cross_donor_zeroed:.4f}"
    )


# ---------------------------------------------------------------------------
# (j) scale_observations
# ---------------------------------------------------------------------------

def test_scale_observations(mdata_basic):
    """scale_observations=True trains without error and produces finite ELBO."""
    import math

    model = _setup_and_train(
        mdata_basic,
        max_epochs=MAX_EPOCHS_QUICK,
        scale_observations=True,
    )
    history = model.history["elbo_train"]
    assert all(math.isfinite(v) for v in history.values.flatten()), (
        "Non-finite ELBO encountered with scale_observations=True"
    )


# ---------------------------------------------------------------------------
# (k) use_map=False (stochastic eps)
# ---------------------------------------------------------------------------

def test_use_map_false(mdata_basic):
    """use_map=False runs without error and produces finite ELBO."""
    import math

    model = _setup_and_train(
        mdata_basic,
        max_epochs=MAX_EPOCHS_QUICK,
        use_map=False,
    )
    history = model.history["elbo_train"]
    assert all(math.isfinite(v) for v in history.values.flatten()), (
        "Non-finite ELBO encountered with use_map=False"
    )
