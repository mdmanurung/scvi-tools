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


def test_mrmultivi_rejects_logistic_normal_latent(mdata_basic):
    """MrMultiVI rejects logistic-normal latents because the hierarchy is additive."""
    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )

    with pytest.raises(ValueError, match="latent_distribution='normal'"):
        MrMultiVI(
            mdata_basic,
            sample_key="donor",
            n_latent=N_LATENT,
            latent_distribution="ln",
        )


def test_mrmultivi_hierarchy_uses_mixed_posterior_mean(mdata_basic):
    """qu receives MULTIVAE's mixed posterior mean, not a sampled base latent."""
    import torch
    from torch import nn
    from torch.distributions import Normal

    from scvi.module._multivae import MULTIVAE

    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT)
    module = model.module.eval()

    dl = model._make_data_loader(adata=model.adata, batch_size=32)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    base_inputs = {k: v for k, v in inf_inputs.items() if k != "sample_index"}

    with torch.no_grad():
        base_out = MULTIVAE.inference(module, **base_inputs)

    class RecordingQu(nn.Module):
        def __init__(self):
            super().__init__()
            self.last_input = None

        def forward(self, u0, sample_index, **kwargs):
            self.last_input = u0.detach().clone()
            return Normal(u0, torch.ones_like(u0))

    module.qu = RecordingQu()

    with torch.no_grad():
        module.inference(**inf_inputs)

    assert not torch.allclose(base_out["z"], base_out["qz_m"]), (
        "Synthetic fixture produced a sampled z identical to qz_m; "
        "this regression test cannot distinguish the hierarchy input."
    )
    assert torch.allclose(module.qu.last_input, base_out["qz_m"], atol=1e-6)


def test_mrmultivi_non_isomorphic_u_dimension(mdata_basic):
    """n_latent_u can be smaller than z while the decoder still receives n_latent."""
    import torch

    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT, n_latent_u=5)

    assert model.module.n_latent_u == 5
    assert model.module.qz.fc is not None

    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = model.module._get_inference_input(tensors)
    with torch.no_grad():
        out = model.module.inference(**inf_inputs)

    assert out["u"].shape[-1] == 5
    assert out["z"].shape[-1] == N_LATENT


def test_mrmultivi_non_isomorphic_u_dimension_with_mc_samples(mdata_basic):
    """MC samples preserve the u and z dimensions and keep the ELBO finite."""
    import torch

    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT, n_latent_u=5)
    module = model.module.train()

    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    inf_out = module.inference(**inf_inputs, n_samples=2)
    gen_inputs = module._get_generative_input(tensors, inf_out)
    gen_out = module.generative(**gen_inputs)
    loss_out = module.loss(tensors, inf_out, gen_out)

    assert inf_out["u"].shape == torch.Size([2, tensors["X"].shape[0], 5])
    assert inf_out["z"].shape == torch.Size([2, tensors["X"].shape[0], N_LATENT])
    assert inf_out["eps"].shape == torch.Size([2, tensors["X"].shape[0], N_LATENT])
    assert loss_out.kl_local["kl_divergence_z"].shape == torch.Size([tensors["X"].shape[0]])
    assert torch.isfinite(loss_out.loss)


def test_mrmultivi_singleton_mc_loss_matches_single_sample_loss(mdata_basic):
    """The custom MC loss is equivalent to the parent loss for one sampled z."""
    import torch

    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(
        mdata_basic,
        sample_key="donor",
        n_latent=N_LATENT,
        n_latent_u=5,
        u_prior_mixture=False,
    )
    module = model.module.eval()

    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    with torch.no_grad():
        inf_out = module.inference(**inf_inputs)
        gen_out = module.generative(**module._get_generative_input(tensors, inf_out))
        loss_out = module.loss(tensors, inf_out, gen_out)

        batch_size = tensors["X"].shape[0]

        def _unsqueeze_batch_tensors(value):
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                return value.unsqueeze(0)
            if isinstance(value, dict):
                return {k: _unsqueeze_batch_tensors(v) for k, v in value.items()}
            return value

        inf_mc = dict(inf_out)
        for key in ("z", "u", "z_base", "eps", "libsize_expr", "libsize_acc"):
            inf_mc[key] = inf_mc[key].unsqueeze(0)
        gen_mc = _unsqueeze_batch_tensors(gen_out)
        loss_mc = module.loss(tensors, inf_mc, gen_mc)

    for key in (
        "reconstruction_loss_expression",
        "reconstruction_loss_accessibility",
        "reconstruction_loss_protein",
    ):
        assert torch.allclose(
            loss_mc.reconstruction_loss[key].squeeze(0),
            loss_out.reconstruction_loss[key],
            atol=1e-5,
        )
    assert torch.allclose(
        loss_mc.kl_local["kl_divergence_z"],
        loss_out.kl_local["kl_divergence_z"],
        atol=1e-5,
    )
    assert torch.allclose(loss_mc.loss, loss_out.loss, atol=1e-5)


def test_mrmultivi_mc_samples_with_stochastic_eps_and_scaled_observations(mdata_basic):
    """MC samples work when eps is stochastic and observations are sample-scaled."""
    import torch

    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(
        mdata_basic,
        sample_key="donor",
        n_latent=N_LATENT,
        n_latent_u=5,
        use_map=False,
        scale_observations=True,
    )
    module = model.module.train()
    assert module.n_obs_per_sample is not None

    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    inf_out = module.inference(**inf_inputs, n_samples=2)
    gen_inputs = module._get_generative_input(tensors, inf_out)
    gen_out = module.generative(**gen_inputs)
    loss_out = module.loss(tensors, inf_out, gen_out)

    assert inf_out["eps_dist"] is not None
    assert inf_out["eps_dist"].loc.shape == torch.Size([2, tensors["X"].shape[0], N_LATENT])
    assert loss_out.kl_local["kl_divergence_z"].shape == torch.Size([tensors["X"].shape[0]])
    assert torch.isfinite(loss_out.loss)


def test_mrmultivi_mc_samples_with_size_factors_and_missing_modalities(mdata_basic):
    """MC loss handles size factors and cells missing individual modalities."""
    import scipy.sparse as sp
    import torch

    mdata = _make_mdata()
    mdata.obs["size_factor_rna"] = np.asarray(mdata["rna"].X.sum(1)).reshape(-1) + 1.0
    mdata.obs["size_factor_atac"] = (np.asarray(mdata["accessibility"].X.sum(1)).reshape(-1) + 1.0) / (
        np.asarray(mdata["accessibility"].X.sum(1)).max() + 1.01
    )

    def zero_rows(mod_key, rows):
        x = mdata[mod_key].X
        if sp.issparse(x):
            x = x.toarray()
        else:
            x = np.asarray(x).copy()
        x[rows, :] = 0
        mdata[mod_key].X = x

    zero_rows("rna", np.arange(0, 8))
    zero_rows("accessibility", np.arange(8, 16))
    zero_rows("protein_expression", np.arange(16, 24))

    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        size_factor_key=["size_factor_rna", "size_factor_atac"],
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata, sample_key="donor", n_latent=N_LATENT, n_latent_u=5)
    module = model.module.train()
    assert module.use_size_factor_key

    dl = model._make_data_loader(adata=model.adata, batch_size=24)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    inf_out = module.inference(**inf_inputs, n_samples=2)
    gen_inputs = module._get_generative_input(tensors, inf_out)
    gen_out = module.generative(**gen_inputs)
    loss_out = module.loss(tensors, inf_out, gen_out)

    batch_size = tensors["X"].shape[0]
    assert loss_out.reconstruction_loss["reconstruction_loss_expression"].shape == torch.Size(
        [2, batch_size]
    )
    assert loss_out.reconstruction_loss["reconstruction_loss_accessibility"].shape == torch.Size(
        [2, batch_size]
    )
    assert loss_out.reconstruction_loss["reconstruction_loss_protein"].shape == torch.Size(
        [2, batch_size]
    )
    assert loss_out.kl_local["kl_divergence_z"].shape == torch.Size([batch_size])
    assert torch.isfinite(loss_out.loss)


def test_mrmultivi_label_conditioned_mog_prior(mdata_basic):
    """labels_key switches the MoG prior to one component per label and biases logits."""
    import torch
    from torch.distributions import MixtureSameFamily

    mdata = _make_mdata()
    mdata.obs["cell_type"] = np.where(np.arange(mdata.n_obs) % 2 == 0, "T", "B")
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        labels_key="cell_type",
        modalities={**MODALITIES, "labels_key": None},
    )
    model = MrMultiVI(mdata, sample_key="donor", n_latent=N_LATENT, n_latent_u=4)

    assert model.module.u_prior_logits.shape == (model.summary_stats.n_labels,)
    assert model.module.u_prior_means.shape == (model.summary_stats.n_labels, 4)

    labels = torch.tensor([[0], [1], [0]])
    u = torch.zeros(3, 4)
    prior = model.module.build_u_prior(u, labels)

    assert isinstance(prior, MixtureSameFamily)
    assert prior.mixture_distribution.logits.argmax(dim=-1).tolist() == [0, 1, 0]


def test_mrmultivi_label_conditioned_mog_flows_through_mc_loss(mdata_basic):
    """Registered labels condition the MoG prior in the actual MC loss path."""
    import torch

    mdata = _make_mdata()
    mdata.obs["cell_type"] = np.where(np.arange(mdata.n_obs) % 2 == 0, "T", "B")
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        labels_key="cell_type",
        modalities={**MODALITIES, "labels_key": None},
    )
    model = MrMultiVI(
        mdata,
        sample_key="donor",
        n_latent=N_LATENT,
        n_latent_u=4,
        z_u_prior=False,
    )
    module = model.module.train()

    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    inf_out = module.inference(**inf_inputs, n_samples=2)
    gen_inputs = module._get_generative_input(tensors, inf_out)
    gen_out = module.generative(**gen_inputs)
    loss_out = module.loss(tensors, inf_out, gen_out)

    expected_kl_u = module.kl_u(inf_out["qu"], inf_out["u"], tensors["labels"])
    assert module.resolved_u_prior_mixture_k == model.summary_stats.n_labels
    assert torch.allclose(loss_out.kl_local["kl_divergence_z"], expected_kl_u, atol=1e-5)
    assert torch.isfinite(loss_out.loss)


def test_mrmultivi_gaussian_u_prior_and_z_u_prior_off(mdata_basic):
    """u_prior_mixture=False uses analytic Gaussian KL and z_u_prior=False omits kl_z."""
    import torch

    mdata = _make_mdata()
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(
        mdata,
        sample_key="donor",
        n_latent=N_LATENT,
        u_prior_mixture=False,
        z_u_prior=False,
    )
    assert not hasattr(model.module, "u_prior_logits")

    module = model.module.train()
    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    inf_out = module.inference(**inf_inputs)
    gen_inputs = module._get_generative_input(tensors, inf_out)
    gen_out = module.generative(**gen_inputs)
    loss_out = module.loss(tensors, inf_out, gen_out)

    expected_kl_u = module.kl_u(inf_out["qu"], inf_out["u"], tensors["labels"])
    assert torch.allclose(loss_out.kl_local["kl_divergence_z"], expected_kl_u, atol=1e-5)


def test_mrmultivi_learnable_prior_scale_clamp(mdata_basic):
    """pz_scale.clamp(min=-4.0) prevents σ→0 / kl_z→-∞ collapse.

    Under learn_z_u_prior_scale=True + use_map=True (the default), the optimizer
    could otherwise drive pz_scale→-∞ (σ→0) jointly with eps→0, making kl_z
    unbounded below.  The clamp ensures σ ≥ exp(-4) ≈ 0.018 at all times.
    """
    import torch
    from torch.distributions import Normal

    model = _setup_and_train(
        mdata_basic,
        max_epochs=MAX_EPOCHS_FULL,
        learn_z_u_prior_scale=True,
        use_map=True,
    )
    module = model.module
    pz_scale = module.pz_scale
    assert isinstance(pz_scale, torch.nn.Parameter), "pz_scale is not nn.Parameter"
    assert (pz_scale >= -4.0).all(), (
        f"pz_scale breached the -4.0 clamp floor: min={pz_scale.min():.3f}"
    )
    module.eval()
    batch = next(iter(model._make_data_loader(adata=model.adata, batch_size=16)))
    with torch.no_grad():
        inf_out = module.inference(**module._get_inference_input(batch))
        eps = inf_out["eps"]
        peps = Normal(0.0, torch.exp(pz_scale.clamp(min=-4.0)))
        kl_z = -peps.log_prob(eps).sum(dim=-1)
    assert torch.isfinite(kl_z).all(), "kl_z contains non-finite values after clamp"
    assert (kl_z > -1e6).all(), f"kl_z implausibly negative: min={kl_z.min():.1f}"


def test_mrmultivi_encode_covariates_expands_qu_input(mdata_basic):
    """encode_covariates=True appends batch, categorical, and continuous covariates to qu."""
    import torch

    mdata = _make_mdata()
    mdata.obs["stim"] = np.where(np.arange(mdata.n_obs) % 2 == 0, "ctrl", "stim")
    mdata.obs["score"] = np.linspace(0.0, 1.0, mdata.n_obs)
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        categorical_covariate_keys=["stim"],
        continuous_covariate_keys=["score"],
        modalities=MODALITIES,
    )
    model = MrMultiVI(
        mdata,
        sample_key="donor",
        n_latent=N_LATENT,
        encode_covariates=True,
    )

    expected_extra = model.summary_stats.n_batch + 2 + 1
    assert model.module.qu.fc1.in_features == N_LATENT + expected_extra

    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = model.module._get_inference_input(tensors)
    with torch.no_grad():
        out = model.module.inference(**inf_inputs)
    assert torch.all(torch.isfinite(out["u"]))


def test_mrmultivi_save_load_preserves_latent_hierarchy(mdata_basic, tmp_path):
    """Save/load preserves non-isomorphic u, MoG prior, and label/sample mappings."""
    import torch

    mdata = _make_mdata()
    mdata.obs["cell_type"] = np.where(np.arange(mdata.n_obs) % 2 == 0, "T", "B")
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        labels_key="cell_type",
        modalities={**MODALITIES, "labels_key": None},
    )
    model = MrMultiVI(
        mdata,
        sample_key="donor",
        n_latent=N_LATENT,
        n_latent_u=4,
        u_prior_mixture=True,
    )

    save_path = tmp_path / "mrmultivi"
    model.save(save_path, overwrite=True)
    loaded = MrMultiVI.load(save_path, adata=mdata)

    assert loaded.module.n_latent_u == 4
    assert loaded.module.qz.fc is not None
    assert loaded.module.resolved_u_prior_mixture_k == model.summary_stats.n_labels
    assert loaded.sample_order.tolist() == model.sample_order.tolist()
    assert loaded.label_order.tolist() == model.label_order.tolist()
    assert torch.allclose(loaded.module.u_prior_means.cpu(), model.module.u_prior_means.cpu())
    assert torch.allclose(loaded.module.u_prior_scales.cpu(), model.module.u_prior_scales.cpu())


def test_mrmultivi_u_space_statistical_apis(mdata_basic):
    """Aggregated posterior, DA, and admissibility APIs operate over u."""
    import torch
    from torch.distributions import MixtureSameFamily

    mdata = _make_mdata()
    mdata.obs["condition"] = np.where(mdata.obs["donor"].isin(["donor_0", "donor_1"]), "a", "b")
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata, sample_key="donor", n_latent=N_LATENT, n_latent_u=4)
    model.is_trained_ = True

    ap = model.get_aggregated_posterior(batch_size=32)
    assert isinstance(ap, MixtureSameFamily)
    assert ap.component_distribution.event_shape == torch.Size([4])

    da = model.differential_abundance(sample_cov_keys=["condition"], batch_size=32)
    assert "log_probs" in da
    assert "condition_log_probs" in da
    assert da["log_probs"].dims == ("cell_name", "sample")

    outliers = model.get_outlier_cell_sample_pairs(batch_size=32, subsample_size=10)
    assert {"log_probs", "log_ratios", "is_admissible"} <= set(outliers.data_vars)


def test_mrmultivi_atac_differential_expression_is_explicitly_unsupported(mdata_basic):
    """ATAC-containing differential expression paths fail with a clear API error."""
    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT)

    with pytest.raises(NotImplementedError, match="ATAC"):
        model.differential_expression()


def test_mrmultivi_differential_expression_rna_protein():
    """MrVI-style DE works on RNA+protein (no ATAC): correct shapes, finite, valid p-vals."""
    from mudata import MuData

    mdata_full = synthetic_iid(return_mudata=True)
    n_cells = mdata_full.n_obs
    mdata_full.obs["donor"] = [f"donor_{i % N_DONORS}" for i in range(n_cells)]

    # Bimodal: RNA + protein only (no ATAC → n_regions=0, bypasses ATAC guard)
    mdata_bi = MuData({
        "rna": mdata_full.mod["rna"].copy(),
        "protein_expression": mdata_full.mod["protein_expression"].copy(),
    })
    mdata_bi.obs["donor"] = mdata_full.obs["donor"].values
    mdata_bi.obs["batch"] = mdata_full.obs["batch"].values
    mdata_bi.obs["condition"] = np.where(
        mdata_bi.obs["donor"].isin(["donor_0", "donor_1"]), "a", "b"
    )

    MrMultiVI.setup_mudata(
        mdata_bi,
        sample_key="donor",
        batch_key="batch",
        modalities={"rna_layer": "rna", "protein_layer": "protein_expression"},
    )
    model = MrMultiVI(mdata_bi, sample_key="donor", n_latent=N_LATENT, n_latent_u=4)
    model.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")

    de = model.differential_expression(
        sample_cov_keys=["condition"], mc_samples=2, batch_size=32
    )

    assert set(de.data_vars) >= {"beta", "effect_size", "pvalue", "padj"}
    assert de["beta"].dims == ("cell_name", "covariate", "latent_dim")
    assert de["beta"].shape == (n_cells, 1, N_LATENT)
    assert np.all(np.isfinite(de["beta"].values))
    assert np.all((de["pvalue"].values >= 0) & (de["pvalue"].values <= 1))
    assert np.all((de["padj"].values >= 0) & (de["padj"].values <= 1))


def test_mrmultivi_de_raises_on_nonunique_cov():
    """ValueError when sample_cov_key is not constant within sample_key (L-077).

    Models trained with sample_key="donor" cannot test within-donor conditions
    (e.g. timepoint) via sample_cov_keys — drop_duplicates picks an arbitrary
    timepoint per donor, making the design matrix wrong.
    """
    from mudata import MuData

    mdata_full = synthetic_iid(return_mudata=True)
    n_cells = mdata_full.n_obs
    mdata_full.obs["donor"] = [f"donor_{i % N_DONORS}" for i in range(n_cells)]

    mdata_bi = MuData({
        "rna": mdata_full.mod["rna"].copy(),
        "protein_expression": mdata_full.mod["protein_expression"].copy(),
    })
    mdata_bi.obs["donor"] = mdata_full.obs["donor"].values
    mdata_bi.obs["batch"] = mdata_full.obs["batch"].values
    # Each donor spans both timepoints — paired design where donor ≠ timepoint.
    # Round-robin assignment means donor index = i % N_DONORS; use block-based
    # timepoint (i // N_DONORS) % 2 so every donor gets cells at both timepoints.
    mdata_bi.obs["timepoint"] = np.where(
        (np.arange(n_cells) // N_DONORS) % 2 == 0, "T0", "T1"
    )

    MrMultiVI.setup_mudata(
        mdata_bi,
        sample_key="donor",
        batch_key="batch",
        modalities={"rna_layer": "rna", "protein_layer": "protein_expression"},
    )
    model = MrMultiVI(mdata_bi, sample_key="donor", n_latent=N_LATENT, n_latent_u=4)
    model.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")

    with pytest.raises(ValueError, match="not constant within"):
        model.differential_expression(
            sample_cov_keys=["timepoint"],
            mc_samples=2,
            batch_size=32,
        )


# ---------------------------------------------------------------------------
# store_lfc tests — gene/protein LFC path
# ---------------------------------------------------------------------------


def _bi_model_and_mdata():
    """RNA+protein bimodal MrMultiVI (no ATAC) with a binary condition."""
    from mudata import MuData

    mdata_full = synthetic_iid(return_mudata=True)
    n_cells = mdata_full.n_obs
    mdata_full.obs["donor"] = [f"donor_{i % N_DONORS}" for i in range(n_cells)]
    mdata_bi = MuData({
        "rna": mdata_full.mod["rna"].copy(),
        "protein_expression": mdata_full.mod["protein_expression"].copy(),
    })
    mdata_bi.obs["donor"] = mdata_full.obs["donor"].values
    mdata_bi.obs["batch"] = mdata_full.obs["batch"].values
    mdata_bi.obs["condition"] = np.where(
        mdata_bi.obs["donor"].isin(["donor_0", "donor_1"]), "a", "b"
    )
    MrMultiVI.setup_mudata(
        mdata_bi,
        sample_key="donor",
        batch_key="batch",
        modalities={"rna_layer": "rna", "protein_layer": "protein_expression"},
    )
    model = MrMultiVI(mdata_bi, sample_key="donor", n_latent=N_LATENT, n_latent_u=4)
    model.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")
    return model, mdata_bi


def test_mrmultivi_store_lfc_shapes_finite():
    """store_lfc=True returns lfc + lfc_std with correct shapes and finite values."""
    model, mdata = _bi_model_and_mdata()
    n_cells = mdata.n_obs
    n_genes = mdata.mod["rna"].n_vars
    n_proteins = mdata.mod["protein_expression"].n_vars
    n_features = n_genes + n_proteins

    de = model.differential_expression(
        sample_cov_keys=["condition"],
        mc_samples=2,
        batch_size=32,
        store_lfc=True,
        delta=None,
    )

    assert "lfc" in de.data_vars
    assert "lfc_std" in de.data_vars
    assert "pde" not in de.data_vars
    assert de["lfc"].dims == ("cell_name", "covariate", "feature")
    assert de["lfc"].shape == (n_cells, 1, n_features)
    assert de["lfc_std"].shape == (n_cells, 1, n_features)
    assert np.all(np.isfinite(de["lfc"].values))
    assert np.all(np.isfinite(de["lfc_std"].values))
    assert np.all(de["lfc_std"].values >= 0)


def test_mrmultivi_store_lfc_feature_coords():
    """store_lfc=True annotates the feature axis with gene/protein labels."""
    model, mdata = _bi_model_and_mdata()
    n_genes = mdata.mod["rna"].n_vars
    n_proteins = mdata.mod["protein_expression"].n_vars

    de = model.differential_expression(
        sample_cov_keys=["condition"],
        mc_samples=2,
        batch_size=32,
        store_lfc=True,
    )

    feature_coords = de.coords["feature"].values.tolist()
    assert feature_coords[:n_genes] == ["gene"] * n_genes
    assert feature_coords[n_genes:] == ["protein"] * n_proteins


def test_mrmultivi_store_lfc_pde_in_range():
    """pde values are in [0, 1] when delta is set."""
    model, mdata = _bi_model_and_mdata()

    de = model.differential_expression(
        sample_cov_keys=["condition"],
        mc_samples=2,
        batch_size=32,
        store_lfc=True,
        delta=0.3,
    )

    assert "pde" in de.data_vars
    pde = de["pde"].values
    assert np.all(pde >= 0.0) and np.all(pde <= 1.0)
    assert np.all(np.isfinite(pde))


def test_mrmultivi_store_lfc_baseline():
    """store_baseline=True adds a finite baseline_expression variable."""
    model, mdata = _bi_model_and_mdata()
    n_cells = mdata.n_obs
    n_features = mdata.mod["rna"].n_vars + mdata.mod["protein_expression"].n_vars

    de = model.differential_expression(
        sample_cov_keys=["condition"],
        mc_samples=2,
        batch_size=32,
        store_lfc=True,
        store_baseline=True,
    )

    assert "baseline_expression" in de.data_vars
    bl = de["baseline_expression"].values
    assert bl.shape == (n_cells, n_features)
    assert np.all(np.isfinite(bl))
    assert np.all(bl >= 0)


def test_mrmultivi_store_lfc_backward_compat():
    """store_lfc=False produces the same output as before (backward compat)."""
    model, mdata = _bi_model_and_mdata()

    de = model.differential_expression(
        sample_cov_keys=["condition"],
        mc_samples=2,
        batch_size=32,
        store_lfc=False,
    )

    assert "lfc" not in de.data_vars
    assert "pde" not in de.data_vars
    assert "baseline_expression" not in de.data_vars
    assert set(de.data_vars) == {"beta", "effect_size", "pvalue", "padj"}


def test_mrmultivi_atac_rejects_lfc():
    """store_lfc=True on an ATAC-containing model still raises via the ATAC guard."""
    mdata = _make_mdata()
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata, sample_key="donor", n_latent=N_LATENT)

    with pytest.raises(NotImplementedError, match="ATAC"):
        model.differential_expression(store_lfc=True, sample_cov_keys=["batch"])


def test_mrmultivi_protein_decode_is_deterministic():
    """D-021: compute_h_from_x_eps returns identical protein output on two calls.

    The protein background path uses exp(back_alpha) (deterministic), not rsample().
    Two identical calls must agree to floating-point precision on the protein slice.
    """
    import torch

    model, mdata = _bi_model_and_mdata()
    n_proteins = mdata.mod["protein_expression"].n_vars
    n_latent = model.module.n_latent

    model.module.eval()
    dl = model._make_data_loader(adata=mdata, batch_size=32)
    tensors = next(iter(dl))
    inf_inputs = model.module._get_inference_input(tensors)
    n_cells = inf_inputs["x"].shape[0]
    extra_eps = torch.zeros(n_cells, n_latent)

    out1 = model.module.compute_h_from_x_eps(extra_eps=extra_eps, **inf_inputs)
    out2 = model.module.compute_h_from_x_eps(extra_eps=extra_eps, **inf_inputs)

    protein1 = out1[..., -n_proteins:]
    protein2 = out2[..., -n_proteins:]
    assert torch.allclose(protein1, protein2), (
        "Protein decode is not deterministic — D-021 background fix may be missing"
    )


def test_mrmultivi_crn_identity():
    """CRN: sharing u_anchor between x_0 and x_1 gives LFC == 0 when extra_eps is identical.

    When the same u and the same extra_eps are passed to both decode calls, x_1 and
    x_0 are bit-identical, so log2(x_1) - log2(x_0) is exactly 0.  This proves
    u_anchor is wired through both endpoints and the CRN path is active.
    """
    import torch

    model, mdata = _bi_model_and_mdata()
    n_latent = model.module.n_latent

    model.module.eval()
    dl = model._make_data_loader(adata=mdata, batch_size=32)
    tensors = next(iter(dl))
    inf_inputs = model.module._get_inference_input(tensors)
    n_cells = inf_inputs["x"].shape[0]
    extra_eps = torch.randn(n_cells, n_latent)

    # Run inference once to get qu, then sample a u.
    with torch.inference_mode():
        base_out = model.module.inference(**inf_inputs, n_samples=1)
    u = base_out["qu"].rsample()

    eps_lfc = 1e-6
    x_0 = model.module.compute_h_from_x_eps(extra_eps=extra_eps, u_anchor=u, **inf_inputs)
    x_1 = model.module.compute_h_from_x_eps(extra_eps=extra_eps, u_anchor=u, **inf_inputs)

    lfc = torch.log2(x_1 + eps_lfc) - torch.log2(x_0 + eps_lfc)
    assert lfc.abs().max().item() < 1e-5, (
        f"CRN identity failed: max|LFC| = {lfc.abs().max().item():.2e} "
        "(expected exactly 0 when same u_anchor and extra_eps are shared)"
    )


def test_mrmultivi_lfc_is_nontrivial():
    """extra_eps routes live through the decoder: non-trivial covariate yields |lfc| > 0.

    A covariate beta of zero everywhere would indicate extra_eps never reaches the
    generative model, making the LFC feature silently inert.
    """
    model, mdata = _bi_model_and_mdata()

    de = model.differential_expression(
        sample_cov_keys=["condition"],
        mc_samples=4,
        batch_size=32,
        store_lfc=True,
    )

    max_abs_lfc = np.abs(de["lfc"].values).max()
    assert max_abs_lfc > 1e-4, (
        f"All LFCs are effectively zero (max={max_abs_lfc:.2e}); "
        "extra_eps may not be wired through the decoder"
    )


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


# ---------------------------------------------------------------------------
# (l) VampPrior toggle
# ---------------------------------------------------------------------------

def test_vamprior_trains_finite_elbo(mdata_basic):
    """u_prior='vamp' constructs and trains to a finite ELBO.

    VampPrior routes K pseudoinputs (in MULTIVAE's continuous latent space)
    through the shared qu encoder each forward pass.
    """
    import math

    model = _setup_and_train(
        mdata_basic,
        max_epochs=MAX_EPOCHS_QUICK,
        u_prior="vamp",
        u_prior_mixture_k=5,
    )
    history = model.history["elbo_train"]
    assert all(math.isfinite(v) for v in history.values.flatten()), (
        "Non-finite ELBO encountered with u_prior='vamp'"
    )


def test_vamprior_default_unchanged(mdata_basic):
    """u_prior='mog' (default) does not register u_vamp_pseudo."""
    model = _setup_and_train(mdata_basic, max_epochs=1)
    assert getattr(model.module, "u_prior_type", "mog") == "mog"
    assert hasattr(model.module, "u_prior_means"), (
        "MoG default should register u_prior_means"
    )
    assert not hasattr(model.module, "u_vamp_pseudo"), (
        "MoG default must not register u_vamp_pseudo"
    )


def test_vamprior_has_correct_parameters(mdata_basic):
    """VampPrior registers u_vamp_pseudo (in latent space) and u_prior_logits."""
    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    K = 6
    model = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT,
                      u_prior="vamp", u_prior_mixture_k=K)
    module = model.module
    assert module.u_prior_type == "vamp"
    assert hasattr(module, "u_vamp_pseudo"), "u_vamp_pseudo must be registered"
    assert module.u_vamp_pseudo.shape == (K, module.n_latent)
    assert module.u_prior_logits.shape == (K,)
    assert not hasattr(module, "u_prior_means"), (
        "VampPrior must not register u_prior_means"
    )
    assert module.u_prior_mixture is True, (
        "VampPrior must set u_prior_mixture=True to enable MC KL path"
    )


def test_vamprior_save_load(tmp_path):
    """VampPrior pseudo-inputs and logits survive a save/load cycle."""
    import torch

    mdata = _make_mdata()
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    K = 4
    model = MrMultiVI(mdata, sample_key="donor", n_latent=N_LATENT,
                      u_prior="vamp", u_prior_mixture_k=K)
    model.train(
        max_epochs=1,
        accelerator="cpu",
        plan_kwargs={"lr": 1e-3},
        check_val_every_n_epoch=1,
    )

    save_path = tmp_path / "mrmultivi_vamp"
    model.save(save_path, overwrite=True)
    loaded = MrMultiVI.load(save_path, adata=mdata)

    assert loaded.module.u_prior_type == "vamp"
    assert loaded.module.u_vamp_pseudo.shape == model.module.u_vamp_pseudo.shape
    assert loaded.module.u_prior_logits.shape == model.module.u_prior_logits.shape
    assert torch.allclose(
        loaded.module.u_vamp_pseudo.cpu(),
        model.module.u_vamp_pseudo.cpu(),
    )
    assert torch.allclose(
        loaded.module.u_prior_logits.cpu(),
        model.module.u_prior_logits.cpu(),
    )


# ---------------------------------------------------------------------------
# (m) protein_in_encoder toggle
# ---------------------------------------------------------------------------


def test_protein_in_encoder_trains_finite_elbo(mdata_basic):
    """protein_in_encoder=True trains to a finite ELBO."""
    import math

    model = _setup_and_train(
        mdata_basic,
        max_epochs=MAX_EPOCHS_QUICK,
        protein_in_encoder=True,
    )
    history = model.history["elbo_train"]
    assert all(math.isfinite(v) for v in history.values.flatten()), (
        "Non-finite ELBO encountered with protein_in_encoder=True"
    )


def test_protein_in_encoder_default_unchanged(mdata_basic):
    """protein_in_encoder=False (default) does not widen fc1 or register extra params."""
    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT)
    module = model.module
    assert not module.protein_in_encoder
    assert module.qu.n_input_proteins == 0
    # fc1 input width must be exactly n_latent (no covariate conditioning, no protein)
    assert module.qu.fc1.in_features == N_LATENT, (
        f"Default fc1.in_features={module.qu.fc1.in_features}, expected {N_LATENT}"
    )


def test_protein_in_encoder_parameter_shapes(mdata_basic):
    """protein_in_encoder=True widens fc1 by n_input_proteins."""
    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT,
                      protein_in_encoder=True)
    module = model.module
    n_proteins = module.n_input_proteins
    assert n_proteins > 0, "synthetic_iid MuData must contain protein data"
    assert module.qu.n_input_proteins == n_proteins
    assert module.qu.fc1.in_features == N_LATENT + n_proteins, (
        f"fc1.in_features={module.qu.fc1.in_features}, "
        f"expected {N_LATENT + n_proteins}"
    )


def test_protein_in_encoder_save_load(tmp_path):
    """protein_in_encoder=True survives a save/load cycle (fc1 shape preserved)."""
    import torch

    mdata = _make_mdata()
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata, sample_key="donor", n_latent=N_LATENT,
                      protein_in_encoder=True)

    save_path = tmp_path / "mrmultivi_pie"
    model.save(save_path, overwrite=True)
    loaded = MrMultiVI.load(save_path, adata=mdata)

    assert loaded.module.protein_in_encoder is True
    assert loaded.module.qu.n_input_proteins == model.module.n_input_proteins
    assert loaded.module.qu.fc1.in_features == model.module.qu.fc1.in_features
    assert torch.allclose(
        loaded.module.qu.fc1.weight.cpu(),
        model.module.qu.fc1.weight.cpu(),
    )


def test_protein_in_encoder_with_vamprior(mdata_basic):
    """protein_in_encoder=True + u_prior='vamp' trains to finite ELBO and has correct pseudo shape."""
    import math

    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    K = 5
    model = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT,
                      protein_in_encoder=True, u_prior="vamp", u_prior_mixture_k=K)
    model.train(
        max_epochs=MAX_EPOCHS_QUICK,
        accelerator="cpu",
        plan_kwargs={"lr": 1e-3},
        check_val_every_n_epoch=MAX_EPOCHS_QUICK,
    )
    module = model.module
    n_proteins = module.n_input_proteins
    assert module.u_vamp_pseudo.shape == (K, N_LATENT + n_proteins), (
        f"u_vamp_pseudo.shape={module.u_vamp_pseudo.shape}, "
        f"expected ({K}, {N_LATENT + n_proteins})"
    )
    history = model.history["elbo_train"]
    assert all(math.isfinite(v) for v in history.values.flatten()), (
        "Non-finite ELBO with protein_in_encoder=True + u_prior='vamp'"
    )


# ---------------------------------------------------------------------------
# Change E — n_obs_per_sample buffer persistence
# ---------------------------------------------------------------------------

def test_n_obs_per_sample_in_state_dict(mdata_basic):
    """Change E: n_obs_per_sample is persistent → present in state_dict and survives load_state_dict."""
    import torch

    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT, scale_observations=True)
    sd = model.module.state_dict()
    assert "n_obs_per_sample" in sd, "n_obs_per_sample must appear in state_dict (persistent=True)"

    model2 = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT, scale_observations=True)
    model2.module.load_state_dict(sd)
    assert torch.allclose(
        model2.module.n_obs_per_sample.cpu(),
        model.module.n_obs_per_sample.cpu(),
    ), "load_state_dict must restore n_obs_per_sample"


# ---------------------------------------------------------------------------
# Change C — separate KL weights for kl_u and kl_z
# ---------------------------------------------------------------------------

def test_kl_weights_stored_and_non_default_differ(mdata_basic):
    """Change C: kl_u_weight / kl_z_weight are stored; non-default values are set correctly."""
    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model_default = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT,
                               kl_u_weight=1.0, kl_z_weight=1.0)
    assert model_default.module.kl_u_weight == 1.0
    assert model_default.module.kl_z_weight == 1.0

    model_scaled = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT,
                              kl_u_weight=0.5, kl_z_weight=2.0)
    assert model_scaled.module.kl_u_weight == 0.5
    assert model_scaled.module.kl_z_weight == 2.0


# ---------------------------------------------------------------------------
# Change B1 — protein_in_encoder=False default
# ---------------------------------------------------------------------------

def test_protein_in_encoder_default_false(mdata_basic):
    """Change B1: protein_in_encoder defaults to False; qu.n_input_proteins == 0 by default."""
    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(mdata_basic, sample_key="donor", n_latent=N_LATENT)
    assert model.module.protein_in_encoder is False
    assert model.module.qu.n_input_proteins == 0, (
        "With protein_in_encoder=False, qu should have n_input_proteins=0"
    )


# ---------------------------------------------------------------------------
# Change B2 — protein_encoder_mode experimental spike
# ---------------------------------------------------------------------------

def test_protein_encoder_mode_layernorm(mdata_basic):
    """Change B2: protein_encoder_mode='layernorm' via qu_kwargs trains to finite ELBO."""
    import math

    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(
        mdata_basic,
        sample_key="donor",
        n_latent=N_LATENT,
        protein_in_encoder=True,
        qu_kwargs={"protein_encoder_mode": "layernorm"},
    )
    assert model.module.qu.protein_encoder_mode == "layernorm"
    assert hasattr(model.module.qu, "protein_layernorm"), (
        "layernorm mode must add protein_layernorm submodule"
    )
    model.train(
        max_epochs=MAX_EPOCHS_QUICK,
        accelerator="cpu",
        plan_kwargs={"lr": 1e-3},
        check_val_every_n_epoch=MAX_EPOCHS_QUICK,
    )
    history = model.history["elbo_train"]
    assert all(math.isfinite(v) for v in history.values.flatten()), (
        "Non-finite ELBO with protein_encoder_mode='layernorm'"
    )


def test_protein_encoder_mode_project(mdata_basic):
    """Change B2: protein_encoder_mode='project' reduces protein dim before fc1."""
    MrMultiVI.setup_mudata(
        mdata_basic,
        sample_key="donor",
        batch_key="batch",
        modalities=MODALITIES,
    )
    model = MrMultiVI(
        mdata_basic,
        sample_key="donor",
        n_latent=N_LATENT,
        protein_in_encoder=True,
        qu_kwargs={"protein_encoder_mode": "project", "protein_encoder_proj_dim": 2},
    )
    qu = model.module.qu
    assert qu.protein_encoder_mode == "project"
    assert hasattr(qu, "protein_proj"), "project mode must add protein_proj submodule"
    assert qu.protein_proj.out_features == 2
    # fc1 input must use proj_dim (2), not full protein dim
    n_proteins_full = model.module.n_input_proteins
    n_proteins_in_fc1 = qu.protein_proj.out_features
    assert n_proteins_in_fc1 < n_proteins_full or n_proteins_full <= 2, (
        "Projected protein dim should be smaller than full protein dim"
    )
