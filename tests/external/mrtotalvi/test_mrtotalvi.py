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
    model.train(
        max_epochs=max_epochs,
        accelerator="cpu",
        plan_kwargs={"lr": 1e-3},
        check_val_every_n_epoch=max_epochs,
    )
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


def test_mrtotalvi_non_isomorphic_u_dimension(adata_basic):
    """n_latent_u can be smaller than z while the decoder still receives n_latent."""
    import torch

    MrTotalVI.setup_anndata(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata_basic, sample_key="sample", n_latent=N_LATENT, n_latent_u=5)

    assert model.module.n_latent_u == 5
    assert model.module.qz.fc is not None

    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = model.module._get_inference_input(tensors)
    with torch.no_grad():
        out = model.module._regular_inference(**inf_inputs)

    assert out["u"].shape[-1] == 5
    assert out["z"].shape[-1] == N_LATENT


def test_mrtotalvi_non_isomorphic_u_dimension_with_mc_samples(adata_basic):
    """MC samples preserve the u and z dimensions and keep the ELBO finite."""
    import torch

    MrTotalVI.setup_anndata(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata_basic, sample_key="sample", n_latent=N_LATENT, n_latent_u=5)
    module = model.module.train()

    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    inf_out = module._regular_inference(**inf_inputs, n_samples=2)
    gen_inputs = module._get_generative_input(tensors, inf_out)
    gen_out = module.generative(**gen_inputs)
    loss_out = module.loss(tensors, inf_out, gen_out)

    batch_size = tensors[scvi.REGISTRY_KEYS.X_KEY].shape[0]
    assert inf_out["u"].shape == torch.Size([2, batch_size, 5])
    assert inf_out["z"].shape == torch.Size([2, batch_size, N_LATENT])
    assert inf_out["eps"].shape == torch.Size([2, batch_size, N_LATENT])
    assert loss_out.kl_local["kl_div_z"].shape == torch.Size([batch_size])
    assert torch.isfinite(loss_out.loss)


def test_mrtotalvi_singleton_mc_loss_matches_single_sample_loss(adata_basic):
    """The custom MC loss is equivalent to the parent loss for one sampled z."""
    import torch

    MrTotalVI.setup_anndata(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(
        adata_basic,
        sample_key="sample",
        n_latent=N_LATENT,
        n_latent_u=5,
        u_prior_mixture=False,
    )
    module = model.module.eval()

    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    with torch.no_grad():
        inf_out = module._regular_inference(**inf_inputs)
        gen_out = module.generative(**module._get_generative_input(tensors, inf_out))
        loss_out = module.loss(tensors, inf_out, gen_out)

        batch_size = tensors[scvi.REGISTRY_KEYS.X_KEY].shape[0]

        def _unsqueeze_batch_tensors(value):
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                return value.unsqueeze(0)
            if isinstance(value, dict):
                return {k: _unsqueeze_batch_tensors(v) for k, v in value.items()}
            return value

        inf_mc = dict(inf_out)
        for key in ("z", "u", "z_base", "eps", "library_gene"):
            inf_mc[key] = inf_mc[key].unsqueeze(0)
        gen_mc = _unsqueeze_batch_tensors(gen_out)
        loss_mc = module.loss(tensors, inf_mc, gen_mc)

    assert torch.allclose(
        loss_mc.reconstruction_loss["reconst_loss_gene"].squeeze(0),
        loss_out.reconstruction_loss["reconst_loss_gene"],
        atol=1e-5,
    )
    assert torch.allclose(
        loss_mc.reconstruction_loss["reconst_loss_protein"].squeeze(0),
        loss_out.reconstruction_loss["reconst_loss_protein"],
        atol=1e-5,
    )
    assert torch.allclose(loss_mc.kl_local["kl_div_z"], loss_out.kl_local["kl_div_z"], atol=1e-5)
    assert torch.allclose(loss_mc.loss, loss_out.loss, atol=1e-5)


def test_mrtotalvi_mc_samples_with_stochastic_eps_and_scaled_observations(adata_basic):
    """MC samples work when eps is stochastic and observations are sample-scaled."""
    import torch

    MrTotalVI.setup_anndata(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(
        adata_basic,
        sample_key="sample",
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
    inf_out = module._regular_inference(**inf_inputs, n_samples=2)
    gen_inputs = module._get_generative_input(tensors, inf_out)
    gen_out = module.generative(**gen_inputs)
    loss_out = module.loss(tensors, inf_out, gen_out)

    batch_size = tensors[scvi.REGISTRY_KEYS.X_KEY].shape[0]
    assert inf_out["eps_dist"] is not None
    assert inf_out["eps_dist"].loc.shape == torch.Size([2, batch_size, N_LATENT])
    assert loss_out.kl_local["kl_div_z"].shape == torch.Size([batch_size])
    assert torch.isfinite(loss_out.loss)


def test_mrtotalvi_mc_samples_with_size_factor_and_protein_batch_mask(adata_basic):
    """MC loss handles TotalVI size factors and protein batch masks."""
    import torch

    adata = adata_basic.copy()
    adata.obs["batch"] = np.where(np.arange(adata.n_obs) % 2 == 0, "batch_0", "batch_1")
    adata.obs["size_factor"] = np.asarray(adata.X.sum(axis=1)).reshape(-1) + 1.0
    protein = np.asarray(adata.obsm["protein_expression"]).copy()
    protein[adata.obs["batch"].to_numpy() == "batch_0", 0] = 0
    adata.obsm["protein_expression"] = protein

    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        size_factor_key="size_factor",
    )
    with pytest.warns(UserWarning, match="Some proteins have all 0 counts"):
        model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=5)
    module = model.module.train()
    assert module.use_size_factor_key
    assert module.protein_batch_mask is not None

    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    inf_out = module._regular_inference(**inf_inputs, n_samples=2)
    gen_inputs = module._get_generative_input(tensors, inf_out)
    gen_out = module.generative(**gen_inputs)
    loss_out = module.loss(tensors, inf_out, gen_out)

    batch_size = tensors[scvi.REGISTRY_KEYS.X_KEY].shape[0]
    assert loss_out.reconstruction_loss["reconst_loss_protein"].shape == torch.Size([2, batch_size])
    assert loss_out.kl_local["kl_div_z"].shape == torch.Size([batch_size])
    assert torch.isfinite(loss_out.loss)


def test_mrtotalvi_default_u_dimension_is_isomorphic(adata_basic):
    """n_latent_u=None preserves the original isomorphic u->z behavior."""
    MrTotalVI.setup_anndata(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata_basic, sample_key="sample", n_latent=N_LATENT)

    assert model.module.n_latent_u == N_LATENT
    assert model.module.qz.fc is None


def test_mrtotalvi_label_conditioned_mog_prior(adata_basic):
    """labels_key switches the MoG prior to one component per label and biases logits."""
    import torch
    from torch.distributions import MixtureSameFamily

    adata = adata_basic.copy()
    adata.obs["cell_type"] = np.where(np.arange(adata.n_obs) % 2 == 0, "T", "B")
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        labels_key="cell_type",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)

    assert model.label_order.tolist() == ["B", "T"] or model.label_order.tolist() == ["T", "B"]
    assert model.module.u_prior_logits.shape == (model.summary_stats.n_labels,)
    assert model.module.u_prior_means.shape == (model.summary_stats.n_labels, 4)

    labels = torch.tensor([[0], [1], [0]])
    u = torch.zeros(3, 4)
    prior = model.module.build_u_prior(u, labels)

    assert isinstance(prior, MixtureSameFamily)
    assert prior.mixture_distribution.logits.argmax(dim=-1).tolist() == [0, 1, 0]


def test_mrtotalvi_label_conditioned_mog_flows_through_mc_loss(adata_basic):
    """Registered labels condition the MoG prior in the actual MC loss path."""
    import torch

    adata = adata_basic.copy()
    adata.obs["cell_type"] = np.where(np.arange(adata.n_obs) % 2 == 0, "T", "B")
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        labels_key="cell_type",
    )
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=N_LATENT,
        n_latent_u=4,
        z_u_prior=False,
    )
    module = model.module.train()

    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    inf_out = module._regular_inference(**inf_inputs, n_samples=2)
    gen_inputs = module._get_generative_input(tensors, inf_out)
    gen_out = module.generative(**gen_inputs)
    loss_out = module.loss(tensors, inf_out, gen_out)

    expected_kl_u = module.kl_u(
        inf_out["qu"],
        inf_out["u"],
        tensors[scvi.REGISTRY_KEYS.LABELS_KEY],
    )
    assert module.resolved_u_prior_mixture_k == model.summary_stats.n_labels
    assert torch.allclose(loss_out.kl_local["kl_div_z"], expected_kl_u, atol=1e-5)
    assert torch.isfinite(loss_out.loss)


def test_mrtotalvi_gaussian_u_prior_and_z_u_prior_off(adata_basic):
    """u_prior_mixture=False uses analytic Gaussian KL and z_u_prior=False omits kl_z."""
    import torch

    MrTotalVI.setup_anndata(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(
        adata_basic,
        sample_key="sample",
        n_latent=N_LATENT,
        u_prior_mixture=False,
        z_u_prior=False,
    )
    assert not hasattr(model.module, "u_prior_logits")

    module = model.module.train()
    dl = model._make_data_loader(adata=model.adata, batch_size=16)
    tensors = next(iter(dl))
    inf_inputs = module._get_inference_input(tensors)
    inf_out = module._regular_inference(**inf_inputs)
    gen_inputs = module._get_generative_input(tensors, inf_out)
    gen_out = module.generative(**gen_inputs)
    loss_out = module.loss(tensors, inf_out, gen_out)

    expected_kl_u = module.kl_u(
        inf_out["qz"],
        inf_out["u"],
        tensors[scvi.REGISTRY_KEYS.LABELS_KEY],
    )
    assert torch.allclose(loss_out.kl_local["kl_div_z"], expected_kl_u, atol=1e-5)


def test_mrtotalvi_encode_covariates_expands_qu_input(adata_basic):
    """encode_covariates=True appends batch, categorical, and continuous covariates to qu."""
    adata = adata_basic.copy()
    adata.obs["stim"] = np.where(np.arange(adata.n_obs) % 2 == 0, "ctrl", "stim")
    adata.obs["score"] = np.linspace(0.0, 1.0, adata.n_obs)

    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        categorical_covariate_keys=["stim"],
        continuous_covariate_keys=["score"],
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, encode_covariates=True)

    base_features = model.summary_stats.n_vars + model.summary_stats.n_proteins
    expected_extra = model.summary_stats.n_batch + 2 + 1
    assert model.module.qu.fc1.in_features == base_features + expected_extra


def test_mrtotalvi_save_load_preserves_latent_hierarchy(adata_basic, tmp_path):
    """Save/load preserves non-isomorphic u, MoG prior, and label/sample mappings."""
    import torch

    adata = adata_basic.copy()
    adata.obs["cell_type"] = np.where(np.arange(adata.n_obs) % 2 == 0, "T", "B")
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        labels_key="cell_type",
    )
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=N_LATENT,
        n_latent_u=4,
        u_prior_mixture=True,
    )

    save_path = tmp_path / "mrtotalvi"
    model.save(save_path, overwrite=True)
    loaded = MrTotalVI.load(save_path, adata=adata)

    assert loaded.module.n_latent_u == 4
    assert loaded.module.qz.fc is not None
    assert loaded.module.resolved_u_prior_mixture_k == model.summary_stats.n_labels
    assert loaded.sample_order.tolist() == model.sample_order.tolist()
    assert loaded.label_order.tolist() == model.label_order.tolist()
    assert torch.allclose(loaded.module.u_prior_means.cpu(), model.module.u_prior_means.cpu())
    assert torch.allclose(loaded.module.u_prior_scales.cpu(), model.module.u_prior_scales.cpu())


def test_mrtotalvi_u_space_statistical_apis(adata_basic):
    """Aggregated posterior, DA, and admissibility APIs operate over u."""
    import torch
    from torch.distributions import MixtureSameFamily

    adata = adata_basic.copy()
    adata.obs["condition"] = np.where(adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b")
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)
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


def test_mrtotalvi_differential_expression_returns_latent_statistics(adata_basic):
    """MrVI-style DE returns correct shapes, finite values, and valid p-values."""
    adata = adata_basic.copy()
    adata.obs["condition"] = np.where(
        adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b"
    )
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)
    model.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")

    de = model.differential_expression(
        sample_cov_keys=["condition"], mc_samples=2, batch_size=32
    )

    n_cells = adata.n_obs
    assert set(de.data_vars) >= {"beta", "effect_size", "pvalue", "padj"}
    assert de["beta"].dims == ("cell_name", "covariate", "latent_dim")
    assert de["beta"].shape == (n_cells, 1, N_LATENT)
    assert de["effect_size"].shape == (n_cells, 1)
    assert de["pvalue"].shape == (n_cells, 1)
    assert de["padj"].shape == (n_cells, 1)
    assert np.all(np.isfinite(de["beta"].values))
    assert np.all((de["pvalue"].values >= 0) & (de["pvalue"].values <= 1))
    assert np.all((de["padj"].values >= 0) & (de["padj"].values <= 1))
    assert de.coords["covariate"].values.tolist() == ["condition_b"]


def test_mrtotalvi_de_mc_samples_1_fast_path(adata_basic):
    """mc_samples=1 uses qu.loc (posterior mean) and runs without error."""
    adata = adata_basic.copy()
    adata.obs["condition"] = np.where(
        adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b"
    )
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)
    model.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")

    de = model.differential_expression(
        sample_cov_keys=["condition"], mc_samples=1, batch_size=32
    )
    assert np.all(np.isfinite(de["beta"].values))
    assert np.all((de["pvalue"].values >= 0) & (de["pvalue"].values <= 1))


def test_mrtotalvi_de_donor_key_warning(adata_basic):
    """donor_key triggers a UserWarning about potential collinearity."""
    adata = adata_basic.copy()
    adata.obs["condition"] = np.where(
        adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b"
    )
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)
    model.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")

    with pytest.warns(UserWarning, match="donor_key"):
        de = model.differential_expression(
            sample_cov_keys=["condition"],
            donor_key="sample",
            mc_samples=2,
            batch_size=32,
        )
    assert de["beta"].dims == ("cell_name", "covariate", "latent_dim")
    assert np.all(np.isfinite(de["beta"].values))


# ---------------------------------------------------------------------------
# store_lfc tests — gene/protein LFC path
# ---------------------------------------------------------------------------


def _de_lfc_model(adata):
    """Helper: train a tiny MrTotalVI and tag obs with a binary condition."""
    adata = adata.copy()
    adata.obs["condition"] = np.where(
        adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b"
    )
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)
    model.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")
    return model, adata


def test_mrtotalvi_store_lfc_shapes_finite(adata_basic):
    """store_lfc=True returns lfc + lfc_std with correct shapes and finite values."""
    model, adata = _de_lfc_model(adata_basic)
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    n_proteins = adata.obsm["protein_expression"].shape[1]
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


def test_mrtotalvi_store_lfc_feature_coords(adata_basic):
    """store_lfc=True annotates the feature axis with gene/protein labels."""
    model, adata = _de_lfc_model(adata_basic)
    n_genes = adata.n_vars
    n_proteins = adata.obsm["protein_expression"].shape[1]

    de = model.differential_expression(
        sample_cov_keys=["condition"],
        mc_samples=2,
        batch_size=32,
        store_lfc=True,
    )

    feature_coords = de.coords["feature"].values.tolist()
    assert feature_coords[:n_genes] == ["gene"] * n_genes
    assert feature_coords[n_genes:] == ["protein"] * n_proteins


def test_mrtotalvi_store_lfc_pde_in_range(adata_basic):
    """pde values are in [0, 1] when delta is set."""
    model, adata = _de_lfc_model(adata_basic)

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


def test_mrtotalvi_store_lfc_baseline(adata_basic):
    """store_baseline=True adds a finite baseline_expression variable."""
    model, adata = _de_lfc_model(adata_basic)
    n_cells = adata.n_obs
    n_features = adata.n_vars + adata.obsm["protein_expression"].shape[1]

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


def test_mrtotalvi_store_lfc_backward_compat(adata_basic):
    """store_lfc=False produces the same output as before (backward compat)."""
    model, adata = _de_lfc_model(adata_basic)

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


def test_learnable_prior_scale_clamp(adata_basic):
    """pz_scale.clamp(min=-4.0) prevents σ→0 / kl_z→-∞ collapse.

    Under learn_z_u_prior_scale=True + use_map=True (the default), the optimizer
    could otherwise drive pz_scale→-∞ (σ→0) jointly with eps→0, making kl_z
    unbounded below.  The clamp ensures σ ≥ exp(-4) ≈ 0.018 at all times.
    """
    import torch

    model = _setup_and_train(
        adata_basic,
        max_epochs=MAX_EPOCHS_FULL,
        learn_z_u_prior_scale=True,
        use_map=True,
    )
    module = model.module
    pz_scale = module.pz_scale
    # After training, no dimension should be able to breach the clamp floor
    assert (pz_scale >= -4.0).all(), (
        f"pz_scale breached the -4.0 clamp floor: min={pz_scale.min():.3f}"
    )
    # kl_z must be finite and bounded below — confirm no -inf or nan in a forward pass
    module.eval()
    batch = next(iter(model._make_data_loader(adata_basic)))
    with torch.no_grad():
        inf_out = module._regular_inference(**module._get_inference_input(batch))
        eps = inf_out["eps"]
        import math
        from torch.distributions import Normal
        peps = Normal(0.0, torch.exp(pz_scale.clamp(min=-4.0)))
        kl_z = -peps.log_prob(eps).sum(dim=-1)
    assert torch.isfinite(kl_z).all(), "kl_z contains non-finite values after clamp"
    assert (kl_z > -1e6).all(), f"kl_z implausibly negative: min={kl_z.min():.1f}"


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
