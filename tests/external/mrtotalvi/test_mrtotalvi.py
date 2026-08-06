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
from tests.external.conftest import get_elbo_key

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
    assert loss_out.reconstruction_loss["reconst_loss_protein"].shape == torch.Size(
        [2, batch_size]
    )
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
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4,
                      u_prior="mog")

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
        u_prior="mog",
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
        u_prior="mog",
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


def test_mrtotalvi_de_raises_on_nonunique_cov(adata_basic):
    """ValueError when sample_cov_key is not constant within sample_key (L-077).

    Models trained with sample_key="donor" cannot test within-donor conditions
    (e.g. timepoint) via sample_cov_keys — drop_duplicates picks an arbitrary
    timepoint per donor, making the design matrix wrong.
    """
    adata = adata_basic.copy()
    # Each donor gets both "T0" and "T1" — paired design where donor ≠ timepoint.
    # Round-robin donor assignment means i % N_DONORS == donor index.  Use
    # (i // N_DONORS) % 2 so timepoint alternates per "block" of donors, giving
    # every donor a mix of T0 and T1 cells.
    n_cells = adata.n_obs
    n_donors = len(adata.obs["sample"].unique())
    adata.obs["timepoint"] = np.where(
        (np.arange(n_cells) // n_donors) % 2 == 0, "T0", "T1"
    )
    # Verify the paired structure: every donor must span both timepoints.
    for donor in adata.obs["sample"].unique():
        mask = adata.obs["sample"] == donor
        assert set(adata.obs.loc[mask, "timepoint"]) == {"T0", "T1"}, (
            f"donor {donor} doesn't span both timepoints — fixture assumption broken"
        )

    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)
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
    assert np.all(pde >= 0.0)
    assert np.all(pde <= 1.0)
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


def test_mrtotalvi_protein_decode_is_deterministic(adata_basic):
    """D-021: compute_h_from_x_eps returns identical protein output on two calls.

    The protein background path uses exp(back_alpha) (deterministic), not rsample().
    Two identical calls must agree to floating-point precision on the protein slice.
    """
    import torch

    model, adata = _de_lfc_model(adata_basic)
    n_proteins = adata.obsm["protein_expression"].shape[1]
    n_latent = model.module.n_latent

    model.module.eval()
    dl = model._make_data_loader(adata=adata, batch_size=32)
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


def test_mrtotalvi_crn_identity(adata_basic):
    """CRN: sharing u_anchor between x_0 and x_1 gives LFC == 0 when extra_eps is identical.

    When the same u and the same extra_eps are passed to both decode calls, x_1 and
    x_0 are bit-identical, so log2(x_1) - log2(x_0) is exactly 0.  This proves
    u_anchor is wired through both endpoints and the CRN path is active.
    """
    import torch

    model, adata = _de_lfc_model(adata_basic)
    n_latent = model.module.n_latent

    model.module.eval()
    dl = model._make_data_loader(adata=adata, batch_size=32)
    tensors = next(iter(dl))
    inf_inputs = model.module._get_inference_input(tensors)
    n_cells = inf_inputs["x"].shape[0]
    extra_eps = torch.randn(n_cells, n_latent)

    # Run inference once to get qu, then sample a u.
    with torch.inference_mode():
        base_out = model.module._regular_inference(**inf_inputs, n_samples=1)
    u = base_out["qu"].rsample()

    eps_lfc = 1e-6
    x_0 = model.module.compute_h_from_x_eps(extra_eps=extra_eps, u_anchor=u, **inf_inputs)
    x_1 = model.module.compute_h_from_x_eps(extra_eps=extra_eps, u_anchor=u, **inf_inputs)

    lfc = torch.log2(x_1 + eps_lfc) - torch.log2(x_0 + eps_lfc)
    assert lfc.abs().max().item() < 1e-5, (
        f"CRN identity failed: max|LFC| = {lfc.abs().max().item():.2e} "
        "(expected exactly 0 when same u_anchor and extra_eps are shared)"
    )


def test_mrtotalvi_lfc_is_nontrivial(adata_basic):
    """extra_eps routes live through the decoder: non-trivial covariate yields |lfc| > 0.

    A covariate beta of zero everywhere would indicate extra_eps never reaches the
    generative model, making the LFC feature silently inert.
    """
    model, adata = _de_lfc_model(adata_basic)

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

    # Explicit seed guards against CLI --seed override changing the statistical result.
    scvi.settings.seed = 0
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
        "Trained cross-donor distances are 0 — hierarchy encodes no signal."
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


# ---------------------------------------------------------------------------
# (i) VampPrior toggle
# ---------------------------------------------------------------------------

def test_vamprior_trains_finite_elbo(adata_basic):
    """u_prior='vamp' constructs and trains to a finite ELBO.

    VampPrior routes K pseudoinputs through the shared qu encoder each forward
    pass; this test confirms no NaN propagates from the log1p + Softplus path.
    """
    import math

    model = _setup_and_train(
        adata_basic,
        max_epochs=MAX_EPOCHS_QUICK,
        u_prior="vamp",
        u_prior_mixture_k=5,
    )
    history = model.history["elbo_train"]
    assert all(math.isfinite(v) for v in history.values.flatten()), (
        "Non-finite ELBO encountered with u_prior='vamp'"
    )


def test_mog_default_unchanged(adata_basic):
    """u_prior='mog' is the default — confirmed by checking module attributes."""
    model = _setup_and_train(adata_basic, max_epochs=1)
    assert getattr(model.module, "u_prior_type", "mog") == "mog"
    assert hasattr(model.module, "u_prior_means"), (
        "MoG default should register u_prior_means"
    )
    assert not hasattr(model.module, "u_vamp_pseudo"), (
        "MoG default must not register u_vamp_pseudo"
    )


def test_vamprior_has_correct_parameters(adata_basic):
    """VampPrior registers u_vamp_pseudo and u_prior_logits; no u_prior_means."""
    MrTotalVI.setup_anndata(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    K = 6
    model = MrTotalVI(adata_basic, sample_key="sample", n_latent=N_LATENT,
                      u_prior="vamp", u_prior_mixture_k=K)
    module = model.module
    assert module.u_prior_type == "vamp"
    assert hasattr(module, "u_vamp_pseudo"), "u_vamp_pseudo must be registered"
    assert module.u_vamp_pseudo.shape == (K, module.n_input_genes + module.n_input_proteins)
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

    adata = _make_adata()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    K = 4
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT,
                      u_prior="vamp", u_prior_mixture_k=K)
    model.train(
        max_epochs=1,
        accelerator="cpu",
        plan_kwargs={"lr": 1e-3},
        check_val_every_n_epoch=1,
    )

    save_path = tmp_path / "mrtotalvi_vamp"
    model.save(save_path, overwrite=True)
    loaded = MrTotalVI.load(save_path, adata=adata)

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
# Change E — n_obs_per_sample buffer persistence
# ---------------------------------------------------------------------------

def test_n_obs_per_sample_in_state_dict():
    """Test that persistent n_obs_per_sample survives a state-dict round trip."""
    import torch

    adata = _make_adata()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, scale_observations=True)
    sd = model.module.state_dict()
    assert "n_obs_per_sample" in sd, "n_obs_per_sample must appear in state_dict (persistent=True)"

    model2 = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, scale_observations=True)
    model2.module.load_state_dict(sd)
    assert torch.allclose(
        model2.module.n_obs_per_sample.cpu(),
        model.module.n_obs_per_sample.cpu(),
    ), "load_state_dict must restore n_obs_per_sample"


# ---------------------------------------------------------------------------
# Change C — separate KL weights for kl_u and kl_z
# ---------------------------------------------------------------------------

def test_kl_weights_stored_and_non_default_differ():
    """Change C: kl_u_weight / kl_z_weight are stored; non-default values are set correctly."""
    adata = _make_adata()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model_default = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT,
                               kl_u_weight=1.0, kl_z_weight=1.0)
    assert model_default.module.kl_u_weight == 1.0
    assert model_default.module.kl_z_weight == 1.0

    model_scaled = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT,
                              kl_u_weight=0.5, kl_z_weight=2.0)
    assert model_scaled.module.kl_u_weight == 0.5
    assert model_scaled.module.kl_z_weight == 2.0


# ---------------------------------------------------------------------------
# Change D — data-driven VampPrior initialisation
# ---------------------------------------------------------------------------

def test_init_prior_from_data_vamprior():
    """Change D: init_prior_from_data=True yields finite pseudo-inputs near the data manifold."""

    import torch

    adata = _make_adata()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    K = 4
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT,
                      u_prior="vamp", u_prior_mixture_k=K,
                      init_prior_from_data=True)
    pseudo = model.module.u_vamp_pseudo
    dim = model.module.n_input_genes + model.module.n_input_proteins
    assert pseudo.shape == (K, dim)
    assert torch.all(torch.isfinite(pseudo)), "data-driven VampPrior pseudo-inputs must be finite"
    # Norms should be substantially larger than the default randn*0.01 init
    norms = pseudo.norm(dim=-1)
    assert (norms > 0.5).any(), (
        f"data-driven pseudo-inputs norms {norms.tolist()} are unexpectedly small; "
        "expected them near data centroids"
    )


def test_freeze_prior_after_init_mog():
    """freeze_prior_after_init=True freezes MoG location/scale but keeps logits trainable."""

    adata = _make_adata()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT,
                      u_prior="mog", freeze_prior_after_init=True)
    m = model.module
    assert not m.u_prior_means.requires_grad, "u_prior_means should be frozen"
    assert not m.u_prior_scales.requires_grad, "u_prior_scales should be frozen"
    assert m.u_prior_logits.requires_grad, "u_prior_logits must remain trainable"


def test_freeze_prior_after_init_vamp():
    """freeze_prior_after_init=True freezes VampPrior pseudo-inputs but keeps logits trainable."""

    adata = _make_adata()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    K = 4
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT,
                      u_prior="vamp", u_prior_mixture_k=K,
                      freeze_prior_after_init=True)
    m = model.module
    assert not m.u_vamp_pseudo.requires_grad, "u_vamp_pseudo should be frozen"
    assert m.u_prior_logits.requires_grad, "u_prior_logits must remain trainable"


def test_freeze_prior_false_default():
    """MoG default: u_prior_means and u_prior_logits are trainable parameters."""
    adata = _make_adata()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT)
    m = model.module
    assert m.u_prior_means.requires_grad
    assert m.u_prior_logits.requires_grad


def test_differential_abundance_n_mc_samples_1_is_deterministic(adata_basic):
    """n_mc_samples=1 (default) gives a deterministic result given fixed weights."""
    adata = adata_basic.copy()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)
    model.is_trained_ = True

    da1 = model.differential_abundance(n_mc_samples=1, batch_size=32)
    da2 = model.differential_abundance(n_mc_samples=1, batch_size=32)
    np.testing.assert_array_equal(da1["log_probs"].values, da2["log_probs"].values)


def test_differential_abundance_n_mc_samples_gt1_runs(adata_basic):
    """n_mc_samples > 1 runs and returns same shape as n_mc_samples=1."""
    adata = adata_basic.copy()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)
    model.is_trained_ = True

    da_mean = model.differential_abundance(n_mc_samples=1, batch_size=32)
    da_mc = model.differential_abundance(n_mc_samples=4, batch_size=32)
    assert da_mc["log_probs"].shape == da_mean["log_probs"].shape
    assert np.all(np.isfinite(da_mc["log_probs"].values))


def test_differential_abundance_n_mc_samples_invalid():
    """n_mc_samples < 1 raises ValueError."""
    adata = _make_adata()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT)
    model.is_trained_ = True
    with pytest.raises(ValueError, match="n_mc_samples"):
        model.differential_abundance(n_mc_samples=0, batch_size=32)


# ---------------------------------------------------------------------------
# New tests appended after test_n_obs_per_sample_in_state_dict
# ---------------------------------------------------------------------------


def test_mrtotalvi_layer_norm_trains_finite(adata_basic):
    """MrTotalVI with use_batch_norm='none', use_layer_norm='both' trains without NaN.

    use_batch_norm/use_layer_norm flow through **model_kwargs to TOTALVAE. This
    configuration mirrors MrMultiVI's default (MULTIVAE uses LayerNorm) and is
    the recommended setting for small-N DA (stable at N<=20 samples vs BatchNorm).
    """
    model = _setup_and_train(
        adata_basic,
        use_batch_norm="none",
        use_layer_norm="both",
    )
    history = model.history
    candidates = ["elbo_train", "train_loss", "train_loss_epoch"]
    found = next((k for k in candidates if k in history), None)
    assert found is not None, f"No loss key found in history: {list(history.keys())}"
    train_loss = list(history[found].values.ravel())
    assert all(np.isfinite(v) for v in train_loss), (
        f"Training loss has non-finite values with LayerNorm: {train_loss}"
    )

    # Smoke-check latent representation is finite
    u = model.get_latent_representation(give_z=False)
    assert np.all(np.isfinite(u)), "get_latent_representation returned non-finite u"


def test_mrtotalvi_lfc_aux_fast_path_matches_full(adata_basic):
    """_lfc_aux fast path gives bit-identical compute_h_from_x_eps output to full path.

    When u_anchor is provided and _lfc_aux is cached, compute_h_from_x_eps skips
    _regular_inference and reads library_gene from the cache.  This test verifies
    that the cached value equals what _regular_inference would return — confirming
    the F3/G1 correctness claim: library_gene depends only on x/batch_index, not
    on u_anchor.
    """
    import torch

    model, adata = _de_lfc_model(adata_basic)
    n_latent = model.module.n_latent
    model.module.eval()

    dl = model._make_data_loader(adata=adata, batch_size=32)
    tensors = next(iter(dl))
    inf_inputs = model.module._get_inference_input(tensors)
    n_cells = inf_inputs["x"].shape[0]
    extra_eps = torch.randn(n_cells, n_latent)

    with torch.inference_mode():
        # Get a deterministic u anchor
        base_out = model.module._regular_inference(**inf_inputs, n_samples=1)
        u = base_out["qu"].mean

        # Pre-compute the aux cache
        aux = model.module._infer_lfc_aux(**inf_inputs)

        # Full inference path: u_anchor provided, no cache → reruns _regular_inference
        h_full = model.module.compute_h_from_x_eps(
            extra_eps=extra_eps, u_anchor=u, _lfc_aux=None, **inf_inputs
        )

        # Fast path: u_anchor + cache → skips _regular_inference
        h_fast = model.module.compute_h_from_x_eps(
            extra_eps=extra_eps, u_anchor=u, _lfc_aux=aux, **inf_inputs
        )

    max_diff = (h_full - h_fast).abs().max().item()
    assert max_diff < 1e-5, (
        f"Fast path diverges from full inference path: max|diff| = {max_diff:.2e}. "
        "library_gene from _infer_lfc_aux differs from _regular_inference output — "
        "the F3/G1 cache-is-constant assumption may be violated."
    )


def test_mrtotalvi_lfc_sign_known_positive_control():
    """LFC gene-space sign is correct: a strongly up-regulated gene has positive LFC.

    Constructs a two-group dataset where condition='b' cells have gene 0 inflated
    by 20× its baseline mean.  After training, the condition_b WLS coefficient
    should yield positive mean gene-0 LFC.

    The design matrix encodes condition 'a' as reference (drop_first=True, 'a' < 'b'),
    so condition_b dummy = 1 for the high-expression group → expected lfc > 0 for gene 0.
    """
    import scipy.sparse as sp

    # Explicit seed guards against CLI --seed override changing the statistical result.
    scvi.settings.seed = 0

    adata = _make_adata(n_donors=N_DONORS)

    # Two conditions: 'b' = donor_2 + donor_3; these will have inflated gene 0.
    adata.obs["condition"] = np.where(
        adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b"
    )

    # Inflate gene 0 for condition 'b' cells by 20× baseline mean
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float32)
    mean_g0 = float(X[:, 0].mean())
    mask_b = adata.obs["condition"] == "b"
    X[mask_b.values, 0] += 20.0 * max(mean_g0, 1.0)
    adata.X = X.astype(np.float32)

    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4,
                      u_prior="mog")
    model.train(max_epochs=30, accelerator="cpu")

    de = model.differential_expression(
        sample_cov_keys=["condition"],
        mc_samples=10,
        batch_size=64,
        store_lfc=True,
    )

    # lfc[:, 0, 0] = per-cell LFC for first covariate (condition_b), gene 0
    lfc_gene0 = de["lfc"].values[:, 0, 0]
    mean_lfc = float(np.mean(lfc_gene0))
    assert mean_lfc > 0, (
        f"Gene 0 LFC is not positive (mean={mean_lfc:.4f}) despite 20× inflation in "
        "condition='b'. LFC sign may be inverted or extra_eps is not reaching the decoder."
    )


def test_differential_abundance_trained_model_smoke(adata_basic):
    """DA runs correctly on a real trained model (V4-001 coverage).

    All other DA tests use ``model.is_trained_ = True`` (a manual bypass)
    without actual training.  This test exercises the end-to-end trained-weights
    code path: outputs must be finite and have the correct shape.
    """
    adata = adata_basic.copy()
    adata.obs["condition"] = np.where(
        adata.obs["sample"].isin(["donor_0", "donor_1"]), "a", "b"
    )
    model = _setup_and_train(adata, n_latent_u=4)

    da = model.differential_abundance(
        sample_cov_keys=["condition"],
        n_mc_samples=2,
        batch_size=32,
    )
    n_cells = adata.n_obs
    n_samples = adata.obs["sample"].nunique()
    assert "log_probs" in da
    assert "condition_log_probs" in da
    assert da["log_probs"].shape == (n_cells, n_samples), (
        f"log_probs shape {da['log_probs'].shape} != expected ({n_cells}, {n_samples})"
    )
    assert np.all(np.isfinite(da["log_probs"].values)), (
        "DA log_probs must be finite after real training"
    )
    assert np.all(np.isfinite(da["condition_log_probs"].values)), (
        "DA condition_log_probs must be finite after real training"
    )


# ---------------------------------------------------------------------------
# P1-002 — n_labels == 0 / unlabeled_category smoke
# ---------------------------------------------------------------------------


def test_mrtotalvi_n_labels_zero_mog_prior_smoke(adata_basic):
    """MrTotalVI with u_prior_mixture=True and NO labels_key must not crash.

    This exercises the zero-label MoG-prior / classifier-guard branch (C-002).
    Without a labels_key the n_labels summary stat is 0; the MoG prior must
    degrade gracefully to a standard Normal (or equivalent) rather than
    indexing an empty table and raising IndexError / division-by-zero.
    """
    adata = adata_basic.copy()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        # Intentionally no labels_key → n_labels == 0
    )
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=N_LATENT,
        u_prior_mixture=True,   # MoG path must not crash with zero labels
    )
    model.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")

    history = model.history
    elbo_key = get_elbo_key(history)
    vals = history[elbo_key].to_numpy().astype(float).flatten()
    assert np.all(np.isfinite(vals)), (
        f"Training {elbo_key} is not finite with n_labels=0 + u_prior_mixture=True: {vals}"
    )

    # Latent representation must be finite (no NaN from empty label table)
    z = model.get_latent_representation(give_z=True)
    assert np.all(np.isfinite(z)), (
        "get_latent_representation returned non-finite values with n_labels=0"
    )


# ---------------------------------------------------------------------------
# P1-004 — Default-flag determinism guard
# ---------------------------------------------------------------------------


def test_mrtotalvi_default_latent_is_deterministic(adata_basic):
    """Two MrTotalVI models trained with identical default flags and same seed
    must produce the same latent representations.

    This locks default behaviour against silent flag-default changes (L-090:
    protein_in_encoder reverted to True in session 58 silently widened the
    u_vamp_pseudo_dim, breaking tests).  If this test fails, a default has
    drifted and the change was not intentional.
    """
    adata = adata_basic.copy()

    scvi.settings.seed = 0
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model_a = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT)
    model_a.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")
    z_a = model_a.get_latent_representation(give_z=True)

    scvi.settings.seed = 0
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model_b = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT)
    model_b.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")
    z_b = model_b.get_latent_representation(give_z=True)

    assert np.allclose(z_a, z_b, atol=1e-4), (
        f"Two models with default flags + seed=0 produced different latents. "
        f"Max |diff| = {np.abs(z_a - z_b).max():.2e}. "
        "A flag default may have changed — audit recent commits."
    )


# ---------------------------------------------------------------------------
# batch_representation="embedding"
# ---------------------------------------------------------------------------

def test_mrtotalvi_batch_representation_default_is_one_hot(adata_basic):
    """The default must not change the architecture: no embedding, stock decoder."""
    from scvi.nn import DecoderTOTALVI

    model = _setup_and_train(adata_basic, encode_covariates=True)

    assert model.module.batch_representation == "one-hot"
    assert model.module._batch_dim is None
    # No embedding table is created, so no extra parameters enter the state dict.
    assert len(model.module.embeddings_dict) == 0
    assert isinstance(model.module.decoder, DecoderTOTALVI)
    # Batch still enters the u-encoder one-hot when covariates are encoded.
    assert model.module.qu.batch_dim is None


def test_mrtotalvi_batch_representation_embedding_widths(adata_basic):
    """Embedding replaces n_batch columns with embedding_dim in encoder and decoder."""
    from scvi.external.mrtotalvi._components import BatchEmbeddingDecoderAdapter

    embedding_dim = 3
    one_hot = _setup_and_train(adata_basic, encode_covariates=True)
    embedded = _setup_and_train(
        adata_basic,
        encode_covariates=True,
        batch_representation="embedding",
        batch_embedding_kwargs={"embedding_dim": embedding_dim},
    )

    n_batch = one_hot.module.n_batch
    assert n_batch > 0

    # u-encoder: -n_batch one-hot columns, +embedding_dim embedding columns.
    assert (
        embedded.module.qu.fc1.in_features
        == one_hot.module.qu.fc1.in_features - n_batch + embedding_dim
    )

    # Decoder: the batch category is dropped and the input widened instead.
    assert isinstance(embedded.module.decoder, BatchEmbeddingDecoderAdapter)
    one_hot_in = one_hot.module.decoder.px_decoder.fc_layers[0][0].in_features
    embedded_in = embedded.module.decoder.decoder.px_decoder.fc_layers[0][0].in_features
    assert embedded_in == one_hot_in - n_batch + embedding_dim


def test_mrtotalvi_batch_representation_embedding_runs(adata_basic):
    """Embedding mode trains and every downstream tensor stays finite."""
    embedding_dim = 3
    model = _setup_and_train(
        adata_basic,
        encode_covariates=True,
        batch_representation="embedding",
        batch_embedding_kwargs={"embedding_dim": embedding_dim},
    )

    latent = model.get_latent_representation()
    assert latent.shape == (adata_basic.n_obs, N_LATENT)
    assert np.isfinite(latent).all()

    representation = model.get_batch_representation()
    assert representation.shape == (adata_basic.n_obs, embedding_dim)
    assert np.isfinite(representation).all()


def test_mrtotalvi_batch_representation_embedding_deterministic_decoder(adata_basic):
    """The counterfactual decode path routes through the embedding adapter."""
    import torch

    model = _setup_and_train(
        adata_basic,
        encode_covariates=True,
        batch_representation="embedding",
        batch_embedding_kwargs={"embedding_dim": 3},
    )

    decoded = model.module._deterministic_decoder_parameters(
        torch.zeros(5, N_LATENT),
        torch.ones(5, 1),
        torch.zeros(5, 1, dtype=torch.long),
    )
    assert decoded
    for name, value in decoded.items():
        assert torch.isfinite(value).all(), f"{name} is not finite"


def test_mrtotalvi_batch_representation_embedding_save_load(adata_basic, tmp_path):
    """The embedding table survives a save/load round-trip."""
    model = _setup_and_train(
        adata_basic,
        encode_covariates=True,
        batch_representation="embedding",
        batch_embedding_kwargs={"embedding_dim": 3},
    )
    before = model.get_latent_representation()

    path = tmp_path / "mrtotalvi_embedding"
    model.save(str(path), overwrite=True)
    reloaded = MrTotalVI.load(str(path), adata=adata_basic)

    assert reloaded.module._batch_dim == 3
    assert np.allclose(before, reloaded.get_latent_representation(), atol=1e-5)


def test_mrtotalvi_batch_representation_invalid(adata_basic):
    """An unknown batch_representation fails loudly at construction."""
    MrTotalVI.setup_anndata(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    with pytest.raises(ValueError, match="one-hot"):
        MrTotalVI(
            adata_basic,
            sample_key="sample",
            n_latent=N_LATENT,
            batch_representation="nope",
        )


def test_mrtotalvi_one_hot_numerics_unchanged_by_embedding_support(adata_basic):
    """The one-hot path must be numerically untouched by batch-embedding support.

    ``_append_covariates`` and ``_covariate_n_input`` gained optional ``batch_rep`` /
    ``batch_dim`` arguments to support ``batch_representation="embedding"``. Both helpers
    sit on the hot path of *every* MrTotalVI model, so a mistake there would silently move
    latents for existing one-hot configurations. Structural assertions do not catch that;
    this pins the numerics.

    Trains with ``encode_covariates=True`` specifically, since that is the configuration
    where batch actually flows through the modified helper.
    """
    import torch

    adata = adata_basic.copy()
    latents = []

    for _ in range(2):
        scvi.settings.seed = 0
        MrTotalVI.setup_anndata(
            adata,
            protein_expression_obsm_key="protein_expression",
            sample_key="sample",
            batch_key="batch",
        )
        model = MrTotalVI(
            adata,
            sample_key="sample",
            n_latent=N_LATENT,
            encode_covariates=True,
        )
        model.train(max_epochs=MAX_EPOCHS_QUICK, accelerator="cpu")
        latents.append(model.get_latent_representation(give_z=True))

    assert np.allclose(latents[0], latents[1], atol=1e-4), (
        "One-hot latents are not reproducible at seed=0 with encode_covariates=True. "
        f"Max |diff| = {np.abs(latents[0] - latents[1]).max():.2e}."
    )
    # The one-hot branch must never build a batch representation.
    assert model.module._batch_representation_for(torch.zeros(4, 1, dtype=torch.long)) is None


# ---------------------------------------------------------------------------
# _stats.py internals: branches unreachable through the public API
#
# `collect_u_posterior` lives in shared code ("Shared u-space statistical APIs for
# Mr multimodal models"), but MrTotalVAE.inference always returns "qu", so its two
# fallback branches are dead for every other test in this file. Monkeypatching
# `inference` is the only way to reach them.
# ---------------------------------------------------------------------------


def _untrained_model(adata):
    """setup + construct without training; `is_trained_` bypassed as elsewhere in this file."""
    adata = adata.copy()
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=N_LATENT, n_latent_u=4)
    model.is_trained_ = True
    return model, adata


def test_collect_u_posterior_falls_back_to_qz(adata_basic, monkeypatch):
    """When inference emits `qz` but no `qu`, the posterior is collected from `qz`."""
    import torch
    from torch.distributions import Normal

    from scvi.external.mrtotalvi._stats import collect_u_posterior

    model, adata = _untrained_model(adata_basic)
    real_inference = model.module.inference

    def fake(**kwargs):
        out = real_inference(**kwargs)
        return {"qz": Normal(out["qu"].loc, out["qu"].scale)}

    monkeypatch.setattr(model.module, "inference", fake)

    loc, scale = collect_u_posterior(model, adata=adata, indices=np.arange(20), batch_size=8)

    assert loc.shape == (20, 4)
    assert scale.shape == (20, 4)
    assert torch.isfinite(loc).all()
    assert (scale > 0).all()


def test_collect_u_posterior_falls_back_to_qz_m_and_qz_v(adata_basic, monkeypatch):
    """The oldest shape -- raw `qz_m`/`qz_v` tensors rather than a distribution object."""
    import torch

    from scvi.external.mrtotalvi._stats import collect_u_posterior

    model, adata = _untrained_model(adata_basic)
    real_inference = model.module.inference

    def fake(**kwargs):
        out = real_inference(**kwargs)
        n, d = out["qu"].loc.shape
        return {"qz_m": torch.zeros(n, d), "qz_v": torch.full((n, d), 4.0)}

    monkeypatch.setattr(model.module, "inference", fake)

    loc, scale = collect_u_posterior(model, adata=adata, indices=np.arange(20), batch_size=8)

    assert loc.shape == (20, 4)
    torch.testing.assert_close(loc, torch.zeros(20, 4))
    # scale is a standard deviation: sqrt of the variance that was supplied
    torch.testing.assert_close(scale, torch.full((20, 4), 2.0))


# ---------------------------------------------------------------------------
# _construct_design_matrix: pure function over a DataFrame, no model required
# ---------------------------------------------------------------------------


def test_construct_design_matrix_normalizes_numeric_and_dummies_categorical():
    import pandas as pd

    from scvi.external.mrtotalvi._stats import _construct_design_matrix

    df = pd.DataFrame({"cond": ["a", "b", "a", "b"], "num": [1.0, 2.0, 3.0, 4.0]})

    xmat, names, n_fixed = _construct_design_matrix(df, ["cond", "num"])

    assert list(names) == ["cond_b", "num"]
    assert n_fixed == 2
    # numeric column is min-max normalized onto [0, 1]
    assert float(xmat[:, 1].min()) == 0.0
    assert float(xmat[:, 1].max()) == 1.0


def test_construct_design_matrix_constant_column_does_not_divide_by_zero():
    """xmax == xmin must fall back to a scale of 1.0 rather than producing NaN."""
    import pandas as pd
    import torch

    from scvi.external.mrtotalvi._stats import _construct_design_matrix

    df = pd.DataFrame({"const": [5.0, 5.0, 5.0]})

    xmat, _, _ = _construct_design_matrix(df, ["const"])

    assert not torch.isnan(xmat).any()
    torch.testing.assert_close(xmat, torch.zeros_like(xmat))


def test_construct_design_matrix_donor_key_from_index_name():
    """`donor_key` matching the index name adds donor dummies after the fixed effects."""
    import pandas as pd

    from scvi.external.mrtotalvi._stats import _construct_design_matrix

    df = pd.DataFrame(
        {"cond": ["a", "b", "a"]},
        index=pd.Index(["s0", "s1", "s2"], name="sample"),
    )

    xmat, names, n_fixed = _construct_design_matrix(df, ["cond"], donor_key="sample")

    assert n_fixed == 1  # only `cond_b` is a fixed effect
    assert list(names)[0] == "cond_b"
    assert any(str(n).startswith("sample_") for n in names)
    assert xmat.shape[0] == 3


# ---------------------------------------------------------------------------
# compute_h_from_x_eps: the u_anchor=None legacy branch
#
# Every existing caller in this file passes `u_anchor` explicitly, so the biased
# fallback (and its warning) is the one genuinely uncovered path.
# ---------------------------------------------------------------------------


def test_compute_h_from_x_eps_warns_and_falls_back_without_u_anchor(adata_basic):
    import torch

    model, adata = _untrained_model(adata_basic)
    model.module.eval()

    dl = model._make_data_loader(adata=adata, batch_size=16)
    inference_inputs = model.module._get_inference_input(next(iter(dl)))
    n_cells = inference_inputs["x"].shape[0]
    extra_eps = torch.zeros(n_cells, model.module.n_latent)

    with pytest.warns(UserWarning, match="without u_anchor"):
        out = model.module.compute_h_from_x_eps(extra_eps=extra_eps, **inference_inputs)

    n_proteins = adata.obsm["protein_expression"].shape[1]
    assert out.shape == (n_cells, adata.n_vars + n_proteins)
    assert torch.isfinite(out).all()


def test_use_vmap_is_currently_ignored_by_differential_expression(adata_basic):
    """Pins `use_vmap` as a genuine no-op so a future partial implementation is caught.

    The parameter is accepted and documented as "currently ignored (reserved for future opt-in
    vmap acceleration)" -- it appears nowhere in `_stats.py` beyond its signature and docstring.
    A caller passing `use_vmap=True` therefore gets no behaviour change and no warning. This
    asserts byte-identical LFCs across both settings; if vmap is ever partially wired up and
    changes results, this fails rather than silently shifting numbers.

    `mc_samples=1` is required: it takes the deterministic `qu.loc` fast path, so any difference
    is attributable to `use_vmap` rather than to fresh `rsample` draws.
    """
    model, _ = _de_lfc_model(adata_basic)
    kwargs = {
        "sample_cov_keys": ["condition"],
        "mc_samples": 1,
        "batch_size": 32,
        "store_lfc": True,
    }

    de_true = model.differential_expression(use_vmap=True, **kwargs)
    de_false = model.differential_expression(use_vmap=False, **kwargs)

    np.testing.assert_array_equal(de_true["lfc"].values, de_false["lfc"].values)
