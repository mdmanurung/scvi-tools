"""Shared u-space statistical APIs for Mr multimodal models."""

from __future__ import annotations

import warnings

import numpy as np
import torch
import xarray as xr
from scipy.special import logsumexp
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from tqdm import tqdm


def _validate_sample_level_covariates(obs, sample_key: str, cov_keys: list[str]) -> None:
    """Raise ValueError if any cov in cov_keys is not constant within sample_key."""
    for cov in cov_keys:
        if cov not in obs.columns:
            continue
        n_per_sample = obs.groupby(sample_key, observed=True)[cov].nunique()
        bad = n_per_sample[n_per_sample > 1]
        if len(bad):
            raise ValueError(
                f"sample_cov_key '{cov}' is not constant within sample_key '{sample_key}' "
                f"for the following samples: {bad.index.tolist()}. "
                f"Each sample (sample_key level) must map to exactly one covariate value; "
                f"otherwise drop_duplicates() picks an arbitrary value and the design matrix "
                f"is incorrect. To test a within-sample condition (e.g. timepoint), retrain "
                f"with a more granular sample_key (e.g. 'donor_timepoint')."
            )


def _rowwise_max_excluding_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(matrix.shape)}.")
    n_cols = matrix.shape[1]
    if n_cols == 1:
        return matrix[:, 0]
    mask = ~torch.eye(n_cols, dtype=torch.bool, device=matrix.device)
    return torch.where(mask, matrix, torch.tensor(-float("inf"), device=matrix.device)).max(
        dim=1
    ).values


def _as_indices(adata, indices):
    if indices is None:
        return np.arange(adata.n_obs)
    return np.asarray(indices)


def collect_u_posterior(model, adata=None, indices=None, batch_size: int = 256):
    """Collect posterior Normal parameters for ``u`` from a model dataloader."""
    adata = model._validate_anndata(adata)
    indices = _as_indices(adata, indices)
    dataloader = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

    locs = []
    scales = []
    was_training = model.module.training
    model.module.eval()
    try:
        with torch.inference_mode():
            for tensors in dataloader:
                inference_inputs = model.module._get_inference_input(tensors)
                outputs = model.module.inference(**inference_inputs)
                qu = outputs.get("qu")
                if qu is None:
                    qu = outputs.get("qz")
                if qu is None:
                    locs.append(outputs["qz_m"])
                    scales.append(torch.sqrt(outputs["qz_v"]))
                else:
                    locs.append(qu.loc)
                    scales.append(qu.scale)
    finally:
        model.module.train(was_training)

    return torch.cat(locs, dim=0), torch.cat(scales, dim=0)


def get_aggregated_posterior(
    model,
    *,
    adata=None,
    sample_key: str,
    sample: str | int | None = None,
    indices=None,
    batch_size: int = 256,
) -> MixtureSameFamily:
    """Compute the aggregated posterior mixture over ``u``."""
    model._check_if_trained(warn=False)
    adata = model._validate_anndata(adata)
    indices = _as_indices(adata, indices)
    if sample is not None:
        indices = np.intersect1d(indices, np.where(adata.obs[sample_key] == sample)[0])
    if len(indices) == 0:
        raise ValueError("Cannot build an aggregated posterior from zero cells.")

    loc, scale = collect_u_posterior(model, adata=adata, indices=indices, batch_size=batch_size)
    weights = torch.ones(loc.shape[0], device=loc.device, dtype=loc.dtype) / loc.shape[0]
    return MixtureSameFamily(Categorical(probs=weights), Independent(Normal(loc, scale), 1))


def differential_abundance(
    model,
    *,
    adata=None,
    sample_key: str,
    sample_cov_keys: list[str] | None = None,
    sample_subset: list[str] | None = None,
    compute_log_enrichment: bool = False,
    omit_original_sample: bool = True,
    donor_key: str | None = None,
    batch_size: int = 128,
) -> xr.Dataset:
    """Compute MrVI-style differential abundance log probabilities over ``u``."""
    adata = model._validate_anndata(adata)

    if sample_cov_keys:
        _validate_sample_level_covariates(adata.obs, sample_key, list(sample_cov_keys))

    sample_values = list(model.sample_order) if hasattr(model, "sample_order") else list(
        adata.obs[sample_key].unique()
    )
    if sample_subset is not None:
        sample_values = [s for s in sample_values if s in set(sample_subset)]

    us = model.get_latent_representation(
        adata,
        give_mean=True,
        give_z=False,
        batch_size=batch_size,
    )

    log_probs = []
    n_splits = max(int(np.ceil(adata.n_obs / batch_size)), 1)
    for sample_name in tqdm(sample_values, desc="Aggregated posterior log probabilities"):
        ap = model.get_aggregated_posterior(
            adata=adata,
            sample=sample_name,
            batch_size=batch_size,
        )
        sample_log_probs = []
        for u_rep in np.array_split(us, n_splits):
            u_tensor = torch.as_tensor(
                u_rep,
                dtype=ap.component_distribution.base_dist.loc.dtype,
                device=ap.component_distribution.base_dist.loc.device,
            )
            sample_log_probs.append(ap.log_prob(u_tensor).detach().cpu().numpy()[:, None])
        log_probs.append(np.concatenate(sample_log_probs, axis=0))

    log_probs = np.concatenate(log_probs, axis=1)

    if donor_key is not None:
        # Within-donor centering: subtract the per-donor mean across that donor's
        # samples so that donor blocks are absorbed before covariate aggregation.
        sample_info_da = (
            adata.obs[[sample_key, donor_key]]
            .drop_duplicates(subset=sample_key)
            .set_index(sample_key)
        )
        for donor_id in sample_info_da[donor_key].unique():
            donor_samples = sample_info_da.index[sample_info_da[donor_key] == donor_id].tolist()
            col_idx = [i for i, s in enumerate(sample_values) if s in set(donor_samples)]
            if len(col_idx) < 2:
                continue
            donor_mean = log_probs[:, col_idx].mean(axis=1, keepdims=True)
            log_probs[:, col_idx] -= donor_mean

    coords = {"cell_name": adata.obs_names.to_numpy(), "sample": np.asarray(sample_values)}
    data_vars = {"log_probs": (["cell_name", "sample"], log_probs)}
    log_probs_ds = xr.Dataset(data_vars, coords=coords)

    if not sample_cov_keys:
        return log_probs_ds

    for key in sample_cov_keys:
        n_cov_values = len(adata.obs[key].unique())
        if n_cov_values > len(sample_values) / 2:
            warnings.warn(
                f"The covariate '{key}' has {n_cov_values} unique values across "
                f"{len(sample_values)} samples; DA covariates should be discrete.",
                UserWarning,
                stacklevel=2,
            )

    sample_info = adata.obs[[sample_key, *sample_cov_keys]].drop_duplicates(subset=sample_key)

    def aggregate(samples: np.ndarray) -> np.ndarray:
        sample_log_probs = log_probs_ds.log_probs.loc[{"sample": samples}].values
        if omit_original_sample:
            sample_one_hot = np.zeros((adata.n_obs, len(samples)))
            for i, sample_name in enumerate(samples):
                sample_one_hot[adata.obs[sample_key].to_numpy() == sample_name, i] = 1
            denom = np.maximum((1 - sample_one_hot).sum(axis=1), 1)
            sample_log_probs = np.where(sample_one_hot, -np.inf, sample_log_probs)
            return logsumexp(sample_log_probs, axis=1) - np.log(denom)
        return logsumexp(sample_log_probs, axis=1) - np.log(sample_log_probs.shape[1])

    cov_log_probs = {}
    cov_log_enrichs = {}
    for key in sample_cov_keys:
        per_value = {}
        per_enrich = {}
        for value in sample_info[key].unique():
            samples = sample_info.loc[sample_info[key] == value, sample_key].to_numpy()
            if sample_subset is not None:
                samples = np.intersect1d(samples, np.asarray(sample_subset))
            if len(samples) == 0:
                continue
            val_log_probs = aggregate(samples)
            per_value[value] = val_log_probs
            if compute_log_enrichment:
                rest = np.setdiff1d(np.asarray(sample_values), samples)
                if len(rest) == 0:
                    continue
                per_enrich[value] = val_log_probs - aggregate(rest)
        values = list(per_value)
        cov_log_probs[key] = (values, np.vstack([per_value[v] for v in values]).T)
        if per_enrich:
            enrich_values = list(per_enrich)
            cov_log_enrichs[key] = (
                enrich_values,
                np.vstack([per_enrich[v] for v in enrich_values]).T,
            )

    coords = {"cell_name": adata.obs_names.to_numpy(), "sample": np.asarray(sample_values)}
    for key, (values, _) in cov_log_probs.items():
        coords[key] = np.asarray(values)

    data_vars = {"log_probs": (["cell_name", "sample"], log_probs)}
    for key, (_, values) in cov_log_probs.items():
        data_vars[f"{key}_log_probs"] = (["cell_name", key], values)
    if compute_log_enrichment:
        for key, (values, arr) in cov_log_enrichs.items():
            enrich_dim = f"{key}_enrichment"
            coords[enrich_dim] = np.asarray(values)
            data_vars[f"{key}_log_enrichs"] = (["cell_name", enrich_dim], arr)

    return xr.Dataset(data_vars, coords=coords)


def _construct_design_matrix(
    sample_info,
    sample_cov_keys: list[str],
    normalize_design_matrix: bool = True,
    donor_key: str | None = None,
) -> tuple:
    """Build a float design matrix from sample-level metadata.

    Parameters
    ----------
    sample_info
        DataFrame indexed by sample name; must contain all ``sample_cov_keys``
        (and ``donor_key`` if provided).
    sample_cov_keys
        Column names for fixed-effect covariates.
    normalize_design_matrix
        Min-max normalize each fixed-effect column to [0, 1].
    donor_key
        Optional column to add donor dummy columns after the fixed effects as
        nuisance covariates (not normalized, not reported in output betas).

    Returns
    -------
    Xmat : ``torch.FloatTensor`` of shape ``(n_samples, n_covariates)``
    covariate_names : ``np.ndarray`` of covariate names
    n_fixed : int — number of reported fixed-effect columns
    """
    import pandas as pd

    parts: list[np.ndarray] = []
    names: list[str] = []

    for key in sample_cov_keys:
        cov = sample_info[key]
        if cov.dtype == object or hasattr(cov, "cat"):
            if hasattr(cov, "cat"):
                cov = cov.cat.remove_unused_categories()
            dummies = pd.get_dummies(cov, drop_first=True)
            parts.append(dummies.values.astype(np.float32))
            names.extend([f"{key}_{c}" for c in dummies.columns])
        else:
            parts.append(cov.values[:, None].astype(np.float32))
            names.append(key)

    if parts:
        xmat_fixed = np.concatenate(parts, axis=1)
    else:
        xmat_fixed = np.zeros((len(sample_info), 0), np.float32)
    names_fixed = np.array(names, dtype=str)

    if normalize_design_matrix and xmat_fixed.shape[1] > 0:
        xmin = xmat_fixed.min(axis=0)
        xmax = xmat_fixed.max(axis=0)
        scale = np.where(xmax - xmin > 1e-6, xmax - xmin, 1.0)
        xmat_fixed = (xmat_fixed - xmin) / scale

    n_fixed = xmat_fixed.shape[1]

    if donor_key is not None:
        # donor_key may equal the DataFrame's index name (when donor_key == model.sample_key).
        if donor_key == sample_info.index.name:
            cov = sample_info.index.to_series()
        else:
            cov = sample_info[donor_key]
        if hasattr(cov, "cat"):
            cov = cov.cat.remove_unused_categories()
        donor_dummies = pd.get_dummies(cov, drop_first=True)
        donor_mat = donor_dummies.values.astype(np.float32)
        donor_names = np.array([f"{donor_key}_{c}" for c in donor_dummies.columns], dtype=str)
        if n_fixed > 0:
            xmat = np.concatenate([xmat_fixed, donor_mat], axis=1)
            names_all = np.concatenate([names_fixed, donor_names])
        else:
            xmat = donor_mat
            names_all = donor_names
    else:
        xmat = xmat_fixed
        names_all = names_fixed

    return torch.tensor(xmat, dtype=torch.float32), names_all, n_fixed


def _differential_expression(
    model,
    *,
    adata=None,
    sample_cov_keys: list[str],
    sample_subset: list[str] | None = None,
    indices=None,
    batch_size: int = 128,
    normalize_design_matrix: bool = True,
    mc_samples: int = 50,
    filter_inadmissible_samples: bool = False,
    store_lfc: bool = False,
    donor_key: str | None = None,
    delta: float | None = 0.3,
    lambd: float = 0.0,
    store_baseline: bool = False,
    eps_lfc: float = 1e-6,
    use_vmap: bool = False,
    **filter_samples_kwargs,
) -> xr.Dataset:
    """MrVI-style latent-space differential expression for Mr multimodal models.

    Fits a cell-specific weighted least-squares linear model on the sample-
    specific residual ``eps_d = z_d - z_base`` (the donor latent shift),
    following the approach of MrVI (Boyeau et al., 2023).

    Parameters
    ----------
    model
        A trained ``MrTotalVI`` or ``MrMultiVI`` instance.
    adata
        AnnData/MuData to compute DE on.  Defaults to the training data.
    sample_cov_keys
        Sample-level covariate column names in ``adata.obs``.
    sample_subset
        Restrict DE to these sample names only.
    indices
        Cell indices (default: all cells).
    batch_size
        Dataloader batch size.
    normalize_design_matrix
        Min-max normalize each covariate column to [0, 1].
    mc_samples
        Monte-Carlo draws from ``q(u)``.  ``1`` uses the posterior mean.
    filter_inadmissible_samples
        Weight out outlier cell-sample pairs via aggregated-posterior scores.
    store_lfc
        If ``True``, compute and store gene/protein log2-fold-changes (LFC) in
        the returned dataset.  Requires ``model.module`` to implement
        ``compute_h_from_x_eps``.  LFC is batch-marginalized by weighting over
        unique batch values observed in each mini-batch.
    donor_key
        Column in ``adata.obs`` identifying the donor.  Adds donor dummy
        columns as nuisance covariates; only fixed-effect betas are returned.
    delta
        Effect-size threshold for the posterior-DE probability (PDE):
        ``pde = P(|lfc| >= delta)``.  Only used when ``store_lfc=True``.
        Set to ``None`` to skip PDE computation.
    lambd
        L2 regularisation added to ``X^T M X`` before inversion.
    store_baseline
        When ``store_lfc=True``, also store the batch-marginalized baseline
        expression ``h(x, eps_mean)`` (the null decode) as
        ``baseline_expression`` in the returned dataset.
    eps_lfc
        Small constant added to expression values before taking log2 to avoid
        log(0).  Default ``1e-6``.
    use_vmap
        Currently ignored (reserved for future opt-in vmap acceleration).
        The LFC path always uses explicit Python loops, which are safe with
        BatchNorm-based decoders.
    **filter_samples_kwargs
        Forwarded to :func:`get_outlier_cell_sample_pairs`.
    """
    import scipy.linalg
    import torch.distributions as tdist
    from scipy.stats import false_discovery_control

    from scvi import REGISTRY_KEYS as RK

    if store_lfc and not hasattr(model.module, "compute_h_from_x_eps"):
        raise NotImplementedError(
            "store_lfc requires model.module.compute_h_from_x_eps, "
            "which is not implemented for this module."
        )
    if not sample_cov_keys:
        raise ValueError("sample_cov_keys must contain at least one sample-level covariate.")
    if donor_key is not None:
        warnings.warn(
            "donor_key adds donor dummies as nuisance covariates.  This is only valid "
            "when each donor spans multiple levels of the sample_cov_keys covariates "
            "(e.g. multiple time points).  If each donor maps to exactly one condition "
            "the design is collinear and fixed-effect betas will be unreliable.",
            UserWarning,
            stacklevel=2,
        )

    model._check_if_trained(warn=False)
    adata = model._validate_anndata(adata)

    if mc_samples < 1:
        raise ValueError("mc_samples must be >= 1.")
    _cov_to_check = list(sample_cov_keys)
    if donor_key and donor_key != model.sample_key:
        _cov_to_check.append(donor_key)
    _validate_sample_level_covariates(adata.obs, model.sample_key, _cov_to_check)

    n_sample = model.summary_stats.n_sample
    sample_mask = (
        np.isin(model.sample_order, list(sample_subset))
        if sample_subset is not None
        else np.ones(n_sample, dtype=bool)
    )
    sample_order_kept = model.sample_order[sample_mask]
    sample_indices_kept = np.where(sample_mask)[0].tolist()
    n_sample_kept = len(sample_indices_kept)

    # Admissibility mask: (n_cells_total, n_sample_kept)
    if filter_inadmissible_samples:
        sample_key = model.sample_key
        outliers = get_outlier_cell_sample_pairs(
            model,
            adata=adata,
            sample_key=sample_key,
            **filter_samples_kwargs,
        )
        admissible_samples = (
            outliers["is_admissible"]
            .sel(sample=sample_order_kept)
            .values.astype(np.float32)
        )
    else:
        admissible_samples = np.ones((adata.n_obs, n_sample_kept), dtype=np.float32)

    # Design matrix from sample-level metadata.
    # Exclude donor_key from the column list when it equals model.sample_key — that
    # column will become the index after set_index(), so selecting it twice would
    # produce duplicate columns and a tuple MultiIndex.
    cov_cols = list(sample_cov_keys) + ([donor_key] if donor_key else [])
    cov_cols_for_select = [c for c in cov_cols if c != model.sample_key]
    sample_info = (
        adata.obs[[model.sample_key] + cov_cols_for_select]
        .drop_duplicates(subset=model.sample_key)
        .set_index(model.sample_key)
        .loc[sample_order_kept]
    )
    Xmat, Xmat_names, n_fixed = _construct_design_matrix(
        sample_info,
        list(sample_cov_keys),
        normalize_design_matrix=normalize_design_matrix,
        donor_key=donor_key,
    )
    n_covariates = Xmat.shape[1]
    device = model.device
    Xmat = Xmat.to(device)

    def _sqrtm_batch(xtmx: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                torch.from_numpy(
                    scipy.linalg.sqrtm(m.cpu().numpy()).real.astype(np.float32)
                ).to(device)
                for m in xtmx
            ]
        )

    scdl = model._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

    beta_list: list[np.ndarray] = []
    effect_size_list: list[np.ndarray] = []
    pvalue_list: list[np.ndarray] = []
    all_cell_names: list[str] = []

    lfc_list: list[np.ndarray] = []
    lfc_std_list: list[np.ndarray] = []
    pde_list: list[np.ndarray] | None = [] if (store_lfc and delta is not None) else None
    baseline_list: list[np.ndarray] | None = [] if (store_lfc and store_baseline) else None

    was_training = model.module.training
    model.module.eval()
    try:
        with torch.inference_mode():
            for tensors in tqdm(scdl, desc="MrVI-style DE"):
                cell_idx = tensors[RK.INDICES_KEY].long().flatten().cpu().numpy()
                n_cells = len(cell_idx)
                all_cell_names.extend(adata.obs_names[cell_idx].tolist())

                inf_inputs = model.module._get_inference_input(tensors)
                base_out = model.module.inference(**inf_inputs)
                qu = base_out.get("qu")
                if qu is None:
                    raise RuntimeError(
                        "model.module.inference() did not return 'qu'; "
                        "cannot compute MC eps samples."
                    )

                def _eps_from_u(u: torch.Tensor) -> torch.Tensor:
                    # For each kept sample d, compute eps_d = qz(u, d)[1].
                    # u is held fixed across donors: counterfactual substitution.
                    eps_list = []
                    for d_idx in sample_indices_kept:
                        cf = torch.full(
                            (n_cells, 1), d_idx, dtype=torch.long, device=u.device
                        )
                        _, eps_d, _ = model.module.qz(u, cf)
                        eps_list.append(eps_d)
                    # (n_cells, n_sample_kept, n_latent)
                    return torch.stack(eps_list, dim=1)

                if mc_samples == 1:
                    u_samples = [qu.mean]
                    eps_batch = _eps_from_u(qu.mean).unsqueeze(0)
                else:
                    u_samples = [qu.rsample() for _ in range(mc_samples)]
                    eps_batch = torch.stack(
                        [_eps_from_u(u_i) for u_i in u_samples], dim=0
                    )
                # eps_batch: (mc, n_cells, n_sample_kept, n_latent)

                # Admissibility for this batch
                admiss = torch.from_numpy(admissible_samples[cell_idx]).to(device)
                # (n_cells, n_sample_kept)
                n_per_cell = admiss.sum(1)  # (n_cells,)

                # X^T M X per cell: einsum over sample dimension s
                # Xmat: (n_sample_kept, n_cov)  →  s,k
                # admiss: (n_cells, n_sample_kept)  →  n,s
                xtmx = (
                    torch.einsum("sk,ns,sl->nkl", Xmat, admiss, Xmat)
                    + lambd * torch.eye(n_covariates, device=device).unsqueeze(0)
                )  # (n_cells, n_cov, n_cov)
                prefactor = _sqrtm_batch(xtmx)  # (n_cells, n_cov, n_cov)
                inv_ = torch.vmap(torch.linalg.pinv)(xtmx)  # (n_cells, n_cov, n_cov)
                # Amat = (X^T M X)^{-1} X^T M  →  (n_cells, n_cov, n_sample_kept)
                Amat = torch.einsum("nkj,sj,ns->nks", inv_, Xmat, admiss)

                # Standardise eps across sample dimension (dim=2)
                eps_mean = eps_batch.mean(dim=2, keepdim=True)
                eps_std = eps_batch.std(dim=2, keepdim=True)
                eps_norm = (eps_batch - eps_mean) / (eps_std + 1e-6)
                # (mc, n_cells, n_sample_kept, n_latent)

                # OLS per cell per MC draw
                # Amat: (n_cells, n_cov, n_sample)  →  n,k,s
                # eps_norm: (mc, n_cells, n_sample, n_latent)  →  a,n,s,d
                # betas: (mc, n_cells, n_cov, n_latent)
                betas = torch.einsum("nks,ansd->ankd", Amat, eps_norm)

                # Wald: prefactor: (n_cells, n_cov, n_cov)  →  n,l,k
                # betas_norm: (mc, n_cells, n_cov, n_latent)  →  a,n,l,d
                betas_norm = torch.einsum("nlk,ankd->anld", prefactor, betas)
                # ts: mean over mc, sum over latent → (n_cells, n_cov)
                ts = (betas_norm**2).sum(-1).mean(0)

                # Chi2 p-values with df = n_admissible_per_cell
                df = n_per_cell.clamp(min=1).float().unsqueeze(-1).expand_as(ts)
                pvals = 1.0 - tdist.Chi2(df).cdf(ts)  # (n_cells, n_cov)

                # Un-standardise betas and average over MC draws
                betas_rescaled = (betas * eps_std).mean(0)  # (n_cells, n_cov, n_latent)

                beta_list.append(betas_rescaled[:, :n_fixed, :].cpu().numpy())
                effect_size_list.append(ts[:, :n_fixed].cpu().numpy())
                pvalue_list.append(pvals[:, :n_fixed].cpu().numpy())

                # ---- gene/protein LFC block ----
                if store_lfc:
                    # betas_eps: per-draw betas in original eps space
                    # shape (mc, n_cells, n_fixed, n_latent)
                    betas_eps = (betas * eps_std)[:, :, :n_fixed, :]

                    # null eps position: mean over MC draws and sample dim
                    # eps_mean: (mc, n_cells, 1, n_latent) → average to (n_cells, n_latent)
                    eps_mean_cell = eps_mean.mean(0).squeeze(1)  # (n_cells, n_latent)

                    # Build kwargs for compute_h_from_x_eps from inference inputs;
                    # batch_index is overridden per counterfactual batch below.
                    h_kwargs = dict(inf_inputs)
                    # _get_inference_input does not return extra_eps or cf_sample,
                    # so h_kwargs can be passed directly to compute_h_from_x_eps.

                    # Batch-weighted LFC: marginalize over unique batch values.
                    batch_index_obs = inf_inputs["batch_index"]  # (n_cells, 1)
                    unique_batches = batch_index_obs.unique()
                    batch_counts = torch.stack(
                        [(batch_index_obs == b).sum().float() for b in unique_batches]
                    )
                    batch_weights = batch_counts / batch_counts.sum()

                    # Accumulators (initialized lazily on first batch value)
                    lfc_wsum: torch.Tensor | None = None   # (n_fixed, n_cells, n_features)
                    lfc_var_wsum: torch.Tensor | None = None
                    pde_wsum: torch.Tensor | None = None
                    baseline_wsum: torch.Tensor | None = None

                    for b_idx, b_val in enumerate(unique_batches):
                        w_b = batch_weights[b_idx]
                        batch_cf = torch.full_like(batch_index_obs, b_val.item())
                        h_kwargs_b = dict(h_kwargs)
                        h_kwargs_b["batch_index"] = batch_cf

                        # Cache inference aux once per batch value.  library_gene (and
                        # qz_m for MrMultiVI) depend only on x/batch_index, not on
                        # u_anchor, so they are identical for all mc_samples draws.
                        # Without this cache, compute_h_from_x_eps would re-run the
                        # full encoder mc_samples*(1+n_fixed) times per batch value.
                        _lfc_aux = None
                        if hasattr(model.module, "_infer_lfc_aux"):
                            _lfc_aux = model.module._infer_lfc_aux(**h_kwargs_b)

                        # CRN: null decode per MC draw, sharing u_samples[mc_idx]
                        # with the contrast decode below.  x_0 varies with u so
                        # lfc_std now captures u-posterior uncertainty (not just
                        # regression uncertainty in beta).
                        x0_mc = [] if store_baseline else None
                        log_x0_mc = []
                        for mc_idx in range(mc_samples):
                            x_0 = model.module.compute_h_from_x_eps(
                                extra_eps=eps_mean_cell,
                                u_anchor=u_samples[mc_idx],
                                _lfc_aux=_lfc_aux,
                                **h_kwargs_b,
                            )  # (n_cells, n_features)
                            if x0_mc is not None:
                                x0_mc.append(x_0)
                            log_x0_mc.append(torch.log2(x_0 + eps_lfc))

                        if lfc_wsum is None:
                            n_features_lfc = log_x0_mc[0].shape[-1]
                            lfc_wsum = torch.zeros(
                                n_fixed, n_cells, n_features_lfc, device=device
                            )
                            lfc_var_wsum = torch.zeros_like(lfc_wsum)
                            if delta is not None:
                                pde_wsum = torch.zeros_like(lfc_wsum)
                            if store_baseline:
                                baseline_wsum = torch.zeros(
                                    n_cells, n_features_lfc, device=device
                                )

                        if store_baseline:
                            baseline_wsum += w_b * torch.stack(x0_mc, 0).mean(0)

                        # Contrast decode per covariate × MC draw; share u_anchor (CRN)
                        lfc_mc_cov_list = []
                        for k in range(n_fixed):
                            lfc_mc_k = []
                            for mc_idx in range(mc_samples):
                                extra = (
                                    betas_eps[mc_idx, :, k, :] + eps_mean_cell
                                )  # (n_cells, n_latent)
                                x_1 = model.module.compute_h_from_x_eps(
                                    extra_eps=extra,
                                    u_anchor=u_samples[mc_idx],
                                    _lfc_aux=_lfc_aux,
                                    **h_kwargs_b,
                                )  # (n_cells, n_features)
                                lfc_mc_k.append(
                                    torch.log2(x_1 + eps_lfc) - log_x0_mc[mc_idx]
                                )
                            lfc_mc_cov_list.append(
                                torch.stack(lfc_mc_k, 0)
                            )  # (mc, n_cells, n_features)
                        lfc_mc_cov = torch.stack(
                            lfc_mc_cov_list, 0
                        )  # (n_fixed, mc, n_cells, n_features)

                        lfc_wsum += w_b * lfc_mc_cov.mean(1)
                        # correction=0 (population variance) because mc_samples=1 is valid;
                        # correction=1 yields NaN via 0/0 when mc_samples=1.
                        # Note: this accumulates within-batch MC variance only; the
                        # between-batch term sum(w*(mu_b-mu)^2) is omitted — matching the
                        # MRVI reference (mrvi_torch/_model.py:1444-1448), so lfc_std is
                        # slightly conservative when batches have different mean LFC.
                        lfc_var_wsum += w_b * lfc_mc_cov.var(1, correction=0)
                        if delta is not None:
                            pde_wsum += w_b * (lfc_mc_cov.abs() >= delta).float().mean(1)

                    # lfc_wsum: (n_fixed, n_cells, n_features) → permute to (n_cells, n_fixed, n_features)
                    lfc_list.append(lfc_wsum.permute(1, 0, 2).cpu().numpy())
                    lfc_std_list.append(
                        torch.sqrt(lfc_var_wsum).permute(1, 0, 2).cpu().numpy()
                    )
                    if delta is not None:
                        pde_list.append(pde_wsum.permute(1, 0, 2).cpu().numpy())
                    if store_baseline:
                        baseline_list.append(baseline_wsum.cpu().numpy())
    finally:
        model.module.train(was_training)

    beta = np.concatenate(beta_list, 0)
    effect_size = np.concatenate(effect_size_list, 0)
    pvalue = np.concatenate(pvalue_list, 0)
    padj = false_discovery_control(pvalue.flatten(), method="bh").reshape(pvalue.shape)

    data_vars: dict = {
        "beta": (["cell_name", "covariate", "latent_dim"], beta),
        "effect_size": (["cell_name", "covariate"], effect_size),
        "pvalue": (["cell_name", "covariate"], pvalue),
        "padj": (["cell_name", "covariate"], padj),
    }
    coords: dict = {
        "cell_name": np.asarray(all_cell_names),
        "covariate": Xmat_names[:n_fixed],
        "latent_dim": np.arange(beta.shape[-1]),
    }

    if store_lfc:
        lfc = np.concatenate(lfc_list, 0)          # (n_cells_total, n_fixed, n_features)
        lfc_std = np.concatenate(lfc_std_list, 0)
        data_vars["lfc"] = (["cell_name", "covariate", "feature"], lfc)
        data_vars["lfc_std"] = (["cell_name", "covariate", "feature"], lfc_std)
        coords["feature"] = np.arange(lfc.shape[-1])
        if pde_list is not None:
            pde = np.concatenate(pde_list, 0)
            data_vars["pde"] = (["cell_name", "covariate", "feature"], pde)
        if baseline_list is not None:
            baseline = np.concatenate(baseline_list, 0)
            data_vars["baseline_expression"] = (["cell_name", "feature"], baseline)

    return xr.Dataset(data_vars, coords=coords)


def get_outlier_cell_sample_pairs(
    model,
    *,
    adata=None,
    sample_key: str,
    subsample_size: int | None = 5_000,
    quantile_threshold: float = 0.05,
    admissibility_threshold: float = 0.0,
    batch_size: int = 256,
) -> xr.Dataset:
    """Compute admissible cell-sample pairs using aggregated posteriors over ``u``."""
    adata = model._validate_anndata(adata)
    us = model.get_latent_representation(
        adata,
        give_mean=True,
        give_z=False,
        batch_size=batch_size,
    )

    sample_values = list(model.sample_order) if hasattr(model, "sample_order") else list(
        adata.obs[sample_key].unique()
    )
    log_probs = []
    thresholds = []
    n_splits = max(int(np.ceil(adata.n_obs / batch_size)), 1)
    for sample_name in tqdm(sample_values, desc="Outlier cell-sample scores"):
        sample_idxs = np.where(adata.obs[sample_key].to_numpy() == sample_name)[0]
        if subsample_size is not None and sample_idxs.shape[0] > subsample_size:
            sample_idxs = np.random.choice(sample_idxs, size=subsample_size, replace=False)

        ap = model.get_aggregated_posterior(
            adata=adata,
            indices=sample_idxs,
            batch_size=batch_size,
        )
        in_sample = torch.as_tensor(
            us[sample_idxs],
            dtype=ap.component_distribution.base_dist.loc.dtype,
            device=ap.component_distribution.base_dist.loc.device,
        )
        in_comp_log_probs = ap.component_distribution.log_prob(in_sample.unsqueeze(-2))
        thresholds.append(_rowwise_max_excluding_diagonal(in_comp_log_probs).detach().cpu())

        sample_log_probs = []
        for u_rep in np.array_split(us, n_splits):
            u_tensor = torch.as_tensor(
                u_rep,
                dtype=ap.component_distribution.base_dist.loc.dtype,
                device=ap.component_distribution.base_dist.loc.device,
            )
            comp_log_probs = ap.component_distribution.log_prob(u_tensor.unsqueeze(-2))
            sample_log_probs.append(comp_log_probs.max(dim=1, keepdim=True).values.cpu().numpy())
        log_probs.append(np.concatenate(sample_log_probs, axis=0))

    threshold_values = torch.cat(thresholds).numpy()
    global_threshold = np.quantile(threshold_values, q=quantile_threshold)
    log_probs = np.concatenate(log_probs, axis=1)
    log_ratios = log_probs - global_threshold

    coords = {"cell_name": adata.obs_names.to_numpy(), "sample": np.asarray(sample_values)}
    return xr.Dataset(
        {
            "log_probs": (["cell_name", "sample"], log_probs),
            "log_ratios": (["cell_name", "sample"], log_ratios),
            "is_admissible": (
                ["cell_name", "sample"],
                log_ratios > admissibility_threshold,
            ),
        },
        coords=coords,
    )
