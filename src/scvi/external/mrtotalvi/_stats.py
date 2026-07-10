"""Shared u-space statistical APIs for Mr multimodal models."""

from __future__ import annotations

import warnings

import numpy as np
import torch
import xarray as xr
from scipy.special import logsumexp
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from tqdm import tqdm


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
    batch_size: int = 128,
) -> xr.Dataset:
    """Compute MrVI-style differential abundance log probabilities over ``u``."""
    adata = model._validate_anndata(adata)
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
