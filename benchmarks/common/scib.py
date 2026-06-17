"""scib-metrics integration benchmarking."""

from __future__ import annotations

import numpy as np

LATENT_OBSM = "X_benchmark"


def _subsample_stratified(
    adata,
    batch_key: str,
    n_per_batch: int,
    seed: int = 0,
):
    """Subsample up to ``n_per_batch`` cells per batch stratum (paper A2 uses 10k)."""
    rng = np.random.default_rng(seed)
    batch = np.asarray(adata.obs[batch_key].astype(str))
    keep = []
    for b in np.unique(batch):
        idx = np.where(batch == b)[0]
        if len(idx) > n_per_batch:
            idx = rng.choice(idx, size=n_per_batch, replace=False)
        keep.append(idx)
    return adata[np.sort(np.concatenate(keep))].copy()


def pca_embedding(adata, n_comps: int = 50, layer: str | None = None, obsm_key: str = LATENT_OBSM):
    """PCA on expression matrix; store in ``obsm[obsm_key]`` for scib-metrics."""
    import scanpy as sc

    a = adata.copy()
    if layer is not None:
        x = np.asarray(a.layers[layer])
    else:
        x = np.asarray(a.X)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    a.X = x
    n_comps = min(n_comps, a.n_vars - 1, a.n_obs - 1)
    n_comps = max(2, n_comps)
    sc.pp.pca(a, n_comps=n_comps, zero_center=True)
    a.obsm[obsm_key] = a.obsm["X_pca"].copy()
    return a


def run_scib_benchmark(
    adata,
    *,
    batch_key: str,
    label_key: str,
    embedding_obsm_key: str = LATENT_OBSM,
    subsample_per_batch: int | None = 10_000,
    seed: int = 0,
    n_jobs: int = 1,
):
    """Run scib-metrics ``Benchmarker`` on an embedding in ``obsm``.

    Returns per-metric scores plus ``batch_correction``, ``bio_conservation``, and ``total`` aggregates.
    """
    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

    a = adata.copy()
    if subsample_per_batch is not None and subsample_per_batch > 0:
        a = _subsample_stratified(a, batch_key, subsample_per_batch, seed=seed)

    if embedding_obsm_key not in a.obsm:
        raise KeyError(f"obsm['{embedding_obsm_key}'] missing — compute an embedding first")

    # scib expects string-like labels
    a.obs[batch_key] = a.obs[batch_key].astype(str)
    a.obs[label_key] = a.obs[label_key].astype(str)

    benchmarker = Benchmarker(
        a,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=[embedding_obsm_key],
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        n_jobs=n_jobs,
    )
    benchmarker.benchmark()
    df = benchmarker.get_results(min_max_scale=False)
    row_key = embedding_obsm_key
    if row_key not in df.index:
        row_key = df.index[0]

    metrics = {
        col: float(df.loc[row_key, col])
        for col in df.columns
        if col != "Metric Type" and not isinstance(df.loc[row_key, col], str)
    }
    return {
        "n_cells": int(a.n_obs),
        "embedding_key": embedding_obsm_key,
        "metrics": metrics,
        "batch_correction": float(df.loc[row_key, "Batch correction"]),
        "bio_conservation": float(df.loc[row_key, "Bio conservation"]),
        "total": float(df.loc[row_key, "Total"]),
    }
