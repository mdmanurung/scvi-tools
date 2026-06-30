"""A3 — semi-synthetic marker imputation (paper Figure S4).

Baselines: CytoVI generative imputation vs KNN (k=10) in expression space.
cyCombinePy is **not** used here (batch correction only; no imputation API).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import KNNImputer

from benchmarks.common.training import SCALED_LAYER, train_cytovi

DEFAULT_EXCLUDE_MARKERS = ("CXCR3", "PD-1", "PD1")
NAN_LAYER = "_nan_mask"
MASK_BATCH = "pseudo_1"


def _pseudo_batches(adata, batch_key: str, seed: int = 0):
    """Randomly split cells into two pseudo-batches."""
    from anndata import concat

    rng = np.random.default_rng(seed)
    assign = rng.choice([0, 1], size=adata.n_obs)
    parts = []
    for bid, tag in enumerate(("pseudo_0", "pseudo_1")):
        sub = adata[assign == bid].copy()
        sub.obs[batch_key] = tag
        parts.append(sub)
    return concat(parts, join="inner")


def _mask_marker(adata, marker: str, batch_key: str, mask_batch: str = MASK_BATCH):
    """Hide one marker in one pseudo-batch via nan mask (CytoVI imputation path)."""
    from scvi.external.cytovi import register_nan_layer

    a = adata.copy()
    midx = list(a.var_names).index(marker)
    cells = a.obs[batch_key].astype(str) == mask_batch
    scaled = np.asarray(a.layers[SCALED_LAYER], dtype=float).copy()
    scaled[cells, midx] = np.nan
    a.layers[SCALED_LAYER] = scaled
    register_nan_layer(a, mask_layer_key=NAN_LAYER, scaled_layer_key=SCALED_LAYER, inplace=True)
    return a


def task_a3_imputation(
    adata,
    *,
    batch_key: str = "batch",
    exclude_markers: tuple[str, ...] = DEFAULT_EXCLUDE_MARKERS,
    max_cells: int = 50_000,
    max_epochs: int = 1000,
    n_latent: int | None = None,
    n_posterior_samples: int = 50,
    knn_neighbors: int = 10,
    seed: int = 0,
    markers: list[str] | None = None,
):
    """Leave-one-marker-out imputation on semi-synthetic pseudo-batches."""
    rng = np.random.default_rng(seed)
    a = adata.copy()
    if a.n_obs > max_cells:
        idx = rng.choice(a.n_obs, size=max_cells, replace=False)
        a = a[idx].copy()

    exclude = {m.upper().replace("-", "") for m in exclude_markers}
    test_markers = [
        m
        for m in a.var_names
        if str(m).upper().replace("-", "") not in exclude
    ]
    if markers is not None:
        test_markers = [m for m in markers if m in a.var_names]

    merged = _pseudo_batches(a, batch_key, seed=seed)
    holdout = (merged.obs[batch_key].astype(str) == MASK_BATCH).to_numpy()

    per_marker = {}
    for marker in test_markers:
        masked = _mask_marker(merged, marker, batch_key)
        model, train_adata = train_cytovi(
            masked,
            batch_key=batch_key,
            nan_layer=NAN_LAYER,
            layer=SCALED_LAYER,
            n_latent=n_latent,
            max_epochs=max_epochs,
        )
        holdout_idx = np.where(holdout)[0]
        imp = model.get_normalized_expression(
            train_adata,
            indices=holdout_idx,
            protein_list=[marker],
            n_samples=n_posterior_samples,
            return_mean=True,
            return_numpy=True,
            nan_warning=False,
        )
        cytovi_imp = np.asarray(imp).reshape(-1)
        if cytovi_imp.shape[0] != len(holdout_idx):
            # fallback: full matrix then slice
            full = np.asarray(
                model.get_normalized_expression(
                    train_adata,
                    protein_list=[marker],
                    n_samples=n_posterior_samples,
                    return_mean=True,
                    return_numpy=True,
                    nan_warning=False,
                )
            ).reshape(-1)
            cytovi_imp = full[holdout_idx]

        x = np.asarray(train_adata.layers[SCALED_LAYER])
        midx = list(train_adata.var_names).index(marker)
        # Inductive KNN: fit on reference cells only, then predict holdout — never sees
        # holdout values during neighbor search (fixes the transductive data-leakage bug).
        knn_imp = KNNImputer(n_neighbors=knn_neighbors).fit(x[~holdout]).transform(x[holdout_idx])[:, midx]

        true = np.asarray(merged.layers[SCALED_LAYER][holdout_idx, list(merged.var_names).index(marker)])
        valid = np.isfinite(true) & np.isfinite(cytovi_imp) & np.isfinite(knn_imp)
        if valid.sum() < 10:
            continue
        t, cv, kv = true[valid], cytovi_imp[valid], knn_imp[valid]
        per_marker[str(marker)] = {
            "n_cells": int(valid.sum()),
            "cytovi_pearson": float(pearsonr(t, cv)[0]),
            "cytovi_spearman": float(spearmanr(t, cv).correlation),
            "knn_pearson": float(pearsonr(t, kv)[0]),
            "knn_spearman": float(spearmanr(t, kv).correlation),
        }

    pearsons_c = [v["cytovi_pearson"] for v in per_marker.values()]
    pearsons_k = [v["knn_pearson"] for v in per_marker.values()]
    return {
        "task": "a3_imputation",
        "seed": seed,
        "max_epochs": max_epochs,
        "n_markers": len(per_marker),
        "n_posterior_samples": n_posterior_samples,
        "mean_cytovi_pearson": float(np.mean(pearsons_c)) if pearsons_c else float("nan"),
        "mean_knn_pearson": float(np.mean(pearsons_k)) if pearsons_k else float("nan"),
        "per_marker": per_marker,
        "note": "cyCombinePy omitted — batch correction only, no imputation API",
    }
