"""Benchmark tasks B1, B2, B3, B5 (B4 continual deferred — needs a case/control axis).

Each task returns a plain dict of metrics; run.py serializes them to JSON.
"""

from __future__ import annotations

import numpy as np

from benchmarks.common.scib import LATENT_OBSM, run_scib_benchmark
from benchmarks.common.training import NAN_LAYER, SCALED_LAYER, latent_obsm, train_cytoanvi

from . import metrics
from .baselines import cytovi_latent_and_knn


def _holdout(labels, unlabeled_category, frac, seed):
    """Stratified mask of cells to blank to the unlabeled category (the holdout to score)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(str)
    held = np.zeros(len(labels), dtype=bool)
    for lab in set(labels):
        if lab == unlabeled_category:
            continue
        idx = np.where(labels == lab)[0]
        k = max(1, int(round(frac * len(idx))))
        held[rng.choice(idx, size=k, replace=False)] = True
    return held


def task_b1_label_transfer(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    holdout_frac=0.2,
    seed=0,
    max_epochs=1000,
    n_latent=None,
):
    """B1: CytoANVI classifier vs CytoVI k-NN at transferring labels to held-out cells."""
    true = np.asarray(adata.obs[labels_key].astype(str))
    held = _holdout(true, unlabeled_category, holdout_frac, seed)

    work = adata.copy()
    masked = true.copy()
    masked[held] = unlabeled_category
    work.obs[labels_key] = masked

    model, a = train_cytoanvi(
        work,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
    )
    cytoanvi_pred = np.asarray(model.predict())

    knn_pred_unlab, _, unlab_mask = cytovi_latent_and_knn(
        work,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
    )
    knn_full = masked.copy()
    knn_full[unlab_mask] = knn_pred_unlab

    return {
        "task": "b1_label_transfer",
        "seed": seed,
        "max_epochs": max_epochs,
        "holdout_frac": holdout_frac,
        "n_held": int(held.sum()),
        "cytoanvi": metrics.label_transfer_metrics(true[held], cytoanvi_pred[held]),
        "cytovi_knn": metrics.label_transfer_metrics(true[held], knn_full[held]),
    }


def task_b2_integration(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    seed=0,
    max_epochs=1000,
    n_latent=None,
    subsample_per_batch=10_000,
):
    """B2: scib-metrics on CytoANVI vs CytoVI latents."""
    labels = np.asarray(adata.obs[labels_key].astype(str))
    keep = labels != unlabeled_category

    model, a = train_cytoanvi(
        adata,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
    )
    anvi_adata = a.copy()
    latent_obsm(anvi_adata, model, obsm_key=LATENT_OBSM)

    _, cytovi_latent, _ = cytovi_latent_and_knn(
        adata,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
    )
    vi_adata = adata.copy()
    vi_adata.obsm[LATENT_OBSM] = cytovi_latent

    # scib over labelled cells only (exclude unlabeled category)
    anvi_sub = anvi_adata[keep].copy()
    vi_sub = vi_adata[keep].copy()

    out = {
        "task": "b2_integration",
        "seed": seed,
        "max_epochs": max_epochs,
        "subsample_per_batch": subsample_per_batch,
        "cytoanvi": run_scib_benchmark(
            anvi_sub,
            batch_key=batch_key,
            label_key=labels_key,
            embedding_obsm_key=LATENT_OBSM,
            subsample_per_batch=subsample_per_batch,
            seed=seed,
        ),
        "cytovi": run_scib_benchmark(
            vi_sub,
            batch_key=batch_key,
            label_key=labels_key,
            embedding_obsm_key=LATENT_OBSM,
            subsample_per_batch=subsample_per_batch,
            seed=seed,
        ),
    }
    return out


def task_b3_panel_divergent(
    p1,
    p2,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    holdout_frac=0.2,
    seed=0,
    max_epochs=1000,
    n_latent=None,
):
    """B3: panel-1 reference -> panel-2 query via panel-aware prep + scArches surgery."""
    from scvi.external import cytovi

    merged = cytovi.merge_batches([p1.copy(), p2.copy()], batch_key="panel_batch")
    labels = np.asarray(merged.obs[labels_key].astype(str))
    is_p2 = np.isin(merged.obs_names, p2.obs_names)
    held = _holdout(labels, unlabeled_category, holdout_frac, seed) & ~is_p2
    masked = labels.copy()
    masked[held | is_p2] = unlabeled_category
    merged.obs[labels_key] = masked

    model, a = train_cytoanvi(
        merged,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=NAN_LAYER,
        n_latent=n_latent,
        max_epochs=max_epochs,
    )
    pred = np.asarray(model.predict())

    out = {
        "task": "b3_panel_divergent",
        "seed": seed,
        "max_epochs": max_epochs,
        "n_p2": int(is_p2.sum()),
        "p1_holdout": metrics.label_transfer_metrics(labels[held], pred[held]),
    }

    knn_pred_unlab, _, unlab_mask = cytovi_latent_and_knn(
        merged,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=NAN_LAYER,
        n_latent=n_latent,
        max_epochs=max_epochs,
    )
    knn_full = masked.copy()
    knn_full[unlab_mask] = knn_pred_unlab
    out["p2_concordance_vs_knn"] = metrics.concordance(pred[is_p2], knn_full[is_p2])
    return out


def task_b5_novelty(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    holdout_type=None,
    seed=0,
    max_epochs=1000,
    n_latent=None,
):
    """B5: does get_uncertainty flag a cell type held out of the reference entirely?"""
    labels = np.asarray(adata.obs[labels_key].astype(str))
    types = sorted(t for t in set(labels) if t != unlabeled_category)
    if holdout_type is None:
        holdout_type = types[0]
    is_novel = labels == holdout_type

    work = adata.copy()
    masked = labels.copy()
    masked[is_novel] = unlabeled_category
    work.obs[labels_key] = masked
    model, a = train_cytoanvi(
        work,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
    )
    unc = model.get_uncertainty()
    return {
        "task": "b5_novelty",
        "seed": seed,
        "max_epochs": max_epochs,
        "holdout_type": holdout_type,
        **metrics.novelty_auroc(unc, is_novel),
    }


def task_b5_holdout_sweep(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    seed=0,
    max_epochs=1000,
    n_latent=None,
):
    """B5 sweep: AUROC for each cell type held out as novel."""
    labels = np.asarray(adata.obs[labels_key].astype(str))
    types = sorted(t for t in set(labels) if t != unlabeled_category)
    per_type = {}
    for ht in types:
        per_type[ht] = task_b5_novelty(
            adata,
            labels_key=labels_key,
            unlabeled_category=unlabeled_category,
            batch_key=batch_key,
            sample_key=sample_key,
            nan_layer=nan_layer,
            holdout_type=ht,
            seed=seed,
            max_epochs=max_epochs,
            n_latent=n_latent,
        )
    aurocs = [v["auroc"] for v in per_type.values() if not np.isnan(v["auroc"])]
    return {
        "task": "b5_holdout_sweep",
        "seed": seed,
        "max_epochs": max_epochs,
        "per_type": per_type,
        "best_auroc": float(max(aurocs)) if aurocs else float("nan"),
        "mean_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
    }
