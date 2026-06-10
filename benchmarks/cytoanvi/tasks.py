"""Benchmark tasks B1, B2, B3, B5 (B4 continual deferred — needs a case/control axis).

Each task returns a plain dict of metrics; run.py serializes them to JSON.
"""

from __future__ import annotations

import numpy as np

from . import metrics
from .baselines import cytovi_latent_and_knn

SCALED_LAYER = "scaled"
NAN_LAYER = "_nan_mask"


def _train_cytoanvi(
    adata,
    labels_key,
    unlabeled_category,
    *,
    batch_key,
    sample_key,
    nan_layer,
    n_latent=10,
    max_epochs=100,
):
    from scvi.external import CytoANVI

    a = adata.copy()
    CytoANVI.setup_anndata(
        a,
        layer=SCALED_LAYER,
        batch_key=batch_key,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        sample_key=sample_key,
        nan_layer=nan_layer,
    )
    model = CytoANVI(a, n_latent=n_latent)
    model.train(max_epochs=max_epochs)
    model.module.eval()
    return model, a


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
    max_epochs=100,
):
    """B1: CytoANVI classifier vs CytoVI k-NN at transferring labels to held-out cells."""
    true = np.asarray(adata.obs[labels_key].astype(str))
    held = _holdout(true, unlabeled_category, holdout_frac, seed)

    work = adata.copy()
    masked = true.copy()
    masked[held] = unlabeled_category
    work.obs[labels_key] = masked

    model, a = _train_cytoanvi(
        work,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
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
        max_epochs=max_epochs,
    )
    knn_full = masked.copy()
    knn_full[unlab_mask] = knn_pred_unlab

    return {
        "task": "b1_label_transfer",
        "seed": seed,
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
    max_epochs=100,
):
    """B2: batch mixing + bio conservation of the CytoANVI vs CytoVI latent."""
    model, a = _train_cytoanvi(
        adata,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        max_epochs=max_epochs,
    )
    cytoanvi_latent = model.get_latent_representation()
    _, cytovi_latent, _ = cytovi_latent_and_knn(
        adata,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        max_epochs=max_epochs,
    )

    batch = np.asarray(adata.obs[batch_key].astype(str))
    labels = np.asarray(adata.obs[labels_key].astype(str))
    keep = labels != unlabeled_category  # bio metrics over labelled cells only
    out = {"task": "b2_integration", "seed": seed}
    for name, lat in (("cytoanvi", cytoanvi_latent), ("cytovi", cytovi_latent)):
        out[name] = {
            **metrics.batch_mixing(lat, batch),
            **metrics.bio_conservation(lat[keep], labels[keep]),
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
    max_epochs=100,
):
    """B3: panel-1 reference -> panel-2 query via panel-aware prep + scArches surgery.

    Reports panel-1 holdout accuracy (hard number) and CytoANVI-vs-CytoVI concordance on panel 2.
    The reference (p1) must carry a nan_layer / backbone split; build it with merge so panel-2's
    missing markers are registered. Here p1/p2 are the *separate* panels and we map p2 onto p1.
    """
    from scvi.external import cytovi

    # reference = merged panels so p1 has the backbone/panel-specific split + labels; then restrict
    # training labels to p1 cells (p2 is unlabelled).
    merged = cytovi.merge_batches([p1.copy(), p2.copy()], batch_key="panel_batch")
    labels = np.asarray(merged.obs[labels_key].astype(str))
    is_p2 = np.isin(merged.obs_names, p2.obs_names)
    # hold out part of p1 labels to measure accuracy; p2 is unlabelled
    held = _holdout(labels, unlabeled_category, holdout_frac, seed) & ~is_p2
    masked = labels.copy()
    masked[held | is_p2] = unlabeled_category
    merged.obs[labels_key] = masked

    model, a = _train_cytoanvi(
        merged,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=NAN_LAYER,
        max_epochs=max_epochs,
    )
    pred = np.asarray(model.predict())

    out = {
        "task": "b3_panel_divergent",
        "seed": seed,
        "n_p2": int(is_p2.sum()),
        "p1_holdout": metrics.label_transfer_metrics(labels[held], pred[held]),
    }

    # CytoVI k-NN baseline on p2, for concordance
    knn_pred_unlab, _, unlab_mask = cytovi_latent_and_knn(
        merged,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=NAN_LAYER,
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
    max_epochs=100,
):
    """B5: does get_uncertainty flag a cell type held out of the reference entirely?"""
    labels = np.asarray(adata.obs[labels_key].astype(str))
    types = sorted(t for t in set(labels) if t != unlabeled_category)
    if holdout_type is None:
        holdout_type = types[0]
    is_novel = labels == holdout_type

    # reference excludes the novel type's labels (mark unlabeled so the classifier never sees it)
    work = adata.copy()
    masked = labels.copy()
    masked[is_novel] = unlabeled_category
    work.obs[labels_key] = masked
    model, a = _train_cytoanvi(
        work,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        max_epochs=max_epochs,
    )
    unc = model.get_uncertainty()
    return {
        "task": "b5_novelty",
        "seed": seed,
        "holdout_type": holdout_type,
        **metrics.novelty_auroc(unc, is_novel),
    }
