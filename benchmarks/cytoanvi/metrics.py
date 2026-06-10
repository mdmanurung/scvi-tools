"""Dependency-light benchmark metrics (scanpy + sklearn only; no scib-metrics).

Label transfer: accuracy / macro-F1 / per-class recall.
Batch mixing: kNN-based iLISI-like score (fraction of cross-batch neighbours, normalised).
Bio conservation: ARI / NMI of Leiden clusters vs labels, plus label silhouette.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def label_transfer_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1, and per-class recall over cells with a (non-unlabeled) ground truth."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    labels = sorted(set(y_true))
    recall = {}
    for lab in labels:
        m = y_true == lab
        recall[str(lab)] = float((y_pred[m] == lab).mean()) if m.any() else float("nan")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n": int(len(y_true)),
        "per_class_recall": recall,
    }


def batch_mixing(latent: np.ndarray, batch, k: int = 30) -> dict:
    """iLISI-like batch mixing: mean fraction of a cell's kNN that come from a *different* batch.

    Normalised by the global cross-batch fraction so 1.0 = perfectly mixed (neighbour batch
    composition matches the dataset), 0.0 = fully segregated. Higher is better integration.
    """
    latent = np.asarray(latent)
    batch = np.asarray(batch)
    n = len(batch)
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    _, idx = nn.kneighbors(latent)
    idx = idx[:, 1:]  # drop self
    neigh_batch = batch[idx]
    cross = (neigh_batch != batch[:, None]).mean()
    # expected cross-batch fraction under random mixing
    _, counts = np.unique(batch, return_counts=True)
    p = counts / n
    expected_cross = 1.0 - float((p**2).sum())
    score = float(cross / expected_cross) if expected_cross > 0 else float("nan")
    return {"batch_mixing_norm": score, "cross_batch_frac": float(cross), "k": k}


def bio_conservation(latent: np.ndarray, labels, resolution: float = 1.0) -> dict:
    """ARI / NMI of Leiden clusters (on the latent kNN graph) vs labels, plus label silhouette."""
    import anndata as ad
    import scanpy as sc

    labels = np.asarray(labels)
    a = ad.AnnData(np.asarray(latent))
    sc.pp.neighbors(a, use_rep="X")
    sc.tl.leiden(a, resolution=resolution, flavor="igraph", n_iterations=2, directed=False)
    clusters = a.obs["leiden"].to_numpy()
    out = {
        "ari": float(adjusted_rand_score(labels, clusters)),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
    }
    if len(set(labels)) > 1:
        out["label_silhouette"] = float(silhouette_score(np.asarray(latent), labels))
    return out


def novelty_auroc(uncertainty: np.ndarray, is_novel) -> dict:
    """AUROC of an uncertainty score for separating novel (held-out-type) from seen cells."""
    is_novel = np.asarray(is_novel).astype(int)
    if is_novel.sum() == 0 or is_novel.sum() == len(is_novel):
        return {"auroc": float("nan"), "n_novel": int(is_novel.sum())}
    return {
        "auroc": float(roc_auc_score(is_novel, np.asarray(uncertainty))),
        "n_novel": int(is_novel.sum()),
    }


def concordance(pred_a, pred_b) -> dict:
    """Fraction of cells where two label-transfer methods agree (e.g. CytoANVI vs CytoVI k-NN)."""
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    return {"agreement": float((pred_a == pred_b).mean()), "n": int(len(pred_a))}
