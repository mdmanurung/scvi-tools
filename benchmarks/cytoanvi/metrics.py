"""Label-transfer and novelty metrics (integration → benchmarks.common.scib)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def label_transfer_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro-F1, and per-class recall."""
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


def _counts(values) -> dict[str, int]:
    arr = np.asarray(values).astype(str)
    labels, counts = np.unique(arr, return_counts=True)
    return {str(k): int(v) for k, v in zip(labels, counts, strict=True)}


def label_transfer_diagnostics(
    y_train,
    y_true,
    y_pred,
    *,
    rare_max_count: int = 25,
) -> dict:
    """Diagnostic counts and collapse flags for label-transfer benchmarks."""
    y_train = np.asarray(y_train).astype(str)
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    observed = sorted(set(y_true))
    predicted = sorted(set(y_pred))
    train_counts = _counts(y_train)
    true_counts = _counts(y_true)
    pred_counts = _counts(y_pred)
    rare_labels = [lab for lab, count in true_counts.items() if count <= rare_max_count]
    rare_mask = np.isin(y_true, rare_labels)
    majority_fraction = (
        float(max(pred_counts.values()) / max(len(y_pred), 1)) if pred_counts else 0.0
    )
    return {
        "train_label_counts": train_counts,
        "heldout_label_counts": true_counts,
        "predicted_label_counts": pred_counts,
        "n_true_labels": len(observed),
        "n_predicted_labels": len(predicted),
        "predicted_label_coverage": float(
            len(set(predicted) & set(observed)) / max(len(observed), 1)
        ),
        "majority_prediction_fraction": majority_fraction,
        "collapse_warning": bool(len(predicted) <= 1 or majority_fraction >= 0.95),
        "rare_labels": rare_labels,
        "rare_macro_f1": (
            float(f1_score(y_true[rare_mask], y_pred[rare_mask], average="macro", zero_division=0))
            if rare_mask.any()
            else None
        ),
    }


def novelty_auroc(uncertainty: np.ndarray, is_novel) -> dict:
    """AUROC of uncertainty for novel (held-out-type) vs seen cells."""
    is_novel = np.asarray(is_novel).astype(int)
    if is_novel.sum() == 0 or is_novel.sum() == len(is_novel):
        return {"auroc": float("nan"), "n_novel": int(is_novel.sum())}
    return {
        "auroc": float(roc_auc_score(is_novel, np.asarray(uncertainty))),
        "n_novel": int(is_novel.sum()),
    }


def concordance(pred_a, pred_b) -> dict:
    """Inter-method label agreement between two predictors.

    Measures the fraction of cells where ``pred_a`` and ``pred_b`` assign the same
    label.  This is **inter-method concordance**, NOT ground-truth accuracy — there
    is no reference ground-truth involved.

    In B3 (cross-panel), both predictors share the CytoVI encoder backbone:
    ``pred_a`` is the CytoANVI classifier and ``pred_b`` is the CytoVI-kNN.  High
    concordance means the two methods agree with each other, not that either is
    correct.  Validating cross-panel label accuracy requires independent manually-gated
    panel-2 labels, which are not available in the current benchmark.
    """
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    return {"agreement": float((pred_a == pred_b).mean()), "n": int(len(pred_a))}


def precision_at_specificity(
    uncertainty: np.ndarray,
    is_novel: np.ndarray,
    *,
    specificity: float = 0.95,
    uncertainty_ref: np.ndarray | None = None,
) -> dict:
    """Novelty-detection precision at a target reference specificity.

    The threshold T is the ``specificity``-th quantile of ``uncertainty_ref`` (or the non-novel
    subset of ``uncertainty`` when ``uncertainty_ref`` is not provided). Cells with
    ``uncertainty > T`` are predicted novel.

    Parameters
    ----------
    uncertainty
        Per-cell uncertainty scores for all evaluated cells.
    is_novel
        Boolean mask — True for cells that are truly novel (held-out type).
    specificity
        Target specificity on reference cells (fraction correctly below threshold).
    uncertainty_ref
        If provided, use this separate reference uncertainty distribution to set T instead of
        the non-novel subset of ``uncertainty``.

    Returns
    -------
    dict with keys: threshold, specificity, precision, recall, n_predicted_novel
    """
    from cytoanvi._uncertainty import get_uncertainty_threshold

    uncertainty = np.asarray(uncertainty, dtype=float)
    is_novel = np.asarray(is_novel, dtype=bool)
    ref_scores = (
        np.asarray(uncertainty_ref, dtype=float)
        if uncertainty_ref is not None
        else uncertainty[~is_novel]
    )
    threshold = get_uncertainty_threshold(ref_scores, specificity=specificity)
    pred_novel = uncertainty > threshold
    tp = int((pred_novel & is_novel).sum())
    fp = int((pred_novel & ~is_novel).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / is_novel.sum()) if is_novel.sum() > 0 else float("nan")
    return {
        "threshold": threshold,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "n_predicted_novel": tp + fp,
    }


def rna_macro_f1_paired(
    adata,
    preds,
    *,
    batch_key: str = "modality",
    sample_key: str = "sample_id",
    rna_modality: str = "RNA",
    cytof_modality: str = "CyTOF",
    eval_label_key: str = "eval_celltype",
    shared_samples: set[str] | frozenset[str] | None = None,
) -> float:
    """Macro-F1 for RNA label transfer on donors present in both modalities."""
    if shared_samples is None:
        rna_samples = set(
            adata.obs.loc[adata.obs[batch_key].astype(str) == rna_modality, sample_key].astype(str)
        )
        cy_samples = set(
            adata.obs.loc[adata.obs[batch_key].astype(str) == cytof_modality, sample_key]
            .astype(str)
        )
        shared_samples = rna_samples & cy_samples
    mask = (adata.obs[batch_key].astype(str) == rna_modality) & (
        adata.obs[sample_key].astype(str).isin(shared_samples)
    )
    if mask.sum() == 0:
        return float("nan")
    return float(
        f1_score(
            adata.obs.loc[mask, eval_label_key],
            np.asarray(preds)[mask.to_numpy()],
            average="macro",
            zero_division=0,
        )
    )
