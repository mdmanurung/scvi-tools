"""Label-transfer and novelty metrics (integration → benchmarks.common.scib)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
)


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
    """Agreement between two label-transfer methods."""
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    return {"agreement": float((pred_a == pred_b).mean()), "n": int(len(pred_a))}
