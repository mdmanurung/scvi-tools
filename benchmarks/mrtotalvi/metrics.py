"""Metrics that keep representation quality separate from sample leakage."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def _validated_embedding(embedding: np.ndarray) -> np.ndarray:
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("embedding must have shape (at least 3 cells, at least 1 dimension).")
    if not np.all(np.isfinite(values)):
        raise ValueError("embedding must contain only finite values.")
    return values


def _encoded_labels(labels, *, n_cells: int, name: str) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != n_cells:
        raise ValueError(f"{name} must be a one-dimensional cell-aligned vector.")
    encoded = LabelEncoder().fit_transform(values)
    if np.unique(encoded).size < 2:
        raise ValueError(f"{name} must contain at least two classes.")
    return encoded


def _cv_balanced_accuracy(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int,
) -> float:
    counts = np.bincount(labels)
    n_splits = min(5, int(counts.min()))
    if n_splits < 2:
        raise ValueError("Every class needs at least two cells for cross-validation.")
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    classifier = LogisticRegression(
        C=1_000.0,
        max_iter=2_000,
        random_state=random_state,
        solver="lbfgs",
    )
    prediction = cross_val_predict(
        classifier,
        embedding,
        labels,
        cv=splitter,
        n_jobs=1,
    )
    return float(balanced_accuracy_score(labels, prediction))


def _knn_label_accuracy(
    embedding: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
) -> float:
    if not 1 <= k < len(embedding):
        raise ValueError("k must be at least one and smaller than the number of cells.")
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
    indices = neighbors.fit(embedding).kneighbors(return_distance=False)
    prediction = np.asarray(
        [
            np.bincount(labels[cell_neighbors]).argmax()
            for cell_neighbors in indices
        ],
        dtype=np.int64,
    )
    return float(balanced_accuracy_score(labels, prediction))


def representation_metrics(
    embedding: np.ndarray,
    *,
    state_labels,
    sample_labels,
    k: int = 15,
    random_state: int = 0,
) -> dict[str, float]:
    """Score cell-state conservation and within-state sample predictability.

    Sample predictability is computed after subtracting each state's centroid,
    preventing cell-state composition from being mistaken for direct sample
    leakage. The two outcomes remain separate.
    """
    values = _validated_embedding(embedding)
    states = _encoded_labels(
        state_labels,
        n_cells=len(values),
        name="state_labels",
    )
    samples = _encoded_labels(
        sample_labels,
        n_cells=len(values),
        name="sample_labels",
    )
    state_residual = values.copy()
    for state in np.unique(states):
        mask = states == state
        state_residual[mask] -= state_residual[mask].mean(axis=0, keepdims=True)
    return {
        "state_balanced_accuracy": _cv_balanced_accuracy(
            values,
            states,
            random_state=random_state,
        ),
        "sample_balanced_accuracy_within_state": _cv_balanced_accuracy(
            state_residual,
            samples,
            random_state=random_state,
        ),
        "knn_state_accuracy": _knn_label_accuracy(
            values,
            states,
            k=k,
        ),
    }


def mean_knn_jaccard(
    first: np.ndarray,
    second: np.ndarray,
    *,
    k: int = 15,
) -> float:
    """Return mean per-cell neighbor-set Jaccard for aligned embeddings."""
    first_values = _validated_embedding(first)
    second_values = _validated_embedding(second)
    if len(first_values) != len(second_values):
        raise ValueError("Aligned embeddings must contain the same number of cells.")
    if not 1 <= k < len(first_values):
        raise ValueError("k must be at least one and smaller than the number of cells.")

    first_neighbors = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
    ).fit(first_values).kneighbors(return_distance=False)
    second_neighbors = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
    ).fit(second_values).kneighbors(return_distance=False)
    scores = []
    for left, right in zip(first_neighbors, second_neighbors, strict=True):
        left_set = set(left.tolist())
        right_set = set(right.tolist())
        scores.append(len(left_set & right_set) / len(left_set | right_set))
    return float(np.mean(scores))


def counterfactual_recovery_metrics(
    predicted: np.ndarray,
    *,
    truth: np.ndarray,
    observed_target_indices: np.ndarray,
) -> dict[str, float | int]:
    """Compare same-cell target-distance ranks against exogenous truth."""
    predicted_values = np.asarray(predicted, dtype=np.float64)
    truth_values = np.asarray(truth, dtype=np.float64)
    if (
        predicted_values.ndim != 3
        or predicted_values.shape != truth_values.shape
        or predicted_values.shape[1] < 3
    ):
        raise ValueError(
            "predicted and truth must share shape "
            "(cells, at least 3 targets, features)."
        )
    if not np.all(np.isfinite(predicted_values)) or not np.all(
        np.isfinite(truth_values)
    ):
        raise ValueError("Counterfactual arrays must contain only finite values.")
    observed = np.asarray(observed_target_indices, dtype=np.int64)
    if observed.shape != (len(predicted_values),):
        raise ValueError("observed_target_indices must be cell-aligned.")
    if np.any(observed < 0) or np.any(observed >= predicted_values.shape[1]):
        raise ValueError("observed_target_indices contain an invalid target.")

    correlations: list[float] = []
    for cell, factual_target in enumerate(observed):
        keep = np.arange(predicted_values.shape[1]) != factual_target
        predicted_distance = np.linalg.norm(
            predicted_values[cell, keep]
            - predicted_values[cell, factual_target],
            axis=1,
        )
        truth_distance = np.linalg.norm(
            truth_values[cell, keep] - truth_values[cell, factual_target],
            axis=1,
        )
        if np.ptp(predicted_distance) == 0.0 or np.ptp(truth_distance) == 0.0:
            continue
        correlation = float(spearmanr(predicted_distance, truth_distance).statistic)
        if np.isfinite(correlation):
            correlations.append(correlation)

    predicted_total = predicted_values.sum(axis=2, keepdims=True)
    truth_total = truth_values.sum(axis=2, keepdims=True)
    predicted_composition = np.divide(
        predicted_values,
        predicted_total,
        out=np.zeros_like(predicted_values),
        where=predicted_total > 0,
    )
    truth_composition = np.divide(
        truth_values,
        truth_total,
        out=np.zeros_like(truth_values),
        where=truth_total > 0,
    )
    rmse = float(
        np.sqrt(np.mean((predicted_composition - truth_composition) ** 2))
    )
    return {
        "median_target_distance_spearman": (
            float(np.median(correlations)) if correlations else float("nan")
        ),
        "mean_target_distance_spearman": (
            float(np.mean(correlations)) if correlations else float("nan")
        ),
        "n_finite_cells": len(correlations),
        "normalized_composition_rmse": rmse,
    }
