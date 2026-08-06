"""Estimand-matched MrTotalVI benchmark metrics."""

from __future__ import annotations

import numpy as np
from benchmarks.mrtotalvi import (
    counterfactual_recovery_metrics,
    mean_knn_jaccard,
    representation_metrics,
)


def test_representation_metrics_separate_state_conservation_from_sample_leakage():
    """Sample leakage can fall while state information and rotation invariance remain."""
    rng = np.random.default_rng(44)
    states = np.repeat(np.arange(3), 80)
    samples = np.tile(np.repeat(np.arange(4), 20), 3)
    state_signal = np.eye(3)[states] * 3.0
    sample_signal = np.eye(4)[samples] * 3.0
    noise = rng.normal(0.0, 0.05, size=(len(states), 2))
    conditioned = np.column_stack([state_signal, sample_signal, noise])
    blind = np.column_stack(
        [
            state_signal,
            np.zeros_like(sample_signal),
            noise,
        ]
    )

    conditioned_metrics = representation_metrics(
        conditioned,
        state_labels=states,
        sample_labels=samples,
        k=7,
        random_state=9,
    )
    blind_metrics = representation_metrics(
        blind,
        state_labels=states,
        sample_labels=samples,
        k=7,
        random_state=9,
    )

    assert conditioned_metrics["state_balanced_accuracy"] >= 0.95
    assert blind_metrics["state_balanced_accuracy"] >= 0.95
    assert (
        conditioned_metrics["sample_balanced_accuracy_within_state"]
        - blind_metrics["sample_balanced_accuracy_within_state"]
        >= 0.5
    )

    rotation, _ = np.linalg.qr(rng.normal(size=(conditioned.shape[1], conditioned.shape[1])))
    rotated = representation_metrics(
        conditioned @ rotation,
        state_labels=states,
        sample_labels=samples,
        k=7,
        random_state=9,
    )
    for name in (
        "state_balanced_accuracy",
        "sample_balanced_accuracy_within_state",
        "knn_state_accuracy",
    ):
        np.testing.assert_allclose(
            rotated[name],
            conditioned_metrics[name],
            atol=1e-10,
        )


def test_counterfactual_rank_and_cross_seed_geometry_ignore_nuisance_coordinates():
    """Rank recovery and kNN stability are invariant to allowed transformations."""
    rng = np.random.default_rng(91)
    truth = rng.lognormal(size=(30, 5, 7))
    observed_targets = np.arange(30) % 5
    predicted = 4.0 * truth + rng.uniform(0.0, 2.0, size=(30, 1, 7))

    recovery = counterfactual_recovery_metrics(
        predicted,
        truth=truth,
        observed_target_indices=observed_targets,
    )
    assert recovery["n_finite_cells"] == 30
    np.testing.assert_allclose(
        recovery["median_target_distance_spearman"],
        1.0,
        atol=1e-12,
    )

    embedding = rng.normal(size=(30, 6))
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    np.testing.assert_allclose(
        mean_knn_jaccard(embedding, embedding @ rotation, k=6),
        1.0,
        atol=1e-12,
    )
