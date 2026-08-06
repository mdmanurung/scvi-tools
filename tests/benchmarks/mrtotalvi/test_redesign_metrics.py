"""Pure contracts for the preregistered MrTotalVI redesign diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.mrtotalvi.diagnostics import (
    cross_seed_knn_metrics,
    heldout_prediction_metrics,
    latent_diagnostics,
    latent_diagnostics_v2,
    linear_cka,
    orthogonal_procrustes_disparity,
    representation_diagnostics,
)


def test_geometry_metrics_are_rotation_reflection_and_common_order_invariant():
    rng = np.random.default_rng(41)
    first = rng.normal(size=(90, 7))
    rotation, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    rotation[:, 0] *= -1
    second = 3.5 * (first @ rotation) + 4.0
    order = rng.permutation(len(first))

    np.testing.assert_allclose(linear_cka(first, second), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        orthogonal_procrustes_disparity(first, second),
        0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        linear_cka(first[order], second[order]),
        linear_cka(first, second),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        orthogonal_procrustes_disparity(first[order], second[order]),
        orthogonal_procrustes_disparity(first, second),
        atol=1e-12,
    )


def test_geometry_accepts_exact_one_ulp_variation_without_an_epsilon_gate():
    """Exact nonconstant evidence remains numerically evaluable."""
    first = np.ones((20, 3), dtype=np.float64)
    first[-1, 0] = np.nextafter(1.0, 2.0)
    second = first.copy()

    assert linear_cka(first, second, scale_safe=True) == pytest.approx(
        1.0,
        abs=1e-15,
    )
    assert orthogonal_procrustes_disparity(
        first,
        second,
        scale_safe=True,
    ) == pytest.approx(
        0.0,
        abs=1e-15,
    )
    diagnostics = latent_diagnostics_v2(
        first,
        posterior_scale=np.ones_like(first),
    )
    assert diagnostics["exact_nonconstant_variation"] == 1.0
    assert diagnostics["effective_rank"] == pytest.approx(1.0)


def test_geometry_and_rank_scale_extreme_finite_values_without_overflow():
    """Finite inputs near float limits are normalized before products."""
    base = np.asarray(
        [
            [-1.0, -0.5],
            [-0.25, 0.75],
            [0.25, -0.75],
            [1.0, 0.5],
        ],
        dtype=np.float64,
    )
    extreme = base * 1e150

    assert np.isfinite(linear_cka(extreme, extreme, scale_safe=True))
    assert np.isfinite(
        orthogonal_procrustes_disparity(
            extreme,
            extreme,
            scale_safe=True,
        )
    )
    diagnostics = latent_diagnostics_v2(
        extreme,
        posterior_scale=np.ones_like(extreme),
    )
    assert np.isfinite(diagnostics["latent_variance"])
    assert np.isfinite(diagnostics["effective_rank"])

    finite_max = np.finfo(np.float64).max
    mixed_scale = np.column_stack(
        [
            np.full(4, finite_max / 2.0),
            np.asarray([-1e-300, -5e-301, 5e-301, 1e-300]),
        ]
    )
    assert np.isfinite(linear_cka(mixed_scale, mixed_scale, scale_safe=True))
    assert np.isfinite(
        orthogonal_procrustes_disparity(
            mixed_scale,
            mixed_scale,
            scale_safe=True,
        )
    )
    near_limit = latent_diagnostics_v2(
        mixed_scale,
        posterior_scale=np.full_like(mixed_scale, finite_max),
    )
    assert near_limit["posterior_scale"] == finite_max
    assert near_limit["effective_rank"] == pytest.approx(1.0)


def test_unrepresentable_derived_variance_has_a_stable_no_call_reason():
    finite_max = np.finfo(np.float64).max
    values = np.asarray(
        [
            [-finite_max, 0.0],
            [0.0, 1.0],
            [finite_max, -1.0],
        ]
    )

    diagnostics = latent_diagnostics_v2(
        values,
        posterior_scale=np.ones_like(values),
    )

    assert diagnostics["representation_all_finite"] == 1.0
    assert diagnostics["exact_nonconstant_variation"] == 1.0
    assert diagnostics["latent_variance"] is None
    assert diagnostics["effective_rank"] is not None
    assert diagnostics["diagnostic_no_call_reasons"] == {
        "latent_variance": [
            "finite_input_latent_variance_not_representable"
        ]
    }


def test_stable_geometry_preserves_ordinary_estimands_to_tight_tolerance():
    """Numerical stabilization does not redefine ordinary-scale geometry."""
    rng = np.random.default_rng(20260731)
    first = rng.normal(size=(80, 5))
    second = first @ rng.normal(size=(5, 5)) + 3.0
    old_left = first - first.mean(axis=0, keepdims=True)
    old_right = second - second.mean(axis=0, keepdims=True)
    expected_cka = np.square(old_left.T @ old_right).sum() / (
        np.linalg.norm(old_left.T @ old_left, ord="fro")
        * np.linalg.norm(old_right.T @ old_right, ord="fro")
    )
    old_left /= np.linalg.norm(old_left, ord="fro")
    old_right /= np.linalg.norm(old_right, ord="fro")
    scipy_linalg = __import__(
        "scipy.linalg",
        fromlist=["orthogonal_procrustes"],
    )
    rotation, _ = scipy_linalg.orthogonal_procrustes(old_left, old_right)
    expected_disparity = np.square(old_left @ rotation - old_right).sum()

    assert linear_cka(first, second, scale_safe=True) == pytest.approx(
        expected_cka,
        abs=1e-12,
    )
    assert orthogonal_procrustes_disparity(
        first,
        second,
        scale_safe=True,
    ) == pytest.approx(
        expected_disparity,
        abs=1e-12,
    )


def test_cross_seed_knn_aligns_exact_cell_ids_and_rejects_bad_lineage():
    rng = np.random.default_rng(7)
    cells = np.asarray([f"cell-{index}" for index in range(50)])
    embedding = rng.normal(size=(len(cells), 5))
    order = rng.permutation(len(cells))

    metrics = cross_seed_knn_metrics(
        {
            0: (cells, embedding),
            1: (cells[order], embedding[order]),
        },
        k=8,
    )
    assert metrics["seed_pairs"] == ["0:1"]
    np.testing.assert_allclose(metrics["pairwise_jaccard"], [1.0], atol=1e-12)
    np.testing.assert_allclose(metrics["mean_jaccard"], 1.0, atol=1e-12)

    with pytest.raises(ValueError, match="duplicate"):
        cross_seed_knn_metrics(
            {
                0: (cells, embedding),
                1: (np.repeat(cells[:1], len(cells)), embedding),
            },
            k=8,
        )
    changed = cells.copy()
    changed[-1] = "different-cell"
    with pytest.raises(ValueError, match="same exact cell-ID set"):
        cross_seed_knn_metrics(
            {0: (cells, embedding), 1: (changed, embedding)},
            k=8,
        )


def test_latent_diagnostics_separate_scale_variance_rank_and_finiteness():
    rng = np.random.default_rng(19)
    embedding = rng.normal(size=(200, 6))
    scale = np.full_like(embedding, 0.25)
    metrics = latent_diagnostics(embedding, posterior_scale=scale)

    np.testing.assert_allclose(metrics["posterior_scale"], 0.25)
    assert 5.0 < metrics["effective_rank"] <= 6.0
    assert metrics["latent_variance"] > 0.8
    assert metrics["all_finite"] == 1.0

    broken = embedding.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        latent_diagnostics(broken, posterior_scale=scale)


def test_representation_diagnostics_use_evaluation_cells_only_and_seeded_null():
    rng = np.random.default_rng(22)
    n_train = 48
    n_eval = 144
    states = np.tile(np.repeat(np.arange(3), 48), 1)
    samples = np.tile(np.arange(4), 36)
    batches = np.tile(np.arange(2), 72)
    state_signal = np.eye(3)[states] * 3.0
    embedding_eval = np.column_stack(
        [state_signal, rng.normal(0.0, 0.08, size=(n_eval, 3))]
    )
    embedding = np.vstack([rng.normal(size=(n_train, 6)), embedding_eval])
    eval_indices = np.arange(n_train, n_train + n_eval)
    cell_ids = np.asarray([f"cell-{index:04d}" for index in range(len(embedding))])
    all_states = np.concatenate([np.zeros(n_train, dtype=int), states])
    all_samples = np.concatenate([np.zeros(n_train, dtype=int), samples])
    all_batches = np.concatenate([np.zeros(n_train, dtype=int), batches])

    first = representation_diagnostics(
        embedding,
        state_labels=all_states,
        sample_labels=all_samples,
        technical_batch_labels=all_batches,
        evaluation_indices=eval_indices,
        cell_ids=cell_ids,
        k=7,
        random_state=13,
        n_permutations=24,
    )
    changed_training_labels = np.arange(n_train) % 7
    second = representation_diagnostics(
        embedding,
        state_labels=np.concatenate([changed_training_labels, states]),
        sample_labels=np.concatenate([changed_training_labels, samples]),
        technical_batch_labels=np.concatenate([changed_training_labels, batches]),
        evaluation_indices=eval_indices,
        cell_ids=cell_ids,
        k=7,
        random_state=13,
        n_permutations=24,
    )

    assert first == second
    assert first["state_balanced_accuracy"] > 0.95
    assert 0.0 <= first["within_state_sample_predictability"] <= 1.0
    assert (
        first["within_state_sample_predictability_permutation_p95"]
        >= 1.0 / len(np.unique(samples))
    )
    assert (
        first["within_state_sample_predictability"]
        <= first["within_state_sample_predictability_permutation_p95"]
    )
    assert 0.0 <= first["technical_batch_mixing"] <= 1.0

    order = rng.permutation(len(embedding))
    reordered_eval = np.flatnonzero(np.isin(order, eval_indices))
    reordered = representation_diagnostics(
        embedding[order],
        state_labels=all_states[order],
        sample_labels=all_samples[order],
        technical_batch_labels=all_batches[order],
        evaluation_indices=reordered_eval,
        cell_ids=cell_ids[order],
        k=7,
        random_state=13,
        n_permutations=24,
    )
    assert reordered == first


def test_heldout_prediction_normalizes_modalities_and_calibrates_separately():
    rng = np.random.default_rng(5)
    rna_observed = rng.poisson(4.0, size=(12, 5))
    protein_observed = rng.poisson(2.0, size=(12, 2))
    rna_log_prob = np.full(rna_observed.shape, -2.0)
    protein_log_prob = np.full(protein_observed.shape, -3.0)
    rna_draws = rng.poisson(
        rna_observed[None, :, :] + 0.1,
        size=(40, *rna_observed.shape),
    )
    protein_draws = rng.poisson(
        protein_observed[None, :, :] + 0.1,
        size=(40, *protein_observed.shape),
    )

    metrics = heldout_prediction_metrics(
        rna_log_prob=rna_log_prob,
        rna_observed=rna_observed,
        rna_predictive_draws=rna_draws,
        protein_log_prob=protein_log_prob,
        protein_observed=protein_observed,
        protein_predictive_draws=protein_draws,
    )
    np.testing.assert_allclose(metrics["rna_negative_log_likelihood"], 2.0)
    np.testing.assert_allclose(metrics["protein_negative_log_likelihood"], 3.0)
    np.testing.assert_allclose(metrics["multimodal_predictive_loss"], 5.0)
    assert 0.0 <= metrics["rna_calibration_error"] <= 1.0
    assert 0.0 <= metrics["protein_calibration_error"] <= 1.0

    duplicated = heldout_prediction_metrics(
        rna_log_prob=np.tile(rna_log_prob, (1, 3)),
        rna_observed=np.tile(rna_observed, (1, 3)),
        rna_predictive_draws=np.tile(rna_draws, (1, 1, 3)),
    )
    np.testing.assert_allclose(duplicated["rna_negative_log_likelihood"], 2.0)
    assert duplicated["protein_negative_log_likelihood"] is None
    assert duplicated["multimodal_predictive_loss"] is None


def test_prediction_metrics_reject_shape_and_observation_mask_errors():
    observed = np.ones((4, 3))
    log_prob = -np.ones_like(observed)
    draws = np.ones((10, 4, 3))
    with pytest.raises(ValueError, match="same shape"):
        heldout_prediction_metrics(
            rna_log_prob=log_prob[:, :2],
            rna_observed=observed,
            rna_predictive_draws=draws,
        )
    with pytest.raises(ValueError, match="at least two draws"):
        heldout_prediction_metrics(
            rna_log_prob=log_prob,
            rna_observed=observed,
            rna_predictive_draws=draws[:1],
        )
