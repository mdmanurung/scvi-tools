"""Pure, estimand-separated diagnostics for the MrTotalVI redesign.

This module intentionally has no model or torch dependency.  It is used by the
Python 3.13 training environment and by the Python 3.14 contract-only check.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.linalg import orthogonal_procrustes

from .metrics import (
    _cv_balanced_accuracy,
    _encoded_labels,
    _knn_label_accuracy,
    _validated_embedding,
    mean_knn_jaccard,
)


def _centered(values: np.ndarray) -> np.ndarray:
    """Return the frozen historical centered embedding."""
    checked = _validated_embedding(values)
    centered = checked - checked.mean(axis=0, keepdims=True)
    if np.linalg.norm(centered, ord="fro") <= np.finfo(np.float64).eps:
        raise ValueError("Embedding has no centered variation.")
    return centered


def _centered_v2(values: np.ndarray) -> np.ndarray:
    """Return a scale-safe, unit-Frobenius centered embedding.

    Exact row equality is the only zero-variation boundary.  Columnwise input
    scaling preserves tiny valid variation beside extreme constant columns;
    long-double relative rescaling then reconstructs the original geometry
    before a final bounded Frobenius normalization.
    """
    checked = _validated_embedding(values)
    if np.all(checked == checked[0]):
        raise ValueError("Embedding has no centered variation.")
    column_scale = np.max(np.abs(checked), axis=0)
    safe_scale = np.where(column_scale == 0.0, 1.0, column_scale)
    scaled = checked / safe_scale
    centered = scaled - scaled.mean(axis=0, keepdims=True)
    centered_column_max = np.max(np.abs(centered), axis=0)
    centered_magnitudes = (
        centered_column_max.astype(np.longdouble)
        * column_scale.astype(np.longdouble)
    )
    centered_scale = np.max(centered_magnitudes)
    if not np.isfinite(centered_scale) or centered_scale == 0.0:
        raise FloatingPointError(
            "Embedding has no numerically resolvable centered variation."
        )
    relative_scale = column_scale.astype(np.longdouble) / centered_scale
    centered = np.asarray(
        centered.astype(np.longdouble) * relative_scale,
        dtype=np.float64,
    )
    frobenius = float(np.linalg.norm(centered, ord="fro"))
    if not np.isfinite(frobenius) or frobenius == 0.0:
        raise FloatingPointError("Embedding centered normalization is invalid.")
    return centered / frobenius


def linear_cka(
    first: np.ndarray,
    second: np.ndarray,
    *,
    scale_safe: bool = False,
) -> float:
    """Return centered linear CKA for cell-aligned embeddings.

    Linear CKA is invariant to isotropic scaling and orthogonal rotation or
    reflection, but it does not silently align cells.
    """
    center = _centered_v2 if scale_safe else _centered
    left = center(first)
    right = center(second)
    if len(left) != len(right):
        raise ValueError("Cell-aligned embeddings must have the same number of rows.")
    cross = left.T @ right
    numerator = float(np.square(cross).sum())
    denominator = float(
        np.linalg.norm(left.T @ left, ord="fro")
        * np.linalg.norm(right.T @ right, ord="fro")
    )
    if scale_safe:
        if (
            not np.isfinite(numerator)
            or not np.isfinite(denominator)
            or denominator == 0.0
        ):
            raise FloatingPointError("Linear CKA denominator is invalid.")
    elif denominator <= np.finfo(np.float64).eps:
        raise ValueError("Linear CKA denominator is zero.")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def orthogonal_procrustes_disparity(
    first: np.ndarray,
    second: np.ndarray,
    *,
    scale_safe: bool = False,
) -> float:
    """Return squared normalized disparity after orthogonal alignment."""
    center = _centered_v2 if scale_safe else _centered
    left = center(first)
    right = center(second)
    if left.shape != right.shape:
        raise ValueError(
            "Orthogonal Procrustes requires identical cell and latent dimensions."
        )
    if not scale_safe:
        left = left / np.linalg.norm(left, ord="fro")
        right = right / np.linalg.norm(right, ord="fro")
    rotation, _ = orthogonal_procrustes(left, right)
    disparity = float(np.square(left @ rotation - right).sum())
    return max(0.0, disparity)


def _validated_cell_embedding(
    cell_ids,
    embedding: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(cell_ids, dtype=str)
    values = _validated_embedding(embedding)
    if cells.ndim != 1 or len(cells) != len(values):
        raise ValueError(f"Seed {seed} cell IDs must align one-to-one with its embedding.")
    if len(np.unique(cells)) != len(cells):
        raise ValueError(f"Seed {seed} contains duplicate cell IDs.")
    return cells, values


def cross_seed_knn_metrics(
    representations: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    k: int = 15,
    scale_safe: bool = False,
) -> dict[str, float | list[float] | list[str]]:
    """Align exact cell IDs and return every pairwise cross-seed kNN Jaccard."""
    if not isinstance(representations, dict) or len(representations) < 2:
        raise ValueError("At least two distinct training seeds are required.")
    seeds = sorted(representations)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Training seeds must be unique.")

    checked = {
        seed: _validated_cell_embedding(*representations[seed], seed=seed)
        for seed in seeds
    }
    reference_cells = checked[seeds[0]][0]
    reference_set = set(reference_cells.tolist())
    aligned: dict[int, np.ndarray] = {}
    for seed in seeds:
        cells, values = checked[seed]
        if set(cells.tolist()) != reference_set:
            raise ValueError("Every seed must contain the same exact cell-ID set.")
        position = {cell: index for index, cell in enumerate(cells.tolist())}
        order = np.fromiter(
            (position[cell] for cell in reference_cells.tolist()),
            dtype=np.int64,
            count=len(reference_cells),
        )
        aligned_values = values[order]
        distance_scale = 0.0
        if scale_safe:
            maximum = float(np.max(np.abs(aligned_values)))
            safe_distance_bound = np.sqrt(
                np.finfo(np.float64).max / aligned_values.shape[1]
            ) / 2.0
            if maximum > safe_distance_bound:
                distance_scale = maximum
        aligned[seed] = (
            aligned_values
            if distance_scale == 0.0
            else aligned_values / distance_scale
        )

    pair_names: list[str] = []
    pair_values: list[float] = []
    for first_seed, second_seed in combinations(seeds, 2):
        pair_names.append(f"{first_seed}:{second_seed}")
        pair_values.append(
            mean_knn_jaccard(
                aligned[first_seed],
                aligned[second_seed],
                k=k,
            )
        )
    return {
        "seed_pairs": pair_names,
        "pairwise_jaccard": pair_values,
        "mean_jaccard": float(np.mean(pair_values)),
    }


def latent_diagnostics(
    embedding: np.ndarray,
    *,
    posterior_scale: np.ndarray,
) -> dict[str, float]:
    """Summarize posterior scale, marginal variance, effective rank, and finiteness."""
    values = _validated_embedding(embedding)
    scale = np.asarray(posterior_scale, dtype=np.float64)
    if scale.shape != values.shape:
        raise ValueError("posterior_scale must have the same shape as the embedding.")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("posterior_scale must contain finite positive values.")

    variance = np.var(values, axis=0, ddof=1)
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= np.finfo(np.float64).eps:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 0.0] / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    return {
        "posterior_scale": float(np.mean(scale)),
        "latent_variance": float(np.mean(variance)),
        "effective_rank": effective_rank,
        "all_finite": 1.0,
    }


def latent_diagnostics_v2(
    embedding: np.ndarray,
    *,
    posterior_scale: np.ndarray | None,
) -> dict[str, object]:
    """Serialize prospective integrity defects instead of raising on them."""
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError(
            "embedding must have shape (at least 3 cells, at least 1 dimension)."
        )
    representation_finite = bool(np.all(np.isfinite(values)))
    exact_nonconstant = bool(not np.all(values == values[0]))
    try:
        scale = np.asarray(posterior_scale, dtype=np.float64)
    except (TypeError, ValueError):
        scale = np.asarray(np.nan)
    scale_valid = bool(
        scale.shape == values.shape
        and np.all(np.isfinite(scale))
        and np.all(scale > 0.0)
    )

    posterior_scale_mean = None
    if scale_valid:
        scale_max = float(np.max(scale))
        scale_fraction = min(1.0, float(np.mean(scale / scale_max)))
        posterior_scale_mean = float(scale_fraction * scale_max)
    latent_variance = None
    effective_rank = None
    diagnostic_no_call_reasons: dict[str, list[str]] = {}
    if representation_finite:
        column_scale = np.max(np.abs(values), axis=0)
        if not np.any(column_scale):
            latent_variance = 0.0
            effective_rank = 0.0
        else:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                safe_scale = np.where(column_scale == 0.0, 1.0, column_scale)
                scaled_variance = np.var(
                    values / safe_scale,
                    axis=0,
                    ddof=1,
                )
                variance = (
                    scaled_variance.astype(np.longdouble)
                    * column_scale.astype(np.longdouble)
                    * column_scale.astype(np.longdouble)
                )
                variance_mean = np.mean(variance, dtype=np.longdouble)
            if np.isfinite(variance_mean) and variance_mean <= np.finfo(
                np.float64
            ).max:
                latent_variance = float(variance_mean)
            else:
                diagnostic_no_call_reasons["latent_variance"] = [
                    "finite_input_latent_variance_not_representable"
                ]

            if not exact_nonconstant:
                effective_rank = 0.0
            else:
                try:
                    normalized = _centered_v2(values)
                    covariance = (normalized.T @ normalized) / (
                        normalized.shape[0] - 1
                    )
                    eigenvalues = np.clip(
                        np.linalg.eigvalsh(covariance),
                        0.0,
                        None,
                    )
                    total = float(eigenvalues.sum())
                    if not np.isfinite(total) or total == 0.0:
                        raise FloatingPointError(
                            "Normalized covariance spectrum is invalid."
                        )
                    probabilities = eigenvalues[eigenvalues > 0.0] / total
                    effective_rank = float(
                        np.exp(
                            -np.sum(probabilities * np.log(probabilities))
                        )
                    )
                    if not np.isfinite(effective_rank):
                        raise FloatingPointError(
                            "Normalized effective rank is invalid."
                        )
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    effective_rank = None
                    diagnostic_no_call_reasons["effective_rank"] = [
                        "finite_input_effective_rank_not_evaluable"
                    ]
    return {
        "posterior_scale": posterior_scale_mean,
        "latent_variance": latent_variance,
        "effective_rank": effective_rank,
        "all_finite": float(representation_finite),
        "representation_all_finite": float(representation_finite),
        "exact_nonconstant_variation": float(exact_nonconstant),
        "posterior_scales_all_valid": float(scale_valid),
        "diagnostic_no_call_reasons": diagnostic_no_call_reasons,
    }


def _validated_evaluation_indices(indices, *, n_cells: int) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("evaluation_indices must contain at least three cells.")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.equal(values, np.floor(values))):
            raise ValueError("evaluation_indices must be integers.")
    values = values.astype(np.int64)
    if len(np.unique(values)) != len(values):
        raise ValueError("evaluation_indices must not contain duplicates.")
    if np.any(values < 0) or np.any(values >= n_cells):
        raise ValueError("evaluation_indices are out of bounds.")
    return values


def _state_centered(values: np.ndarray, states: np.ndarray) -> np.ndarray:
    residual = values.copy()
    for state in np.unique(states):
        mask = states == state
        residual[mask] -= residual[mask].mean(axis=0, keepdims=True)
    return residual


def _technical_batch_mixing(
    residual: np.ndarray,
    batches: np.ndarray,
    *,
    random_state: int,
) -> float:
    predictability = _cv_balanced_accuracy(
        residual,
        batches,
        random_state=random_state,
    )
    chance = 1.0 / len(np.unique(batches))
    excess = max(0.0, (predictability - chance) / (1.0 - chance))
    return float(1.0 - min(1.0, excess))


def representation_diagnostics(
    embedding: np.ndarray,
    *,
    state_labels,
    sample_labels,
    technical_batch_labels,
    evaluation_indices,
    cell_ids=None,
    k: int = 15,
    random_state: int = 0,
    n_permutations: int = 100,
) -> dict[str, float]:
    """Evaluate state, sample leakage, its null, and batch mixing on held-out cells."""
    all_values = _validated_embedding(embedding)
    evaluation = _validated_evaluation_indices(
        evaluation_indices,
        n_cells=len(all_values),
    )
    if cell_ids is not None:
        all_cells = np.asarray(cell_ids, dtype=str)
        if all_cells.ndim != 1 or len(all_cells) != len(all_values):
            raise ValueError("cell_ids must align with the full embedding.")
        evaluation_cells = all_cells[evaluation]
        if len(np.unique(evaluation_cells)) != len(evaluation_cells):
            raise ValueError("Evaluation cell IDs must be unique.")
        evaluation = evaluation[
            np.argsort(evaluation_cells, kind="stable")
        ]
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive.")
    values = all_values[evaluation]

    def subset_labels(labels, name: str) -> np.ndarray:
        raw = np.asarray(labels)
        if raw.ndim != 1 or len(raw) != len(all_values):
            raise ValueError(f"{name} must align with the full embedding.")
        return raw[evaluation]

    states = _encoded_labels(
        subset_labels(state_labels, "state_labels"),
        n_cells=len(values),
        name="state_labels",
    )
    samples = _encoded_labels(
        subset_labels(sample_labels, "sample_labels"),
        n_cells=len(values),
        name="sample_labels",
    )
    batches = _encoded_labels(
        subset_labels(technical_batch_labels, "technical_batch_labels"),
        n_cells=len(values),
        name="technical_batch_labels",
    )
    state_residual = _state_centered(values, states)
    sample_score = _cv_balanced_accuracy(
        state_residual,
        samples,
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    null_scores = np.empty(n_permutations, dtype=np.float64)
    for permutation_index in range(n_permutations):
        permuted = samples.copy()
        for state in np.unique(states):
            mask = np.flatnonzero(states == state)
            permuted[mask] = permuted[mask][rng.permutation(len(mask))]
        null_scores[permutation_index] = _cv_balanced_accuracy(
            state_residual,
            permuted,
            random_state=random_state + permutation_index + 1,
        )

    return {
        "state_balanced_accuracy": _cv_balanced_accuracy(
            values,
            states,
            random_state=random_state,
        ),
        "knn_state_accuracy": _knn_label_accuracy(values, states, k=k),
        "within_state_sample_predictability": sample_score,
        "within_state_sample_predictability_permutation_p95": float(
            np.quantile(null_scores, 0.95)
        ),
        "technical_batch_mixing": _technical_batch_mixing(
            state_residual,
            batches,
            random_state=random_state,
        ),
    }


def _calibration_error(
    observed: np.ndarray,
    predictive_draws: np.ndarray,
    mask: np.ndarray,
) -> float:
    draws = np.asarray(predictive_draws, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[1:] != observed.shape:
        raise ValueError(
            "Predictive draws must have shape (draws, cells, features)."
        )
    if draws.shape[0] < 2:
        raise ValueError("Posterior predictive calibration needs at least two draws.")
    if not np.all(np.isfinite(draws)):
        raise ValueError("Posterior predictive draws must be finite.")
    errors = []
    for nominal in (0.5, 0.8, 0.9):
        tail = (1.0 - nominal) / 2.0
        lower = np.quantile(draws, tail, axis=0)
        upper = np.quantile(draws, 1.0 - tail, axis=0)
        coverage = np.mean(
            ((observed >= lower) & (observed <= upper))[mask]
        )
        errors.append(abs(float(coverage) - nominal))
    return float(np.mean(errors))


def _modality_prediction_metrics(
    *,
    log_prob: np.ndarray,
    observed: np.ndarray,
    predictive_draws: np.ndarray,
    observed_mask: np.ndarray | None,
    name: str,
) -> tuple[float, float]:
    values = np.asarray(observed, dtype=np.float64)
    log_values = np.asarray(log_prob, dtype=np.float64)
    if values.ndim != 2 or log_values.shape != values.shape:
        raise ValueError(f"{name} log probabilities and observations must have the same shape.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} observations must be finite and nonnegative.")
    if observed_mask is None:
        mask = np.ones(values.shape, dtype=bool)
    else:
        mask = np.asarray(observed_mask)
        if mask.shape != values.shape or mask.dtype != np.bool_:
            raise ValueError(f"{name} observed mask must be a shape-matched boolean array.")
    if not np.any(mask):
        raise ValueError(f"{name} observed mask selects no entries.")
    if not np.all(np.isfinite(log_values[mask])):
        raise ValueError(f"{name} observed log probabilities must be finite.")
    negative_log_likelihood = float(-np.mean(log_values[mask]))
    calibration = _calibration_error(values, predictive_draws, mask)
    return negative_log_likelihood, calibration


def heldout_prediction_metrics(
    *,
    rna_log_prob: np.ndarray,
    rna_observed: np.ndarray,
    rna_predictive_draws: np.ndarray,
    rna_observed_mask: np.ndarray | None = None,
    protein_log_prob: np.ndarray | None = None,
    protein_observed: np.ndarray | None = None,
    protein_predictive_draws: np.ndarray | None = None,
    protein_observed_mask: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Return feature-normalized held-out likelihood and calibration by modality.

    The multimodal score is the sum of the two independently normalized
    modality scores.  It is intentionally unavailable for RNA-only models.
    """
    rna_nll, rna_calibration = _modality_prediction_metrics(
        log_prob=rna_log_prob,
        observed=rna_observed,
        predictive_draws=rna_predictive_draws,
        observed_mask=rna_observed_mask,
        name="RNA",
    )
    protein_arguments = (
        protein_log_prob,
        protein_observed,
        protein_predictive_draws,
    )
    if all(value is None for value in protein_arguments):
        protein_nll = None
        protein_calibration = None
        multimodal = None
    elif any(value is None for value in protein_arguments):
        raise ValueError("Protein prediction inputs must be supplied together.")
    else:
        protein_nll, protein_calibration = _modality_prediction_metrics(
            log_prob=protein_log_prob,
            observed=protein_observed,
            predictive_draws=protein_predictive_draws,
            observed_mask=protein_observed_mask,
            name="protein",
        )
        multimodal = rna_nll + protein_nll
    return {
        "rna_negative_log_likelihood": rna_nll,
        "protein_negative_log_likelihood": protein_nll,
        "multimodal_predictive_loss": multimodal,
        "rna_calibration_error": rna_calibration,
        "protein_calibration_error": protein_calibration,
    }
