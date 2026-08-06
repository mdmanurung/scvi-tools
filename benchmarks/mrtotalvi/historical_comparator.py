"""Bounded historical-human MrTotalVI engineering sensitivity.

This module deliberately excludes the disputed ``pass_qc`` field. Its outputs
are not canonical, biological validation, candidate-selection evidence, or a
reason to change the legacy default.
"""

from __future__ import annotations

import math
import resource
import time
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import h5py
import numpy as np

from .config import candidate_configs
from .metrics import mean_knn_jaccard, representation_metrics

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal


ALLOWED_ENGINEERING_ANNOTATIONS = (
    "cell_label_l1",
    "cell_label_l1p5",
    "cell_label_l2",
    "cell_label_l3",
)
EVALUATION_SEED = 20260726


def _decode_strings(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset.asstr()[:], dtype=str)


def read_selected_categorical(
    source_h5ad: str | Path,
    *,
    selected_cell_ids: tuple[str, ...],
    column: str,
) -> np.ndarray:
    """Read one allowed annotation for exact selected cells in requested order.

    The implementation opens only ``obs/_index`` and the explicitly allowed
    categorical group. In particular, it cannot be used to read ``pass_qc``.
    """
    if column not in ALLOWED_ENGINEERING_ANNOTATIONS:
        raise ValueError(
            f"{column!r} is not an allowed engineering annotation; choose one "
            f"of {ALLOWED_ENGINEERING_ANNOTATIONS}."
        )
    if not selected_cell_ids or len(selected_cell_ids) != len(set(selected_cell_ids)):
        raise ValueError("selected_cell_ids must be non-empty and unique.")

    with h5py.File(source_h5ad, "r") as source:
        index_path = "obs/_index"
        column_path = f"obs/{column}"
        missing = [
            path for path in (index_path, column_path) if path not in source
        ]
        if missing:
            raise ValueError(f"Source H5AD is missing paths {missing}.")
        source_ids = _decode_strings(source[index_path])
        if len(source_ids) != len(set(source_ids.tolist())):
            raise ValueError("Source H5AD cell identifiers are not unique.")

        group = source[column_path]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"{column_path!r} must be categorical.")
        if not {"categories", "codes"}.issubset(group.keys()):
            raise ValueError(f"{column_path!r} lacks categorical codes/categories.")
        categories = _decode_strings(group["categories"])
        codes = np.asarray(group["codes"][:], dtype=np.int64)
        if codes.shape != source_ids.shape:
            raise ValueError(f"{column_path!r} is not cell-aligned.")
        if np.any(codes < 0) or np.any(codes >= len(categories)):
            raise ValueError(f"{column_path!r} contains missing or invalid codes.")

    source_lookup = {cell_id: index for index, cell_id in enumerate(source_ids)}
    missing_cells = [
        cell_id for cell_id in selected_cell_ids if cell_id not in source_lookup
    ]
    if missing_cells:
        raise ValueError(
            f"Selected cells are absent from the annotation source: {missing_cells[:5]}."
        )
    positions = np.asarray(
        [source_lookup[cell_id] for cell_id in selected_cell_ids],
        dtype=np.int64,
    )
    return categories[codes[positions]]


def _metric_summary(results: list[dict]) -> dict:
    metric_names = sorted(
        {
            name
            for result in results
            for name in result.get("metrics", {})
        }
    )
    summary = {}
    for name in metric_names:
        values = [
            float(result["metrics"][name])
            for result in results
            if (
                isinstance(result.get("metrics", {}).get(name), int | float)
                and not isinstance(result["metrics"][name], bool)
                and math.isfinite(float(result["metrics"][name]))
            )
        ]
        if not values:
            summary[name] = {"mean": None, "sd": None, "n": 0}
            continue
        array = np.asarray(values, dtype=np.float64)
        summary[name] = {
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if len(array) > 1 else None,
            "n": len(array),
        }
    return summary


def aggregate_historical_results(
    results: list[dict],
    *,
    representations: dict[tuple[str, int], np.ndarray],
    expected_candidates: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    fixture_sha256: str,
    k: int = 15,
) -> dict:
    """Aggregate one exact historical candidate-by-seed grid without selection."""
    if not expected_candidates or len(expected_candidates) != len(
        set(expected_candidates)
    ):
        raise ValueError("expected_candidates must be non-empty and unique.")
    if not expected_seeds or len(expected_seeds) != len(set(expected_seeds)):
        raise ValueError("expected_seeds must be non-empty and unique.")
    expected = {
        (candidate, seed)
        for candidate in expected_candidates
        for seed in expected_seeds
    }
    indexed: dict[tuple[str, int], dict] = {}
    for result in results:
        if (
            result.get("schema_version")
            != "mrtotalvi-historical-comparator-result-v1"
        ):
            raise ValueError("Unexpected result schema_version.")
        if result.get("fixture_sha256") != fixture_sha256:
            raise ValueError("Historical result fixture SHA-256 mismatch.")
        key = (result.get("candidate"), result.get("seed"))
        if key in indexed:
            raise ValueError(f"Duplicate result for {key}.")
        if key not in expected:
            raise ValueError(f"Unexpected result for {key}.")
        indexed[key] = result
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"Missing result entries: {missing}.")
    if set(representations) != expected:
        missing_representations = sorted(expected - set(representations))
        extra_representations = sorted(set(representations) - expected)
        raise ValueError(
            "Representation grid mismatch: "
            f"missing={missing_representations}, extra={extra_representations}."
        )

    first_shape = None
    for key, representation in representations.items():
        values = np.asarray(representation, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"Representation for {key} must be a finite matrix.")
        if first_shape is None:
            first_shape = values.shape
        elif values.shape != first_shape:
            raise ValueError("Historical representations are not cell/dimension aligned.")

    candidate_summaries = {}
    for candidate in expected_candidates:
        candidate_results = [
            indexed[(candidate, seed)] for seed in expected_seeds
        ]
        modes = {repr(result.get("mode")) for result in candidate_results}
        if len(modes) != 1:
            raise ValueError(f"Candidate {candidate} changed mode across seeds.")
        pairwise = []
        for first_seed, second_seed in combinations(expected_seeds, 2):
            pairwise.append(
                {
                    "seed_a": first_seed,
                    "seed_b": second_seed,
                    "value": mean_knn_jaccard(
                        representations[(candidate, first_seed)],
                        representations[(candidate, second_seed)],
                        k=k,
                    ),
                }
            )
        pairwise_values = np.asarray(
            [item["value"] for item in pairwise],
            dtype=np.float64,
        )
        candidate_summaries[candidate] = {
            "mode": candidate_results[0]["mode"],
            "metrics": _metric_summary(candidate_results),
            "cross_seed_knn_jaccard": {
                "mean": (
                    float(pairwise_values.mean())
                    if len(pairwise_values)
                    else None
                ),
                "sd": (
                    float(pairwise_values.std(ddof=1))
                    if len(pairwise_values) > 1
                    else None
                ),
                "pairwise": pairwise,
            },
            "per_seed": {
                str(seed): indexed[(candidate, seed)]
                for seed in expected_seeds
            },
        }
    return {
        "schema_version": "mrtotalvi-historical-comparator-aggregate-v1",
        "fixture_sha256": fixture_sha256,
        "expected_candidates": list(expected_candidates),
        "expected_seeds": list(expected_seeds),
        "grid_complete": True,
        "selection_rule": "none",
        "candidates": candidate_summaries,
        "scientific_scope": (
            "historical comparator; not canonical; not QC-pass; not biological "
            "validation; not promotion evidence"
        ),
    }


@dataclass(frozen=True)
class HistoricalRunConfig:
    """One bounded training request against the sealed 500-cell fixture."""

    candidate: Literal["C0", "C1", "C2", "C3", "C4"]
    seed: int
    max_epochs: int = 3
    batch_size: int = 64
    train_size: float = 0.8
    n_latent: int = 20
    n_prior_components: int = 20

    def __post_init__(self) -> None:
        """Validate the bounded request."""
        if self.candidate not in candidate_configs():
            raise ValueError(f"Unknown candidate {self.candidate!r}.")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative.")
        for name in ("max_epochs", "batch_size", "n_latent", "n_prior_components"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if not 0.5 <= self.train_size < 1.0:
            raise ValueError("train_size must be inside [0.5, 1.0).")


def _max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def run_historical_candidate(
    fixture_h5ad: str | Path,
    *,
    state_labels: np.ndarray,
    fixture_sha256: str,
    config: HistoricalRunConfig,
) -> tuple[dict, np.ndarray]:
    """Train and score one noncanonical historical-human sensitivity run."""
    import anndata as ad
    import torch

    import scvi
    from scvi.external import MrTotalVI

    started = time.perf_counter()
    rss_before = _max_rss_bytes()
    adata = ad.read_h5ad(fixture_h5ad)
    labels = np.asarray(state_labels, dtype=str)
    if labels.shape != (adata.n_obs,):
        raise ValueError("state_labels must align exactly to the historical fixture.")
    if np.unique(labels).size < 2:
        raise ValueError("state_labels must contain at least two classes.")
    if "pass_qc" in adata.obs:
        raise ValueError("The bounded historical fixture must not contain pass_qc.")

    scvi.settings.seed = config.seed
    torch.set_num_threads(min(8, torch.get_num_threads()))
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        protein_names_uns_key="protein_names_engineering",
        sample_key="donor_timepoint",
        batch_key="batch",
        layer="counts",
    )
    candidate = candidate_configs()[config.candidate]
    model = MrTotalVI(
        adata,
        sample_key="donor_timepoint",
        n_latent=config.n_latent,
        n_latent_u=config.n_latent,
        n_latent_sample=config.n_latent,
        z_u_prior=True,
        u_prior_mixture=True,
        u_prior_mixture_k=config.n_prior_components,
        use_map=True,
        hierarchy_mode=candidate.hierarchy_mode,
        u_encoder_mode=candidate.u_encoder_mode,
        scale_observations=candidate.scale_observations,
        u_prior=candidate.u_prior,
        init_prior_from_data=candidate.init_prior_from_data,
        freeze_prior_after_init=candidate.freeze_prior_after_init,
        encode_covariates=True,
    )
    model.train(
        max_epochs=config.max_epochs,
        accelerator="cpu",
        devices=1,
        train_size=config.train_size,
        validation_size=1.0 - config.train_size,
        shuffle_set_split=True,
        batch_size=config.batch_size,
        early_stopping=False,
        check_val_every_n_epoch=1,
        reduce_lr_on_plateau=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )

    validation_indices = np.asarray(model.validation_indices, dtype=np.int64)
    if len(validation_indices) == 0:
        raise RuntimeError("The historical sensitivity produced no validation cells.")
    heldout_elbo = float(
        model.get_elbo(
            indices=validation_indices,
            batch_size=config.batch_size,
        )
    )
    observed_samples = adata.obs["donor_timepoint"].astype(str).to_numpy()
    sample_elbos = []
    for sample in model.sample_order:
        sample_validation = validation_indices[
            observed_samples[validation_indices] == str(sample)
        ]
        if len(sample_validation):
            sample_elbos.append(
                float(
                    model.get_elbo(
                        indices=sample_validation,
                        batch_size=config.batch_size,
                    )
                )
            )
    representation_indices = np.arange(adata.n_obs, dtype=np.int64)
    u = np.asarray(
        model.get_latent_representation(
            indices=representation_indices,
            give_z=False,
            give_mean=True,
            batch_size=config.batch_size,
        ),
        dtype=np.float32,
    )
    metrics = representation_metrics(
        u,
        state_labels=labels,
        sample_labels=observed_samples,
        k=min(15, adata.n_obs - 1),
        random_state=EVALUATION_SEED,
    )

    if candidate.hierarchy_mode == "centered_v2":
        latent = model.get_counterfactual_latent(
            indices=validation_indices,
            target_samples=[str(value) for value in model.sample_order],
            inference_mode="latent_mean",
            n_draws=1,
            reference_indices=representation_indices,
            batch_size=config.batch_size,
            random_state=EVALUATION_SEED,
        )
        centering_max_abs = float(
            np.max(
                np.abs(
                    latent["eps_centered"]
                    .isel(draw=0)
                    .mean("target_sample")
                    .to_numpy()
                )
            )
        )
    else:
        centering_max_abs = float("nan")

    sample_elbo_values = np.asarray(sample_elbos, dtype=np.float64)
    result = {
        "schema_version": "mrtotalvi-historical-comparator-result-v1",
        "candidate": config.candidate,
        "seed": config.seed,
        "fixture_sha256": fixture_sha256,
        "evaluation_seed": EVALUATION_SEED,
        "n_train_cells": len(model.train_indices),
        "n_validation_cells": len(validation_indices),
        "n_representation_cells": adata.n_obs,
        "mode": {
            "hierarchy_mode": candidate.hierarchy_mode,
            "u_encoder_mode": candidate.u_encoder_mode,
            "scale_observations": candidate.scale_observations,
        },
        "metrics": {
            "heldout_elbo": heldout_elbo,
            "heldout_sample_elbo_sd": (
                float(sample_elbo_values.std(ddof=1))
                if len(sample_elbo_values) > 1
                else 0.0
            ),
            "heldout_sample_elbo_range": (
                float(np.ptp(sample_elbo_values))
                if len(sample_elbo_values)
                else float("nan")
            ),
            **metrics,
            "centering_max_abs": centering_max_abs,
        },
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_increase_bytes": max(0, _max_rss_bytes() - rss_before),
        "scientific_scope": (
            "historical comparator; not canonical; not QC-pass; not biological "
            "validation; not promotion evidence"
        ),
    }
    return result, u
