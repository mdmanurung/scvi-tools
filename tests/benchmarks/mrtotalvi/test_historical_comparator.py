"""Contracts for the bounded historical-human comparator sensitivity."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from benchmarks.mrtotalvi.historical_comparator import (
    aggregate_historical_results,
    read_selected_categorical,
)
from benchmarks.mrtotalvi.run_historical_comparator import _assessment


def test_annotation_reader_is_ordered_and_cannot_access_qc(tmp_path):
    """The sensitivity may read a frozen annotation, never the disputed QC field."""
    source = tmp_path / "source.h5ad"
    with h5py.File(source, "w") as handle:
        obs = handle.create_group("obs")
        obs.create_dataset(
            "_index",
            data=np.asarray(["c0", "c1", "c2"], dtype=h5py.string_dtype()),
        )
        labels = obs.create_group("cell_label_l2")
        labels.create_dataset(
            "categories",
            data=np.asarray(["T", "B"], dtype=h5py.string_dtype()),
        )
        labels.create_dataset("codes", data=np.asarray([0, 1, 0]))
        obs.create_dataset("pass_qc", data=np.asarray([False, False, False]))

    observed = read_selected_categorical(
        source,
        selected_cell_ids=("c2", "c0"),
        column="cell_label_l2",
    )
    np.testing.assert_array_equal(observed, np.asarray(["T", "T"]))

    with pytest.raises(ValueError, match="allowed engineering annotation"):
        read_selected_categorical(
            source,
            selected_cell_ids=("c0",),
            column="pass_qc",
        )


def _result(candidate: str, seed: int, leakage: float) -> dict:
    return {
        "schema_version": "mrtotalvi-historical-comparator-result-v1",
        "candidate": candidate,
        "seed": seed,
        "fixture_sha256": "a" * 64,
        "mode": {"candidate": candidate},
        "metrics": {
            "heldout_elbo": -10.0 + seed,
            "sample_balanced_accuracy_within_state": leakage,
        },
    }


def test_historical_aggregation_requires_exact_grid_and_keeps_seed_geometry():
    """No seed may disappear and cross-seed geometry remains candidate-specific."""
    results = [
        _result(candidate, seed, leakage=0.8 if candidate == "C2" else 0.3)
        for candidate in ("C2", "C4")
        for seed in (0, 1, 2)
    ]
    base = np.arange(30, dtype=np.float64).reshape(10, 3)
    representations = {
        (candidate, seed): base + seed
        for candidate in ("C2", "C4")
        for seed in (0, 1, 2)
    }

    aggregate = aggregate_historical_results(
        results,
        representations=representations,
        expected_candidates=("C2", "C4"),
        expected_seeds=(0, 1, 2),
        fixture_sha256="a" * 64,
        k=3,
    )
    assert aggregate["grid_complete"] is True
    assert aggregate["selection_rule"] == "none"
    assert aggregate["candidates"]["C4"]["metrics"][
        "sample_balanced_accuracy_within_state"
    ] == {"mean": 0.3, "sd": 0.0, "n": 3}
    assert aggregate["candidates"]["C2"]["cross_seed_knn_jaccard"]["mean"] == 1.0
    assert len(
        aggregate["candidates"]["C2"]["cross_seed_knn_jaccard"]["pairwise"]
    ) == 3

    with pytest.raises(ValueError, match="Missing result"):
        aggregate_historical_results(
            results[:-1],
            representations=representations,
            expected_candidates=("C2", "C4"),
            expected_seeds=(0, 1, 2),
            fixture_sha256="a" * 64,
            k=3,
        )


def test_historical_assessment_uses_only_frozen_verdict_vocabulary():
    """Directional diagnostics cannot become new promotion-like verdict labels."""
    seeds = (0, 1, 2)
    candidates = ("C1", "C2", "C3", "C4")
    per_candidate = {}
    for candidate in candidates:
        per_seed = {}
        for seed in seeds:
            per_seed[str(seed)] = {
                "metrics": {
                    "centering_max_abs": (
                        1e-8 if candidate in {"C2", "C3", "C4"} else float("nan")
                    ),
                    "heldout_elbo": -10.0,
                    "heldout_sample_elbo_sd": 1.0,
                    "state_balanced_accuracy": 0.8,
                    "sample_balanced_accuracy_within_state": (
                        0.2 if candidate == "C4" else 0.8
                    ),
                    "knn_state_accuracy": 0.8,
                }
            }
        per_candidate[candidate] = {"per_seed": per_seed}

    assessment = _assessment(
        {"candidates": per_candidate},
        candidates=candidates,
        seeds=seeds,
    )
    allowed = {
        "mechanism-supported",
        "mechanism-not-supported",
        "engineering-feasible",
        "inconclusive",
        "blocked",
    }
    assert {
        item["verdict"] for item in assessment["hypotheses"].values()
    } <= allowed
    leakage = assessment["hypotheses"]["H3_sample_blind_leakage_sensitivity"]
    assert leakage["verdict"] == "inconclusive"
    assert leakage["direction_consistent"] is True
