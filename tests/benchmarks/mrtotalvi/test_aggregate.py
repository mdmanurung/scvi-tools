"""Explicit-grid aggregation for MrTotalVI fixture results."""

from __future__ import annotations

import pytest
from benchmarks.mrtotalvi import aggregate_fixture_results


def _result(candidate, seed, value):
    return {
        "schema_version": "mrtotalvi-fixture-result-v1",
        "candidate": candidate,
        "scenario": "mixed",
        "seed": seed,
        "mode": {"hierarchy_mode": "centered_v2"},
        "metrics": {
            "heldout_elbo": value,
            "state_balanced_accuracy": 0.8 + value / 100,
            "not_available": float("nan"),
        },
    }


def test_aggregate_accepts_only_the_explicit_candidate_seed_grid():
    """Missing, duplicate, or unexpected runs cannot enter the summary."""
    results = [
        _result(candidate, seed, 10 * candidate_index + seed)
        for candidate_index, candidate in enumerate(("C2", "C4"))
        for seed in (0, 1, 2)
    ]
    aggregate = aggregate_fixture_results(
        results,
        expected_candidates=("C2", "C4"),
        expected_seeds=(0, 1, 2),
        scenario="mixed",
    )
    assert aggregate["grid_complete"] is True
    assert aggregate["candidates"]["C2"]["metrics"]["heldout_elbo"] == {
        "mean": 1.0,
        "sd": 1.0,
        "n": 3,
    }
    assert aggregate["candidates"]["C4"]["metrics"]["heldout_elbo"] == {
        "mean": 11.0,
        "sd": 1.0,
        "n": 3,
    }
    assert aggregate["candidates"]["C2"]["metrics"]["not_available"] == {
        "mean": None,
        "sd": None,
        "n": 0,
    }

    with pytest.raises(ValueError, match="Missing result"):
        aggregate_fixture_results(
            results[:-1],
            expected_candidates=("C2", "C4"),
            expected_seeds=(0, 1, 2),
            scenario="mixed",
        )
    with pytest.raises(ValueError, match="Duplicate result"):
        aggregate_fixture_results(
            results + [results[0]],
            expected_candidates=("C2", "C4"),
            expected_seeds=(0, 1, 2),
            scenario="mixed",
        )
    with pytest.raises(ValueError, match="Unexpected result"):
        aggregate_fixture_results(
            results + [_result("C3", 0, 3)],
            expected_candidates=("C2", "C4"),
            expected_seeds=(0, 1, 2),
            scenario="mixed",
        )
