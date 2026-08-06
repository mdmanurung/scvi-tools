"""Explicit-grid aggregation for bounded MrTotalVI fixture runs."""

from __future__ import annotations

import math

import numpy as np


def aggregate_fixture_results(
    results: list[dict],
    *,
    expected_candidates: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    scenario: str,
) -> dict:
    """Aggregate only a complete, explicit candidate-by-seed grid."""
    if not expected_candidates or len(set(expected_candidates)) != len(
        expected_candidates
    ):
        raise ValueError("expected_candidates must be non-empty and unique.")
    if not expected_seeds or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError("expected_seeds must be non-empty and unique.")

    expected = {
        (candidate, seed)
        for candidate in expected_candidates
        for seed in expected_seeds
    }
    indexed: dict[tuple[str, int], dict] = {}
    for result in results:
        if result.get("schema_version") != "mrtotalvi-fixture-result-v1":
            raise ValueError("Unexpected result schema_version.")
        if result.get("scenario") != scenario:
            raise ValueError(
                f"Unexpected result scenario {result.get('scenario')!r}."
            )
        key = (result.get("candidate"), result.get("seed"))
        if key in indexed:
            raise ValueError(f"Duplicate result for {key}.")
        if key not in expected:
            raise ValueError(f"Unexpected result for {key}.")
        indexed[key] = result
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"Missing result entries: {missing}.")

    candidate_summaries = {}
    for candidate in expected_candidates:
        per_seed = {
            str(seed): indexed[(candidate, seed)]
            for seed in expected_seeds
        }
        modes = {
            repr(indexed[(candidate, seed)].get("mode"))
            for seed in expected_seeds
        }
        if len(modes) != 1:
            raise ValueError(f"Candidate {candidate} changed mode across seeds.")
        metric_names = sorted(
            {
                name
                for seed in expected_seeds
                for name in indexed[(candidate, seed)].get("metrics", {})
            }
        )
        metric_summary = {}
        for name in metric_names:
            values = []
            for seed in expected_seeds:
                value = indexed[(candidate, seed)].get("metrics", {}).get(name)
                if (
                    isinstance(value, int | float)
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    values.append(float(value))
            if values:
                array = np.asarray(values, dtype=np.float64)
                metric_summary[name] = {
                    "mean": float(array.mean()),
                    "sd": (
                        float(array.std(ddof=1))
                        if len(array) > 1
                        else None
                    ),
                    "n": len(array),
                }
            else:
                metric_summary[name] = {"mean": None, "sd": None, "n": 0}
        candidate_summaries[candidate] = {
            "mode": indexed[(candidate, expected_seeds[0])].get("mode"),
            "metrics": metric_summary,
            "per_seed": per_seed,
        }
    return {
        "schema_version": "mrtotalvi-fixture-aggregate-v1",
        "scenario": scenario,
        "expected_candidates": list(expected_candidates),
        "expected_seeds": list(expected_seeds),
        "grid_complete": True,
        "candidates": candidate_summaries,
        "scientific_scope": (
            "synthetic mechanism pilot; descriptive mean and sample SD across "
            "training seeds; not publication calibration"
        ),
    }
