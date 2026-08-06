"""Fail-closed contracts for the preregistered RDX-03 diagnosis."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from benchmarks.mrtotalvi import convergence as convergence_module
from benchmarks.mrtotalvi.convergence import (
    aggregate_diagnosis_grid,
    assess_convergence,
    assess_latent_collapse,
    convergence_control,
    d0_downstream_gate_decision,
    diagnosis_fit_spec,
    validate_diagnosis_grid,
)
from benchmarks.mrtotalvi.versioning import (
    historical_redesign_contract_adapter,
)


def _history(n_checks: int, *, last_epoch: int):
    step = last_epoch / n_checks
    return [
        {
            "epoch": int(round(step * (index + 1))),
            "value": 100.0 - index,
        }
        for index in range(n_checks)
    ]


def _fit_result(candidate_id: str, fixture_id: str, seed: int):
    return {
        "schema_version": "mrtotalvi-convergence-fit-v2",
        "candidate_id": candidate_id,
        "fixture_id": fixture_id,
        "training_seed": seed,
        "control": convergence_control().to_dict(),
        "convergence": {"status": "converged"},
        "collapse": {"failed": False},
        "metrics": {},
    }


def test_convergence_control_and_fit_specs_are_exact_and_candidate_invariant():
    control = convergence_control()
    assert control.to_dict() == {
        "check_every_epochs": 5,
        "minimum_epochs": 50,
        "maximum_epochs": 400,
        "patience_checks": 30,
        "restore_best_checkpoint": True,
        "candidate_specific_retuning": False,
        "monitor": "elbo_validation",
        "mode": "min",
        "min_delta": 0.0,
    }

    specs = {
        candidate_id: diagnosis_fit_spec(candidate_id)
        for candidate_id in ("B1", "B2", "B3", "D0")
    }
    assert specs["B1"].model_family == "totalvi"
    assert specs["B2"].legacy_candidate == "C0"
    assert specs["B3"].legacy_candidate == "C2"
    assert specs["D0"].legacy_candidate == "C4"
    assert len({spec.control for spec in specs.values()}) == 1
    with pytest.raises(ValueError, match="diagnosis row"):
        diagnosis_fit_spec("D1")


def test_convergence_assessment_requires_minimum_epochs_and_a_real_plateau():
    converged = assess_convergence(
        _history(31, last_epoch=155),
        trainer_epochs=155,
        stopped_early=True,
    )
    assert converged == {
        "status": "converged",
        "stable_validation_plateau": True,
        "stopped_early": True,
        "trainer_epochs": 155,
        "validation_checks": 31,
        "reached_minimum_epochs": True,
        "reached_maximum_epochs": False,
    }

    maximum = assess_convergence(
        _history(80, last_epoch=400),
        trainer_epochs=400,
        stopped_early=False,
    )
    assert maximum["status"] == "non_converged_at_maximum"
    assert maximum["stable_validation_plateau"] is False
    assert maximum["reached_maximum_epochs"] is True

    with pytest.raises(ValueError, match="minimum"):
        assess_convergence(
            _history(5, last_epoch=25),
            trainer_epochs=25,
            stopped_early=True,
        )
    with pytest.raises(ValueError, match="inconsistent"):
        assess_convergence(
            _history(31, last_epoch=155),
            trainer_epochs=155,
            stopped_early=False,
        )


def test_latent_collapse_assessment_is_candidate_aware_and_noncompensatory():
    b1 = assess_latent_collapse(
        {
            "factual_z_effective_rank": 6.0,
            "factual_z_latent_variance": 0.2,
            "factual_z_posterior_scale": 0.3,
            "latent_all_finite": 1.0,
        },
        candidate_id="B1",
        latent_dimension=10,
    )
    assert b1["failed"] is False
    assert b1["required_representations"] == ["factual_z"]

    d0_metrics = {
        "u_effective_rank": 5.0,
        "factual_z_effective_rank": 5.0,
        "u_latent_variance": 0.2,
        "factual_z_latent_variance": 0.2,
        "u_posterior_scale": 0.3,
        "factual_z_posterior_scale": 0.3,
        "registered_residual_gradient_coverage": 1.0,
        "latent_all_finite": 1.0,
    }
    d0 = assess_latent_collapse(
        d0_metrics,
        candidate_id="D0",
        latent_dimension=10,
    )
    assert d0["failed"] is False
    assert d0["required_representations"] == ["u", "factual_z"]

    for metric_id, bad_value in (
        ("u_effective_rank", 4.99),
        ("u_latent_variance", 0.0),
        ("u_posterior_scale", 0.0),
        ("registered_residual_gradient_coverage", 0.99),
        ("latent_all_finite", 0.0),
    ):
        changed = dict(d0_metrics)
        changed[metric_id] = bad_value
        assessed = assess_latent_collapse(
            changed,
            candidate_id="D0",
            latent_dimension=10,
        )
        assert assessed["failed"] is True
        assert metric_id in assessed["failed_metrics"]


def test_diagnosis_grid_is_exact_and_d0_cannot_preclaim_downstream_da():
    rows = ("B1", "B2", "B3", "D0")
    fixtures = (
        "mixed",
        "unequal_cells",
        "sealed_500",
        "canonical_human_if_available",
    )
    results = [
        _fit_result(candidate_id, fixture_id, seed)
        for fixture_id in fixtures
        for candidate_id in rows
        for seed in (0, 1, 2)
    ]
    validated = validate_diagnosis_grid(results)
    assert len(validated) == 48

    with pytest.raises(ValueError, match="Missing"):
        validate_diagnosis_grid(results[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_diagnosis_grid([*results, deepcopy(results[0])])
    changed = deepcopy(results)
    changed[0]["control"]["maximum_epochs"] = 401
    with pytest.raises(ValueError, match="control"):
        validate_diagnosis_grid(changed)

    decision = d0_downstream_gate_decision(results)
    assert decision == {
        "d0_passes_all_downstream_gates": False,
        "action": "continue_to_d1_d5",
        "reason_codes": ["milo_calibration_not_available_at_rdx03"],
    }


def test_diagnosis_aggregate_binds_comparisons_and_all_cross_seed_pairs(
    monkeypatch,
):
    rows = ("B1", "B2", "B3", "D0")
    fixtures = (
        "mixed",
        "unequal_cells",
        "sealed_500",
        "canonical_human_if_available",
    )
    results = [
        _fit_result(candidate_id, fixture_id, seed)
        for fixture_id in fixtures
        for candidate_id in rows
        for seed in (0, 1, 2)
    ]
    cells = np.asarray([f"cell-{index}" for index in range(20)])
    base = np.arange(80, dtype=np.float64).reshape(20, 4)
    representations = {}
    for result in results:
        candidate_id = result["candidate_id"]
        key = (
            candidate_id,
            result["fixture_id"],
            result["training_seed"],
        )
        payload = {
            "factual_z": {
                "cell_ids": cells,
                "values": base + result["training_seed"],
            }
        }
        if candidate_id != "B1":
            payload["u"] = {
                "cell_ids": cells,
                "values": 2.0 * base + result["training_seed"],
            }
        representations[key] = payload

    aggregate = aggregate_diagnosis_grid(
        results,
        representations=representations,
    )

    assert aggregate["schema_version"] == "mrtotalvi-convergence-aggregate-v2"
    assert aggregate["n_fits"] == 48
    assert len(aggregate["paired_comparisons"]) == 48
    assert len(aggregate["cross_seed"]) == 16
    assert {
        item["metrics"]["factual_z_cross_seed_knn_jaccard_k15"]
        for item in aggregate["cross_seed"]
    } == {1.0}
    d0_mixed_seed0 = next(
        item
        for item in aggregate["paired_comparisons"]
        if item["candidate_id"] == "D0"
        and item["fixture_id"] == "mixed"
        and item["training_seed"] == 0
    )
    assert d0_mixed_seed0["reference_candidate_ids"] == {
        "u": "B2",
        "factual_z": "B1",
    }
    assert aggregate["d0_decision"]["action"] == "continue_to_d1_d5"

    historical = historical_redesign_contract_adapter()

    def unavailable_live_contract():
        raise AssertionError("historical replay consulted the live contract")

    monkeypatch.setattr(
        convergence_module,
        "redesign_run_contract",
        unavailable_live_contract,
    )
    replayed = aggregate_diagnosis_grid(
        results,
        representations=representations,
        contract_adapter=historical,
    )
    assert replayed == aggregate

    changed = deepcopy(representations)
    changed[("D0", "mixed", 2)]["u"]["cell_ids"] = cells.copy()
    changed[("D0", "mixed", 2)]["u"]["cell_ids"][-1] = "changed"
    with pytest.raises(ValueError, match="same exact cell-ID set"):
        aggregate_diagnosis_grid(
            results,
            representations=changed,
            contract_adapter=historical,
        )


def test_failed_representations_are_explicit_no_calls_not_geometry_inputs():
    rows = ("B1", "B2", "B3", "D0")
    fixtures = (
        "mixed",
        "unequal_cells",
        "sealed_500",
        "canonical_human_if_available",
    )
    results = [
        _fit_result(candidate_id, fixture_id, seed)
        for fixture_id in fixtures
        for candidate_id in rows
        for seed in (0, 1, 2)
    ]
    b1_failed = next(
        result
        for result in results
        if (
            result["candidate_id"],
            result["fixture_id"],
            result["training_seed"],
        )
        == ("B1", "mixed", 0)
    )
    b1_failed["convergence"]["status"] = "non_converged_at_maximum"
    b2_u_failed = next(
        result
        for result in results
        if (
            result["candidate_id"],
            result["fixture_id"],
            result["training_seed"],
        )
        == ("B2", "mixed", 0)
    )
    b2_u_failed["collapse"] = {
        "failed": True,
        "failed_metrics": ["u_effective_rank"],
    }

    cells = np.asarray([f"cell-{index}" for index in range(20)])
    base = np.arange(80, dtype=np.float64).reshape(20, 4)
    representations = {}
    for result in results:
        candidate_id = result["candidate_id"]
        key = (
            candidate_id,
            result["fixture_id"],
            result["training_seed"],
        )
        payload = {
            "factual_z": {
                "cell_ids": cells,
                "values": base + result["training_seed"],
            }
        }
        if candidate_id != "B1":
            payload["u"] = {
                "cell_ids": cells,
                "values": 2.0 * base + result["training_seed"],
            }
        representations[key] = payload

    # Invalid values prove the aggregate never evaluates stopped representations.
    representations[("B1", "mixed", 0)]["factual_z"]["values"] = (
        np.full_like(base, np.nan)
    )
    representations[("B2", "mixed", 0)]["u"]["values"] = np.full_like(
        base,
        np.nan,
    )

    aggregate = aggregate_diagnosis_grid(
        results,
        representations=representations,
    )

    b3_pair = next(
        item
        for item in aggregate["paired_comparisons"]
        if (
            item["candidate_id"],
            item["fixture_id"],
            item["training_seed"],
        )
        == ("B3", "mixed", 0)
    )
    assert b3_pair["metrics"]["factual_z_linear_cka"] is None
    assert b3_pair["metrics"]["u_linear_cka"] is None
    assert b3_pair["representation_status"]["factual_z"] == {
        "status": "not_evaluated_hard_failure",
        "reference_candidate_id": "B1",
        "reason_codes": ["reference_B1_non_converged_at_maximum"],
    }
    assert b3_pair["representation_status"]["u"] == {
        "status": "not_evaluated_hard_failure",
        "reference_candidate_id": "B2",
        "reason_codes": [
            "reference_B2_latent_collapse:u_effective_rank"
        ],
    }

    b2_cross_seed = next(
        item
        for item in aggregate["cross_seed"]
        if (item["candidate_id"], item["fixture_id"]) == ("B2", "mixed")
    )
    assert (
        b2_cross_seed["metrics"]["u_cross_seed_knn_jaccard_k15"]
        is None
    )
    assert (
        b2_cross_seed["metrics"][
            "factual_z_cross_seed_knn_jaccard_k15"
        ]
        == 1.0
    )
    assert b2_cross_seed["representation_status"]["u"] == {
        "status": "not_evaluated_hard_failure",
        "reason_codes": [
            "candidate_seed0_latent_collapse:u_effective_rank"
        ],
    }
    assert b2_cross_seed["representation_status"]["factual_z"] == {
        "status": "evaluated",
        "reason_codes": [],
    }
