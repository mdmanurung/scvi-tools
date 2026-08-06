"""Prospective latent-integrity behavior without changing historical v1."""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.mrtotalvi.convergence import (
    aggregate_diagnosis_grid_v2,
    assess_latent_collapse,
    assess_latent_integrity_v2,
    convergence_control,
)
from benchmarks.mrtotalvi.diagnostics import latent_diagnostics_v2
from benchmarks.mrtotalvi.versioning import (
    prospective_redesign_contract_adapter,
    version_binding_fields,
)


def _valid_v2_metrics() -> dict[str, float]:
    return {
        "u_representation_all_finite": 1.0,
        "u_exact_nonconstant_variation": 1.0,
        "u_posterior_scales_all_valid": 1.0,
        "u_effective_rank": 4.99,
        "factual_z_representation_all_finite": 1.0,
        "factual_z_exact_nonconstant_variation": 1.0,
        "factual_z_posterior_scales_all_valid": 1.0,
        "factual_z_effective_rank": 5.0,
        "registered_residual_gradient_coverage": 1.0,
    }


def _assessment_metrics(
    u: dict[str, float | None],
    factual_z: dict[str, float | None],
    *,
    gradient_coverage: float | None = 1.0,
) -> dict[str, float | None]:
    selected = {
        "representation_all_finite",
        "exact_nonconstant_variation",
        "posterior_scales_all_valid",
        "effective_rank",
    }
    return {
        **{
            f"u_{key}": value
            for key, value in u.items()
            if key in selected
        },
        **{
            f"factual_z_{key}": value
            for key, value in factual_z.items()
            if key in selected
        },
        "registered_residual_gradient_coverage": gradient_coverage,
    }


def _complete_v2_grid(values: np.ndarray):
    adapter = prospective_redesign_contract_adapter()
    bindings = version_binding_fields(adapter)
    cells = np.asarray([f"cell-{index}" for index in range(len(values))])
    results = []
    representations = {}
    for fixture_id in (
        "mixed",
        "unequal_cells",
        "sealed_500",
        "canonical_human_if_available",
    ):
        for candidate_id in ("B1", "B2", "B3", "D0"):
            for seed in (0, 1, 2):
                metrics = _valid_v2_metrics()
                assessment = {
                    **assess_latent_integrity_v2(
                        metrics,
                        candidate_id=candidate_id,
                        latent_dimension=values.shape[1],
                    ),
                    **bindings,
                }
                results.append(
                    {
                        "schema_version": "mrtotalvi-convergence-fit-v3",
                        **bindings,
                        "candidate_id": candidate_id,
                        "fixture_id": fixture_id,
                        "training_seed": seed,
                        "control": convergence_control().to_dict(),
                        "convergence": {"status": "converged"},
                        "latent_integrity": assessment,
                        "metrics": {},
                    }
                )
                payload = {
                    "factual_z": {
                        "cell_ids": cells.copy(),
                        "values": values.copy(),
                    }
                }
                if candidate_id != "B1":
                    payload["u"] = {
                        "cell_ids": cells.copy(),
                        "values": 2.0 * values,
                    }
                representations[(candidate_id, fixture_id, seed)] = payload
    return results, representations


def test_one_ulp_exact_nonconstant_representation_completes_full_v2_aggregate():
    values = np.ones((20, 4), dtype=np.float64)
    values[-1, 0] = np.nextafter(1.0, 2.0)
    results, representations = _complete_v2_grid(values)

    aggregate = aggregate_diagnosis_grid_v2(
        results,
        representations=representations,
        contract_adapter=prospective_redesign_contract_adapter(),
    )

    assert aggregate["n_fits"] == 48
    assert aggregate["terminal_integrity_failed"] is False
    assert all(
        status["status"] == "evaluated"
        for record in aggregate["paired_comparisons"]
        for status in record["representation_status"].values()
    )


def test_prospective_derived_geometry_failure_no_calls_only_affected_representation(
    monkeypatch,
):
    values = np.arange(80, dtype=np.float64).reshape(20, 4)
    results, representations = _complete_v2_grid(values)
    original = __import__(
        "benchmarks.mrtotalvi.convergence",
        fromlist=["linear_cka"],
    ).linear_cka

    def fail_u_only(first, second, **kwargs):
        if np.max(first) > 100.0:
            raise FloatingPointError("synthetic derived u failure")
        return original(first, second, **kwargs)

    monkeypatch.setattr(
        "benchmarks.mrtotalvi.convergence.linear_cka",
        fail_u_only,
    )
    aggregate = aggregate_diagnosis_grid_v2(
        results,
        representations=representations,
        contract_adapter=prospective_redesign_contract_adapter(),
    )
    record = next(
        item
        for item in aggregate["paired_comparisons"]
        if (item["candidate_id"], item["fixture_id"], item["training_seed"])
        == ("D0", "mixed", 0)
    )

    assert record["metrics"]["u_linear_cka"] is None
    assert record["representation_status"]["u"] == {
        "status": "not_evaluated_derived_diagnostic_failure",
        "reference_candidate_id": "B2",
        "reason_codes": ["u_derived_geometry_not_evaluable"],
    }
    assert record["metrics"]["factual_z_linear_cka"] is not None
    assert record["representation_status"]["factual_z"]["status"] == "evaluated"


def test_prospective_geometry_still_fails_closed_on_cell_lineage_drift():
    values = np.arange(80, dtype=np.float64).reshape(20, 4)
    results, representations = _complete_v2_grid(values)
    representations[("D0", "mixed", 0)]["factual_z"]["cell_ids"][-1] = (
        "changed-cell"
    )

    with pytest.raises(ValueError, match="same exact cell-ID set"):
        aggregate_diagnosis_grid_v2(
            results,
            representations=representations,
            contract_adapter=prospective_redesign_contract_adapter(),
        )


def test_low_rank_fails_v1_but_is_alert_only_and_geometry_eligible_v2():
    """Rank is historical terminal evidence but prospective descriptive evidence."""
    historical = assess_latent_collapse(
        {
            "u_effective_rank": 4.99,
            "factual_z_effective_rank": 5.0,
            "u_latent_variance": 0.2,
            "factual_z_latent_variance": 0.2,
            "u_posterior_scale": 0.3,
            "factual_z_posterior_scale": 0.3,
            "registered_residual_gradient_coverage": 1.0,
            "latent_all_finite": 1.0,
        },
        candidate_id="D0",
        latent_dimension=10,
    )
    prospective = assess_latent_integrity_v2(
        _valid_v2_metrics(),
        candidate_id="D0",
        latent_dimension=10,
    )

    assert historical["failed"] is True
    assert historical["failed_metrics"] == ["u_effective_rank"]
    assert prospective == {
        "schema_version": "mrtotalvi-latent-integrity-assessment-v3",
        "terminal_integrity_failed": False,
        "effective_rank_screen_flags": ["u"],
        "required_representations": ["u", "factual_z"],
        "latent_dimension": 10,
        "representations": {
            "u": {
                "terminal_failed": False,
                "terminal_failure_reasons": [],
                "effective_rank": 4.99,
                "effective_rank_threshold": 5.0,
                "low_rank_alert": True,
                "eligible_for_geometry": True,
            },
            "factual_z": {
                "terminal_failed": False,
                "terminal_failure_reasons": [],
                "effective_rank": 5.0,
                "effective_rank_threshold": 5.0,
                "low_rank_alert": False,
                "eligible_for_geometry": True,
            },
        },
    }


def test_invertible_anisotropic_rescaling_changes_rank_not_v2_eligibility():
    """No rank threshold is allowed to masquerade as an integrity threshold."""
    base = np.vstack([np.eye(4), -np.eye(4)])
    scaled = base @ np.diag([1_000.0, 0.001, 0.001, 0.001])
    posterior_scale = np.ones_like(base)

    original = latent_diagnostics_v2(
        base,
        posterior_scale=posterior_scale,
    )
    anisotropic = latent_diagnostics_v2(
        scaled,
        posterior_scale=posterior_scale,
    )

    assert original["effective_rank"] > 2.0
    assert anisotropic["effective_rank"] < 2.0
    for diagnostics in (original, anisotropic):
        assert diagnostics["representation_all_finite"] == 1.0
        assert diagnostics["exact_nonconstant_variation"] == 1.0
        assert diagnostics["posterior_scales_all_valid"] == 1.0
        metrics = {
            f"{representation}_{metric_id}": value
            for representation in ("u", "factual_z")
            for metric_id, value in diagnostics.items()
            if metric_id
            in {
                "representation_all_finite",
                "exact_nonconstant_variation",
                "posterior_scales_all_valid",
                "effective_rank",
            }
        }
        metrics["registered_residual_gradient_coverage"] = 1.0
        assessment = assess_latent_integrity_v2(
            metrics,
            candidate_id="D0",
            latent_dimension=4,
        )
        assert assessment["terminal_integrity_failed"] is False
        assert all(
            item["eligible_for_geometry"]
            for item in assessment["representations"].values()
        )


def test_nonfinite_and_exact_constant_fail_only_the_affected_representation():
    """Integrity uses exact evidence, including a no-epsilon variation rule."""
    valid_values = np.vstack([np.eye(4), -np.eye(4)])
    valid = latent_diagnostics_v2(
        valid_values,
        posterior_scale=np.ones_like(valid_values),
    )
    nonfinite_values = valid_values.copy()
    nonfinite_values[0, 0] = np.nan
    nonfinite = latent_diagnostics_v2(
        nonfinite_values,
        posterior_scale=np.ones_like(nonfinite_values),
    )
    constant_values = np.ones((8, 4), dtype=np.float64)
    constant = latent_diagnostics_v2(
        constant_values,
        posterior_scale=np.ones_like(constant_values),
    )
    one_ulp_values = constant_values.copy()
    one_ulp_values[-1, 0] = np.nextafter(1.0, 2.0)
    one_ulp = latent_diagnostics_v2(
        one_ulp_values,
        posterior_scale=np.ones_like(one_ulp_values),
    )

    assert nonfinite["representation_all_finite"] == 0.0
    assert constant["exact_nonconstant_variation"] == 0.0
    assert one_ulp["exact_nonconstant_variation"] == 1.0

    for affected, reason in (
        (nonfinite, "u_representation_all_finite"),
        (constant, "u_exact_nonconstant_variation"),
    ):
        assessment = assess_latent_integrity_v2(
            _assessment_metrics(affected, valid),
            candidate_id="D0",
            latent_dimension=4,
        )
        assert assessment["terminal_integrity_failed"] is True
        assert assessment["representations"]["u"][
            "terminal_failure_reasons"
        ] == [reason]
        assert (
            assessment["representations"]["u"]["eligible_for_geometry"]
            is False
        )
        assert (
            assessment["representations"]["factual_z"]["terminal_failed"]
            is False
        )
        assert (
            assessment["representations"]["factual_z"][
                "eligible_for_geometry"
            ]
            is True
        )

    reverse = assess_latent_integrity_v2(
        _assessment_metrics(valid, constant),
        candidate_id="D0",
        latent_dimension=4,
    )
    assert reverse["representations"]["u"]["terminal_failed"] is False
    assert reverse["representations"]["factual_z"][
        "terminal_failure_reasons"
    ] == ["factual_z_exact_nonconstant_variation"]

    one_ulp_assessment = assess_latent_integrity_v2(
        _assessment_metrics(one_ulp, valid),
        candidate_id="D0",
        latent_dimension=4,
    )
    assert one_ulp_assessment["terminal_integrity_failed"] is False
    assert one_ulp_assessment["representations"]["u"][
        "eligible_for_geometry"
    ] is True


@pytest.mark.parametrize(
    "posterior_scale",
    [
        None,
        np.ones((8, 3)),
        np.vstack([np.full((1, 4), np.nan), np.ones((7, 4))]),
        np.vstack([np.full((1, 4), np.inf), np.ones((7, 4))]),
        np.vstack([np.zeros((1, 4)), np.ones((7, 4))]),
        np.vstack([-np.ones((1, 4)), np.ones((7, 4))]),
    ],
)
def test_every_posterior_scale_element_must_be_present_finite_and_positive(
    posterior_scale,
):
    """A valid mean cannot hide one missing or invalid posterior scale."""
    values = np.vstack([np.eye(4), -np.eye(4)])
    invalid = latent_diagnostics_v2(
        values,
        posterior_scale=posterior_scale,
    )
    valid = latent_diagnostics_v2(
        values,
        posterior_scale=np.ones_like(values),
    )

    assert invalid["posterior_scales_all_valid"] == 0.0
    assert invalid["posterior_scale"] is None
    assessment = assess_latent_integrity_v2(
        _assessment_metrics(invalid, valid),
        candidate_id="D0",
        latent_dimension=4,
    )
    assert assessment["representations"]["u"][
        "terminal_failure_reasons"
    ] == ["u_posterior_scales_all_valid"]
    assert assessment["representations"]["factual_z"][
        "eligible_for_geometry"
    ] is True


@pytest.mark.parametrize(
    "coverage",
    [None, np.nan, 0.0, 0.99, 0.9999999999999999, 1.0000000000000002],
)
def test_mrtotalvi_gradient_coverage_requires_exactly_one(coverage):
    """Gradient coverage is exact and scoped to affected factual z."""
    metrics = _valid_v2_metrics()
    metrics["registered_residual_gradient_coverage"] = coverage

    assessment = assess_latent_integrity_v2(
        metrics,
        candidate_id="D0",
        latent_dimension=10,
    )

    assert assessment["terminal_integrity_failed"] is True
    assert assessment["representations"]["factual_z"][
        "terminal_failure_reasons"
    ] == ["registered_residual_gradient_coverage"]
    assert assessment["representations"]["u"][
        "eligible_for_geometry"
    ] is True

    b1 = assess_latent_integrity_v2(
        metrics,
        candidate_id="B1",
        latent_dimension=10,
    )
    assert b1["terminal_integrity_failed"] is False


def test_exact_mrtotalvi_gradient_coverage_passes():
    assessment = assess_latent_integrity_v2(
        _valid_v2_metrics(),
        candidate_id="D0",
        latent_dimension=10,
    )

    assert assessment["representations"]["factual_z"]["terminal_failed"] is False


def test_low_rank_alerts_remain_representation_specific_through_aggregate(
    monkeypatch,
):
    """Alerted candidates and references retain paired and cross-seed geometry."""
    adapter = prospective_redesign_contract_adapter()
    bindings = version_binding_fields(adapter)
    rows = ("B1", "B2", "B3", "D0")
    fixtures = (
        "mixed",
        "unequal_cells",
        "sealed_500",
        "canonical_human_if_available",
    )
    results = []
    for fixture_id in fixtures:
        for candidate_id in rows:
            for seed in (0, 1, 2):
                metrics = _valid_v2_metrics()
                if (candidate_id, fixture_id, seed) == ("B2", "mixed", 0):
                    metrics["u_effective_rank"] = 1.99
                if (candidate_id, fixture_id, seed) == ("B1", "mixed", 0):
                    metrics["factual_z_effective_rank"] = 1.99
                assessment = assess_latent_integrity_v2(
                    metrics,
                    candidate_id=candidate_id,
                    latent_dimension=4,
                )
                assessment = {**assessment, **bindings}
                results.append(
                    {
                        "schema_version": "mrtotalvi-convergence-fit-v3",
                        **bindings,
                        "candidate_id": candidate_id,
                        "fixture_id": fixture_id,
                        "training_seed": seed,
                        "control": convergence_control().to_dict(),
                        "convergence": {"status": "converged"},
                        "latent_integrity": assessment,
                        "metrics": {
                            "factual_z_effective_rank": metrics[
                                "factual_z_effective_rank"
                            ],
                            "u_effective_rank": (
                                metrics["u_effective_rank"]
                                if candidate_id != "B1"
                                else None
                            ),
                        },
                    }
                )

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

    monkeypatch.setattr(
        "benchmarks.mrtotalvi.convergence.redesign_run_contract",
        lambda: (_ for _ in ()).throw(
            AssertionError("v2 aggregate consulted a live default contract")
        ),
    )
    aggregate = aggregate_diagnosis_grid_v2(
        results,
        representations=representations,
        contract_adapter=prospective_redesign_contract_adapter(),
    )

    assert aggregate["schema_version"] == "mrtotalvi-convergence-aggregate-v3"
    assert {
        name: aggregate[name] for name in version_binding_fields(adapter)
    } == version_binding_fields(adapter)
    assert aggregate["terminal_integrity_failed"] is False
    assert aggregate["all_fits_converged_with_terminal_integrity"] is True
    assert {
        (
            item["candidate_id"],
            item["fixture_id"],
            item["training_seed"],
            tuple(item["representations"]),
        )
        for item in aggregate["effective_rank_screen_flags"]
    } == {
        ("B1", "mixed", 0, ("factual_z",)),
        ("B2", "mixed", 0, ("u",)),
    }

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
    assert b3_pair["metrics"]["u_linear_cka"] is not None
    assert b3_pair["metrics"]["factual_z_linear_cka"] is not None
    assert b3_pair["representation_status"]["u"]["status"] == "evaluated"
    assert (
        b3_pair["representation_status"]["factual_z"]["status"]
        == "evaluated"
    )

    b2_cross_seed = next(
        item
        for item in aggregate["cross_seed"]
        if (item["candidate_id"], item["fixture_id"]) == ("B2", "mixed")
    )
    assert b2_cross_seed["metrics"]["u_cross_seed_knn_jaccard_k15"] == 1.0
    assert (
        b2_cross_seed["metrics"]["factual_z_cross_seed_knn_jaccard_k15"]
        == 1.0
    )
    assert next(
        result
        for result in results
        if (
            result["candidate_id"],
            result["fixture_id"],
            result["training_seed"],
        )
        == ("B1", "mixed", 0)
    )["metrics"]["factual_z_effective_rank"] == 1.99


def test_terminal_failure_and_nonconvergence_serialize_scoped_no_calls():
    """Unsafe geometry is skipped without aborting the exact 48-fit grid."""
    adapter = prospective_redesign_contract_adapter()
    bindings = version_binding_fields(adapter)
    rows = ("B1", "B2", "B3", "D0")
    fixtures = (
        "mixed",
        "unequal_cells",
        "sealed_500",
        "canonical_human_if_available",
    )
    values = np.vstack([np.eye(10), -np.eye(10)])
    valid = latent_diagnostics_v2(
        values,
        posterior_scale=np.ones_like(values),
    )
    invalid_values = values.copy()
    invalid_values[0, 0] = np.nan
    invalid = latent_diagnostics_v2(
        invalid_values,
        posterior_scale=np.ones_like(invalid_values),
    )
    results = []
    representations = {}
    cells = np.asarray([f"cell-{index}" for index in range(20)])
    for fixture_id in fixtures:
        for candidate_id in rows:
            for seed in (0, 1, 2):
                is_invalid_u = (
                    candidate_id,
                    fixture_id,
                    seed,
                ) == ("B2", "mixed", 0)
                assessment = assess_latent_integrity_v2(
                    _assessment_metrics(
                        invalid if is_invalid_u else valid,
                        valid,
                    ),
                    candidate_id=candidate_id,
                    latent_dimension=10,
                )
                assessment = {**assessment, **bindings}
                nonconverged = (
                    candidate_id,
                    fixture_id,
                    seed,
                ) == ("B3", "unequal_cells", 0)
                result = {
                    "schema_version": "mrtotalvi-convergence-fit-v3",
                    **bindings,
                    "candidate_id": candidate_id,
                    "fixture_id": fixture_id,
                    "training_seed": seed,
                    "control": convergence_control().to_dict(),
                    "convergence": {
                        "status": (
                            "non_converged_at_maximum"
                            if nonconverged
                            else "converged"
                        )
                    },
                    "latent_integrity": assessment,
                    "metrics": {},
                }
                results.append(result)
                payload = {
                    "factual_z": {
                        "cell_ids": cells,
                        "values": (
                            np.full_like(values, np.nan)
                            if nonconverged
                            else values + seed
                        ),
                    }
                }
                if candidate_id != "B1":
                    payload["u"] = {
                        "cell_ids": cells,
                        "values": (
                            np.full_like(values, np.nan)
                            if nonconverged or is_invalid_u
                            else 2.0 * values + seed
                        ),
                    }
                representations[(candidate_id, fixture_id, seed)] = payload

    aggregate = aggregate_diagnosis_grid_v2(
        results,
        representations=representations,
        contract_adapter=prospective_redesign_contract_adapter(),
    )

    assert aggregate["n_fits"] == 48
    assert aggregate["terminal_integrity_failed"] is True
    assert aggregate["all_fits_converged_with_terminal_integrity"] is False

    d0_mixed_seed0 = next(
        item
        for item in aggregate["paired_comparisons"]
        if (
            item["candidate_id"],
            item["fixture_id"],
            item["training_seed"],
        )
        == ("D0", "mixed", 0)
    )
    assert d0_mixed_seed0["metrics"]["u_linear_cka"] is None
    assert d0_mixed_seed0["metrics"]["factual_z_linear_cka"] is not None
    assert d0_mixed_seed0["representation_status"]["u"][
        "reason_codes"
    ] == [
        "reference_B2_terminal_integrity:u_representation_all_finite"
    ]
    assert (
        d0_mixed_seed0["representation_status"]["factual_z"]["status"]
        == "evaluated"
    )

    b2_cross_seed = next(
        item
        for item in aggregate["cross_seed"]
        if (item["candidate_id"], item["fixture_id"]) == ("B2", "mixed")
    )
    assert b2_cross_seed["metrics"]["u_cross_seed_knn_jaccard_k15"] is None
    assert (
        b2_cross_seed["metrics"]["factual_z_cross_seed_knn_jaccard_k15"]
        == 1.0
    )
    assert b2_cross_seed["representation_status"]["u"]["reason_codes"] == [
        "candidate_seed0_terminal_integrity:u_representation_all_finite"
    ]

    b3_nonconverged = next(
        item
        for item in aggregate["paired_comparisons"]
        if (
            item["candidate_id"],
            item["fixture_id"],
            item["training_seed"],
        )
        == ("B3", "unequal_cells", 0)
    )
    assert b3_nonconverged["metrics"]["u_linear_cka"] is None
    assert b3_nonconverged["metrics"]["factual_z_linear_cka"] is None
    assert b3_nonconverged["representation_status"]["u"][
        "reason_codes"
    ] == ["candidate_non_converged_at_maximum"]
    assert b3_nonconverged["representation_status"]["factual_z"][
        "reason_codes"
    ] == ["candidate_non_converged_at_maximum"]
