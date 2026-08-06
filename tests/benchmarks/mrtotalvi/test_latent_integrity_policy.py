"""Prospective latent-integrity v2 policy contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.mrtotalvi.latent_integrity import (
    latent_integrity_policy_digest_v2,
    latent_integrity_policy_v2,
    validate_latent_integrity_policy_v2,
)


def test_latent_integrity_v2_policy_is_complete_and_canonical():
    """The prospective policy is one complete machine-readable decision."""
    expected = {
        "schema_version": "mrtotalvi-latent-integrity-policy-v2",
        "policy_id": "mrtotalvi-rdx03-latent-integrity-v2",
        "scope": "prospective_v2_only",
        "supersedes": "adr-0008-effective-rank-terminal-clause-only",
        "historical_behavior": "preserve_exact_v1_replay",
        "terminal_integrity": {
            "invalid_diagnostics": "explicit_no_calls_continue_grid",
            "nonfinite_integrity_inputs": "fail",
            "exact_zero_centered_variation": "fail",
            "posterior_scale_elements": (
                "all_present_finite_strictly_positive"
            ),
            "registered_residual_gradient_coverage": {
                "applicable_model_families": ["mrtotalvi"],
                "comparison": "absolute_eq",
                "failure_scope": "affected_representation_only",
                "representation": "factual_z",
                "value": 1.0,
            },
        },
        "convergence": {
            "separate_hard_failure": True,
            "non_convergence_geometry": "no_call_all_representations",
        },
        "effective_rank": {
            "comparison": "configured_dimension_fraction_lt",
            "value": 0.5,
            "disposition": "low_rank_alert_only",
            "suppresses_geometry": False,
        },
        "representation_isolation": {
            "representations": ["u", "factual_z"],
            "terminal_failure_scope": "affected_representation_only",
        },
        "metric_preprocessing": "frozen_v1_unchanged",
        "downstream_scientific_gates": "frozen_v1_unchanged",
        "outcome_derived_thresholds": False,
        "knn_rescue_rule": False,
        "factual_human_da": "locked_not_computed_or_inspected",
    }

    assert latent_integrity_policy_v2() == expected
    assert latent_integrity_policy_digest_v2() == (
        "d8e8513768514aa598f7881b3ff6012b06b62b467af3214d2c6c44eb162af2a7"
    )
    policy_path = Path(
        "benchmarks/mrtotalvi/latent_integrity_policy_v2.json"
    )
    assert json.loads(policy_path.read_text(encoding="utf-8")) == expected


def test_latent_integrity_v2_policy_validation_fails_closed_on_drift():
    """Only the complete canonical prospective policy is accepted."""
    canonical = latent_integrity_policy_v2()
    assert validate_latent_integrity_policy_v2(canonical) == canonical

    changed = latent_integrity_policy_v2()
    changed["effective_rank"]["disposition"] = "terminal_failure"
    with pytest.raises(ValueError, match="does not match"):
        validate_latent_integrity_policy_v2(changed)

    changed = latent_integrity_policy_v2()
    changed["undeclared"] = True
    with pytest.raises(ValueError, match="does not match"):
        validate_latent_integrity_policy_v2(changed)
