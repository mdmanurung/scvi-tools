"""Frozen governance contracts for the MrTotalVI stable-latent redesign."""

from __future__ import annotations

from copy import deepcopy

import pytest
from benchmarks.mrtotalvi import (
    RedesignVerdict,
    redesign_candidate_configs,
    redesign_config_digest,
    redesign_run_contract,
    redesign_run_contract_digest,
    validate_redesign_candidate,
    validate_redesign_run_contract,
)


def test_redesign_candidates_are_exact_and_reject_undeclared_axes():
    """B0-B3/D0-D5 encode exactly the preregistered factorial."""
    candidates = redesign_candidate_configs()

    assert tuple(candidates) == (
        "B0",
        "B1",
        "B2",
        "B3",
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    )
    assert candidates["B0"].model_family == "scvi"
    assert candidates["B0"].modality_scope == "rna"
    assert candidates["B1"].model_family == "totalvi"
    assert candidates["B1"].modality_scope == "rna_protein"
    assert candidates["B2"].hierarchy_mode == "legacy"
    assert candidates["B3"].biological_sample_conditioning == "conditioned"

    d0 = candidates["D0"].model_axes()
    expected_changes = {
        "D1": {"input_transform": "totalvi_per_modality"},
        "D2": {
            "input_transform": "totalvi_per_modality",
            "posterior_trunk": "totalvi_fclayers",
        },
        "D3": {
            "input_transform": "totalvi_per_modality",
            "posterior_trunk": "totalvi_fclayers",
            "u_prior": "mog_trainable",
        },
        "D4": {
            "input_transform": "totalvi_per_modality",
            "posterior_trunk": "totalvi_fclayers",
            "observation_weighting": "sample_equal",
        },
        "D5": {
            "input_transform": "totalvi_per_modality",
            "posterior_trunk": "totalvi_fclayers",
            "u_prior": "mog_trainable",
            "observation_weighting": "sample_equal",
        },
    }
    for candidate_id, expected in expected_changes.items():
        observed = {
            key: value
            for key, value in candidates[candidate_id].model_axes().items()
            if d0[key] != value
        }
        assert observed == expected

    d2 = candidates["D2"].to_dict()
    assert validate_redesign_candidate(d2) == candidates["D2"]
    with pytest.raises(ValueError, match="Unknown redesign candidate"):
        validate_redesign_candidate(d2 | {"candidate_id": "D6"})
    with pytest.raises(ValueError, match="Unknown redesign candidate fields"):
        validate_redesign_candidate(d2 | {"decoder_capacity": "larger"})
    with pytest.raises(ValueError, match="does not match frozen axes"):
        validate_redesign_candidate(d2 | {"u_prior": "mog_trainable"})

    digest = redesign_config_digest()
    assert len(digest) == 64
    int(digest, 16)


@pytest.mark.parametrize(
    ("candidate_id", "public_mode"),
    [
        ("D1", "sample_blind_scaled"),
        ("D2", "sample_blind_totalvi"),
        ("D3", "sample_blind_totalvi"),
        ("D4", "sample_blind_totalvi"),
        ("D5", "sample_blind_totalvi"),
    ],
)
def test_candidate_verdict_requires_matching_eligible_public_mode(
    candidate_id,
    public_mode,
):
    """A candidate verdict freezes code/config evidence and one eligible mode."""
    verdict = RedesignVerdict.from_dict(
        {
            "schema_version": "mrtotalvi-redesign-verdict-v1",
            "verdict": "candidate",
            "candidate_id": candidate_id,
            "public_mode": public_mode,
            "reason_codes": [],
            "code_digest": "a" * 64,
            "config_digest": "b" * 64,
            "evidence_digest": "c" * 64,
        }
    )
    assert verdict.candidate_id == candidate_id
    assert verdict.public_mode == public_mode


def test_verdict_schema_accepts_only_candidate_stop_or_blocked():
    """Terminal governance cannot promote D0 or malformed/no-evidence outcomes."""
    common = {
        "schema_version": "mrtotalvi-redesign-verdict-v1",
        "candidate_id": None,
        "public_mode": None,
        "reason_codes": ["no_redesign_passed"],
        "code_digest": "a" * 64,
        "config_digest": "b" * 64,
        "evidence_digest": "c" * 64,
    }
    assert RedesignVerdict.from_dict(common | {"verdict": "stop"}).verdict == "stop"
    assert (
        RedesignVerdict.from_dict(
            common
            | {
                "verdict": "blocked",
                "reason_codes": ["missing_required_dependency"],
            }
        ).verdict
        == "blocked"
    )

    invalid_payloads = [
        common | {"verdict": "inconclusive"},
        common
        | {
            "verdict": "candidate",
            "candidate_id": "D0",
            "public_mode": None,
            "reason_codes": [],
        },
        common | {"verdict": "stop", "candidate_id": "D2"},
        common | {"verdict": "blocked", "reason_codes": []},
        common | {"verdict": "stop", "extra": True},
        common | {"verdict": "stop", "code_digest": "not-a-digest"},
    ]
    for payload in invalid_payloads:
        with pytest.raises(ValueError):
            RedesignVerdict.from_dict(payload)

    with pytest.raises(ValueError, match="tuple of unique non-empty strings"):
        RedesignVerdict(
            schema_version="mrtotalvi-redesign-verdict-v1",
            verdict="blocked",
            candidate_id=None,
            public_mode=None,
            reason_codes=(1,),
            code_digest="a" * 64,
            config_digest="b" * 64,
            evidence_digest="c" * 64,
        )


def test_run_contract_freezes_metrics_gates_grid_seeds_and_environment():
    """The complete preregistered screen fails closed before any new fit."""
    contract = redesign_run_contract()
    payload = contract.to_dict()

    assert payload["schema_version"] == "mrtotalvi-redesign-run-contract-v1"
    assert payload["evaluation_rows"] == [
        "B0",
        "B1",
        "B2",
        "B3",
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    ]
    assert payload["candidate_ids"] == ["D1", "D2", "D3", "D4", "D5"]
    assert payload["rng_streams"] == ["truth", "training", "evaluation"]
    assert payload["stage_a"]["training_seeds"] == [0]
    assert payload["stage_a"]["instances_per_scenario"] == 3
    assert payload["stage_b"]["training_seeds"] == [0, 1, 2]
    assert payload["stage_b"]["instances_per_scenario"] == 10
    assert payload["stage_a"]["max_redesign_survivors"] == 2
    assert payload["diagnosis"] == {
        "rows": ["B1", "B2", "B3", "D0"],
        "training_seeds": [0, 1, 2],
        "fixtures": [
            "mixed",
            "unequal_cells",
            "sealed_500",
            "canonical_human_if_available",
        ],
        "canonical_human_requires_lineage_gate": True,
    }
    assert payload["convergence"] == {
        "check_every_epochs": 5,
        "minimum_epochs": 50,
        "maximum_epochs": 400,
        "patience_checks": 30,
        "restore_best_checkpoint": True,
        "candidate_specific_retuning": False,
    }
    assert payload["selection"]["primary_metric"] == (
        "milo_localization_spatialfdr_0_10"
    )
    assert payload["selection"]["tie_tolerance"] == 0.02
    assert payload["runtime"]["primary_python"] == "3.13.11"
    assert payload["runtime"]["contract_python_minors"] == ["3.13", "3.14"]
    assert payload["runtime"]["r_runtime"].endswith("/R4_51/bin/Rscript")

    metric_ids = set(payload["metric_ids"])
    assert "multimodal_heldout_predictive_loss" in metric_ids
    assert "milo_fdp_spatialfdr_0_10" in metric_ids
    assert "trainable_parameter_count" in metric_ids
    assert len(metric_ids) == len(payload["metric_ids"])
    assert len(payload["hard_gates"]) == 24

    digest = redesign_run_contract_digest()
    assert len(digest) == 64
    int(digest, 16)
    assert validate_redesign_run_contract(payload) == contract


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_metric",
        "threshold",
        "stage_seed",
        "tie_break",
        "environment",
        "diagnosis_grid",
        "extra_axis",
    ],
)
def test_run_contract_rejects_every_preregistered_axis_change(mutation):
    """No endpoint or execution-control drift can pass schema validation."""
    payload = deepcopy(redesign_run_contract().to_dict())
    if mutation == "extra_metric":
        payload["metric_ids"].append("unregistered_composite_score")
    elif mutation == "threshold":
        payload["hard_gates"][0]["value"] = 1e-5
    elif mutation == "stage_seed":
        payload["stage_a"]["training_seeds"] = [0, 1]
    elif mutation == "tie_break":
        payload["selection"]["tie_break_metrics"].reverse()
    elif mutation == "environment":
        payload["runtime"]["primary_python"] = "3.14.0"
    elif mutation == "diagnosis_grid":
        payload["diagnosis"]["fixtures"].remove("sealed_500")
    else:
        payload["decoder_capacity"] = "larger"

    with pytest.raises(ValueError, match="frozen run contract"):
        validate_redesign_run_contract(payload)
