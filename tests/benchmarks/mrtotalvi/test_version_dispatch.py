"""Explicit version dispatch for historical and prospective RDX-03 evidence."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from benchmarks.mrtotalvi import (
    run_convergence_diagnosis as convergence_execution,
)
from benchmarks.mrtotalvi.convergence import assess_latent_collapse
from benchmarks.mrtotalvi.redesign_contract import (
    redesign_run_contract,
    redesign_run_contract_digest,
    redesign_run_contract_digest_v2,
    redesign_run_contract_v2,
)
from benchmarks.mrtotalvi.run_convergence_diagnosis import (
    DiagnosisExecutionRequest,
    _partial_aggregate,
)
from benchmarks.mrtotalvi.versioning import (
    LEGACY_EXECUTION_SCHEMA_V1,
    PROSPECTIVE_EXECUTION_SCHEMA_V2,
    REDESIGN_RUN_CONTRACT_V1_DIGESTS,
    canonical_payload_digest,
    historical_redesign_contract_adapter,
    prospective_redesign_contract_adapter,
    registered_redesign_run_contract_v1,
    resolve_redesign_contract_adapter,
    validate_version_binding,
    version_binding_fields,
)

_CURRENT_DIGEST = (
    "fe650e8c6275568a0c1a2174a9078d0990e6b4c2400443e6828f7544d8e8c26b"
)
_PRE_DIAGNOSIS_DIGEST = (
    "7cccca9b0b1863a9c345ba570bd39100193a1515ca23462375d46511cdc7402c"
)
_GOVERNANCE_ROOT = (
    Path(__file__).parents[3]
    / ".scratch"
    / "mrtotalvi-v2-redesign"
    / "governance-runs"
)


def _historical_payloads() -> tuple[dict, dict]:
    paths = (
        _GOVERNANCE_ROOT
        / "20260726T113218Z-a70f2b5ffdec"
        / "redesign-run-contract.json",
        _GOVERNANCE_ROOT
        / "20260726T113947Z-1773319a2735"
        / "redesign-run-contract.json",
    )
    return tuple(
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    )


def test_both_historical_v1_contract_payloads_replay_exactly():
    """Both immutable v1 variants are selected only by their exact digest."""
    pre_diagnosis, current = _historical_payloads()

    assert REDESIGN_RUN_CONTRACT_V1_DIGESTS == (
        _PRE_DIAGNOSIS_DIGEST,
        _CURRENT_DIGEST,
    )
    for payload, expected_digest in (
        (pre_diagnosis, _PRE_DIAGNOSIS_DIGEST),
        (current, _CURRENT_DIGEST),
    ):
        original = deepcopy(payload)
        adapter = registered_redesign_run_contract_v1(payload)

        assert canonical_payload_digest(payload) == expected_digest
        assert adapter.integrity_version == "v1"
        assert adapter.run_contract_schema_version == (
            "mrtotalvi-redesign-run-contract-v1"
        )
        assert adapter.run_contract_digest == expected_digest
        assert adapter.assessment_entry_point == "assess_latent_collapse"
        assert adapter.run_contract_payload() == original
        assert payload == original


def test_historical_v1_registry_is_anchored_to_immutable_file_bytes():
    """The golden variants cannot be reconstructed from the same live factory."""
    paths = (
        _GOVERNANCE_ROOT
        / "20260726T113218Z-a70f2b5ffdec"
        / "redesign-run-contract.json",
        _GOVERNANCE_ROOT
        / "20260726T113947Z-1773319a2735"
        / "redesign-run-contract.json",
    )
    assert tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths) == (
        "1e7b022d2d4a967c7f9bf7ebc5f38826ac02fe057eb87acaf704c034ad250095",
        "6f81c023b37c9dc8003cbf31ec3870c2bbd90c96f82c1702ade5e167ec90be48",
    )
    assert tuple(
        canonical_payload_digest(
            json.loads(path.read_text(encoding="utf-8"))
        )
        for path in paths
    ) == REDESIGN_RUN_CONTRACT_V1_DIGESTS


def test_current_v1_contract_writer_payload_is_byte_stable():
    """The live v1 factory still writes the exact authoritative v1 bytes."""
    historical = (
        _GOVERNANCE_ROOT
        / "20260726T113947Z-1773319a2735"
        / "redesign-run-contract.json"
    ).read_bytes()
    current = (
        json.dumps(
            redesign_run_contract().to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()

    assert current == historical
    assert canonical_payload_digest(redesign_run_contract().to_dict()) == (
        _CURRENT_DIGEST
    )
    assert redesign_run_contract_digest() == _CURRENT_DIGEST


def test_v1_rank_failure_replays_the_exact_historical_classification():
    """Explicit compatibility work cannot change the old rank hard failure."""
    assessment = assess_latent_collapse(
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

    assert assessment == {
        "failed": True,
        "failed_metrics": ["u_effective_rank"],
        "required_representations": ["u", "factual_z"],
        "latent_dimension": 10,
        "observed": {
            "u_effective_rank": 4.99,
            "u_latent_variance": 0.2,
            "u_posterior_scale": 0.3,
            "factual_z_effective_rank": 5.0,
            "factual_z_latent_variance": 0.2,
            "factual_z_posterior_scale": 0.3,
            "latent_all_finite": 1.0,
            "registered_residual_gradient_coverage": 1.0,
        },
    }


def test_legacy_execution_schema_resolves_only_to_registered_v1():
    """The policy-free execution schema has exactly one compatibility path."""
    pre_diagnosis, current = _historical_payloads()

    for payload in (pre_diagnosis, current):
        adapter = resolve_redesign_contract_adapter(
            {
                "schema_version": LEGACY_EXECUTION_SCHEMA_V1,
                "redesign_run_contract_digest": canonical_payload_digest(
                    payload
                ),
            },
        )
        assert adapter.integrity_version == "v1"
        assert adapter.latent_integrity_policy_id is None
        assert adapter.latent_integrity_policy_digest is None

        supplied = resolve_redesign_contract_adapter(
            {
                "schema_version": LEGACY_EXECUTION_SCHEMA_V1,
                "redesign_run_contract_digest": canonical_payload_digest(
                    payload
                ),
            },
            run_contract_payload=payload,
        )
        assert supplied.run_contract_payload() == payload


def test_legacy_execution_rejects_swapped_registered_digest_and_payload():
    """Two valid v1 identities cannot be cross-substituted."""
    pre_diagnosis, current = _historical_payloads()

    with pytest.raises(ValueError, match="does not match"):
        resolve_redesign_contract_adapter(
            {
                "schema_version": LEGACY_EXECUTION_SCHEMA_V1,
                "redesign_run_contract_digest": canonical_payload_digest(
                    pre_diagnosis
                ),
            },
            run_contract_payload=current,
        )


def test_historical_v1_artifact_classifications_are_unchanged():
    """Compatibility replay cannot make old evidence newly promotable."""
    probe = (
        Path(__file__).parents[3]
        / ".scratch"
        / "mrtotalvi-v2-redesign"
        / "convergence-runs"
        / "20260726T160431Z-d5e85d96-d79ef38b-e6006ef7"
    )
    configuration = json.loads(
        (probe / "configuration.json").read_text(encoding="utf-8")
    )
    assert (
        resolve_redesign_contract_adapter(configuration).integrity_version
        == "v1"
    )
    manifest = json.loads(
        (probe / "run-manifest.json").read_text(encoding="utf-8")
    )
    aggregate = json.loads(
        (probe / "aggregate.json").read_text(encoding="utf-8")
    )
    assert (manifest["evidence_tier"], manifest["status"]) == (
        "pilot_cache",
        "inconclusive",
    )
    assert aggregate["authoritative_full_grid"] is False
    assert aggregate["rdx03_completion"] == "prohibited"
    assert aggregate["d0_decision"] == (
        "not_available_for_non_authoritative_grid"
    )

    preserved = (
        Path(__file__).parents[3]
        / ".scratch"
        / "mrtotalvi-v2-redesign"
        / "PRESERVED-20260726-partial-run-42of48"
    )
    assert not (preserved / "run-manifest.json").exists()
    assert not (preserved / "aggregate.json").exists()


def test_verifier_recomputes_every_valid_sealed_fit_identity(monkeypatch):
    """A valid checkpoint must reach independent fit-identity recomputation."""
    probe = (
        Path(__file__).parents[3]
        / ".scratch"
        / "mrtotalvi-v2-redesign"
        / "convergence-runs"
        / "20260726T160431Z-d5e85d96-d79ef38b-e6006ef7"
    )

    def reached(*_args, **_kwargs):
        raise RuntimeError("fit identity recomputation reached")

    monkeypatch.setattr(
        convergence_execution,
        "_validate_recomputed_fit_identity",
        reached,
    )

    with pytest.raises(RuntimeError, match="recomputation reached"):
        convergence_execution.verify_convergence_run(probe)


def test_prospective_v2_contract_replaces_stale_global_integrity_gates():
    """Prospective terminal gates are representation-specific and exact."""
    historical = redesign_run_contract()
    prospective = redesign_run_contract_v2()

    assert prospective.schema_version == "mrtotalvi-redesign-run-contract-v2"
    assert {
        gate.gate_id for gate in historical.hard_gates
    } - {gate.gate_id for gate in prospective.hard_gates} == {
        "u_finite",
        "factual_z_finite",
        "u_effective_rank",
        "factual_z_effective_rank",
    }
    prospective_gates = {gate.gate_id: gate for gate in prospective.hard_gates}
    for representation in ("u", "factual_z"):
        for indicator in (
            "representation_all_finite",
            "exact_nonconstant_variation",
            "posterior_scales_all_valid",
        ):
            gate_id = f"{representation}_{indicator}"
            gate = prospective_gates[gate_id]
            assert gate.metric_id == gate_id
            assert gate.comparison == "absolute_eq"
            assert gate.value == 1.0
            assert gate.scope == (representation,)
    assert prospective_gates["registered_residual_gradients"].scope == (
        "factual_z",
    )
    assert len(prospective.hard_gates) == len(historical.hard_gates) + 2
    assert prospective.metric_ids == (
        *historical.metric_ids,
        "u_representation_all_finite",
        "factual_z_representation_all_finite",
        "u_exact_nonconstant_variation",
        "factual_z_exact_nonconstant_variation",
        "u_posterior_scales_all_valid",
        "factual_z_posterior_scales_all_valid",
    )
    assert "latent_collapse" not in prospective.stage_a.prune_on
    assert "terminal_integrity_failure" in prospective.stage_a.prune_on
    assert prospective.stage_a.prune_on == prospective.stage_b.prune_on
    assert redesign_run_contract_digest_v2() == canonical_payload_digest(
        prospective.to_dict()
    )
    assert redesign_run_contract_digest_v2() == (
        "4d75b395cb1faf644be09405542b1b1973093e553498754ff61af868696d36af"
    )
    assert redesign_run_contract_digest() == _CURRENT_DIGEST


def test_prospective_dispatch_requires_exact_contract_and_policy_bindings():
    """V3 selection requires all four stored identities and both full payloads."""
    contract = redesign_run_contract_v2().to_dict()
    policy = json.loads(
        (
            Path(__file__).parents[3]
            / "benchmarks"
            / "mrtotalvi"
            / "latent_integrity_policy_v2.json"
        ).read_text(encoding="utf-8")
    )
    execution = {
        "schema_version": PROSPECTIVE_EXECUTION_SCHEMA_V2,
        "redesign_run_contract_schema_version": contract["schema_version"],
        "redesign_run_contract_digest": canonical_payload_digest(contract),
        "latent_integrity_policy_id": policy["policy_id"],
        "latent_integrity_policy_digest": canonical_payload_digest(policy),
    }

    adapter = resolve_redesign_contract_adapter(
        execution,
        run_contract_payload=contract,
        latent_integrity_policy_payload=policy,
    )

    assert adapter.integrity_version == "v2"
    assert adapter.run_contract_payload() == contract
    assert adapter.latent_integrity_policy_id == policy["policy_id"]
    assert adapter.latent_integrity_policy_digest == canonical_payload_digest(
        policy
    )


def test_subordinate_v3_payload_requires_exact_schema_and_version_bindings():
    """Every prospective subordinate payload carries all four exact identities."""
    adapter = prospective_redesign_contract_adapter()
    payload = {
        "schema_version": "mrtotalvi-convergence-fit-v3",
        **version_binding_fields(adapter),
        "candidate_id": "B1",
    }

    assert validate_version_binding(
        payload,
        adapter,
        expected_schema="mrtotalvi-convergence-fit-v3",
    ) == payload

    for field in version_binding_fields(adapter):
        missing = dict(payload)
        del missing[field]
        with pytest.raises(ValueError, match=field):
            validate_version_binding(
                missing,
                adapter,
                expected_schema="mrtotalvi-convergence-fit-v3",
            )

        wrong = dict(payload)
        wrong[field] = "wrong"
        with pytest.raises(ValueError, match=field):
            validate_version_binding(
                wrong,
                adapter,
                expected_schema="mrtotalvi-convergence-fit-v3",
            )

    for schema in (None, "mrtotalvi-convergence-fit-v2"):
        changed = dict(payload)
        if schema is None:
            del changed["schema_version"]
        else:
            changed["schema_version"] = schema
        with pytest.raises(ValueError, match="schema_version"):
            validate_version_binding(
                changed,
                adapter,
                expected_schema="mrtotalvi-convergence-fit-v3",
            )


def test_version_binding_rejects_cross_version_substitution():
    """A valid payload for one integrity version cannot bind to the other."""
    prospective = prospective_redesign_contract_adapter()
    historical = historical_redesign_contract_adapter(_CURRENT_DIGEST)
    v3_payload = {
        "schema_version": "mrtotalvi-convergence-fit-v3",
        **version_binding_fields(prospective),
    }
    v1_payload = {
        "schema_version": "mrtotalvi-convergence-fit-v2",
        "redesign_run_contract_digest": historical.run_contract_digest,
    }

    with pytest.raises(ValueError, match="Historical v1"):
        validate_version_binding(
            v3_payload,
            historical,
            expected_schema="mrtotalvi-convergence-fit-v3",
        )
    with pytest.raises(ValueError):
        validate_version_binding(
            v1_payload,
            prospective,
            expected_schema="mrtotalvi-convergence-fit-v2",
        )


def test_prospective_partial_aggregate_separates_rank_alerts_from_failures():
    """Probe evidence preserves alert-only rank findings without failing them."""
    adapter = prospective_redesign_contract_adapter()
    request = DiagnosisExecutionRequest(
        fixtures=("mixed",),
        rows=("B1", "D0"),
        seeds=(0,),
        authoritative_full_grid=False,
    )
    results = [
        {
            "fixture_id": "mixed",
            "candidate_id": "B1",
            "training_seed": 0,
            "convergence": {"status": "converged"},
            "latent_integrity": {
                "terminal_integrity_failed": False,
                "effective_rank_screen_flags": ["factual_z"],
            },
        },
        {
            "fixture_id": "mixed",
            "candidate_id": "D0",
            "training_seed": 0,
            "convergence": {"status": "converged"},
            "latent_integrity": {
                "terminal_integrity_failed": True,
                "effective_rank_screen_flags": [],
            },
        },
    ]

    aggregate = _partial_aggregate(
        results,
        request=request,
        contract_adapter=adapter,
    )

    assert aggregate["terminal_integrity_failed"] is True
    assert aggregate["effective_rank_screen_flags"] == [
        {
            "fixture_id": "mixed",
            "candidate_id": "B1",
            "training_seed": 0,
            "representations": ["factual_z"],
        }
    ]
    assert aggregate["fit_failures"] == [
        {
            "fixture_id": "mixed",
            "candidate_id": "D0",
            "training_seed": 0,
            "convergence_status": "converged",
            "terminal_integrity_failed": True,
        }
    ]
    assert aggregate["all_fits_converged_with_terminal_integrity"] is False


def test_digest_only_v1_binding_does_not_call_live_contract_factories(
    monkeypatch,
):
    """Historical replay is selected only by its immutable registered digest."""

    def unavailable():
        raise AssertionError("live prospective factory was called")

    for factory in (
        "redesign_run_contract_v2",
        "redesign_run_contract_digest_v2",
        "latent_integrity_policy_v2",
        "latent_integrity_policy_digest_v2",
    ):
        monkeypatch.setattr(
            f"benchmarks.mrtotalvi.versioning.{factory}",
            unavailable,
        )
    adapter = historical_redesign_contract_adapter(_CURRENT_DIGEST)
    payload = {
        "schema_version": "mrtotalvi-convergence-fit-v2",
        "redesign_run_contract_digest": _CURRENT_DIGEST,
    }

    assert validate_version_binding(
        payload,
        adapter,
        expected_schema="mrtotalvi-convergence-fit-v2",
    ) == payload


def test_v1_execution_helpers_use_registered_sections_not_live_defaults(
    monkeypatch,
):
    """Digest-selected v1 control and grid replay from immutable registry data."""
    adapter = historical_redesign_contract_adapter(_CURRENT_DIGEST)

    def unavailable():
        raise AssertionError("historical execution consulted a live default")

    monkeypatch.setattr(
        convergence_execution,
        "redesign_run_contract",
        unavailable,
    )

    assert convergence_execution._convergence_control_payload(adapter) == {
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
    request = convergence_execution.validate_diagnosis_request(
        fixtures=("mixed",),
        rows=("B1",),
        seeds=(0,),
        contract_adapter=adapter,
    )
    assert request.authoritative_full_grid is False
    assert request.n_fits == 1


def test_sealed_v2_dispatch_does_not_call_live_contract_factories(
    monkeypatch,
):
    """Internal replay resolves canonical sealed bytes by registered digests."""
    adapter = prospective_redesign_contract_adapter()
    execution = {
        "schema_version": PROSPECTIVE_EXECUTION_SCHEMA_V2,
        **version_binding_fields(adapter),
    }
    contract = adapter.run_contract_payload()
    policy = adapter.latent_integrity_policy_payload()

    def unavailable():
        raise AssertionError("live prospective factory was called")

    for factory in (
        "redesign_run_contract_v2",
        "redesign_run_contract_digest_v2",
        "latent_integrity_policy_v2",
        "latent_integrity_policy_digest_v2",
    ):
        monkeypatch.setattr(
            f"benchmarks.mrtotalvi.versioning.{factory}",
            unavailable,
        )

    resolved = resolve_redesign_contract_adapter(
        execution,
        run_contract_payload=contract,
        latent_integrity_policy_payload=policy,
    )

    assert resolved.run_contract_payload() == contract
    assert resolved.latent_integrity_policy_payload() == policy
    assert version_binding_fields(resolved) == version_binding_fields(adapter)


@pytest.mark.parametrize(
    ("execution_mutation", "contract_mutation", "policy_payload"),
    [
        ({"redesign_run_contract_digest": None}, {}, None),
        ({"schema_version": "unknown"}, {}, None),
        ({"redesign_run_contract_digest": "0" * 64}, {}, None),
        ({}, {"schema_version": "mrtotalvi-redesign-run-contract-v2"}, None),
        ({}, {"human": {"selected_genes": 4_999}}, None),
        ({"latent_integrity_policy_id": "v2"}, {}, None),
        ({"latent_integrity_policy_digest": "1" * 64}, {}, None),
        ({"redesign_run_contract_schema_version": "v1"}, {}, None),
        ({}, {}, {"schema_version": "mrtotalvi-latent-integrity-policy-v2"}),
    ],
)
def test_v1_dispatch_rejects_missing_unknown_tampered_or_cross_version_metadata(
    execution_mutation,
    contract_mutation,
    policy_payload,
):
    """Dispatch fails closed without inferring a version from payload shape."""
    _, contract = _historical_payloads()
    contract_mutation = deepcopy(contract_mutation)
    if "human" in contract_mutation:
        contract["human"].update(contract_mutation.pop("human"))
    contract.update(contract_mutation)
    execution = {
        "schema_version": LEGACY_EXECUTION_SCHEMA_V1,
        "redesign_run_contract_digest": _CURRENT_DIGEST,
    }
    execution.update(execution_mutation)

    with pytest.raises(ValueError):
        resolve_redesign_contract_adapter(
            execution,
            run_contract_payload=contract,
            latent_integrity_policy_payload=policy_payload,
        )
