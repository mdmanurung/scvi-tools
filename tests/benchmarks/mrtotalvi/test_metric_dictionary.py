"""Strict schema tests for the preregistered redesign metric dictionary."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from benchmarks.mrtotalvi import redesign_contract
from benchmarks.mrtotalvi.metric_schema import (
    load_metric_dictionary,
    metric_payload_template,
    validate_metric_dictionary,
    validate_metric_payload,
)
from benchmarks.mrtotalvi.redesign_contract import redesign_run_contract
from benchmarks.mrtotalvi.versioning import (
    canonical_payload_digest,
    historical_redesign_contract_adapter,
    prospective_redesign_contract_adapter,
    version_binding_fields,
)

METRIC_DICTIONARY = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "mrtotalvi"
    / "metric_dictionary.json"
)


def test_metric_schema_is_selected_by_explicit_versioned_contract_adapter():
    """Historical and prospective metric inventories cannot use a live default."""
    historical = historical_redesign_contract_adapter()
    prospective = prospective_redesign_contract_adapter()

    historical_payload = metric_payload_template(
        contract_adapter=historical
    )
    prospective_payload = metric_payload_template(
        contract_adapter=prospective
    )
    assert tuple(historical_payload) == historical.metric_ids
    assert tuple(prospective_payload) == prospective.metric_ids
    assert len(prospective_payload) == len(historical_payload) + 6

    dictionary = load_metric_dictionary(contract_adapter=prospective)
    assert dictionary["schema_version"] == (
        "mrtotalvi-redesign-metric-dictionary-v3"
    )
    assert {
        name: dictionary[name] for name in version_binding_fields(prospective)
    } == version_binding_fields(prospective)
    assert (
        validate_metric_dictionary(
            dictionary,
            contract_adapter=prospective,
        )
        == dictionary
    )


def _valid_b0_per_fit_payload(contract_adapter):
    payload = metric_payload_template(contract_adapter=contract_adapter)
    history = [
        {"epoch": 0, "value": 3.0},
        {"epoch": 1, "value": 2.0},
    ]
    payload.update(
        {
            "validation_objective_history": history,
            "best_checkpoint_identity": {
                "monitor": "elbo_validation",
                "mode": "min",
                "epoch": 1,
                "value": 2.0,
                "state_digest": "a" * 64,
                "artifact_name": "best-0001",
            },
            "rna_reconstruction_loss": 2.0,
            "kl_z": 1.0,
            "factual_z_posterior_scale": 0.5,
            "factual_z_latent_variance": 0.7,
            "factual_z_effective_rank": 2.0,
            "trainable_parameter_count": 100,
            "wall_time_seconds": 1.0,
            "peak_memory_bytes": 1000,
            "factual_z_state_balanced_accuracy": 0.5,
            "factual_z_knn_state_accuracy_k15": 0.5,
            "factual_z_technical_batch_mixing": 0.5,
            "rna_heldout_negative_log_likelihood": 1.0,
            "rna_posterior_predictive_calibration": 0.1,
            "latent_all_finite": 1.0,
        }
    )
    if contract_adapter.integrity_version == "v2":
        payload.update(
            {
                "factual_z_representation_all_finite": 1.0,
                "factual_z_exact_nonconstant_variation": 1.0,
                "factual_z_posterior_scales_all_valid": 1.0,
            }
        )
    return payload


def test_metric_schema_requires_an_explicit_adapter_and_no_live_default(
    monkeypatch,
):
    """Selected metric identities never fall back to the live v1 factory."""
    historical = historical_redesign_contract_adapter()
    prospective = prospective_redesign_contract_adapter()

    with pytest.raises(TypeError):
        load_metric_dictionary()
    with pytest.raises(TypeError):
        metric_payload_template()
    with pytest.raises(TypeError):
        validate_metric_dictionary({})

    def unexpected_live_contract_call():
        raise AssertionError("metric validation consulted the live contract")

    monkeypatch.setattr(
        redesign_contract,
        "redesign_run_contract",
        unexpected_live_contract_call,
    )
    for adapter in (historical, prospective):
        dictionary = load_metric_dictionary(contract_adapter=adapter)
        assert validate_metric_dictionary(
            dictionary,
            contract_adapter=adapter,
        ) == dictionary
        assert tuple(
            metric_payload_template(contract_adapter=adapter)
        ) == adapter.metric_ids


def test_v3_dictionary_freezes_six_integrity_indicators_and_per_fit_use():
    """The prospective inventory is complete, ordered, bound, and required."""
    historical = load_metric_dictionary(
        contract_adapter=historical_redesign_contract_adapter()
    )
    adapter = prospective_redesign_contract_adapter()
    dictionary = load_metric_dictionary(contract_adapter=adapter)
    expected_integrity_ids = (
        "u_representation_all_finite",
        "factual_z_representation_all_finite",
        "u_exact_nonconstant_variation",
        "factual_z_exact_nonconstant_variation",
        "u_posterior_scales_all_valid",
        "factual_z_posterior_scales_all_valid",
    )

    assert len(dictionary["metrics"]) == 51
    assert tuple(dictionary["metrics"])[-6:] == expected_integrity_ids
    assert dictionary["metrics"]["u_effective_rank"][
        "selection_use"
    ] == "alert_only"
    assert dictionary["metrics"]["factual_z_effective_rank"][
        "selection_use"
    ] == "alert_only"
    assert "alert only" in dictionary["metrics"]["u_effective_rank"][
        "missing_or_no_call"
    ]
    assert "alert only" in dictionary["metrics"][
        "factual_z_effective_rank"
    ]["missing_or_no_call"]
    assert historical["metrics"]["u_effective_rank"][
        "selection_use"
    ] == "hard_gate"
    assert historical["metrics"]["factual_z_effective_rank"][
        "selection_use"
    ] == "hard_gate"
    assert dictionary["metrics"]["latent_all_finite"]["selection_use"] == (
        "legacy_diagnostic_not_authoritative"
    )
    gradient = dictionary["metrics"][
        "registered_residual_gradient_coverage"
    ]
    assert gradient["representation"] == "factual_z"
    assert gradient["selection_use"] == "terminal_integrity_gate"
    scale = dictionary["metrics"]["factual_z_posterior_scale"]
    assert "analytic qz.scale" in scale["estimand"]
    assert "32" in scale["aggregation"]
    assert "ddof=1" in scale["aggregation"]
    assert "evaluation-seed" in scale["aggregation"]
    assert canonical_payload_digest(dictionary) == (
        "9866d667e424c6bc9bd74d27cd37033b85dd623c3c008149658ed84f1ac94216"
    )
    for metric_id in expected_integrity_ids:
        definition = dictionary["metrics"][metric_id]
        assert definition["selection_use"] == "terminal_integrity_gate"
        assert definition["value_type"] == "scalar"
        assert definition["direction"] == "higher"
        if metric_id.startswith("u_"):
            assert definition["applicable_model_families"] == ["mrtotalvi"]
        else:
            assert definition["applicable_model_families"] == [
                "scvi",
                "totalvi",
                "mrtotalvi",
            ]

    payload = _valid_b0_per_fit_payload(adapter)
    assert validate_metric_payload(
        payload,
        contract_adapter=adapter,
        candidate_id="B0",
        lifecycle="per_fit",
        metric_dictionary_payload=dictionary,
    ) == payload
    payload["factual_z_posterior_scales_all_valid"] = None
    with pytest.raises(ValueError, match="required"):
        validate_metric_payload(
            payload,
            contract_adapter=adapter,
            candidate_id="B0",
            lifecycle="per_fit",
            metric_dictionary_payload=dictionary,
        )


def test_metric_dictionary_binding_and_cross_version_substitution_fail_closed():
    historical = historical_redesign_contract_adapter()
    prospective = prospective_redesign_contract_adapter()
    historical_dictionary = load_metric_dictionary(
        contract_adapter=historical
    )
    prospective_dictionary = load_metric_dictionary(
        contract_adapter=prospective
    )

    with pytest.raises(ValueError):
        validate_metric_dictionary(
            prospective_dictionary,
            contract_adapter=historical,
        )
    with pytest.raises(ValueError):
        validate_metric_dictionary(
            historical_dictionary,
            contract_adapter=prospective,
        )

    for field in version_binding_fields(prospective):
        changed = deepcopy(prospective_dictionary)
        changed[field] = "0" * 64
        with pytest.raises(ValueError, match=field):
            validate_metric_dictionary(
                changed,
                contract_adapter=prospective,
            )


def test_metric_dictionary_freezes_every_redesign_endpoint_and_estimand_boundary():
    adapter = historical_redesign_contract_adapter()
    payload = load_metric_dictionary(
        METRIC_DICTIONARY,
        contract_adapter=adapter,
    )
    validated = validate_metric_dictionary(
        payload,
        contract_adapter=adapter,
    )
    contract = redesign_run_contract()

    assert validated["schema_version"] == "mrtotalvi-redesign-metric-dictionary-v2"
    assert tuple(validated["metrics"]) == contract.metric_ids
    assert validated["multimodal_ranking_rule"] == (
        "RNA-only scVI is excluded from protein, multimodal predictive-loss, "
        "and multimodal ELBO rankings."
    )
    required = {
        "estimand",
        "direction",
        "unit",
        "split",
        "aggregation",
        "missing_or_no_call",
        "value_type",
        "applicable_model_families",
        "representation",
        "selection_use",
    }
    for metric_id, definition in validated["metrics"].items():
        assert set(definition) == required, metric_id
        assert definition["direction"] in {
            "higher",
            "lower",
            "descriptive",
        }
        assert definition["value_type"] in {"scalar", "series", "object"}
        assert set(definition["applicable_model_families"]) <= {
            "scvi",
            "totalvi",
            "mrtotalvi",
        }

    assert validated["metrics"]["rna_heldout_negative_log_likelihood"][
        "unit"
    ] == "negative log probability per observed RNA entry"
    assert validated["metrics"]["protein_heldout_negative_log_likelihood"][
        "unit"
    ] == "negative log probability per observed protein entry"
    assert validated["metrics"]["multimodal_heldout_predictive_loss"][
        "applicable_model_families"
    ] == ["totalvi", "mrtotalvi"]
    assert validated["metrics"]["u_state_balanced_accuracy"][
        "applicable_model_families"
    ] == ["mrtotalvi"]
    assert validated["metrics"]["factual_z_state_balanced_accuracy"][
        "applicable_model_families"
    ] == ["scvi", "totalvi", "mrtotalvi"]


def test_metric_dictionary_rejects_drift_and_scvi_multimodal_applicability():
    adapter = historical_redesign_contract_adapter()
    payload = json.loads(METRIC_DICTIONARY.read_text(encoding="utf-8"))
    changed = json.loads(json.dumps(payload))
    changed["metrics"].pop("kl_u")
    with pytest.raises(ValueError, match="metric IDs"):
        validate_metric_dictionary(changed, contract_adapter=adapter)

    changed = json.loads(json.dumps(payload))
    changed["metrics"]["multimodal_heldout_predictive_loss"][
        "applicable_model_families"
    ].append("scvi")
    with pytest.raises(ValueError, match="RNA-only scVI"):
        validate_metric_dictionary(changed, contract_adapter=adapter)


def test_metric_payload_schema_requires_exact_per_fit_evidence_and_applicability():
    adapter = historical_redesign_contract_adapter()
    b0 = _valid_b0_per_fit_payload(adapter)
    assert validate_metric_payload(
        b0,
        contract_adapter=adapter,
        candidate_id="B0",
        lifecycle="per_fit",
    ) == b0

    changed = dict(b0)
    changed["protein_heldout_negative_log_likelihood"] = 2.0
    with pytest.raises(ValueError, match="not applicable"):
        validate_metric_payload(
            changed,
            contract_adapter=adapter,
            candidate_id="B0",
            lifecycle="per_fit",
        )

    changed = dict(b0)
    changed.pop("kl_u")
    with pytest.raises(ValueError, match="metric IDs"):
        validate_metric_payload(
            changed,
            contract_adapter=adapter,
            candidate_id="B0",
            lifecycle="per_fit",
        )

    with pytest.raises(ValueError, match="required"):
        validate_metric_payload(
            metric_payload_template(contract_adapter=adapter),
            contract_adapter=adapter,
            candidate_id="B0",
            lifecycle="per_fit",
        )

    changed = _valid_b0_per_fit_payload(adapter)
    changed["rna_reconstruction_loss"] = None
    with pytest.raises(ValueError, match="required"):
        validate_metric_payload(
            changed,
            contract_adapter=adapter,
            candidate_id="B0",
            lifecycle="per_fit",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epoch", 0, "epoch does not match"),
        ("value", 3.0, "value does not match"),
        ("state_digest", "not-a-digest", "SHA-256"),
    ],
)
def test_metric_payload_rejects_checkpoint_identity_drift(field, value, message):
    adapter = historical_redesign_contract_adapter()
    payload = _valid_b0_per_fit_payload(adapter)
    payload["best_checkpoint_identity"] = dict(
        payload["best_checkpoint_identity"]
    )
    payload["best_checkpoint_identity"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_metric_payload(
            payload,
            contract_adapter=adapter,
            candidate_id="B0",
            lifecycle="per_fit",
        )

    malformed = _valid_b0_per_fit_payload(adapter)
    malformed["best_checkpoint_identity"] = {"nonsense": 1}
    with pytest.raises(ValueError, match="requires exactly"):
        validate_metric_payload(
            malformed,
            contract_adapter=adapter,
            candidate_id="B0",
            lifecycle="per_fit",
        )


@pytest.mark.parametrize(
    ("lifecycle", "metric_ids"),
    [
        (
            "comparison",
            {
                "factual_z_linear_cka",
                "factual_z_orthogonal_procrustes_disparity",
            },
        ),
        ("cross_seed", {"factual_z_cross_seed_knn_jaccard_k15"}),
        (
            "milo",
            {
                "milo_primary_failed_fit_count",
                "milo_primary_na_fit_count",
                "milo_fdp_spatialfdr_0_10",
                "milo_power_spatialfdr_0_10",
                "milo_localization_spatialfdr_0_10",
                "milo_seed_stability",
            },
        ),
    ],
)
def test_metric_lifecycles_require_only_currently_computable_endpoints(
    lifecycle,
    metric_ids,
):
    adapter = historical_redesign_contract_adapter()
    payload = metric_payload_template(contract_adapter=adapter)
    for metric_id in metric_ids:
        payload[metric_id] = 0.5
    assert validate_metric_payload(
        payload,
        contract_adapter=adapter,
        candidate_id="B0",
        lifecycle=lifecycle,
    ) == payload
    payload[next(iter(metric_ids))] = None
    with pytest.raises(ValueError, match="required"):
        validate_metric_payload(
            payload,
            contract_adapter=adapter,
            candidate_id="B0",
            lifecycle=lifecycle,
        )
