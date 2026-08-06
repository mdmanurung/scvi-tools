"""Strict schema and applicability checks for redesign metric payloads."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .redesign_contract import redesign_candidate_configs
from .versioning import RedesignContractAdapter, version_binding_fields

if TYPE_CHECKING:
    from collections.abc import Mapping

_DEFINITION_FIELDS = {
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
_DIRECTIONS = {"higher", "lower", "descriptive"}
_VALUE_TYPES = {"scalar", "series", "object"}
_MODEL_FAMILIES = {"scvi", "totalvi", "mrtotalvi"}
_REPRESENTATIONS = {
    "optimization",
    "u",
    "factual_z",
    "prediction",
    "runtime",
    "milo",
}
_SCVI_FORBIDDEN = {
    "protein_reconstruction_loss",
    "protein_heldout_negative_log_likelihood",
    "multimodal_heldout_predictive_loss",
    "protein_posterior_predictive_calibration",
}
_U_ONLY_COMMON = {
    "kl_u",
    "registered_residual_magnitude",
    "registered_residual_gradient_norm",
    "registered_residual_gradient_coverage",
    "centering_max_abs",
}
_VERSION_BINDING_FIELDS = {
    "redesign_run_contract_schema_version",
    "redesign_run_contract_digest",
    "latent_integrity_policy_id",
    "latent_integrity_policy_digest",
}
_V2_DICTIONARY_SCHEMA = "mrtotalvi-redesign-metric-dictionary-v2"
_V3_DICTIONARY_SCHEMA = "mrtotalvi-redesign-metric-dictionary-v3"
_V3_DEFINITION_OVERRIDES = {
    "u_effective_rank": {
        "missing_or_no_call": (
            "null for stock models; missing or nonfinite rank is an explicit "
            "reason-coded no-call and does not replace the u integrity "
            "gates; rank below half the configured dimension is alert only"
        ),
        "selection_use": "alert_only",
    },
    "factual_z_effective_rank": {
        "missing_or_no_call": (
            "missing or nonfinite rank is an explicit reason-coded no-call "
            "and does not replace the factual-z integrity gates; rank below "
            "half the configured dimension is alert only"
        ),
        "selection_use": "alert_only",
    },
    "latent_all_finite": {
        "missing_or_no_call": (
            "legacy combined finite indicator retained for replay "
            "comparability; the six representation-specific integrity "
            "indicators are authoritative in v3"
        ),
        "selection_use": "legacy_diagnostic_not_authoritative",
    },
    "registered_residual_gradient_coverage": {
        "estimand": (
            "fraction of registered factual-z residual embedding rows with "
            "a finite nonzero accumulated gradient"
        ),
        "missing_or_no_call": (
            "null for stock models; any untrained row is a terminal "
            "factual-z integrity failure for MrTotalVI"
        ),
        "representation": "factual_z",
        "selection_use": "terminal_integrity_gate",
    },
    "factual_z_posterior_scale": {
        "estimand": (
            "model-specific factual-z posterior scale: analytic qz.scale "
            "for stock scVI/TotalVI and seeded Monte Carlo standard "
            "deviation for MrTotalVI"
        ),
        "aggregation": (
            "stock mean analytic qz.scale over cells and coordinates; "
            "MrTotalVI mean of 32 evaluation-seed posterior draws' sample "
            "standard deviation with ddof=1 over cells and coordinates"
        ),
    },
}
_V3_INTEGRITY_DEFINITIONS = {
    "u_representation_all_finite": {
        "estimand": "indicator that every evaluated u coordinate is finite",
        "direction": "higher",
        "unit": "binary indicator",
        "split": "all evaluation cells",
        "aggregation": "minimum across all u coordinates",
        "missing_or_no_call": "missing evidence is a terminal u integrity failure",
        "value_type": "scalar",
        "applicable_model_families": ["mrtotalvi"],
        "representation": "u",
        "selection_use": "terminal_integrity_gate",
    },
    "factual_z_representation_all_finite": {
        "estimand": "indicator that every evaluated factual-z coordinate is finite",
        "direction": "higher",
        "unit": "binary indicator",
        "split": "all evaluation cells",
        "aggregation": "minimum across all factual-z coordinates",
        "missing_or_no_call": (
            "missing evidence is a terminal factual-z integrity failure"
        ),
        "value_type": "scalar",
        "applicable_model_families": ["scvi", "totalvi", "mrtotalvi"],
        "representation": "factual_z",
        "selection_use": "terminal_integrity_gate",
    },
    "u_exact_nonconstant_variation": {
        "estimand": "exact indicator that evaluated u rows are not all identical",
        "direction": "higher",
        "unit": "binary indicator",
        "split": "all evaluation cells",
        "aggregation": "exact row equality with no epsilon threshold",
        "missing_or_no_call": "missing evidence is a terminal u integrity failure",
        "value_type": "scalar",
        "applicable_model_families": ["mrtotalvi"],
        "representation": "u",
        "selection_use": "terminal_integrity_gate",
    },
    "factual_z_exact_nonconstant_variation": {
        "estimand": (
            "exact indicator that evaluated factual-z rows are not all identical"
        ),
        "direction": "higher",
        "unit": "binary indicator",
        "split": "all evaluation cells",
        "aggregation": "exact row equality with no epsilon threshold",
        "missing_or_no_call": (
            "missing evidence is a terminal factual-z integrity failure"
        ),
        "value_type": "scalar",
        "applicable_model_families": ["scvi", "totalvi", "mrtotalvi"],
        "representation": "factual_z",
        "selection_use": "terminal_integrity_gate",
    },
    "u_posterior_scales_all_valid": {
        "estimand": (
            "indicator that every evaluated u posterior-scale element is "
            "present, finite, and strictly positive"
        ),
        "direction": "higher",
        "unit": "binary indicator",
        "split": "all evaluation cells",
        "aggregation": "minimum across all u posterior-scale elements",
        "missing_or_no_call": "missing evidence is a terminal u integrity failure",
        "value_type": "scalar",
        "applicable_model_families": ["mrtotalvi"],
        "representation": "u",
        "selection_use": "terminal_integrity_gate",
    },
    "factual_z_posterior_scales_all_valid": {
        "estimand": (
            "indicator that every evaluated factual-z posterior-scale element "
            "is present, finite, and strictly positive"
        ),
        "direction": "higher",
        "unit": "binary indicator",
        "split": "all evaluation cells",
        "aggregation": (
            "minimum across all factual-z posterior-scale elements"
        ),
        "missing_or_no_call": (
            "missing evidence is a terminal factual-z integrity failure"
        ),
        "value_type": "scalar",
        "applicable_model_families": ["scvi", "totalvi", "mrtotalvi"],
        "representation": "factual_z",
        "selection_use": "terminal_integrity_gate",
    },
}
MetricLifecycle = Literal[
    "per_fit",
    "comparison",
    "cross_seed",
    "milo",
    "complete",
]
_LIFECYCLES = {
    "per_fit",
    "comparison",
    "cross_seed",
    "milo",
    "complete",
}
_PER_FIT_COMMON = {
    "validation_objective_history",
    "best_checkpoint_identity",
    "rna_reconstruction_loss",
    "kl_z",
    "factual_z_posterior_scale",
    "factual_z_latent_variance",
    "factual_z_effective_rank",
    "trainable_parameter_count",
    "wall_time_seconds",
    "peak_memory_bytes",
    "factual_z_state_balanced_accuracy",
    "factual_z_knn_state_accuracy_k15",
    "factual_z_technical_batch_mixing",
    "rna_heldout_negative_log_likelihood",
    "rna_posterior_predictive_calibration",
    "latent_all_finite",
    "factual_z_representation_all_finite",
    "factual_z_exact_nonconstant_variation",
    "factual_z_posterior_scales_all_valid",
}
_PER_FIT_MULTIMODAL = {
    "protein_reconstruction_loss",
    "protein_heldout_negative_log_likelihood",
    "multimodal_heldout_predictive_loss",
    "protein_posterior_predictive_calibration",
}
_PER_FIT_MRTOTALVI = {
    "kl_u",
    "u_posterior_scale",
    "u_latent_variance",
    "u_effective_rank",
    "registered_residual_magnitude",
    "registered_residual_gradient_norm",
    "registered_residual_gradient_coverage",
    "u_state_balanced_accuracy",
    "u_knn_state_accuracy_k15",
    "u_within_state_sample_predictability",
    "u_within_state_sample_predictability_permutation_p95",
    "u_technical_batch_mixing",
    "u_representation_all_finite",
    "u_exact_nonconstant_variation",
    "u_posterior_scales_all_valid",
}
_COMPARISON_COMMON = {
    "factual_z_linear_cka",
    "factual_z_orthogonal_procrustes_disparity",
}
_COMPARISON_MRTOTALVI = {
    "u_linear_cka",
    "u_orthogonal_procrustes_disparity",
}
_CROSS_SEED_COMMON = {"factual_z_cross_seed_knn_jaccard_k15"}
_CROSS_SEED_MRTOTALVI = {"u_cross_seed_knn_jaccard_k15"}
_MILO = {
    "milo_primary_failed_fit_count",
    "milo_primary_na_fit_count",
    "milo_fdp_spatialfdr_0_10",
    "milo_power_spatialfdr_0_10",
    "milo_localization_spatialfdr_0_10",
    "milo_seed_stability",
}
_CHECKPOINT_FIELDS = {
    "monitor",
    "mode",
    "epoch",
    "value",
    "state_digest",
    "artifact_name",
}


def _require_contract_adapter(
    contract_adapter: RedesignContractAdapter,
) -> RedesignContractAdapter:
    if not isinstance(contract_adapter, RedesignContractAdapter):
        raise TypeError(
            "contract_adapter must be an explicit RedesignContractAdapter."
        )
    return contract_adapter


def _tracked_v2_metric_dictionary() -> dict:
    return json.loads(
        Path(__file__).with_name("metric_dictionary.json").read_text(
            encoding="utf-8"
        )
    )


def build_metric_dictionary_v3(
    *,
    contract_adapter: RedesignContractAdapter,
    base_payload: dict | None = None,
) -> dict:
    """Build the complete prospective dictionary from frozen v2 definitions."""
    adapter = _require_contract_adapter(contract_adapter)
    if adapter.integrity_version != "v2":
        raise ValueError("Metric dictionary v3 requires the prospective adapter.")
    base = deepcopy(
        _tracked_v2_metric_dictionary()
        if base_payload is None
        else base_payload
    )
    if (
        not isinstance(base, dict)
        or base.get("schema_version") != _V2_DICTIONARY_SCHEMA
        or not isinstance(base.get("metrics"), dict)
    ):
        raise ValueError("Metric dictionary v3 requires the frozen v2 base.")
    historical_ids = tuple(base["metrics"])
    if adapter.metric_ids[: len(historical_ids)] != historical_ids:
        raise ValueError(
            "Prospective metric IDs do not preserve the historical prefix."
        )
    new_ids = adapter.metric_ids[len(historical_ids) :]
    if new_ids != tuple(_V3_INTEGRITY_DEFINITIONS):
        raise ValueError("Prospective integrity metric order drifted.")
    base["schema_version"] = _V3_DICTIONARY_SCHEMA
    base.update(version_binding_fields(adapter))
    for metric_id, override in _V3_DEFINITION_OVERRIDES.items():
        base["metrics"][metric_id].update(override)
    base["metrics"].update(deepcopy(_V3_INTEGRITY_DEFINITIONS))
    return base


def load_metric_dictionary(
    path: str | Path | None = None,
    *,
    contract_adapter: RedesignContractAdapter,
) -> dict:
    """Load the selected historical or prospective metric dictionary."""
    adapter = _require_contract_adapter(contract_adapter)
    if path is None:
        if adapter.integrity_version == "v1":
            return _tracked_v2_metric_dictionary()
        if adapter.integrity_version == "v2":
            return build_metric_dictionary_v3(contract_adapter=adapter)
        raise ValueError(
            f"Unsupported metric contract version {adapter.integrity_version!r}."
        )
    dictionary_path = (
        Path(path)
    )
    return json.loads(dictionary_path.read_text(encoding="utf-8"))


def validate_metric_dictionary(
    payload: dict,
    *,
    contract_adapter: RedesignContractAdapter,
) -> dict:
    """Reject endpoint, definition, applicability, or top-level schema drift."""
    adapter = _require_contract_adapter(contract_adapter)
    if not isinstance(payload, dict):
        raise ValueError("Metric dictionary must be a mapping.")
    expected_top = {
        "schema_version",
        "scientific_scope",
        "selection_rule",
        "tie_break",
        "multimodal_ranking_rule",
        "metrics",
        "legacy_pilot_metrics",
    }
    if adapter.integrity_version == "v1":
        expected_schema = _V2_DICTIONARY_SCHEMA
    elif adapter.integrity_version == "v2":
        expected_schema = _V3_DICTIONARY_SCHEMA
        expected_top.update(_VERSION_BINDING_FIELDS)
    else:
        raise ValueError(
            f"Unsupported metric contract version {adapter.integrity_version!r}."
        )
    if set(payload) != expected_top:
        raise ValueError(
            "Metric dictionary top-level fields differ from the frozen schema."
        )
    if payload["schema_version"] != expected_schema:
        raise ValueError("Unknown metric dictionary schema version.")
    if adapter.integrity_version == "v2":
        for name, expected in version_binding_fields(adapter).items():
            if payload.get(name) != expected:
                raise ValueError(f"Metric dictionary {name} drifted.")
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("Metric definitions must be a mapping.")
    expected_ids = adapter.metric_ids
    if tuple(metrics) != expected_ids:
        raise ValueError("Metric dictionary metric IDs or order differ from the run contract.")
    u_only = {
        metric_id
        for metric_id in expected_ids
        if metric_id.startswith("u_")
    } | _U_ONLY_COMMON
    for metric_id, definition in metrics.items():
        if not isinstance(definition, dict) or set(definition) != _DEFINITION_FIELDS:
            raise ValueError(f"Metric {metric_id!r} has an invalid definition schema.")
        for field in (
            "estimand",
            "unit",
            "split",
            "aggregation",
            "missing_or_no_call",
            "representation",
            "selection_use",
        ):
            if not isinstance(definition[field], str) or not definition[field]:
                raise ValueError(f"Metric {metric_id!r} has invalid {field}.")
        if definition["direction"] not in _DIRECTIONS:
            raise ValueError(f"Metric {metric_id!r} has an invalid direction.")
        if definition["value_type"] not in _VALUE_TYPES:
            raise ValueError(f"Metric {metric_id!r} has an invalid value type.")
        families = definition["applicable_model_families"]
        if (
            not isinstance(families, list)
            or not families
            or len(families) != len(set(families))
            or not set(families) <= _MODEL_FAMILIES
        ):
            raise ValueError(
                f"Metric {metric_id!r} has invalid model-family applicability."
            )
        if definition["representation"] not in _REPRESENTATIONS:
            raise ValueError(f"Metric {metric_id!r} has an invalid representation.")
        if metric_id in _SCVI_FORBIDDEN and "scvi" in families:
            raise ValueError(
                f"RNA-only scVI cannot be applicable to {metric_id!r}."
            )
        if metric_id in u_only and families != ["mrtotalvi"]:
            raise ValueError(
                f"Metric {metric_id!r} must apply only to MrTotalVI."
            )
    legacy = payload["legacy_pilot_metrics"]
    if not isinstance(legacy, list) or len(legacy) != len(set(legacy)):
        raise ValueError("legacy_pilot_metrics must be a unique ordered list.")
    return payload


def metric_payload_template(
    *,
    contract_adapter: RedesignContractAdapter,
) -> dict[str, None]:
    """Return the selected endpoint keys with explicit no-calls."""
    adapter = _require_contract_adapter(contract_adapter)
    return dict.fromkeys(adapter.metric_ids)


def _required_metric_ids(
    *,
    candidate_id: str,
    lifecycle: MetricLifecycle,
    contract_adapter: RedesignContractAdapter,
) -> set[str]:
    candidate = redesign_candidate_configs()[candidate_id]
    per_fit = set(_PER_FIT_COMMON)
    comparison = set(_COMPARISON_COMMON)
    cross_seed = set(_CROSS_SEED_COMMON)
    if candidate.model_family in {"totalvi", "mrtotalvi"}:
        per_fit.update(_PER_FIT_MULTIMODAL)
    if candidate.model_family == "mrtotalvi":
        per_fit.update(_PER_FIT_MRTOTALVI)
        comparison.update(_COMPARISON_MRTOTALVI)
        cross_seed.update(_CROSS_SEED_MRTOTALVI)
        if candidate.hierarchy_mode == "centered_v2":
            per_fit.add("centering_max_abs")
    required = {
        "per_fit": per_fit,
        "comparison": comparison,
        "cross_seed": cross_seed,
        "milo": set(_MILO),
    }
    if lifecycle == "complete":
        selected = set().union(*required.values())
    else:
        selected = required[lifecycle]
    return selected & set(contract_adapter.metric_ids)


def _validate_history_series(value: object) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(
            "Metric 'validation_objective_history' must be a non-empty series."
        )
    epochs: list[float] = []
    for record in value:
        if not isinstance(record, dict) or set(record) != {"epoch", "value"}:
            raise ValueError(
                "Validation history records require exactly epoch and value."
            )
        epoch = record["epoch"]
        objective = record["value"]
        for name, numeric in (("epoch", epoch), ("value", objective)):
            if (
                isinstance(numeric, (bool, np.bool_))
                or not isinstance(
                    numeric,
                    (int, float, np.integer, np.floating),
                )
                or not np.isfinite(float(numeric))
            ):
                raise ValueError(
                    f"Validation history {name} values must be finite numeric values."
                )
        epochs.append(float(epoch))
    if any(
        next_epoch <= epoch
        for epoch, next_epoch in zip(epochs, epochs[1:], strict=False)
    ):
        raise ValueError("Validation history epochs must be strictly increasing.")


def _validate_checkpoint_object(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_FIELDS:
        raise ValueError(
            "Metric 'best_checkpoint_identity' requires exactly monitor, mode, "
            "epoch, value, state_digest, and artifact_name."
        )
    if value["monitor"] != "elbo_validation" or value["mode"] != "min":
        raise ValueError(
            "Best checkpoint must minimize the elbo_validation monitor."
        )
    for name in ("epoch", "value"):
        numeric = value[name]
        if (
            isinstance(numeric, (bool, np.bool_))
            or not isinstance(
                numeric,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(float(numeric))
        ):
            raise ValueError(f"Checkpoint {name} must be finite and numeric.")
    digest = value["state_digest"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Checkpoint state_digest must be a SHA-256 digest.")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(
            "Checkpoint state_digest must be hexadecimal."
        ) from error
    artifact_name = value["artifact_name"]
    if (
        not isinstance(artifact_name, str)
        or not artifact_name
        or Path(artifact_name).name != artifact_name
    ):
        raise ValueError("Checkpoint artifact_name must be one basename.")


def _validate_checkpoint_history_binding(payload: dict) -> None:
    history = payload["validation_objective_history"]
    checkpoint = payload["best_checkpoint_identity"]
    if history is None or checkpoint is None:
        return
    best = min(history, key=lambda record: float(record["value"]))
    if checkpoint["epoch"] != best["epoch"]:
        raise ValueError(
            "Best checkpoint epoch does not match the minimum validation record."
        )
    if float(checkpoint["value"]) != float(best["value"]):
        raise ValueError(
            "Best checkpoint value does not match the minimum validation record."
        )


def validate_metric_payload(
    payload: dict,
    *,
    contract_adapter: RedesignContractAdapter,
    candidate_id: str,
    lifecycle: MetricLifecycle,
    metric_dictionary_payload: dict | None = None,
    required_no_call_reasons: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """Require exact keys, lifecycle evidence, and explicit applicability no-calls."""
    adapter = _require_contract_adapter(contract_adapter)
    if not isinstance(payload, dict):
        raise ValueError("Metric payload must be a mapping.")
    expected_ids = adapter.metric_ids
    if tuple(payload) != expected_ids:
        raise ValueError("Metric payload metric IDs or order differ from the run contract.")
    configs = redesign_candidate_configs()
    if candidate_id not in configs:
        raise ValueError(f"Unknown redesign row {candidate_id!r}.")
    if lifecycle not in _LIFECYCLES:
        raise ValueError(f"Unknown metric lifecycle {lifecycle!r}.")
    family = configs[candidate_id].model_family
    dictionary = validate_metric_dictionary(
        (
            load_metric_dictionary(contract_adapter=adapter)
            if metric_dictionary_payload is None
            else metric_dictionary_payload
        ),
        contract_adapter=adapter,
    )
    required = _required_metric_ids(
        candidate_id=candidate_id,
        lifecycle=lifecycle,
        contract_adapter=adapter,
    )
    no_call_reasons = (
        {}
        if required_no_call_reasons is None
        else dict(required_no_call_reasons)
    )
    if not set(no_call_reasons) <= required:
        raise ValueError(
            "Required no-call reasons may name only lifecycle-required metrics."
        )
    for metric_id, reason_codes in no_call_reasons.items():
        if (
            isinstance(reason_codes, (str, bytes))
            or not isinstance(reason_codes, Sequence)
            or not reason_codes
            or any(
                not isinstance(reason, str) or not reason
                for reason in reason_codes
            )
        ):
            raise ValueError(
                f"Required no-call reasons for {metric_id!r} must be "
                "a non-empty sequence of reason codes."
            )
    for metric_id, value in payload.items():
        definition = dictionary["metrics"][metric_id]
        if family not in definition["applicable_model_families"]:
            if value is not None:
                raise ValueError(
                    f"Metric {metric_id!r} is not applicable to {candidate_id}."
                )
            continue
        if value is None:
            if metric_id in required and metric_id not in no_call_reasons:
                raise ValueError(
                    f"Metric {metric_id!r} is required for lifecycle {lifecycle!r}."
                )
            continue
        if metric_id in no_call_reasons:
            raise ValueError(
                f"Metric {metric_id!r} has both a value and a no-call reason."
            )
        value_type = definition["value_type"]
        if value_type == "scalar":
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(float(value))
            ):
                raise ValueError(f"Metric {metric_id!r} must be a finite scalar.")
        elif value_type == "series":
            _validate_history_series(value)
        elif metric_id == "best_checkpoint_identity":
            _validate_checkpoint_object(value)
        elif not isinstance(value, dict) or not value:
            raise ValueError(f"Metric {metric_id!r} must be a non-empty object.")
    _validate_checkpoint_history_binding(payload)
    return payload
