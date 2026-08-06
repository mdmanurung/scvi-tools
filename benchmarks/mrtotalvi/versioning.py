"""Fail-closed version dispatch for immutable RDX-03 evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from .latent_integrity import (
    latent_integrity_policy_digest_v2,
    latent_integrity_policy_v2,
)
from .redesign_contract import (
    redesign_run_contract_digest_v2,
    redesign_run_contract_v2,
)

LEGACY_EXECUTION_SCHEMA_V1 = "mrtotalvi-convergence-execution-v2"
PROSPECTIVE_EXECUTION_SCHEMA_V2 = "mrtotalvi-convergence-execution-v3"
PROSPECTIVE_RUN_CONTRACT_SCHEMA_V2 = (
    "mrtotalvi-redesign-run-contract-v2"
)
PROSPECTIVE_RUN_CONTRACT_DIGEST_V2 = (
    "4d75b395cb1faf644be09405542b1b1973093e553498754ff61af868696d36af"
)
PROSPECTIVE_LATENT_INTEGRITY_POLICY_ID_V2 = (
    "mrtotalvi-rdx03-latent-integrity-v2"
)
PROSPECTIVE_LATENT_INTEGRITY_POLICY_DIGEST_V2 = (
    "d8e8513768514aa598f7881b3ff6012b06b62b467af3214d2c6c44eb162af2a7"
)
REDESIGN_RUN_CONTRACT_V1_DIGESTS = (
    "7cccca9b0b1863a9c345ba570bd39100193a1515ca23462375d46511cdc7402c",
    "fe650e8c6275568a0c1a2174a9078d0990e6b4c2400443e6828f7544d8e8c26b",
)
_REGISTERED_V1_CONVERGENCE = {
    "candidate_specific_retuning": False,
    "check_every_epochs": 5,
    "maximum_epochs": 400,
    "minimum_epochs": 50,
    "patience_checks": 30,
    "restore_best_checkpoint": True,
}
_REGISTERED_V1_DIAGNOSIS_BY_DIGEST = {
    REDESIGN_RUN_CONTRACT_V1_DIGESTS[1]: {
        "canonical_human_requires_lineage_gate": True,
        "fixtures": [
            "mixed",
            "unequal_cells",
            "sealed_500",
            "canonical_human_if_available",
        ],
        "rows": ["B1", "B2", "B3", "D0"],
        "training_seeds": [0, 1, 2],
    }
}
_REGISTERED_V1_RUNTIME = {
    "accelerator_policy": "cpu_unless_separately_authorized",
}
_REDESIGN_METRIC_IDS_V1 = (
    "validation_objective_history",
    "best_checkpoint_identity",
    "rna_reconstruction_loss",
    "protein_reconstruction_loss",
    "kl_z",
    "kl_u",
    "u_posterior_scale",
    "factual_z_posterior_scale",
    "u_latent_variance",
    "factual_z_latent_variance",
    "u_effective_rank",
    "factual_z_effective_rank",
    "registered_residual_magnitude",
    "registered_residual_gradient_norm",
    "registered_residual_gradient_coverage",
    "trainable_parameter_count",
    "wall_time_seconds",
    "peak_memory_bytes",
    "u_linear_cka",
    "u_orthogonal_procrustes_disparity",
    "factual_z_linear_cka",
    "factual_z_orthogonal_procrustes_disparity",
    "u_cross_seed_knn_jaccard_k15",
    "factual_z_cross_seed_knn_jaccard_k15",
    "u_state_balanced_accuracy",
    "u_knn_state_accuracy_k15",
    "factual_z_state_balanced_accuracy",
    "factual_z_knn_state_accuracy_k15",
    "u_within_state_sample_predictability",
    "u_within_state_sample_predictability_permutation_p95",
    "u_technical_batch_mixing",
    "factual_z_technical_batch_mixing",
    "rna_heldout_negative_log_likelihood",
    "protein_heldout_negative_log_likelihood",
    "multimodal_heldout_predictive_loss",
    "rna_posterior_predictive_calibration",
    "protein_posterior_predictive_calibration",
    "centering_max_abs",
    "latent_all_finite",
    "milo_primary_failed_fit_count",
    "milo_primary_na_fit_count",
    "milo_fdp_spatialfdr_0_10",
    "milo_power_spatialfdr_0_10",
    "milo_localization_spatialfdr_0_10",
    "milo_seed_stability",
)


def canonical_payload_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize one mapping using the frozen digest representation."""
    if not isinstance(payload, Mapping):
        raise ValueError("Versioned payload must be a mapping.")
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Versioned payload is not canonical JSON.") from error


def canonical_payload_digest(payload: Mapping[str, object]) -> str:
    """Return the canonical SHA-256 digest for one versioned payload."""
    return hashlib.sha256(canonical_payload_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class RedesignContractAdapter:
    """An immutable, metadata-selected contract and assessment binding."""

    integrity_version: str
    execution_schema_version: str
    run_contract_schema_version: str
    run_contract_digest: str
    assessment_entry_point: str
    latent_integrity_policy_id: str | None
    latent_integrity_policy_digest: str | None
    _run_contract_payload_bytes: bytes | None
    _latent_integrity_policy_payload_bytes: bytes | None
    _metric_ids: tuple[str, ...]

    def run_contract_payload(self) -> dict[str, object]:
        """Return a detached replay of the exact registered payload."""
        if self._run_contract_payload_bytes is None:
            raise ValueError(
                "This historical execution stores only a contract digest."
            )
        return json.loads(self._run_contract_payload_bytes)

    def latent_integrity_policy_payload(self) -> dict[str, object]:
        """Return a detached replay of the selected policy payload."""
        if self._latent_integrity_policy_payload_bytes is None:
            raise ValueError("Historical v1 has no latent-integrity policy.")
        return json.loads(self._latent_integrity_policy_payload_bytes)

    def run_contract_section(self, name: str) -> object:
        """Return one detached section from full bytes or the v1 registry."""
        if not isinstance(name, str) or not name:
            raise ValueError("Run-contract section name must be non-empty.")
        if self._run_contract_payload_bytes is not None:
            payload = json.loads(self._run_contract_payload_bytes)
            if name not in payload:
                raise ValueError(
                    f"Selected run contract has no {name!r} section."
                )
            return json.loads(canonical_payload_bytes({"value": payload[name]}))[
                "value"
            ]
        if self.integrity_version != "v1":
            raise ValueError("Selected contract has no registered sections.")
        if name == "convergence":
            section = _REGISTERED_V1_CONVERGENCE
        elif name == "diagnosis":
            section = _REGISTERED_V1_DIAGNOSIS_BY_DIGEST.get(
                self.run_contract_digest
            )
            if section is None:
                raise ValueError(
                    "Registered historical contract predates the diagnosis grid."
                )
        elif name == "runtime":
            section = _REGISTERED_V1_RUNTIME
        else:
            raise ValueError(
                f"Historical registry has no {name!r} section."
            )
        return json.loads(canonical_payload_bytes({"value": section}))["value"]

    @property
    def metric_ids(self) -> tuple[str, ...]:
        """Return the exact metric order from the selected contract."""
        return self._metric_ids


def _registered_v1_digest_adapter(digest: object) -> RedesignContractAdapter:
    if digest not in REDESIGN_RUN_CONTRACT_V1_DIGESTS:
        raise ValueError(
            "Run-contract payload digest is not a registered historical v1 "
            "variant."
        )
    return RedesignContractAdapter(
        integrity_version="v1",
        execution_schema_version=LEGACY_EXECUTION_SCHEMA_V1,
        run_contract_schema_version="mrtotalvi-redesign-run-contract-v1",
        run_contract_digest=digest,
        assessment_entry_point="assess_latent_collapse",
        latent_integrity_policy_id=None,
        latent_integrity_policy_digest=None,
        _run_contract_payload_bytes=None,
        _latent_integrity_policy_payload_bytes=None,
        _metric_ids=_REDESIGN_METRIC_IDS_V1,
    )


def registered_redesign_run_contract_v1(
    payload: Mapping[str, object],
) -> RedesignContractAdapter:
    """Replay either exact historical v1 payload without upgrading it."""
    if not isinstance(payload, Mapping):
        raise ValueError("Redesign run contract must be a mapping.")
    detached = json.loads(canonical_payload_bytes(payload))
    if (
        detached.get("schema_version")
        != "mrtotalvi-redesign-run-contract-v1"
    ):
        raise ValueError("Run-contract schema is not registered as v1.")
    digest = canonical_payload_digest(detached)
    adapter = _registered_v1_digest_adapter(digest)
    return RedesignContractAdapter(
        **{
            **adapter.__dict__,
            "_run_contract_payload_bytes": canonical_payload_bytes(detached),
            "_metric_ids": tuple(detached["metric_ids"]),
        }
    )


def historical_redesign_contract_adapter(
    digest: str = REDESIGN_RUN_CONTRACT_V1_DIGESTS[1],
) -> RedesignContractAdapter:
    """Return one exact registered historical adapter by stored digest."""
    return _registered_v1_digest_adapter(digest)


def prospective_redesign_contract_adapter() -> RedesignContractAdapter:
    """Return the complete canonical prospective contract/policy adapter."""
    contract = redesign_run_contract_v2().to_dict()
    policy = latent_integrity_policy_v2()
    execution = {
        "schema_version": PROSPECTIVE_EXECUTION_SCHEMA_V2,
        "redesign_run_contract_schema_version": contract["schema_version"],
        "redesign_run_contract_digest": redesign_run_contract_digest_v2(),
        "latent_integrity_policy_id": policy["policy_id"],
        "latent_integrity_policy_digest": (
            latent_integrity_policy_digest_v2()
        ),
    }
    return resolve_redesign_contract_adapter(
        execution,
        run_contract_payload=contract,
        latent_integrity_policy_payload=policy,
    )


def version_binding_fields(
    adapter: RedesignContractAdapter,
) -> dict[str, str]:
    """Return the four mandatory prospective subordinate bindings."""
    if (
        adapter.integrity_version != "v2"
        or adapter.latent_integrity_policy_id is None
        or adapter.latent_integrity_policy_digest is None
    ):
        raise ValueError("Only prospective v2 has four-field bindings.")
    return {
        "redesign_run_contract_schema_version": (
            adapter.run_contract_schema_version
        ),
        "redesign_run_contract_digest": adapter.run_contract_digest,
        "latent_integrity_policy_id": adapter.latent_integrity_policy_id,
        "latent_integrity_policy_digest": (
            adapter.latent_integrity_policy_digest
        ),
    }


def validate_version_binding(
    payload: Mapping[str, object],
    adapter: RedesignContractAdapter,
    *,
    expected_schema: str,
) -> dict[str, object]:
    """Validate one subordinate payload against its selected version."""
    if not isinstance(payload, Mapping):
        raise ValueError("Version-bound payload must be a mapping.")
    if not isinstance(expected_schema, str) or not expected_schema:
        raise ValueError("expected_schema must be a non-empty string.")
    if payload.get("schema_version") != expected_schema:
        raise ValueError(
            "Version-bound payload schema_version does not match "
            f"{expected_schema!r}."
        )
    if not isinstance(adapter, RedesignContractAdapter):
        raise ValueError("Version binding requires a contract adapter.")

    if adapter.integrity_version == "v1":
        forbidden = (
            "redesign_run_contract_schema_version",
            "latent_integrity_policy_id",
            "latent_integrity_policy_digest",
        )
        if any(field in payload for field in forbidden):
            raise ValueError(
                "Historical v1 payload cannot carry prospective version "
                "bindings."
            )
        if (
            payload.get("redesign_run_contract_digest")
            != adapter.run_contract_digest
        ):
            raise ValueError(
                "Historical v1 redesign_run_contract_digest drifted."
            )
        return dict(payload)

    if adapter.integrity_version != "v2":
        raise ValueError(
            f"Unsupported integrity version {adapter.integrity_version!r}."
        )
    for field, expected in version_binding_fields(adapter).items():
        if payload.get(field) != expected:
            raise ValueError(f"Prospective payload {field} drifted.")
    return dict(payload)


def resolve_redesign_contract_adapter(
    execution_payload: Mapping[str, object],
    *,
    run_contract_payload: Mapping[str, object] | None = None,
    latent_integrity_policy_payload: Mapping[str, object] | None = None,
) -> RedesignContractAdapter:
    """Resolve only declared schema, digest, and policy metadata combinations."""
    if not isinstance(execution_payload, Mapping):
        raise ValueError("Execution payload must be a mapping.")
    schema = execution_payload.get("schema_version")
    if schema == LEGACY_EXECUTION_SCHEMA_V1:
        if latent_integrity_policy_payload is not None or any(
            key in execution_payload
            for key in (
                "redesign_run_contract_schema_version",
                "latent_integrity_policy_id",
                "latent_integrity_policy_digest",
            )
        ):
            raise ValueError(
                "The legacy execution schema cannot carry policy metadata."
            )

        declared_digest = execution_payload.get(
            "redesign_run_contract_digest"
        )
        adapter = _registered_v1_digest_adapter(declared_digest)
        if run_contract_payload is None:
            return adapter
        supplied = registered_redesign_run_contract_v1(
            run_contract_payload
        )
        if declared_digest != supplied.run_contract_digest:
            raise ValueError(
                "Execution run-contract digest does not match its exact "
                "payload."
            )
        return supplied

    if schema != PROSPECTIVE_EXECUTION_SCHEMA_V2:
        raise ValueError(f"Unsupported execution schema {schema!r}.")
    if run_contract_payload is None or latent_integrity_policy_payload is None:
        raise ValueError(
            "Prospective execution requires complete contract and policy "
            "payloads."
        )
    contract = json.loads(canonical_payload_bytes(run_contract_payload))
    policy = json.loads(
        canonical_payload_bytes(latent_integrity_policy_payload)
    )
    if (
        contract.get("schema_version")
        != PROSPECTIVE_RUN_CONTRACT_SCHEMA_V2
        or canonical_payload_digest(contract)
        != PROSPECTIVE_RUN_CONTRACT_DIGEST_V2
    ):
        raise ValueError(
            "Prospective run contract does not match the canonical v2 "
            "payload."
        )
    if (
        policy.get("policy_id")
        != PROSPECTIVE_LATENT_INTEGRITY_POLICY_ID_V2
        or canonical_payload_digest(policy)
        != PROSPECTIVE_LATENT_INTEGRITY_POLICY_DIGEST_V2
    ):
        raise ValueError(
            "Prospective latent-integrity policy does not match the "
            "canonical v2 payload."
        )
    bindings = {
        "redesign_run_contract_schema_version": contract["schema_version"],
        "redesign_run_contract_digest": (
            PROSPECTIVE_RUN_CONTRACT_DIGEST_V2
        ),
        "latent_integrity_policy_id": policy["policy_id"],
        "latent_integrity_policy_digest": (
            PROSPECTIVE_LATENT_INTEGRITY_POLICY_DIGEST_V2
        ),
    }
    for name, expected in bindings.items():
        if execution_payload.get(name) != expected:
            raise ValueError(f"Prospective execution {name} drifted.")
    return RedesignContractAdapter(
        integrity_version="v2",
        execution_schema_version=PROSPECTIVE_EXECUTION_SCHEMA_V2,
        run_contract_schema_version=contract["schema_version"],
        run_contract_digest=bindings["redesign_run_contract_digest"],
        assessment_entry_point="assess_latent_integrity_v2",
        latent_integrity_policy_id=bindings[
            "latent_integrity_policy_id"
        ],
        latent_integrity_policy_digest=bindings[
            "latent_integrity_policy_digest"
        ],
        _run_contract_payload_bytes=canonical_payload_bytes(contract),
        _latent_integrity_policy_payload_bytes=canonical_payload_bytes(policy),
        _metric_ids=tuple(contract["metric_ids"]),
    )
