"""Pure contracts for the RDX-03 convergence diagnosis."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from .diagnostics import (
    cross_seed_knn_metrics,
    linear_cka,
    orthogonal_procrustes_disparity,
)
from .metric_schema import (
    metric_payload_template,
    validate_metric_payload,
)
from .redesign_contract import redesign_candidate_configs, redesign_run_contract
from .versioning import (
    RedesignContractAdapter,
    historical_redesign_contract_adapter,
    validate_version_binding,
    version_binding_fields,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal


LATENT_INTEGRITY_ASSESSMENT_SCHEMA = (
    "mrtotalvi-latent-integrity-assessment-v3"
)


@dataclass(frozen=True)
class ConvergenceControl:
    """Candidate-invariant optimization controls for every diagnosis fit."""

    check_every_epochs: int
    minimum_epochs: int
    maximum_epochs: int
    patience_checks: int
    restore_best_checkpoint: bool
    candidate_specific_retuning: bool
    monitor: Literal["elbo_validation"]
    mode: Literal["min"]
    min_delta: float

    def to_dict(self) -> dict[str, int | bool | str | float]:
        """Return the exact serialized control."""
        return asdict(self)


def convergence_control() -> ConvergenceControl:
    """Return the frozen RDX-03 control with explicit monitor semantics."""
    frozen = redesign_run_contract().convergence
    return ConvergenceControl(
        check_every_epochs=frozen.check_every_epochs,
        minimum_epochs=frozen.minimum_epochs,
        maximum_epochs=frozen.maximum_epochs,
        patience_checks=frozen.patience_checks,
        restore_best_checkpoint=frozen.restore_best_checkpoint,
        candidate_specific_retuning=frozen.candidate_specific_retuning,
        monitor="elbo_validation",
        mode="min",
        min_delta=0.0,
    )


def _convergence_control_from_adapter(
    contract_adapter: RedesignContractAdapter | None,
) -> ConvergenceControl:
    """Resolve explicit adapters from sealed bytes, never a live default."""
    if contract_adapter is None:
        return convergence_control()
    if not isinstance(contract_adapter, RedesignContractAdapter):
        raise TypeError("contract_adapter must be a RedesignContractAdapter.")
    frozen = contract_adapter.run_contract_section("convergence")
    return ConvergenceControl(
        check_every_epochs=frozen["check_every_epochs"],
        minimum_epochs=frozen["minimum_epochs"],
        maximum_epochs=frozen["maximum_epochs"],
        patience_checks=frozen["patience_checks"],
        restore_best_checkpoint=frozen["restore_best_checkpoint"],
        candidate_specific_retuning=frozen[
            "candidate_specific_retuning"
        ],
        monitor="elbo_validation",
        mode="min",
        min_delta=0.0,
    )


def _diagnosis_grid_from_adapter(
    contract_adapter: RedesignContractAdapter | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    """Resolve an explicit historical or prospective diagnosis from sealed bytes."""
    if contract_adapter is None:
        diagnosis = redesign_run_contract().diagnosis
        return (
            diagnosis.fixtures,
            diagnosis.rows,
            diagnosis.training_seeds,
        )
    if not isinstance(contract_adapter, RedesignContractAdapter):
        raise TypeError("contract_adapter must be a RedesignContractAdapter.")
    diagnosis = contract_adapter.run_contract_section("diagnosis")
    if not isinstance(diagnosis, Mapping):
        raise ValueError("Selected run contract has no diagnosis grid.")
    return (
        tuple(diagnosis["fixtures"]),
        tuple(diagnosis["rows"]),
        tuple(diagnosis["training_seeds"]),
    )


@dataclass(frozen=True)
class DiagnosisFitSpec:
    """One exact B1-B3/D0 model identity with a shared control."""

    candidate_id: Literal["B1", "B2", "B3", "D0"]
    model_family: Literal["totalvi", "mrtotalvi"]
    legacy_candidate: Literal["C0", "C2", "C4"] | None
    control: ConvergenceControl


_LEGACY_CANDIDATES = {
    "B1": None,
    "B2": "C0",
    "B3": "C2",
    "D0": "C4",
}


def diagnosis_fit_spec(candidate_id: str) -> DiagnosisFitSpec:
    """Map only the preregistered diagnosis rows to implemented model identities."""
    if candidate_id not in _LEGACY_CANDIDATES:
        raise ValueError(
            f"{candidate_id!r} is not an RDX-03 diagnosis row."
        )
    candidate = redesign_candidate_configs()[candidate_id]
    return DiagnosisFitSpec(
        candidate_id=candidate_id,
        model_family=candidate.model_family,
        legacy_candidate=_LEGACY_CANDIDATES[candidate_id],
        control=convergence_control(),
    )


def _validated_history(
    validation_history: Sequence[Mapping],
) -> list[dict[str, float | int]]:
    if not isinstance(validation_history, list) or not validation_history:
        raise ValueError("Validation history must be a non-empty list.")
    records = []
    previous_epoch = -np.inf
    for record in validation_history:
        if not isinstance(record, dict) or set(record) != {"epoch", "value"}:
            raise ValueError(
                "Validation history records require exactly epoch and value."
            )
        epoch = record["epoch"]
        value = record["value"]
        if (
            isinstance(epoch, (bool, np.bool_))
            or not isinstance(epoch, (int, float, np.integer, np.floating))
            or not np.isfinite(float(epoch))
            or float(epoch) <= previous_epoch
        ):
            raise ValueError(
                "Validation-history epochs must be finite and strictly increasing."
            )
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(float(value))
        ):
            raise ValueError("Validation-history values must be finite.")
        previous_epoch = float(epoch)
        records.append({"epoch": epoch, "value": float(value)})
    return records


def assess_convergence(
    validation_history: list[dict],
    *,
    trainer_epochs: int,
    stopped_early: bool,
    control: ConvergenceControl | Mapping[str, object] | None = None,
) -> dict[str, int | bool | str]:
    """Classify a completed fit without extending the frozen epoch budget."""
    if control is None:
        selected_control = convergence_control()
    elif isinstance(control, ConvergenceControl):
        selected_control = control
    elif isinstance(control, Mapping):
        try:
            selected_control = ConvergenceControl(**dict(control))
        except TypeError as error:
            raise ValueError(
                "Serialized convergence control has invalid fields."
            ) from error
    else:
        raise TypeError("control must be a ConvergenceControl or mapping.")
    control = selected_control
    records = _validated_history(validation_history)
    if (
        isinstance(trainer_epochs, bool)
        or not isinstance(trainer_epochs, (int, np.integer))
        or trainer_epochs < control.minimum_epochs
    ):
        raise ValueError(
            "A completed diagnosis fit must reach the frozen minimum epochs."
        )
    if trainer_epochs > control.maximum_epochs:
        raise ValueError("trainer_epochs exceed the frozen maximum.")
    if float(records[-1]["epoch"]) > trainer_epochs:
        raise ValueError("Validation history extends beyond trainer_epochs.")
    if stopped_early:
        if trainer_epochs >= control.maximum_epochs:
            raise ValueError(
                "Early-stopping evidence is inconsistent with maximum epochs."
            )
        if len(records) < control.patience_checks + 1:
            raise ValueError(
                "Early-stopping evidence has fewer checks than frozen patience."
            )
    elif trainer_epochs < control.maximum_epochs:
        raise ValueError(
            "Non-early-stopped fit below maximum epochs is inconsistent."
        )
    reached_maximum = trainer_epochs == control.maximum_epochs
    plateau = bool(stopped_early and not reached_maximum)
    return {
        "status": (
            "converged" if plateau else "non_converged_at_maximum"
        ),
        "stable_validation_plateau": plateau,
        "stopped_early": bool(stopped_early),
        "trainer_epochs": int(trainer_epochs),
        "validation_checks": len(records),
        "reached_minimum_epochs": True,
        "reached_maximum_epochs": reached_maximum,
    }


def assess_latent_collapse(
    metrics: Mapping[str, object],
    *,
    candidate_id: str,
    latent_dimension: int,
) -> dict[str, object]:
    """Apply only preregistered rank/finite/gradient and structural-zero gates."""
    spec = diagnosis_fit_spec(candidate_id)
    if (
        isinstance(latent_dimension, bool)
        or not isinstance(latent_dimension, (int, np.integer))
        or latent_dimension < 1
    ):
        raise ValueError("latent_dimension must be a positive integer.")
    representations = (
        ("factual_z",)
        if spec.model_family == "totalvi"
        else ("u", "factual_z")
    )
    required_metrics: list[tuple[str, str, float]] = []
    for representation in representations:
        required_metrics.extend(
            [
                (
                    f"{representation}_effective_rank",
                    "minimum",
                    0.5 * latent_dimension,
                ),
                (
                    f"{representation}_latent_variance",
                    "positive",
                    0.0,
                ),
                (
                    f"{representation}_posterior_scale",
                    "positive",
                    0.0,
                ),
            ]
        )
    required_metrics.append(("latent_all_finite", "equal", 1.0))
    if spec.model_family == "mrtotalvi":
        required_metrics.append(
            (
                "registered_residual_gradient_coverage",
                "equal",
                1.0,
            )
        )

    failed: list[str] = []
    observed: dict[str, float] = {}
    for metric_id, comparison, threshold in required_metrics:
        value = metrics.get(metric_id)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(float(value))
        ):
            failed.append(metric_id)
            continue
        numeric = float(value)
        observed[metric_id] = numeric
        passes = {
            "minimum": numeric >= threshold,
            "positive": numeric > threshold,
            "equal": numeric == threshold,
        }[comparison]
        if not passes:
            failed.append(metric_id)
    return {
        "failed": bool(failed),
        "failed_metrics": failed,
        "required_representations": list(representations),
        "latent_dimension": int(latent_dimension),
        "observed": observed,
    }


def assess_latent_integrity_v2(
    metrics: Mapping[str, object],
    *,
    candidate_id: str,
    latent_dimension: int,
) -> dict[str, object]:
    """Record low effective rank as a prospective alert, never a failure."""
    if candidate_id not in _LEGACY_CANDIDATES:
        raise ValueError(
            f"{candidate_id!r} is not an RDX-03 diagnosis row."
        )
    model_family = "totalvi" if candidate_id == "B1" else "mrtotalvi"
    if (
        isinstance(latent_dimension, bool)
        or not isinstance(latent_dimension, (int, np.integer))
        or latent_dimension < 1
    ):
        raise ValueError("latent_dimension must be a positive integer.")
    representations = (
        ("factual_z",)
        if model_family == "totalvi"
        else ("u", "factual_z")
    )
    threshold = 0.5 * int(latent_dimension)
    assessed = {}
    alerts = []
    for representation in representations:
        terminal_reasons = []
        for indicator in (
            "representation_all_finite",
            "exact_nonconstant_variation",
            "posterior_scales_all_valid",
        ):
            indicator_id = f"{representation}_{indicator}"
            value = metrics.get(indicator_id)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(
                    value,
                    (int, float, np.integer, np.floating),
                )
                or not np.isfinite(float(value))
                or float(value) != 1.0
            ):
                terminal_reasons.append(indicator_id)
        if representation == "factual_z" and model_family == "mrtotalvi":
            coverage = metrics.get(
                "registered_residual_gradient_coverage"
            )
            if (
                isinstance(coverage, (bool, np.bool_))
                or not isinstance(
                    coverage,
                    (int, float, np.integer, np.floating),
                )
                or not np.isfinite(float(coverage))
                or float(coverage) != 1.0
            ):
                terminal_reasons.append(
                    "registered_residual_gradient_coverage"
                )
        metric_id = f"{representation}_effective_rank"
        value = metrics.get(metric_id)
        valid_rank = not (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(float(value))
        )
        effective_rank = float(value) if valid_rank else None
        low_rank_alert = bool(
            effective_rank is not None and effective_rank < threshold
        )
        if low_rank_alert:
            alerts.append(representation)
        terminal_failed = bool(terminal_reasons)
        assessed[representation] = {
            "terminal_failed": terminal_failed,
            "terminal_failure_reasons": terminal_reasons,
            "effective_rank": effective_rank,
            "effective_rank_threshold": threshold,
            "low_rank_alert": low_rank_alert,
            "eligible_for_geometry": not terminal_failed,
        }
    terminal_failed = any(
        evidence["terminal_failed"] for evidence in assessed.values()
    )
    return {
        "schema_version": LATENT_INTEGRITY_ASSESSMENT_SCHEMA,
        "terminal_integrity_failed": terminal_failed,
        "effective_rank_screen_flags": alerts,
        "required_representations": list(representations),
        "latent_dimension": int(latent_dimension),
        "representations": assessed,
    }


def validate_diagnosis_grid(
    results: list[dict],
    *,
    contract_adapter: RedesignContractAdapter | None = None,
) -> list[dict]:
    """Require the exact 4-row by 3-seed by 4-fixture diagnosis grid."""
    fixtures, rows, training_seeds = _diagnosis_grid_from_adapter(
        contract_adapter
    )
    if not isinstance(results, list):
        raise ValueError("Diagnosis results must be a list.")
    expected = {
        (candidate_id, fixture_id, seed)
        for fixture_id in fixtures
        for candidate_id in rows
        for seed in training_seeds
    }
    indexed: dict[tuple[str, str, int], dict] = {}
    frozen_control = _convergence_control_from_adapter(
        contract_adapter
    ).to_dict()
    for result in results:
        if (
            not isinstance(result, dict)
            or result.get("schema_version")
            != "mrtotalvi-convergence-fit-v2"
        ):
            raise ValueError("Unexpected diagnosis result schema.")
        key = (
            result.get("candidate_id"),
            result.get("fixture_id"),
            result.get("training_seed"),
        )
        if key in indexed:
            raise ValueError(f"Duplicate diagnosis result for {key}.")
        if key not in expected:
            raise ValueError(f"Unexpected diagnosis result for {key}.")
        if result.get("control") != frozen_control:
            raise ValueError(f"Diagnosis control drift for {key}.")
        convergence = result.get("convergence")
        collapse = result.get("collapse")
        if (
            not isinstance(convergence, dict)
            or convergence.get("status")
            not in {"converged", "non_converged_at_maximum"}
        ):
            raise ValueError(f"Invalid convergence evidence for {key}.")
        if not isinstance(collapse, dict) or not isinstance(
            collapse.get("failed"),
            bool,
        ):
            raise ValueError(f"Invalid collapse evidence for {key}.")
        indexed[key] = result
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"Missing diagnosis results: {missing}.")
    return [indexed[key] for key in sorted(indexed)]


def d0_downstream_gate_decision(
    results: list[dict],
    *,
    contract_adapter: RedesignContractAdapter | None = None,
) -> dict[str, object]:
    """Prevent RDX-03 optimization evidence from preclaiming a DA pass."""
    validated = validate_diagnosis_grid(
        results,
        contract_adapter=contract_adapter,
    )
    d0 = [result for result in validated if result["candidate_id"] == "D0"]
    failed_diagnosis = any(
        result["convergence"]["status"] != "converged"
        or result["collapse"]["failed"]
        for result in d0
    )
    reason_codes = (
        ["d0_convergence_or_collapse_failure"]
        if failed_diagnosis
        else ["milo_calibration_not_available_at_rdx03"]
    )
    return {
        "d0_passes_all_downstream_gates": False,
        "action": "continue_to_d1_d5",
        "reason_codes": reason_codes,
    }


def _aligned_representation_pair(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Align a candidate embedding to the exact reference cell order."""

    def checked(
        payload: Mapping[str, object],
        *,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(payload, Mapping) or set(payload) != {
            "cell_ids",
            "values",
        }:
            raise ValueError(
                f"{name} representation requires exactly cell_ids and values."
            )
        cells = np.asarray(payload["cell_ids"], dtype=str)
        values = np.asarray(payload["values"], dtype=np.float64)
        if (
            cells.ndim != 1
            or values.ndim != 2
            or len(cells) != len(values)
            or len(np.unique(cells)) != len(cells)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                f"{name} representation must be finite and cell aligned."
            )
        return cells, values

    reference_cells, reference_values = checked(
        reference,
        name="reference",
    )
    candidate_cells, candidate_values = checked(
        candidate,
        name="candidate",
    )
    if set(reference_cells) != set(candidate_cells):
        raise ValueError(
            "Every compared representation must contain the same exact "
            "cell-ID set."
        )
    positions = {
        cell_id: index
        for index, cell_id in enumerate(candidate_cells.tolist())
    }
    order = np.fromiter(
        (positions[cell_id] for cell_id in reference_cells.tolist()),
        dtype=np.int64,
        count=len(reference_cells),
    )
    return reference_values, candidate_values[order]


def _representation_failure_reasons(
    result: Mapping[str, object],
    *,
    representation_name: str,
    role: str,
    integrity_version: str,
) -> list[str]:
    """Return noncompensatory failure reasons for one named representation."""
    if representation_name not in {"u", "factual_z"}:
        raise ValueError("Unknown diagnosis representation.")
    reasons = []
    convergence = result["convergence"]
    status = convergence["status"]
    if status != "converged":
        reasons.append(f"{role}_{status}")

    collapse = result["collapse"]
    if not collapse["failed"]:
        return reasons
    failed_metrics = collapse.get("failed_metrics")
    if not isinstance(failed_metrics, list) or not failed_metrics:
        return [*reasons, f"{role}_latent_collapse:unspecified"]
    relevant = []
    for metric_id in failed_metrics:
        if (
            metric_id == "latent_all_finite"
            or metric_id.startswith(f"{representation_name}_")
            or (
                representation_name
                == ("factual_z" if integrity_version == "v2" else "u")
                and metric_id == "registered_residual_gradient_coverage"
            )
        ):
            relevant.append(metric_id)
    reasons.extend(
        f"{role}_latent_collapse:{metric_id}"
        for metric_id in relevant
    )
    return reasons


def _geometry_metric_ids(
    representation_name: str,
    *,
    lifecycle: str,
) -> tuple[str, ...]:
    """Return the frozen geometry endpoints for one representation/lifecycle."""
    prefix = "u" if representation_name == "u" else "factual_z"
    if lifecycle == "comparison":
        return (
            f"{prefix}_linear_cka",
            f"{prefix}_orthogonal_procrustes_disparity",
        )
    if lifecycle == "cross_seed":
        return (f"{prefix}_cross_seed_knn_jaccard_k15",)
    raise ValueError("Unknown geometry lifecycle.")


def aggregate_diagnosis_grid(
    results: list[dict],
    *,
    representations: Mapping[
        tuple[str, str, int],
        Mapping[str, Mapping[str, object]],
    ],
    contract_adapter: RedesignContractAdapter | None = None,
) -> dict[str, object]:
    """Aggregate only the exact RDX-03 grid with lineage-locked geometry."""
    adapter = (
        historical_redesign_contract_adapter()
        if contract_adapter is None
        else contract_adapter
    )
    if not isinstance(adapter, RedesignContractAdapter):
        raise TypeError("contract_adapter must be a RedesignContractAdapter.")
    validated = validate_diagnosis_grid(
        results,
        contract_adapter=adapter,
    )
    fixtures, rows, training_seeds = _diagnosis_grid_from_adapter(adapter)
    indexed = {
        (
            result["candidate_id"],
            result["fixture_id"],
            result["training_seed"],
        ): result
        for result in validated
    }
    if not isinstance(representations, Mapping):
        raise ValueError("representations must be a mapping.")
    if set(representations) != set(indexed):
        missing = sorted(set(indexed) - set(representations))
        extra = sorted(set(representations) - set(indexed))
        raise ValueError(
            "Diagnosis representations do not match the exact grid; "
            f"missing={missing}, extra={extra}."
        )

    comparisons = []
    cross_seed = []
    for fixture_id in fixtures:
        for candidate_id in rows:
            expected_representations = (
                ("factual_z",)
                if candidate_id == "B1"
                else ("u", "factual_z")
            )
            for seed in training_seeds:
                key = (candidate_id, fixture_id, seed)
                candidate_payload = representations[key]
                candidate_result = indexed[key]
                if set(candidate_payload) != set(expected_representations):
                    raise ValueError(
                        f"Unexpected representation set for {key}: "
                        f"{sorted(candidate_payload)}."
                    )
                metrics = metric_payload_template(
                    contract_adapter=adapter,
                )
                reference_candidate_ids = {}
                representation_status = {}
                no_call_reasons = {}
                for representation_name in expected_representations:
                    reference_candidate_id = (
                        "B2" if representation_name == "u" else "B1"
                    )
                    prefix = (
                        "u"
                        if representation_name == "u"
                        else "factual_z"
                    )
                    reference_candidate_ids[prefix] = (
                        reference_candidate_id
                    )
                    reference_key = (
                        reference_candidate_id,
                        fixture_id,
                        seed,
                    )
                    reasons = _representation_failure_reasons(
                        candidate_result,
                        representation_name=representation_name,
                        role="candidate",
                        integrity_version=adapter.integrity_version,
                    )
                    if reference_key != key:
                        reasons.extend(
                            _representation_failure_reasons(
                                indexed[reference_key],
                                representation_name=representation_name,
                                role=(
                                    f"reference_{reference_candidate_id}"
                                ),
                                integrity_version=adapter.integrity_version,
                            )
                        )
                    metric_ids = _geometry_metric_ids(
                        representation_name,
                        lifecycle="comparison",
                    )
                    if reasons:
                        for metric_id in metric_ids:
                            no_call_reasons[metric_id] = reasons
                        representation_status[representation_name] = {
                            "status": "not_evaluated_hard_failure",
                            "reference_candidate_id": (
                                reference_candidate_id
                            ),
                            "reason_codes": reasons,
                        }
                        continue
                    reference = representations[reference_key][
                        representation_name
                    ]
                    reference_values, candidate_values = (
                        _aligned_representation_pair(
                            reference,
                            candidate_payload[representation_name],
                        )
                    )
                    try:
                        metrics[metric_ids[0]] = linear_cka(
                            reference_values,
                            candidate_values,
                            scale_safe=adapter.integrity_version == "v2",
                        )
                        metrics[metric_ids[1]] = (
                            orthogonal_procrustes_disparity(
                                reference_values,
                                candidate_values,
                                scale_safe=adapter.integrity_version == "v2",
                            )
                        )
                    except (FloatingPointError, np.linalg.LinAlgError):
                        if adapter.integrity_version != "v2":
                            raise
                        derived_reasons = [
                            f"{representation_name}_derived_geometry_not_evaluable"
                        ]
                        for metric_id in metric_ids:
                            metrics[metric_id] = None
                            no_call_reasons[metric_id] = derived_reasons
                        representation_status[representation_name] = {
                            "status": "not_evaluated_derived_diagnostic_failure",
                            "reference_candidate_id": reference_candidate_id,
                            "reason_codes": derived_reasons,
                        }
                        continue
                    representation_status[representation_name] = {
                        "status": "evaluated",
                        "reference_candidate_id": (
                            reference_candidate_id
                        ),
                        "reason_codes": [],
                    }
                validate_metric_payload(
                    metrics,
                    candidate_id=candidate_id,
                    lifecycle="comparison",
                    contract_adapter=adapter,
                    required_no_call_reasons=no_call_reasons,
                )
                comparisons.append(
                    {
                        "fixture_id": fixture_id,
                        "candidate_id": candidate_id,
                        "reference_candidate_ids": (
                            reference_candidate_ids
                        ),
                        "training_seed": seed,
                        "metrics": metrics,
                        "representation_status": representation_status,
                    }
                )

            metrics = metric_payload_template(
                contract_adapter=adapter,
            )
            pairwise = {}
            representation_status = {}
            no_call_reasons = {}
            for representation_name in expected_representations:
                reasons = []
                for seed in training_seeds:
                    reasons.extend(
                        _representation_failure_reasons(
                            indexed[(candidate_id, fixture_id, seed)],
                            representation_name=representation_name,
                            role=f"candidate_seed{seed}",
                            integrity_version=adapter.integrity_version,
                        )
                    )
                metric_id = _geometry_metric_ids(
                    representation_name,
                    lifecycle="cross_seed",
                )[0]
                if reasons:
                    no_call_reasons[metric_id] = reasons
                    representation_status[representation_name] = {
                        "status": "not_evaluated_hard_failure",
                        "reason_codes": reasons,
                    }
                    continue
                seed_payloads = {
                    seed: (
                        np.asarray(
                            representations[
                                (candidate_id, fixture_id, seed)
                            ][representation_name]["cell_ids"],
                            dtype=str,
                        ),
                        np.asarray(
                            representations[
                                (candidate_id, fixture_id, seed)
                            ][representation_name]["values"],
                            dtype=np.float64,
                        ),
                    )
                    for seed in training_seeds
                }
                try:
                    stability = cross_seed_knn_metrics(
                        seed_payloads,
                        k=15,
                        scale_safe=adapter.integrity_version == "v2",
                    )
                except (FloatingPointError, np.linalg.LinAlgError):
                    if adapter.integrity_version != "v2":
                        raise
                    derived_reasons = [
                        f"{representation_name}_derived_geometry_not_evaluable"
                    ]
                    no_call_reasons[metric_id] = derived_reasons
                    representation_status[representation_name] = {
                        "status": "not_evaluated_derived_diagnostic_failure",
                        "reason_codes": derived_reasons,
                    }
                    continue
                prefix = (
                    "u"
                    if representation_name == "u"
                    else "factual_z"
                )
                metrics[metric_id] = stability["mean_jaccard"]
                pairwise[representation_name] = stability
                representation_status[representation_name] = {
                    "status": "evaluated",
                    "reason_codes": [],
                }
            validate_metric_payload(
                metrics,
                candidate_id=candidate_id,
                lifecycle="cross_seed",
                contract_adapter=adapter,
                required_no_call_reasons=no_call_reasons,
            )
            cross_seed.append(
                {
                    "fixture_id": fixture_id,
                    "candidate_id": candidate_id,
                    "training_seeds": list(training_seeds),
                    "metrics": metrics,
                    "pairwise": pairwise,
                    "representation_status": representation_status,
                }
            )

    failures = [
        {
            "fixture_id": result["fixture_id"],
            "candidate_id": result["candidate_id"],
            "training_seed": result["training_seed"],
            "convergence_status": result["convergence"]["status"],
            "collapse_failed": result["collapse"]["failed"],
        }
        for result in validated
        if result["convergence"]["status"] != "converged"
        or result["collapse"]["failed"]
    ]
    return {
        "schema_version": "mrtotalvi-convergence-aggregate-v2",
        "n_fits": len(validated),
        "rows": list(rows),
        "fixtures": list(fixtures),
        "training_seeds": list(training_seeds),
        "paired_comparisons": comparisons,
        "cross_seed": cross_seed,
        "fit_failures": failures,
        "all_fits_converged_without_collapse": not failures,
        "d0_decision": d0_downstream_gate_decision(
            validated,
            contract_adapter=adapter,
        ),
        "scientific_scope": (
            "RDX-03 convergence diagnosis; no redesign selection; "
            "Milo and factual human DA unavailable"
        ),
    }


def _v2_result_as_v1_geometry_adapter(result: Mapping[str, object]) -> dict:
    """Adapt terminal-only v2 eligibility to the frozen v1 geometry engine."""
    if (
        not isinstance(result, Mapping)
        or result.get("schema_version") != "mrtotalvi-convergence-fit-v3"
    ):
        raise ValueError("Unexpected prospective diagnosis result schema.")
    integrity = result.get("latent_integrity")
    if (
        not isinstance(integrity, Mapping)
        or integrity.get("schema_version")
        != LATENT_INTEGRITY_ASSESSMENT_SCHEMA
        or not isinstance(integrity.get("terminal_integrity_failed"), bool)
    ):
        raise ValueError("Invalid prospective latent-integrity evidence.")
    representations = integrity.get("representations")
    if not isinstance(representations, Mapping):
        raise ValueError("Prospective representation evidence is missing.")
    failed_metrics = []
    for representation in integrity.get("required_representations", []):
        evidence = representations.get(representation)
        if (
            not isinstance(evidence, Mapping)
            or not isinstance(evidence.get("terminal_failed"), bool)
            or not isinstance(evidence.get("terminal_failure_reasons"), list)
        ):
            raise ValueError(
                f"Invalid prospective integrity evidence for {representation}."
            )
        if evidence["terminal_failed"]:
            failed_metrics.extend(evidence["terminal_failure_reasons"])

    adapted = deepcopy(dict(result))
    adapted["schema_version"] = "mrtotalvi-convergence-fit-v2"
    adapted["collapse"] = {
        "failed": integrity["terminal_integrity_failed"],
        "failed_metrics": failed_metrics,
    }
    del adapted["latent_integrity"]
    return adapted


def aggregate_diagnosis_grid_v2(
    results: list[dict],
    *,
    representations: Mapping[
        tuple[str, str, int],
        Mapping[str, Mapping[str, object]],
    ],
    contract_adapter: RedesignContractAdapter,
) -> dict[str, object]:
    """Run frozen geometry while treating v2 rank findings as alerts only."""
    if (
        not isinstance(contract_adapter, RedesignContractAdapter)
        or contract_adapter.integrity_version != "v2"
    ):
        raise ValueError(
            "Prospective aggregation requires the v2 contract adapter."
        )
    if not isinstance(results, list):
        raise ValueError("Diagnosis results must be a list.")
    for result in results:
        validate_version_binding(
            result,
            contract_adapter,
            expected_schema="mrtotalvi-convergence-fit-v3",
        )
        validate_version_binding(
            result.get("latent_integrity"),
            contract_adapter,
            expected_schema=LATENT_INTEGRITY_ASSESSMENT_SCHEMA,
        )
    adapted = [_v2_result_as_v1_geometry_adapter(result) for result in results]
    aggregate = aggregate_diagnosis_grid(
        adapted,
        representations=representations,
        contract_adapter=contract_adapter,
    )
    for collection in ("paired_comparisons", "cross_seed"):
        for record in aggregate[collection]:
            for status in record["representation_status"].values():
                status["reason_codes"] = [
                    reason.replace(
                        "_latent_collapse:",
                        "_terminal_integrity:",
                    )
                    for reason in status["reason_codes"]
                ]

    failures = []
    alerts = []
    terminal_failed = False
    for result in results:
        integrity = result["latent_integrity"]
        flagged = integrity["effective_rank_screen_flags"]
        if flagged:
            alerts.append(
                {
                    "fixture_id": result["fixture_id"],
                    "candidate_id": result["candidate_id"],
                    "training_seed": result["training_seed"],
                    "representations": list(flagged),
                }
            )
        failed = integrity["terminal_integrity_failed"]
        terminal_failed = terminal_failed or failed
        if result["convergence"]["status"] != "converged" or failed:
            failures.append(
                {
                    "fixture_id": result["fixture_id"],
                    "candidate_id": result["candidate_id"],
                    "training_seed": result["training_seed"],
                    "convergence_status": result["convergence"]["status"],
                    "terminal_integrity_failed": failed,
                }
            )

    aggregate["schema_version"] = "mrtotalvi-convergence-aggregate-v3"
    aggregate.update(version_binding_fields(contract_adapter))
    aggregate["fit_failures"] = failures
    aggregate["terminal_integrity_failed"] = terminal_failed
    aggregate["effective_rank_screen_flags"] = alerts
    aggregate["all_fits_converged_with_terminal_integrity"] = not failures
    del aggregate["all_fits_converged_without_collapse"]
    if aggregate["d0_decision"]["reason_codes"] == [
        "d0_convergence_or_collapse_failure"
    ]:
        aggregate["d0_decision"]["reason_codes"] = [
            "d0_convergence_or_latent_integrity_failure"
        ]
    return aggregate
