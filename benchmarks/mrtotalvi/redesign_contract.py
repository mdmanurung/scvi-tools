"""Frozen candidate and verdict contracts for the MrTotalVI redesign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal


_CANDIDATE_ORDER = (
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
_ELIGIBLE_PUBLIC_MODES = {
    "D1": "sample_blind_scaled",
    "D2": "sample_blind_totalvi",
    "D3": "sample_blind_totalvi",
    "D4": "sample_blind_totalvi",
    "D5": "sample_blind_totalvi",
}


def _validate_digest(name: str, value: object) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal.") from error


@dataclass(frozen=True)
class RedesignCandidateConfig:
    """One fully declared comparator or redesign configuration."""

    candidate_id: Literal[
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
    model_family: Literal["scvi", "totalvi", "mrtotalvi"]
    representation_set: Literal["factual_z", "u_and_factual_z"]
    modality_scope: Literal["rna", "rna_protein"]
    hierarchy_mode: Literal["stock", "legacy", "centered_v2"]
    input_transform: Literal[
        "stock_scvi",
        "stock_totalvi",
        "raw_log1p",
        "totalvi_per_modality",
    ]
    posterior_trunk: Literal[
        "stock_scvi",
        "stock_totalvi",
        "current_mlp",
        "totalvi_fclayers",
    ]
    biological_sample_conditioning: Literal["stock", "conditioned", "blind"]
    u_prior: Literal["not_applicable", "mog_trainable", "vamp_frozen_initialized"]
    observation_weighting: Literal["cell_equal", "sample_equal"]
    scientific_role: Literal[
        "rna_contextual_comparator",
        "primary_factual_z_comparator",
        "legacy_baseline",
        "centered_conditioned_control",
        "convergence_control",
        "redesign",
    ]

    def to_dict(self) -> dict[str, str]:
        """Return the complete strict payload."""
        return asdict(self)

    def model_axes(self) -> dict[str, str]:
        """Return every declared model axis, excluding identity and role."""
        payload = self.to_dict()
        return {
            key: value
            for key, value in payload.items()
            if key not in {"candidate_id", "scientific_role"}
        }


def redesign_candidate_configs() -> dict[str, RedesignCandidateConfig]:
    """Return B0-B3 and D0-D5 in their frozen evaluation order."""
    d0 = {
        "model_family": "mrtotalvi",
        "representation_set": "u_and_factual_z",
        "modality_scope": "rna_protein",
        "hierarchy_mode": "centered_v2",
        "input_transform": "raw_log1p",
        "posterior_trunk": "current_mlp",
        "biological_sample_conditioning": "blind",
        "u_prior": "vamp_frozen_initialized",
        "observation_weighting": "cell_equal",
    }
    configs = {
        "B0": RedesignCandidateConfig(
            candidate_id="B0",
            model_family="scvi",
            representation_set="factual_z",
            modality_scope="rna",
            hierarchy_mode="stock",
            input_transform="stock_scvi",
            posterior_trunk="stock_scvi",
            biological_sample_conditioning="stock",
            u_prior="not_applicable",
            observation_weighting="cell_equal",
            scientific_role="rna_contextual_comparator",
        ),
        "B1": RedesignCandidateConfig(
            candidate_id="B1",
            model_family="totalvi",
            representation_set="factual_z",
            modality_scope="rna_protein",
            hierarchy_mode="stock",
            input_transform="stock_totalvi",
            posterior_trunk="stock_totalvi",
            biological_sample_conditioning="stock",
            u_prior="not_applicable",
            observation_weighting="cell_equal",
            scientific_role="primary_factual_z_comparator",
        ),
        "B2": RedesignCandidateConfig(
            candidate_id="B2",
            model_family="mrtotalvi",
            representation_set="u_and_factual_z",
            modality_scope="rna_protein",
            hierarchy_mode="legacy",
            input_transform="raw_log1p",
            posterior_trunk="current_mlp",
            biological_sample_conditioning="conditioned",
            u_prior="mog_trainable",
            observation_weighting="cell_equal",
            scientific_role="legacy_baseline",
        ),
        "B3": RedesignCandidateConfig(
            candidate_id="B3",
            model_family="mrtotalvi",
            representation_set="u_and_factual_z",
            modality_scope="rna_protein",
            hierarchy_mode="centered_v2",
            input_transform="raw_log1p",
            posterior_trunk="current_mlp",
            biological_sample_conditioning="conditioned",
            u_prior="vamp_frozen_initialized",
            observation_weighting="cell_equal",
            scientific_role="centered_conditioned_control",
        ),
        "D0": RedesignCandidateConfig(
            candidate_id="D0",
            **d0,
            scientific_role="convergence_control",
        ),
        "D1": RedesignCandidateConfig(
            candidate_id="D1",
            **(d0 | {"input_transform": "totalvi_per_modality"}),
            scientific_role="redesign",
        ),
        "D2": RedesignCandidateConfig(
            candidate_id="D2",
            **(
                d0
                | {
                    "input_transform": "totalvi_per_modality",
                    "posterior_trunk": "totalvi_fclayers",
                }
            ),
            scientific_role="redesign",
        ),
        "D3": RedesignCandidateConfig(
            candidate_id="D3",
            **(
                d0
                | {
                    "input_transform": "totalvi_per_modality",
                    "posterior_trunk": "totalvi_fclayers",
                    "u_prior": "mog_trainable",
                }
            ),
            scientific_role="redesign",
        ),
        "D4": RedesignCandidateConfig(
            candidate_id="D4",
            **(
                d0
                | {
                    "input_transform": "totalvi_per_modality",
                    "posterior_trunk": "totalvi_fclayers",
                    "observation_weighting": "sample_equal",
                }
            ),
            scientific_role="redesign",
        ),
        "D5": RedesignCandidateConfig(
            candidate_id="D5",
            **(
                d0
                | {
                    "input_transform": "totalvi_per_modality",
                    "posterior_trunk": "totalvi_fclayers",
                    "u_prior": "mog_trainable",
                    "observation_weighting": "sample_equal",
                }
            ),
            scientific_role="redesign",
        ),
    }
    assert tuple(configs) == _CANDIDATE_ORDER
    return configs


def validate_redesign_candidate(payload: dict) -> RedesignCandidateConfig:
    """Reject unknown candidates, fields, or changes to any frozen axis."""
    if not isinstance(payload, dict):
        raise ValueError("Redesign candidate payload must be a mapping.")
    expected_fields = {field.name for field in fields(RedesignCandidateConfig)}
    missing = expected_fields - set(payload)
    unknown = set(payload) - expected_fields
    if missing:
        raise ValueError(f"Redesign candidate fields missing: {sorted(missing)}")
    if unknown:
        raise ValueError(
            f"Unknown redesign candidate fields: {sorted(unknown)}"
        )
    candidate_id = payload.get("candidate_id")
    configs = redesign_candidate_configs()
    if candidate_id not in configs:
        raise ValueError(f"Unknown redesign candidate {candidate_id!r}.")
    expected = configs[candidate_id]
    if payload != expected.to_dict():
        changed = sorted(
            key
            for key in expected_fields
            if payload.get(key) != expected.to_dict()[key]
        )
        raise ValueError(
            f"Candidate {candidate_id} does not match frozen axes: {changed}."
        )
    return expected


def redesign_config_digest() -> str:
    """Hash the canonical complete B0-B3/D0-D5 configuration payload."""
    payload = [
        redesign_candidate_configs()[candidate_id].to_dict()
        for candidate_id in _CANDIDATE_ORDER
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HardGateContract:
    """One noncompensatory selection gate."""

    gate_id: str
    metric_id: str
    comparison: str
    value: float
    reference: str | None
    aggregation: str
    scope: tuple[str, ...]


@dataclass(frozen=True)
class AdaptiveStageContract:
    """One exact adaptive-screen grid."""

    stage_id: str
    fixed_rows: tuple[str, ...]
    candidate_pool: tuple[str, ...]
    training_seeds: tuple[int, ...]
    instances_per_scenario: int
    scenarios: tuple[str, ...]
    max_redesign_survivors: int
    prune_on: tuple[str, ...]


@dataclass(frozen=True)
class ConvergenceContract:
    """Shared convergence controls for every candidate."""

    check_every_epochs: int
    minimum_epochs: int
    maximum_epochs: int
    patience_checks: int
    restore_best_checkpoint: bool
    candidate_specific_retuning: bool


@dataclass(frozen=True)
class DiagnosisContract:
    """Exact RDX-03 rows, seeds, and fixture grid."""

    rows: tuple[str, ...]
    training_seeds: tuple[int, ...]
    fixtures: tuple[str, ...]
    canonical_human_requires_lineage_gate: bool


@dataclass(frozen=True)
class MiloContract:
    """Frozen replicate-level Milo estimator."""

    graph_k: int
    graph_d: int
    neighborhood_prop: float
    refined: bool
    refinement_scheme: str
    sample_column: str
    formula: str
    fdr_weighting: str
    glmm_solver: str
    reml: bool
    normalization: str
    fail_on_error: bool
    nominal_spatial_fdr: float


@dataclass(frozen=True)
class SelectionContract:
    """Frozen order-invariant candidate tie-break."""

    primary_metric: str
    primary_aggregation: str
    tie_tolerance: float
    tie_break_metrics: tuple[str, ...]
    tie_break_directions: tuple[str, ...]
    require_order_invariance: bool
    require_independent_implementation: bool
    d0_only_verdict: str


@dataclass(frozen=True)
class RuntimeContract:
    """Tracked runtimes and permission boundary."""

    primary_python: str
    primary_environment: str
    contract_python_minors: tuple[str, ...]
    r_runtime: str
    accelerator_policy: str
    dependency_installation_authorized: bool
    external_environment_writes_authorized: bool
    network_authorized: bool
    scheduler_authorized: bool


@dataclass(frozen=True)
class HumanContract:
    """Frozen human development dimensions and outcome lock."""

    expected_cells: int
    expected_complete_donors: int
    selected_genes: int
    selected_non_isotype_proteins: int
    latent_dimension: int
    training_seeds: tuple[int, ...]
    factual_da_locked_until_candidate_freeze: bool
    safety_perturbations: tuple[str, ...]


@dataclass(frozen=True)
class RedesignRunContract:
    """Complete machine-frozen redesign screen contract."""

    schema_version: str
    evaluation_rows: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    hard_gates: tuple[HardGateContract, ...]
    stage_a: AdaptiveStageContract
    stage_b: AdaptiveStageContract
    convergence: ConvergenceContract
    diagnosis: DiagnosisContract
    milo: MiloContract
    selection: SelectionContract
    runtime: RuntimeContract
    human: HumanContract
    rng_streams: tuple[str, ...]

    def to_dict(self) -> dict:
        """Return a stable JSON-compatible representation."""
        return json.loads(json.dumps(asdict(self)))


_REDESIGN_METRIC_IDS = (
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
_SCENARIOS = (
    "null",
    "da_only",
    "de_only",
    "mixed",
    "rare_state",
    "unequal_cells",
    "continuous",
    "batch_confounded",
)


def _gate(
    gate_id: str,
    metric_id: str,
    comparison: str,
    value: float,
    *,
    reference: str | None,
    aggregation: str,
    scope: tuple[str, ...],
) -> HardGateContract:
    return HardGateContract(
        gate_id=gate_id,
        metric_id=metric_id,
        comparison=comparison,
        value=value,
        reference=reference,
        aggregation=aggregation,
        scope=scope,
    )


def redesign_run_contract() -> RedesignRunContract:
    """Return the complete preregistered run contract."""
    hard_gates = (
        _gate(
            "centering_identity",
            "centering_max_abs",
            "absolute_lte",
            1e-6,
            reference=None,
            aggregation="maximum_within_run",
            scope=("u",),
        ),
        _gate(
            "u_sample_leakage",
            "u_within_state_sample_predictability",
            "reference_plus_lte",
            0.02,
            reference=(
                "u_within_state_sample_predictability_permutation_p95"
            ),
            aggregation="each_run",
            scope=("u",),
        ),
        _gate(
            "u_state_noninferiority",
            "u_state_balanced_accuracy",
            "reference_minus_gte",
            0.02,
            reference="B1",
            aggregation="each_seed",
            scope=("u",),
        ),
        _gate(
            "u_knn_state_noninferiority",
            "u_knn_state_accuracy_k15",
            "reference_minus_gte",
            0.02,
            reference="B1",
            aggregation="each_seed",
            scope=("u",),
        ),
        _gate(
            "u_seed_stability_absolute",
            "u_cross_seed_knn_jaccard_k15",
            "absolute_gte",
            0.60,
            reference=None,
            aggregation="all_declared_seed_pairs",
            scope=("u",),
        ),
        _gate(
            "u_seed_stability_noninferiority",
            "u_cross_seed_knn_jaccard_k15",
            "reference_minus_gte",
            0.05,
            reference="B1",
            aggregation="all_declared_seed_pairs",
            scope=("u",),
        ),
        _gate(
            "factual_z_total_prediction",
            "multimodal_heldout_predictive_loss",
            "reference_multiplier_lte",
            1.02,
            reference="B1",
            aggregation="each_seed",
            scope=("factual_z",),
        ),
        _gate(
            "factual_z_rna_prediction",
            "rna_heldout_negative_log_likelihood",
            "reference_multiplier_lte",
            1.03,
            reference="B1",
            aggregation="each_seed",
            scope=("factual_z",),
        ),
        _gate(
            "factual_z_protein_prediction",
            "protein_heldout_negative_log_likelihood",
            "reference_multiplier_lte",
            1.03,
            reference="B1",
            aggregation="each_seed",
            scope=("factual_z",),
        ),
        _gate(
            "factual_z_state_noninferiority",
            "factual_z_state_balanced_accuracy",
            "reference_minus_gte",
            0.02,
            reference="B1",
            aggregation="each_seed",
            scope=("factual_z",),
        ),
        _gate(
            "factual_z_knn_state_noninferiority",
            "factual_z_knn_state_accuracy_k15",
            "reference_minus_gte",
            0.02,
            reference="B1",
            aggregation="each_seed",
            scope=("factual_z",),
        ),
        _gate(
            "factual_z_seed_stability",
            "factual_z_cross_seed_knn_jaccard_k15",
            "absolute_gte",
            0.60,
            reference=None,
            aggregation="all_declared_seed_pairs",
            scope=("factual_z",),
        ),
        _gate(
            "u_finite",
            "latent_all_finite",
            "absolute_eq",
            1.0,
            reference=None,
            aggregation="all_coordinates",
            scope=("u",),
        ),
        _gate(
            "factual_z_finite",
            "latent_all_finite",
            "absolute_eq",
            1.0,
            reference=None,
            aggregation="all_coordinates",
            scope=("factual_z",),
        ),
        _gate(
            "u_effective_rank",
            "u_effective_rank",
            "configured_dimension_fraction_gte",
            0.5,
            reference="configured_latent_dimension",
            aggregation="each_seed",
            scope=("u",),
        ),
        _gate(
            "factual_z_effective_rank",
            "factual_z_effective_rank",
            "configured_dimension_fraction_gte",
            0.5,
            reference="configured_latent_dimension",
            aggregation="each_seed",
            scope=("factual_z",),
        ),
        _gate(
            "registered_residual_gradients",
            "registered_residual_gradient_coverage",
            "absolute_eq",
            1.0,
            reference=None,
            aggregation="all_registered_rows",
            scope=("optimization",),
        ),
        _gate(
            "milo_no_failed_fits",
            "milo_primary_failed_fit_count",
            "absolute_eq",
            0.0,
            reference=None,
            aggregation="each_run",
            scope=("da",),
        ),
        _gate(
            "milo_no_na_fits",
            "milo_primary_na_fit_count",
            "absolute_eq",
            0.0,
            reference=None,
            aggregation="each_run",
            scope=("da",),
        ),
        _gate(
            "milo_fdp",
            "milo_fdp_spatialfdr_0_10",
            "absolute_lte",
            0.15,
            reference=None,
            aggregation="median",
            scope=("null", "de_only"),
        ),
        _gate(
            "milo_power_noninferiority",
            "milo_power_spatialfdr_0_10",
            "reference_minus_gte",
            0.05,
            reference="best_of_PCA_and_B1",
            aggregation="median",
            scope=("known_truth_da",),
        ),
        _gate(
            "milo_localization_noninferiority",
            "milo_localization_spatialfdr_0_10",
            "reference_minus_gte",
            0.05,
            reference="best_of_PCA_and_B1",
            aggregation="median",
            scope=("known_truth_da",),
        ),
        _gate(
            "milo_improvement_over_b2",
            "milo_localization_spatialfdr_0_10",
            "any_scenario_reference_plus_gte",
            0.05,
            reference="B2",
            aggregation="median",
            scope=("known_truth_da",),
        ),
        _gate(
            "milo_no_material_loss_vs_b2",
            "milo_localization_spatialfdr_0_10",
            "all_scenarios_reference_minus_gte",
            0.05,
            reference="B2",
            aggregation="median",
            scope=("known_truth_da",),
        ),
    )
    prune_on = (
        "contract_failure",
        "non_convergence",
        "nonfinite_output",
        "latent_collapse",
        "u_leakage_failure",
        "factual_z_noninferiority_failure",
    )
    return RedesignRunContract(
        schema_version="mrtotalvi-redesign-run-contract-v1",
        evaluation_rows=_CANDIDATE_ORDER,
        candidate_ids=("D1", "D2", "D3", "D4", "D5"),
        metric_ids=_REDESIGN_METRIC_IDS,
        hard_gates=hard_gates,
        stage_a=AdaptiveStageContract(
            stage_id="A",
            fixed_rows=("B1", "B2", "B3", "D0"),
            candidate_pool=("D1", "D2", "D3", "D4", "D5"),
            training_seeds=(0,),
            instances_per_scenario=3,
            scenarios=_SCENARIOS,
            max_redesign_survivors=2,
            prune_on=prune_on,
        ),
        stage_b=AdaptiveStageContract(
            stage_id="B",
            fixed_rows=("B1", "B2"),
            candidate_pool=("stage_a_survivors",),
            training_seeds=(0, 1, 2),
            instances_per_scenario=10,
            scenarios=_SCENARIOS,
            max_redesign_survivors=2,
            prune_on=prune_on,
        ),
        convergence=ConvergenceContract(
            check_every_epochs=5,
            minimum_epochs=50,
            maximum_epochs=400,
            patience_checks=30,
            restore_best_checkpoint=True,
            candidate_specific_retuning=False,
        ),
        diagnosis=DiagnosisContract(
            rows=("B1", "B2", "B3", "D0"),
            training_seeds=(0, 1, 2),
            fixtures=(
                "mixed",
                "unequal_cells",
                "sealed_500",
                "canonical_human_if_available",
            ),
            canonical_human_requires_lineage_gate=True,
        ),
        milo=MiloContract(
            graph_k=30,
            graph_d=20,
            neighborhood_prop=0.1,
            refined=True,
            refinement_scheme="graph",
            sample_column="donor_timepoint",
            formula="~ timepoint + (1|donor)",
            fdr_weighting="graph-overlap",
            glmm_solver="Fisher",
            reml=True,
            normalization="TMM",
            fail_on_error=False,
            nominal_spatial_fdr=0.10,
        ),
        selection=SelectionContract(
            primary_metric="milo_localization_spatialfdr_0_10",
            primary_aggregation="median_known_truth",
            tie_tolerance=0.02,
            tie_break_metrics=(
                "multimodal_heldout_predictive_loss",
                "u_cross_seed_knn_jaccard_k15",
                "trainable_parameter_count",
            ),
            tie_break_directions=("lower", "higher", "lower"),
            require_order_invariance=True,
            require_independent_implementation=True,
            d0_only_verdict="stop",
        ),
        runtime=RuntimeContract(
            primary_python="3.13.11",
            primary_environment=(
                "/exports/archive/hg-funcgenom-research/mdmanurung/conda/"
                "envs/scvi-test/bin/python"
            ),
            contract_python_minors=("3.13", "3.14"),
            r_runtime=(
                "/exports/archive/hg-funcgenom-research/mdmanurung/conda/"
                "envs/R4_51/bin/Rscript"
            ),
            accelerator_policy="cpu_unless_separately_authorized",
            dependency_installation_authorized=False,
            external_environment_writes_authorized=False,
            network_authorized=False,
            scheduler_authorized=False,
        ),
        human=HumanContract(
            expected_cells=46_817,
            expected_complete_donors=10,
            selected_genes=5_000,
            selected_non_isotype_proteins=130,
            latent_dimension=20,
            training_seeds=(0, 1, 2),
            factual_da_locked_until_candidate_freeze=True,
            safety_perturbations=(
                "within_donor_timepoint_label_permutation",
                "human_geometry_null",
                "human_geometry_da_only",
                "human_geometry_de_only",
            ),
        ),
        rng_streams=("truth", "training", "evaluation"),
    )


def validate_redesign_run_contract(payload: dict) -> RedesignRunContract:
    """Reject any endpoint, gate, grid, seed, or environment drift."""
    if not isinstance(payload, dict):
        raise ValueError("Redesign run contract must be a mapping.")
    expected = redesign_run_contract()
    expected_payload = expected.to_dict()
    if payload != expected_payload:
        changed = sorted(
            key
            for key in set(payload) | set(expected_payload)
            if payload.get(key) != expected_payload.get(key)
        )
        raise ValueError(
            "Payload does not match the frozen run contract; changed "
            f"sections: {changed}."
        )
    return expected


def redesign_run_contract_digest() -> str:
    """Hash the canonical endpoint, gate, grid, and runtime contract."""
    encoded = json.dumps(
        redesign_run_contract().to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


_V2_INTEGRITY_METRIC_IDS = (
    "u_representation_all_finite",
    "factual_z_representation_all_finite",
    "u_exact_nonconstant_variation",
    "factual_z_exact_nonconstant_variation",
    "u_posterior_scales_all_valid",
    "factual_z_posterior_scales_all_valid",
)


def redesign_run_contract_v2() -> RedesignRunContract:
    """Return the prospective contract with rank separated from integrity."""
    historical = redesign_run_contract()
    prune_on = tuple(
        "terminal_integrity_failure" if item == "latent_collapse" else item
        for item in historical.stage_a.prune_on
    )
    integrity_gates = tuple(
        _gate(
            f"{representation}_{indicator}",
            f"{representation}_{indicator}",
            "absolute_eq",
            1.0,
            reference=None,
            aggregation=(
                "all_coordinates"
                if indicator == "representation_all_finite"
                else (
                    "all_evaluation_rows"
                    if indicator == "exact_nonconstant_variation"
                    else "all_posterior_scale_elements"
                )
            ),
            scope=(representation,),
        )
        for indicator in (
            "representation_all_finite",
            "exact_nonconstant_variation",
            "posterior_scales_all_valid",
        )
        for representation in ("u", "factual_z")
    )
    hard_gates = []
    removed = {
        "u_finite",
        "factual_z_finite",
        "u_effective_rank",
        "factual_z_effective_rank",
    }
    for gate in historical.hard_gates:
        if gate.gate_id == "u_finite":
            hard_gates.extend(integrity_gates)
        if gate.gate_id in removed:
            continue
        if gate.gate_id == "registered_residual_gradients":
            gate = replace(gate, scope=("factual_z",))
        hard_gates.append(gate)
    return replace(
        historical,
        schema_version="mrtotalvi-redesign-run-contract-v2",
        metric_ids=(*historical.metric_ids, *_V2_INTEGRITY_METRIC_IDS),
        hard_gates=tuple(hard_gates),
        stage_a=replace(historical.stage_a, prune_on=prune_on),
        stage_b=replace(historical.stage_b, prune_on=prune_on),
    )


def redesign_run_contract_digest_v2() -> str:
    """Hash the exact canonical prospective run contract."""
    encoded = json.dumps(
        redesign_run_contract_v2().to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RedesignVerdict:
    """One strict terminal candidate, stop, or blocked result."""

    schema_version: str
    verdict: Literal["candidate", "stop", "blocked"]
    candidate_id: Literal["D1", "D2", "D3", "D4", "D5"] | None
    public_mode: Literal["sample_blind_scaled", "sample_blind_totalvi"] | None
    reason_codes: tuple[str, ...]
    code_digest: str
    config_digest: str
    evidence_digest: str

    def __post_init__(self) -> None:
        """Validate the terminal verdict and its frozen evidence identity."""
        if self.schema_version != "mrtotalvi-redesign-verdict-v1":
            raise ValueError(
                f"Unsupported verdict schema {self.schema_version!r}."
            )
        if self.verdict not in {"candidate", "stop", "blocked"}:
            raise ValueError(f"Unsupported redesign verdict {self.verdict!r}.")
        for name in ("code_digest", "config_digest", "evidence_digest"):
            _validate_digest(name, getattr(self, name))
        if (
            not isinstance(self.reason_codes, tuple)
            or any(
                not isinstance(reason, str) or not reason.strip()
                for reason in self.reason_codes
            )
            or len(set(self.reason_codes)) != len(self.reason_codes)
        ):
            raise ValueError(
                "reason_codes must be a tuple of unique non-empty strings."
            )

        if self.verdict == "candidate":
            expected_mode = _ELIGIBLE_PUBLIC_MODES.get(self.candidate_id)
            if expected_mode is None:
                raise ValueError(
                    "A candidate verdict requires one of D1-D5; D0 cannot "
                    "create a new public mode."
                )
            if self.public_mode != expected_mode:
                raise ValueError(
                    f"{self.candidate_id} requires public_mode={expected_mode!r}."
                )
        else:
            if self.candidate_id is not None or self.public_mode is not None:
                raise ValueError(
                    "stop/blocked verdicts cannot name a candidate or public mode."
                )
            if not self.reason_codes:
                raise ValueError(
                    "stop/blocked verdicts require at least one reason code."
                )

    def to_dict(self) -> dict:
        """Return the strict JSON-compatible verdict payload."""
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> RedesignVerdict:
        """Parse a verdict without accepting missing or unknown fields."""
        if not isinstance(payload, dict):
            raise ValueError("Verdict payload must be a mapping.")
        expected = {field.name for field in fields(cls)}
        missing = expected - set(payload)
        unknown = set(payload) - expected
        if missing:
            raise ValueError(f"Verdict fields missing: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Unknown verdict fields: {sorted(unknown)}")
        reasons = payload["reason_codes"]
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) for reason in reasons
        ):
            raise ValueError("reason_codes must be a list of strings.")
        return cls(
            schema_version=payload["schema_version"],
            verdict=payload["verdict"],
            candidate_id=payload["candidate_id"],
            public_mode=payload["public_mode"],
            reason_codes=tuple(reasons),
            code_digest=payload["code_digest"],
            config_digest=payload["config_digest"],
            evidence_digest=payload["evidence_digest"],
        )
