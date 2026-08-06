"""Model and fixture runners for the preregistered RDX-03 diagnosis."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import numpy as np

from .comparator import (
    MRTOTALVI_FACTUAL_Z_POSTERIOR_DRAWS,
    ComparatorRunConfig,
    _realized_totalvi_optimization_identity,
    best_checkpoint_identity,
    collect_mrtotalvi_diagnostics,
    frozen_totalvi_optimization_kwargs,
    run_stock_comparator,
    serialize_training_history,
    state_dict_digest,
)
from .config import candidate_configs
from .convergence import (
    assess_convergence,
    assess_latent_integrity_v2,
    convergence_control,
    diagnosis_fit_spec,
)
from .historical_comparator import EVALUATION_SEED
from .human_lineage import make_within_sample_hash_split, sha256_lines
from .simulation import ScenarioConfig, generate_scenario
from .versioning import version_binding_fields

if TYPE_CHECKING:
    from anndata import AnnData

    from .versioning import RedesignContractAdapter


SEALED_500_H5AD = Path(
    ".scratch/mrtotalvi-v2/engineering-runs/"
    "20260725T233701Z-seed0-520ff544/checkpoint/adata.h5ad"
)
SEALED_500_SHA256 = (
    "b63a7df6b57d4db5bf0ce9e091ca36db9d19ad7c6ea798c7224a52f3a7d51dff"
)
SEALED_500_ANNOTATION_SOURCE = Path(
    "/exports/para-lipg-hpc/mdmanurung/schisto_citeseq/"
    "analysis/harmonized_integration/outputs/human_immune_joint.h5ad"
)
SEALED_500_ANNOTATION_SOURCE_SHA256 = (
    "520ff544daae6192efd7f3501669e05b0122e6fbaf8e9de0246122cecd1de2da"
)
SEALED_500_STATE_ANNOTATIONS = Path(
    "benchmarks/mrtotalvi/sealed_500_state_annotations.json"
)
SEALED_500_STATE_ANNOTATIONS_SHA256 = (
    "66e6af5d163dc3917907ea633805cd8d5787266b256d912ec9015fe7d9c608e2"
)
SEALED_500_STATE_ANNOTATION_SHA256 = (
    "420db70e69a1007a9c4b406da298dc07960856f6276951b22672ce7c5dcb7c3c"
)
CANONICAL_HUMAN_RUN_ID = (
    "20260731T081355Z-991ec740-b50f4e3a-e6ce6542"
)
CANONICAL_HUMAN_H5AD = Path(
    ".scratch/mrtotalvi-v2-redesign/human-lineage-runs/"
    f"{CANONICAL_HUMAN_RUN_ID}/human-w00-w22.h5ad"
)
CANONICAL_HUMAN_SHA256 = (
    "37198b29dd5bbd9013969639cd7cbe99f1ed67269a52a17ef78fb994ffabee9b"
)
CANONICAL_HUMAN_CACHE_ENV = "MRTOTALVI_CANONICAL_HUMAN_H5AD_CACHE"
CANONICAL_HUMAN_CONTRACT_SHA256 = (
    "b50f4e3a4322b0c2f16337e40508f8eb3cd2b910963cd36096868e383a6bba78"
)
CANONICAL_HUMAN_SPLIT_SHA256 = (
    "68c2c95a74650ed02cac88053de63de030becf88a7448ec13a0534021cb286b9"
)
DIAGNOSIS_SPLIT_SALT = "mrtotalvi-rdx03-diagnosis-v1"
SYNTHETIC_FIXTURE_SEED = 0
HUMAN_EVALUATION_SEED = 20_260_726


@dataclass(frozen=True)
class PreparedDiagnosisFixture:
    """One exact in-memory fixture and its model/evaluation boundary."""

    fixture_id: str
    adata: AnnData
    train_indices: np.ndarray
    validation_indices: np.ndarray
    state_labels: np.ndarray
    state_annotation_id: str
    sample_labels: np.ndarray
    technical_batch_labels: np.ndarray
    count_layer: str
    protein_obsm_key: str
    protein_names_uns_key: str
    technical_batch_key: str
    sample_key: str
    n_latent: int
    n_hidden: int
    n_layers: int
    n_prior_components: int
    batch_size: int
    learning_rate: float
    evaluation_seed: int
    source_data_digest: str
    state_annotation_digest: str
    data_digest: str
    split_digest: str

    def evaluation_annotations(self) -> dict[str, np.ndarray]:
        """Return evaluation-only labels aligned to validation cells."""
        validation = self.validation_indices
        return {
            "state_labels": self.state_labels[validation],
            "sample_labels": self.sample_labels[validation],
            "technical_batch_labels": self.technical_batch_labels[
                validation
            ],
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_human_read_path(authority_path: Path) -> Path:
    """Resolve an exact local read cache only after hashing the authority."""
    authority = Path(authority_path)
    if _sha256_file(authority) != CANONICAL_HUMAN_SHA256:
        raise ValueError("Canonical-human fixture SHA-256 drifted.")
    cache_value = os.environ.get(CANONICAL_HUMAN_CACHE_ENV)
    if cache_value is None:
        return authority
    cache_path = Path(cache_value)
    if not cache_path.is_absolute():
        raise ValueError("Canonical-human cache path must be absolute.")
    if not cache_path.is_file():
        raise ValueError("Canonical-human cache path is not a file.")
    if _sha256_file(cache_path) != CANONICAL_HUMAN_SHA256:
        raise ValueError("Canonical-human cache SHA-256 drifted.")
    return cache_path


def _canonical_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _state_annotation_digest(labels) -> str:
    """Hash ordered evaluation annotations without object-array ambiguity."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("State annotations must be one-dimensional.")
    return sha256_lines(tuple(str(value) for value in values))


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal.") from error
    return value


def _require_exact_keys(
    payload: object,
    *,
    expected: set[str],
    name: str,
) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object.")
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing or unknown:
        raise ValueError(
            f"{name} keys drifted: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}."
        )
    return payload


def _load_sealed_500_state_annotations(
    payload_path: str | Path,
    *,
    fixture_path: str | Path,
    expected_cell_ids: tuple[str, ...],
    expected_payload_sha256: str,
    expected_fixture_sha256: str,
    expected_source_sha256: str,
    expected_state_digest: str,
) -> np.ndarray:
    """Load the exact ordered 500-cell labels without reopening their source.

    The compact payload is a pre-authority closure of the external annotation
    source. Its raw bytes, the sealed fixture bytes, ordered cell inventory,
    ordered labels, and recorded source identity are all independently bound.
    """
    payload_file = Path(payload_path)
    fixture_file = Path(fixture_path)
    for name, path in (
        ("Sealed annotation payload", payload_file),
        ("Sealed fixture", fixture_file),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name} must be a regular non-symlink file.")

    expected_payload = _require_sha256(
        "expected_payload_sha256", expected_payload_sha256
    )
    expected_fixture = _require_sha256(
        "expected_fixture_sha256", expected_fixture_sha256
    )
    expected_source = _require_sha256(
        "expected_source_sha256", expected_source_sha256
    )
    expected_state = _require_sha256(
        "expected_state_digest", expected_state_digest
    )
    if _sha256_file(payload_file) != expected_payload:
        raise ValueError("Sealed annotation payload SHA-256 drifted.")
    if _sha256_file(fixture_file) != expected_fixture:
        raise ValueError("Sealed fixture SHA-256 drifted.")

    try:
        payload = json.loads(payload_file.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Sealed annotation payload is not valid UTF-8 JSON.") from error
    root = _require_exact_keys(
        payload,
        expected={"schema_version", "fixture", "annotation", "records"},
        name="Sealed annotation payload",
    )
    if (
        root["schema_version"]
        != "mrtotalvi-sealed-500-state-annotations-v1"
    ):
        raise ValueError("Sealed annotation payload schema drifted.")

    fixture = _require_exact_keys(
        root["fixture"],
        expected={
            "fixture_id",
            "h5ad_path",
            "h5ad_sha256",
            "ordered_cell_ids_sha256",
        },
        name="Sealed annotation fixture record",
    )
    if fixture["fixture_id"] != "sealed_500":
        raise ValueError("Sealed annotation fixture ID drifted.")
    recorded_fixture_path = fixture["h5ad_path"]
    if not isinstance(recorded_fixture_path, str):
        raise ValueError("Sealed annotation fixture path must be a string.")
    relative_fixture = PurePosixPath(recorded_fixture_path)
    if (
        relative_fixture.is_absolute()
        or recorded_fixture_path in {"", "."}
        or ".." in relative_fixture.parts
    ):
        raise ValueError("Sealed annotation fixture path must be repository-relative.")
    if fixture["h5ad_sha256"] != expected_fixture:
        raise ValueError("Sealed annotation fixture identity drifted.")

    if (
        not expected_cell_ids
        or len(expected_cell_ids) != len(set(expected_cell_ids))
        or any(
            not isinstance(cell_id, str)
            or not cell_id
            or "\n" in cell_id
            or "\r" in cell_id
            for cell_id in expected_cell_ids
        )
    ):
        raise ValueError("Expected sealed cell IDs must be non-empty and unique.")
    expected_cell_digest = sha256_lines(expected_cell_ids)
    if fixture["ordered_cell_ids_sha256"] != expected_cell_digest:
        raise ValueError("Sealed annotation ordered cell-ID digest drifted.")

    annotation = _require_exact_keys(
        root["annotation"],
        expected={
            "column",
            "source_h5ad_path",
            "source_h5ad_sha256",
            "ordered_labels_sha256",
            "ordered_selection_sha256",
        },
        name="Sealed annotation source record",
    )
    if annotation["column"] != "cell_label_l1p5":
        raise ValueError("Sealed annotation column drifted.")
    recorded_source_path = annotation["source_h5ad_path"]
    if (
        not isinstance(recorded_source_path, str)
        or not Path(recorded_source_path).is_absolute()
    ):
        raise ValueError("Sealed annotation source path must be absolute.")
    if annotation["source_h5ad_sha256"] != expected_source:
        raise ValueError("Sealed annotation source identity drifted.")

    records = root["records"]
    if not isinstance(records, list):
        raise ValueError("Sealed annotation records must be a JSON array.")
    observed_cell_ids: list[str] = []
    labels: list[str] = []
    for index, value in enumerate(records):
        record = _require_exact_keys(
            value,
            expected={"cell_id", "label"},
            name=f"Sealed annotation record {index}",
        )
        cell_id = record["cell_id"]
        label = record["label"]
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or "\n" in cell_id
            or "\r" in cell_id
            or not isinstance(label, str)
            or not label
            or "\n" in label
            or "\r" in label
        ):
            raise ValueError("Sealed annotation records require single-line strings.")
        observed_cell_ids.append(cell_id)
        labels.append(label)
    if tuple(observed_cell_ids) != expected_cell_ids:
        raise ValueError("Sealed annotation cell inventory or order drifted.")
    if len(observed_cell_ids) != len(set(observed_cell_ids)):
        raise ValueError("Sealed annotation cell IDs are not unique.")

    labels_digest = sha256_lines(labels)
    selection_digest = sha256_lines(
        tuple(
            f"{cell_id}\t{label}"
            for cell_id, label in zip(
                observed_cell_ids,
                labels,
                strict=True,
            )
        )
    )
    if annotation["ordered_labels_sha256"] != labels_digest:
        raise ValueError("Sealed annotation ordered-label digest drifted.")
    if annotation["ordered_selection_sha256"] != selection_digest:
        raise ValueError("Sealed annotation ordered-selection digest drifted.")
    if labels_digest != expected_state:
        raise ValueError("Sealed annotation state digest drifted.")
    return np.asarray(labels, dtype=str)


def _split_indices(
    *,
    cell_ids: np.ndarray,
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    split = make_within_sample_hash_split(
        cell_ids=tuple(cell_ids.astype(str)),
        samples=tuple(samples.astype(str)),
        train_fraction=0.8,
        salt=DIAGNOSIS_SPLIT_SALT,
    )
    assignments = np.asarray(split.assignments, dtype=str)
    train = np.flatnonzero(assignments == "train").astype(np.int64)
    validation = np.flatnonzero(assignments == "heldout").astype(np.int64)
    return train, validation, split.assignment_sha256


def _synthetic_fixture(fixture_id: str) -> PreparedDiagnosisFixture:
    scenario = "mixed" if fixture_id == "mixed" else "unequal_cells"
    config = ScenarioConfig(
        scenario=scenario,
        n_donors=3,
        cells_per_sample=48,
        n_states=3,
        n_genes=24,
        n_proteins=8,
        latent_truth_dim=4,
    )
    simulated = generate_scenario(config, seed=SYNTHETIC_FIXTURE_SEED)
    adata = simulated.adata
    samples = np.asarray(adata.obs["sample"].astype(str), dtype=str)
    train, validation, split_digest = _split_indices(
        cell_ids=np.asarray(adata.obs_names, dtype=str),
        samples=samples,
    )
    source_data_digest = _canonical_digest(
        {
            "fixture_id": fixture_id,
            "fixture_seed": SYNTHETIC_FIXTURE_SEED,
            "scenario": {
                "scenario": config.scenario,
                "n_donors": config.n_donors,
                "cells_per_sample": config.cells_per_sample,
                "n_states": config.n_states,
                "n_genes": config.n_genes,
                "n_proteins": config.n_proteins,
                "latent_truth_dim": config.latent_truth_dim,
            },
            "truth_seed": simulated.truth.truth_seed,
            "split_digest": split_digest,
        }
    )
    states = np.asarray(simulated.truth.cell_states)
    state_digest = _state_annotation_digest(states)
    data_digest = _canonical_digest(
        {
            "source_data_digest": source_data_digest,
            "state_annotation_id": "known_truth_cell_state",
            "state_annotation_digest": state_digest,
        }
    )
    return PreparedDiagnosisFixture(
        fixture_id=fixture_id,
        adata=adata,
        train_indices=train,
        validation_indices=validation,
        state_labels=states,
        state_annotation_id="known_truth_cell_state",
        sample_labels=np.asarray(
            simulated.truth.observed_sample_indices,
        ),
        technical_batch_labels=np.asarray(
            simulated.truth.sample_batches[
                simulated.truth.observed_sample_indices
            ],
        ),
        count_layer="counts",
        protein_obsm_key="protein_expression",
        protein_names_uns_key="protein_names",
        technical_batch_key="technical_batch",
        sample_key="sample",
        n_latent=4,
        n_hidden=32,
        n_layers=1,
        n_prior_components=8,
        batch_size=64,
        learning_rate=1e-3,
        evaluation_seed=int(simulated.truth.evaluation_seed),
        source_data_digest=source_data_digest,
        state_annotation_digest=state_digest,
        data_digest=data_digest,
        split_digest=split_digest,
    )


def _sealed_500_fixture(repo_root: Path) -> PreparedDiagnosisFixture:
    import anndata as ad

    path = repo_root / SEALED_500_H5AD
    if _sha256_file(path) != SEALED_500_SHA256:
        raise ValueError("Sealed 500-cell fixture SHA-256 drifted.")
    adata = ad.read_h5ad(path)
    if adata.shape != (500, 1000):
        raise ValueError("Sealed 500-cell fixture dimensions drifted.")
    if "pass_qc" in adata.obs:
        raise ValueError("Sealed engineering fixture unexpectedly contains pass_qc.")
    protein_names = np.asarray(
        adata.uns["protein_names_engineering"],
        dtype=str,
    )
    if len(protein_names) != adata.obsm["protein_expression"].shape[1]:
        raise ValueError("Sealed engineering protein names are not aligned.")
    adata.uns["protein_names"] = protein_names.copy()
    cell_ids = np.asarray(adata.obs_names, dtype=str)
    samples = np.asarray(adata.obs["donor_timepoint"].astype(str), dtype=str)
    train, validation, split_digest = _split_indices(
        cell_ids=cell_ids,
        samples=samples,
    )
    states = _load_sealed_500_state_annotations(
        repo_root / SEALED_500_STATE_ANNOTATIONS,
        fixture_path=path,
        expected_cell_ids=tuple(cell_ids),
        expected_payload_sha256=SEALED_500_STATE_ANNOTATIONS_SHA256,
        expected_fixture_sha256=SEALED_500_SHA256,
        expected_source_sha256=SEALED_500_ANNOTATION_SOURCE_SHA256,
        expected_state_digest=SEALED_500_STATE_ANNOTATION_SHA256,
    )
    state_digest = _state_annotation_digest(states)
    if state_digest != SEALED_500_STATE_ANNOTATION_SHA256:
        raise ValueError("Sealed 500-cell state annotations drifted.")
    data_digest = _canonical_digest(
        {
            "source_data_digest": SEALED_500_SHA256,
            "state_annotation_id": "cell_label_l1p5",
            "state_annotation_digest": state_digest,
            "state_annotation_source_recorded_sha256": (
                SEALED_500_ANNOTATION_SOURCE_SHA256
            ),
        }
    )
    return PreparedDiagnosisFixture(
        fixture_id="sealed_500",
        adata=adata,
        train_indices=train,
        validation_indices=validation,
        state_labels=states,
        state_annotation_id="cell_label_l1p5",
        sample_labels=samples,
        technical_batch_labels=np.asarray(
            adata.obs["batch"].astype(str),
            dtype=str,
        ),
        count_layer="counts",
        protein_obsm_key="protein_expression",
        protein_names_uns_key="protein_names",
        technical_batch_key="batch",
        sample_key="donor_timepoint",
        n_latent=20,
        n_hidden=128,
        n_layers=1,
        n_prior_components=20,
        batch_size=64,
        learning_rate=1e-3,
        evaluation_seed=EVALUATION_SEED,
        source_data_digest=SEALED_500_SHA256,
        state_annotation_digest=state_digest,
        data_digest=data_digest,
        split_digest=split_digest,
    )


def _canonical_human_fixture(repo_root: Path) -> PreparedDiagnosisFixture:
    import anndata as ad

    authority_path = repo_root / CANONICAL_HUMAN_H5AD
    read_path = _canonical_human_read_path(authority_path)
    adata = ad.read_h5ad(read_path)
    if adata.shape != (46_817, 5_000):
        raise ValueError("Canonical-human fixture dimensions drifted.")
    assignments = np.asarray(adata.obs["lineage_split"].astype(str), dtype=str)
    if set(assignments) != {"train", "heldout"}:
        raise ValueError("Canonical-human split labels drifted.")
    train = np.flatnonzero(assignments == "train").astype(np.int64)
    validation = np.flatnonzero(assignments == "heldout").astype(np.int64)
    lineage = adata.uns.get("mrtotalvi_human_lineage")
    if not isinstance(lineage, dict):
        raise ValueError("Canonical-human embedded lineage record is missing.")
    if lineage.get("run_id") != CANONICAL_HUMAN_RUN_ID:
        raise ValueError("Canonical-human embedded run ID drifted.")
    if (
        lineage.get("contract_sha256")
        != CANONICAL_HUMAN_CONTRACT_SHA256
    ):
        raise ValueError("Canonical-human embedded contract digest drifted.")
    if (
        lineage.get("factual_human_da")
        != "locked_not_computed_or_inspected"
    ):
        raise ValueError("Canonical-human factual DA lock drifted.")
    split_digest = lineage.get("split_assignment_sha256")
    if split_digest != CANONICAL_HUMAN_SPLIT_SHA256:
        raise ValueError("Canonical-human embedded split digest drifted.")
    if len(train) != 37_447 or len(validation) != 9_370:
        raise ValueError("Canonical-human train/held-out counts drifted.")
    states = np.asarray(
        adata.obs["cell_label_l2"].astype(str),
        dtype=str,
    )
    state_digest = _state_annotation_digest(states)
    data_digest = _canonical_digest(
        {
            "source_data_digest": CANONICAL_HUMAN_SHA256,
            "state_annotation_id": "cell_label_l2",
            "state_annotation_digest": state_digest,
        }
    )
    return PreparedDiagnosisFixture(
        fixture_id="canonical_human_if_available",
        adata=adata,
        train_indices=train,
        validation_indices=validation,
        state_labels=states,
        state_annotation_id="cell_label_l2",
        sample_labels=np.asarray(
            adata.obs["donor_timepoint"].astype(str),
            dtype=str,
        ),
        technical_batch_labels=np.asarray(
            adata.obs["batch"].astype(str),
            dtype=str,
        ),
        count_layer="counts",
        protein_obsm_key="protein_expression",
        protein_names_uns_key="protein_names",
        technical_batch_key="batch",
        sample_key="donor_timepoint",
        n_latent=20,
        n_hidden=128,
        n_layers=1,
        n_prior_components=20,
        batch_size=128,
        learning_rate=1e-3,
        evaluation_seed=HUMAN_EVALUATION_SEED,
        source_data_digest=CANONICAL_HUMAN_SHA256,
        state_annotation_digest=state_digest,
        data_digest=data_digest,
        split_digest=split_digest,
    )


def _validate_evaluation_labels(
    labels,
    *,
    validation_indices,
    name: str,
) -> np.ndarray:
    """Require every held-out class to support stratified cross-validation."""
    values = np.asarray(labels)
    validation = np.asarray(validation_indices)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if (
        validation.ndim != 1
        or not np.issubdtype(validation.dtype, np.integer)
        or np.any(validation < 0)
        or np.any(validation >= len(values))
    ):
        raise ValueError("validation_indices are invalid for evaluation labels.")
    unique, counts = np.unique(values[validation], return_counts=True)
    if len(unique) < 2:
        raise ValueError(f"{name} must contain at least two held-out classes.")
    if np.any(counts < 2):
        sparse = sorted(
            str(label)
            for label, count in zip(unique, counts, strict=True)
            if count < 2
        )
        raise ValueError(
            f"{name} classes require at least two held-out cells: {sparse}."
        )
    return values


def _validate_prepared_fixture(
    fixture: PreparedDiagnosisFixture,
) -> PreparedDiagnosisFixture:
    """Fail before fitting if any frozen evaluation boundary is unusable."""
    n_obs = fixture.adata.n_obs
    combined = np.concatenate(
        [fixture.train_indices, fixture.validation_indices]
    )
    if (
        len(combined) != n_obs
        or len(np.unique(combined)) != n_obs
        or not np.array_equal(
            np.sort(combined),
            np.arange(n_obs, dtype=np.int64),
        )
    ):
        raise ValueError(
            "Diagnosis train/validation indices must partition every cell."
        )
    for labels, name in (
        (fixture.state_labels, "state_labels"),
        (fixture.sample_labels, "sample_labels"),
        (fixture.technical_batch_labels, "technical_batch_labels"),
    ):
        if len(labels) != n_obs:
            raise ValueError(f"{name} must align with every fixture cell.")
        _validate_evaluation_labels(
            labels,
            validation_indices=fixture.validation_indices,
            name=name,
        )
    for name, digest in (
        ("source_data_digest", fixture.source_data_digest),
        ("state_annotation_digest", fixture.state_annotation_digest),
        ("data_digest", fixture.data_digest),
        ("split_digest", fixture.split_digest),
    ):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} must be a SHA-256 digest.")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(f"{name} must be hexadecimal.") from error
    if fixture.evaluation_seed in {0, 1, 2}:
        raise ValueError(
            "Evaluation seed must be distinct from every diagnosis training seed."
        )
    return fixture


def prepare_diagnosis_fixture(
    fixture_id: str,
    *,
    repo_root: str | Path,
) -> PreparedDiagnosisFixture:
    """Load or generate one exact preregistered diagnosis fixture."""
    root = Path(repo_root)
    if fixture_id in {"mixed", "unequal_cells"}:
        fixture = _synthetic_fixture(fixture_id)
    elif fixture_id == "sealed_500":
        fixture = _sealed_500_fixture(root)
    elif fixture_id == "canonical_human_if_available":
        fixture = _canonical_human_fixture(root)
    else:
        raise ValueError(f"Unknown diagnosis fixture {fixture_id!r}.")
    return _validate_prepared_fixture(fixture)


def _max_rss_bytes() -> int:
    multiplier = 1 if __import__("sys").platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier)


def _mrtotalvi_training_kwargs(
    fixture: PreparedDiagnosisFixture,
    *,
    callback: object,
) -> dict:
    """Return the exact shared TotalVI/MrTotalVI optimization control."""
    control = convergence_control()
    return {
        "max_epochs": control.maximum_epochs,
        "min_epochs": control.minimum_epochs,
        "accelerator": "cpu",
        "devices": 1,
        "train_size": None,
        "validation_size": None,
        "shuffle_set_split": False,
        "batch_size": fixture.batch_size,
        "early_stopping": True,
        "early_stopping_monitor": control.monitor,
        "early_stopping_mode": control.mode,
        "early_stopping_patience": control.patience_checks,
        "early_stopping_min_delta": control.min_delta,
        "external_indexing": [
            fixture.train_indices,
            fixture.validation_indices,
            np.asarray([], dtype=np.int64),
        ],
        "lr": fixture.learning_rate,
        "reduce_lr_on_plateau": False,
        **frozen_totalvi_optimization_kwargs(fixture.adata.n_obs),
        "check_val_every_n_epoch": control.check_every_epochs,
        "callbacks": [callback],
        "enable_checkpointing": True,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }


def _run_mrtotalvi_fit(
    fixture: PreparedDiagnosisFixture,
    *,
    candidate_id: str,
    training_seed: int,
    checkpoint_dir: Path,
    contract_adapter: RedesignContractAdapter,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    import scvi
    from scvi.external import MrTotalVI
    from scvi.train import SaveCheckpoint

    spec = diagnosis_fit_spec(candidate_id)
    if spec.model_family != "mrtotalvi" or spec.legacy_candidate is None:
        raise ValueError("MrTotalVI fit requires B2, B3, or D0.")
    control = spec.control
    scvi.settings.seed = training_seed
    MrTotalVI.setup_anndata(
        fixture.adata,
        layer=fixture.count_layer,
        protein_expression_obsm_key=fixture.protein_obsm_key,
        protein_names_uns_key=fixture.protein_names_uns_key,
        sample_key=fixture.sample_key,
        batch_key=fixture.technical_batch_key,
    )
    candidate = candidate_configs()[spec.legacy_candidate]
    model = MrTotalVI(
        fixture.adata,
        sample_key=fixture.sample_key,
        n_latent=fixture.n_latent,
        n_latent_u=fixture.n_latent,
        n_latent_sample=fixture.n_latent,
        z_u_prior=True,
        u_prior_mixture=True,
        u_prior_mixture_k=fixture.n_prior_components,
        use_map=True,
        hierarchy_mode=candidate.hierarchy_mode,
        u_encoder_mode=candidate.u_encoder_mode,
        scale_observations=candidate.scale_observations,
        u_prior=candidate.u_prior,
        init_prior_from_data=candidate.init_prior_from_data,
        freeze_prior_after_init=candidate.freeze_prior_after_init,
        use_batch_norm="none",
        use_layer_norm="both",
        encode_covariates=True,
        n_hidden=fixture.n_hidden,
        n_layers_encoder=fixture.n_layers,
        n_layers_decoder=fixture.n_layers,
        qu_kwargs={"n_hidden": fixture.n_hidden, "n_layers": fixture.n_layers},
        qz_kwargs={"n_hidden": fixture.n_hidden, "n_layers": fixture.n_layers},
    )
    callback = SaveCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:04d}-{elbo_validation:.8f}",
        monitor=control.monitor,
        mode=control.mode,
        save_top_k=1,
        load_best_on_end=control.restore_best_checkpoint,
    )
    started = time.perf_counter()
    rss_before = _max_rss_bytes()
    model.train(**_mrtotalvi_training_kwargs(fixture, callback=callback))
    wall_seconds = float(time.perf_counter() - started)
    history = serialize_training_history(model.history)
    checkpoint_path = Path(callback.best_model_path)
    identity = best_checkpoint_identity(
        history,
        monitor=control.monitor,
        mode=control.mode,
        state_digest=state_dict_digest(model.module.state_dict()),
        artifact_name=checkpoint_path.name,
    )
    optimization_identity = _realized_totalvi_optimization_identity(
        model,
        n_obs=fixture.adata.n_obs,
        learning_rate=fixture.learning_rate,
    )
    diagnostics = collect_mrtotalvi_diagnostics(
        model,
        candidate_id=candidate_id,
        validation_indices=fixture.validation_indices,
        gradient_indices=fixture.train_indices,
        batch_size=fixture.batch_size,
        posterior_samples=MRTOTALVI_FACTUAL_Z_POSTERIOR_DRAWS,
        posterior_predictive_draws=32,
        evaluation_seed=fixture.evaluation_seed,
        evaluation_annotations=fixture.evaluation_annotations(),
        training_history=model.history,
        checkpoint_identity=identity,
        checkpoint_path=checkpoint_path,
        wall_time_seconds=wall_seconds,
        peak_memory_bytes=max(0, _max_rss_bytes() - rss_before),
        contract_adapter=contract_adapter,
    )
    trainer_epochs = int(model.trainer.current_epoch)
    stopped_early = bool(
        model.trainer.should_stop
        and trainer_epochs < control.maximum_epochs
    )
    return {
        "trainer_epochs": trainer_epochs,
        "stopped_early": stopped_early,
        "optimization_identity": optimization_identity,
        **diagnostics,
    }, diagnostics["representations"]


def run_diagnosis_fit(
    fixture: PreparedDiagnosisFixture,
    *,
    candidate_id: str,
    training_seed: int,
    checkpoint_dir: str | Path,
    contract_adapter: RedesignContractAdapter,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Run one candidate-invariant convergence-controlled diagnosis fit."""
    if contract_adapter.integrity_version != "v2":
        raise ValueError("New diagnosis fits require the prospective v2 adapter.")
    spec = diagnosis_fit_spec(candidate_id)
    control = convergence_control()
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists():
        raise FileExistsError(
            f"Diagnosis checkpoint directory exists: {checkpoint_path}"
        )
    if spec.model_family == "totalvi":
        stock = run_stock_comparator(
            fixture.adata,
            ComparatorRunConfig(
                candidate_id="B1",
                train_indices=tuple(fixture.train_indices),
                validation_indices=tuple(fixture.validation_indices),
                max_epochs=control.maximum_epochs,
                n_latent=fixture.n_latent,
                n_hidden=fixture.n_hidden,
                n_layers=fixture.n_layers,
                batch_size=fixture.batch_size,
                learning_rate=fixture.learning_rate,
                technical_batch_key=fixture.technical_batch_key,
                training_seed=training_seed,
                evaluation_seed=fixture.evaluation_seed,
                check_val_every_n_epoch=control.check_every_epochs,
                minimum_epochs=control.minimum_epochs,
                early_stopping_patience=control.patience_checks,
                early_stopping_min_delta=control.min_delta,
                posterior_predictive_draws=32,
                explicit_totalvi_optimization=True,
            ),
            checkpoint_dir=checkpoint_path,
            evaluation_annotations=fixture.evaluation_annotations(),
            contract_adapter=contract_adapter,
        )
        fit = stock
        representations = stock["representations"]
    else:
        fit, representations = _run_mrtotalvi_fit(
            fixture,
            candidate_id=candidate_id,
            training_seed=training_seed,
            checkpoint_dir=checkpoint_path,
            contract_adapter=contract_adapter,
        )

    convergence = assess_convergence(
        fit["training_history"]["elbo_validation"],
        trainer_epochs=fit["trainer_epochs"],
        stopped_early=fit["stopped_early"],
    )
    latent_integrity = assess_latent_integrity_v2(
        fit["metrics"],
        candidate_id=candidate_id,
        latent_dimension=fixture.n_latent,
    )
    bindings = version_binding_fields(contract_adapter)
    latent_integrity = {**latent_integrity, **bindings}
    result = {
        "schema_version": "mrtotalvi-convergence-fit-v3",
        **bindings,
        "candidate_id": candidate_id,
        "fixture_id": fixture.fixture_id,
        "training_seed": training_seed,
        "evaluation_seed": fixture.evaluation_seed,
        "data_digest": fixture.data_digest,
        "source_data_digest": fixture.source_data_digest,
        "state_annotation_digest": fixture.state_annotation_digest,
        "split_digest": fixture.split_digest,
        "n_cells": fixture.adata.n_obs,
        "n_genes": fixture.adata.n_vars,
        "n_proteins": fixture.adata.obsm[
            fixture.protein_obsm_key
        ].shape[1],
        "n_train_cells": len(fixture.train_indices),
        "n_validation_cells": len(fixture.validation_indices),
        "latent_dimension": fixture.n_latent,
        "learning_rate": fixture.learning_rate,
        "control": control.to_dict(),
        "model_spec": {
            "model_family": spec.model_family,
            "legacy_candidate": spec.legacy_candidate,
            "technical_batch_key": fixture.technical_batch_key,
            "sample_key": (
                fixture.sample_key
                if spec.model_family == "mrtotalvi"
                else None
            ),
            "state_annotation_id": fixture.state_annotation_id,
        },
        "convergence": convergence,
        "latent_integrity": latent_integrity,
        "training_history": fit["training_history"],
        "best_checkpoint_identity": fit["best_checkpoint_identity"],
        "optimization_identity": fit["optimization_identity"],
        "metrics": fit["metrics"],
        "metric_no_call_reasons": fit.get(
            "metric_no_call_reasons",
            {},
        ),
        "scientific_scope": (
            "RDX-03 convergence diagnosis; no candidate selection; "
            "no factual human DA"
        ),
    }
    return result, representations
