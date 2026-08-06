"""Fail-closed human-cohort lineage utilities for the MrTotalVI redesign."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


HUMAN_TIMEPOINTS = ("W00", "W22")
SHARED_PARENT_METADATA = ("batch", "donor", "species", "timepoint")
COVARIATE_LEVEL_COLUMNS = (
    "batch",
    "donor",
    "donor_timepoint",
    "species",
    "timepoint",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_RUN_ID_PATTERN = re.compile(
    r"^\d{8}T\d{6}Z-[0-9a-f]{8}-[0-9a-f]{8}-[0-9a-f]{8}$"
)
_BASE_COMMIT = "d8c8e997a67997a53f55923eb3ab14e6cf06f94c"
_SOURCE_SHA256 = {
    "harmonized_joint": (
        "520ff544daae6192efd7f3501669e05b0122e6fbaf8e9de0246122cecd1de2da"
    ),
    "final_parent": (
        "33fb1dc456df30244daecac05ab836f7b15db9a127eac3dac4191a85e0acc59a"
    ),
    "source_registry": (
        "7514d25bf171ec2489677a649b6a6314ce3a432dc3c42d51fad40bcb572dbca4"
    ),
    "protein_panel": (
        "90a467f72abe9b347784ceae0eeb1e4fb14476bbe65a3e736f77785f95cd4570"
    ),
    "migration_inventory": (
        "ff40f444eb6c18396156d8c15a5f842a61b5bd1e80f0a522734d40130f6a145d"
    ),
    "selected_run_state": (
        "52da8f8993f6ea86e73a528b8af3e3addc847a6e81cca7d733a39b997038ad99"
    ),
}
_MIGRATION_INVENTORY_SUPERSESSION = {
    "contract_schema_version": "mrtotalvi-human-lineage-contract-v1",
    "changed_source_role": "migration_inventory",
    "prior_sha256": (
        "ff40f444eb6c18396156d8c15a5f842a61b5bd1e80f0a522734d40130f6a145d"
    ),
    "replacement_sha256": (
        "a321d71d15c7f256de22f32dd1d8742b9ca4df64c8dd0a3f5bbcee2cdb418481"
    ),
    "authority_role": "migration_inventory_supersession",
    "authority_correction_id": (
        "S06E-s01-baseline-inventory-rebinding-completion"
    ),
}
_SOURCE_SHA256_V2 = {
    **_SOURCE_SHA256,
    "migration_inventory": _MIGRATION_INVENTORY_SUPERSESSION[
        "replacement_sha256"
    ],
    "migration_inventory_supersession": (
        "3817dbe8b683af9aa0dceda2ed4daf791cc4e79ba433ac9d5fad3ea4458afcc7"
    ),
}
_REQUIRED_METADATA = (
    "L1",
    "L1.5",
    "L2",
    "L3",
    "batch",
    "cell_label_l1",
    "cell_label_l1p5",
    "cell_label_l2",
    "cell_label_l3",
    "compartment",
    "disposition",
    "donor",
    "donor_timepoint",
    "species",
    "timepoint",
)
_DIGEST_NAMES = (
    "ordered_cells",
    "split_assignments",
    "ordered_training_cells",
    "ordered_heldout_cells",
    "ordered_source_genes",
    "ordered_selected_genes",
    "ordered_source_proteins",
    "ordered_selected_proteins",
    "selected_rna_counts",
    "selected_protein_counts",
    "shared_parent_metadata",
)


def sha256_lines(values: Sequence[str]) -> str:
    """Hash an ordered string sequence using the repository newline contract."""
    normalized = []
    for value in values:
        if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
            raise ValueError("Hash inputs must be non-empty single-line strings.")
        normalized.append(value)
    payload = "".join(f"{value}\n" for value in normalized).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def human_lineage_contract() -> dict:
    """Return the exact RDX-01 derivation contract."""
    return {
        "schema_version": "mrtotalvi-human-lineage-contract-v1",
        "source_sha256": dict(_SOURCE_SHA256),
        "expected": {
            "cells": 46_817,
            "complete_donors": 10,
            "samples": 20,
            "joint_cells": 86_681,
            "parent_cells": 126_299,
            "rna_features": 36_601,
            "source_proteins": 137,
            "selected_genes": 5_000,
            "selected_proteins": 130,
            "ordered_cell_sha256": (
                "dc93b2b4a78c8a3fff5e6aaee1627c81e56bff0f5364bebd1d930bba658e920c"
            ),
            "ordered_protein_sha256": (
                "09bbe2e870a7005d411b9557711266e7e8d4ac3d3a5e2804ddd658467873418b"
            ),
            "timepoints": list(HUMAN_TIMEPOINTS),
        },
        "cell_universe": {
            "order": "harmonized_joint",
            "parent_join": "exact_cell_id",
            "qc_field": "final_parent.obs.pass_qc",
            "retain": "W00_W22_complete_donors",
            "missing_joint_parent_id": "blocked",
            "fallback_universe": "forbidden",
        },
        "split": {
            "group_key": "donor_timepoint",
            "method": "sha256_rank_within_group",
            "salt": "mrtotalvi-v2-redesign-human-lineage-v1",
            "train_fraction": 0.8,
        },
        "hvg": {
            "batch_key": None,
            "chunk_size": 128,
            "clip": "sqrt_training_cells",
            "flavor": "pearson_residuals",
            "n_top_genes": 5_000,
            "theta": 100.0,
            "training_cells_only": True,
        },
        "protein": {
            "control_field": "is_control",
            "order": "panel_and_source_order",
            "retain": "non_isotype",
        },
        "shared_parent_metadata": list(SHARED_PARENT_METADATA),
        "covariate_level_columns": list(COVARIATE_LEVEL_COLUMNS),
        "required_metadata": list(_REQUIRED_METADATA),
        "count_verification": (
            "joint_parent_exact_on_selected_cells_all_36601_rna_and_137_proteins"
        ),
        "output": {
            "format": "h5ad",
            "immutable_new_directory": True,
            "latest_pointer": False,
        },
        "factual_human_da": "locked_not_computed_or_inspected",
    }


def human_lineage_contract_digest() -> str:
    """Hash the canonical RDX-01 contract."""
    return _canonical_json_digest(human_lineage_contract())


def human_lineage_contract_v2() -> dict:
    """Return the RDX-01 contract with the reviewed source supersession."""
    contract = human_lineage_contract()
    contract["schema_version"] = "mrtotalvi-human-lineage-contract-v2"
    contract["source_sha256"] = dict(_SOURCE_SHA256_V2)
    contract["supersedes"] = dict(_MIGRATION_INVENTORY_SUPERSESSION)
    return contract


def human_lineage_contract_digest_v2() -> str:
    """Hash the canonical reviewed-supersession RDX-01 contract."""
    return _canonical_json_digest(human_lineage_contract_v2())


def sha256_csr_matrix(counts: sp.spmatrix | np.ndarray) -> str:
    """Hash a sparse integer count matrix in canonical logical form."""
    matrix = _validate_count_matrix(counts)
    canonical = sp.csr_matrix(
        (
            np.rint(matrix.data).astype("<i8", copy=False),
            matrix.indices.astype("<i8", copy=False),
            matrix.indptr.astype("<i8", copy=False),
        ),
        shape=matrix.shape,
    )
    canonical.sum_duplicates()
    canonical.eliminate_zeros()
    canonical.sort_indices()
    digest = hashlib.sha256(b"mrtotalvi-csr-counts-v1\0")
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.indptr.astype("<i8", copy=False).tobytes())
    digest.update(canonical.indices.astype("<i8", copy=False).tobytes())
    digest.update(canonical.data.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def verify_shared_metadata(
    *,
    cell_ids: Sequence[str],
    joint_metadata: Mapping[str, Sequence[str]],
    parent_metadata: Mapping[str, Sequence[str]],
) -> str:
    """Require exact joint/parent shared metadata after cell-ID alignment."""
    ids = _string_tuple("cell_ids", cell_ids)
    _unique("cell_ids", ids)
    lines = []
    for field in SHARED_PARENT_METADATA:
        if field not in joint_metadata or field not in parent_metadata:
            raise ValueError(f"Missing shared metadata field {field!r}.")
        joint_values = _string_tuple(
            f"joint_metadata[{field!r}]",
            joint_metadata[field],
            expected=len(ids),
        )
        parent_values = _string_tuple(
            f"parent_metadata[{field!r}]",
            parent_metadata[field],
            expected=len(ids),
        )
        if joint_values != parent_values:
            mismatches = sum(
                left != right
                for left, right in zip(
                    joint_values,
                    parent_values,
                    strict=True,
                )
            )
            raise ValueError(
                "Joint/parent shared metadata disagree for "
                f"{field!r} on {mismatches} selected cells."
            )
    for position, cell_id in enumerate(ids):
        lines.append(
            "\t".join(
                (
                    cell_id,
                    *(
                        f"{field}={joint_metadata[field][position]}"
                        for field in SHARED_PARENT_METADATA
                    ),
                )
            )
        )
    return sha256_lines(lines)


def covariate_levels(
    metadata: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """Return the sorted unique registered biological/technical levels."""
    levels = {}
    lengths = set()
    for column in COVARIATE_LEVEL_COLUMNS:
        if column not in metadata:
            raise ValueError(f"Missing covariate-level column {column!r}.")
        values = _string_tuple(f"metadata[{column!r}]", metadata[column])
        lengths.add(len(values))
        levels[column] = tuple(sorted(set(values)))
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Covariate columns must be non-empty and cell-aligned.")
    return levels


def _string_tuple(
    name: str,
    values: Sequence[str],
    *,
    expected: int | None = None,
) -> tuple[str, ...]:
    normalized = tuple(values)
    if expected is not None and len(normalized) != expected:
        raise ValueError(
            f"{name} must contain {expected} values; found {len(normalized)}."
        )
    if any(
        not isinstance(value, str)
        or not value
        or "\n" in value
        or "\r" in value
        for value in normalized
    ):
        raise ValueError(f"{name} must contain non-empty single-line strings.")
    return normalized


def _unique(name: str, values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique.")


@dataclass(frozen=True)
class HumanCellUniverse:
    """Ordered harmonized cells after final-parent QC and complete-pair gates."""

    cell_ids: tuple[str, ...]
    joint_positions: tuple[int, ...]
    parent_positions: tuple[int, ...]
    donors: tuple[str, ...]
    timepoints: tuple[str, ...]
    samples: tuple[str, ...]
    complete_donors: tuple[str, ...]
    ordered_cell_sha256: str


@dataclass(frozen=True)
class HumanHashSplit:
    """Deterministic train/held-out assignments aligned to cohort order."""

    cell_ids: tuple[str, ...]
    samples: tuple[str, ...]
    assignments: tuple[str, ...]
    scores: tuple[str, ...]
    train_fraction: float
    salt: str
    assignment_sha256: str


@dataclass(frozen=True)
class PearsonResidualHVGSelection:
    """Training-only Pearson-residual variance ranking."""

    gene_names: tuple[str, ...]
    selected_gene_names: tuple[str, ...]
    selected_positions: tuple[int, ...]
    ranking_positions: tuple[int, ...]
    residual_variances: tuple[float, ...]
    training_cell_count: int
    theta: float
    clip: float
    ordered_gene_sha256: str


@dataclass(frozen=True)
class BiologicalProteinSelection:
    """Ordered non-control proteins shared by panel and both sources."""

    names: tuple[str, ...]
    positions: tuple[int, ...]
    ordered_protein_sha256: str


def derive_human_cell_universe(
    *,
    joint_cell_ids: Sequence[str],
    joint_donors: Sequence[str],
    joint_timepoints: Sequence[str],
    joint_samples: Sequence[str],
    parent_cell_ids: Sequence[str],
    parent_pass_qc: Sequence[bool],
    expected_cells: int,
    expected_complete_donors: int,
    expected_ordered_cell_sha256: str | None = None,
) -> HumanCellUniverse:
    """Derive the exact human universe while preserving harmonized order."""
    if (
        isinstance(expected_cells, bool)
        or not isinstance(expected_cells, int)
        or expected_cells <= 0
    ):
        raise ValueError("expected_cells must be a positive integer.")
    if (
        isinstance(expected_complete_donors, bool)
        or not isinstance(expected_complete_donors, int)
        or expected_complete_donors <= 0
    ):
        raise ValueError("expected_complete_donors must be a positive integer.")

    cell_ids = _string_tuple("joint_cell_ids", joint_cell_ids)
    _unique("joint_cell_ids", cell_ids)
    n_joint = len(cell_ids)
    donors = _string_tuple("joint_donors", joint_donors, expected=n_joint)
    timepoints = _string_tuple(
        "joint_timepoints", joint_timepoints, expected=n_joint
    )
    samples = _string_tuple("joint_samples", joint_samples, expected=n_joint)

    parent_ids = _string_tuple("parent_cell_ids", parent_cell_ids)
    _unique("parent_cell_ids", parent_ids)
    parent_qc = tuple(parent_pass_qc)
    if len(parent_qc) != len(parent_ids):
        raise ValueError(
            "parent_pass_qc must have the same length as parent_cell_ids."
        )
    if any(not isinstance(value, (bool, np.bool_)) for value in parent_qc):
        raise ValueError("parent_pass_qc must contain booleans.")

    parent_lookup = {cell_id: position for position, cell_id in enumerate(parent_ids)}
    missing_from_parent = [cell_id for cell_id in cell_ids if cell_id not in parent_lookup]
    if missing_from_parent:
        raise ValueError(
            "Every harmonized cell must exist in the final parent; missing "
            f"{len(missing_from_parent)} cells."
        )

    eligible = [
        position
        for position, (cell_id, timepoint) in enumerate(
            zip(cell_ids, timepoints, strict=True)
        )
        if timepoint in HUMAN_TIMEPOINTS
        and parent_qc[parent_lookup[cell_id]]
    ]
    paired_timepoints: dict[str, set[str]] = {}
    for position in eligible:
        paired_timepoints.setdefault(donors[position], set()).add(
            timepoints[position]
        )
    complete_donors = tuple(
        sorted(
            donor
            for donor, observed in paired_timepoints.items()
            if observed == set(HUMAN_TIMEPOINTS)
        )
    )
    if len(complete_donors) != expected_complete_donors:
        raise ValueError(
            "Complete-donor count mismatch: expected "
            f"{expected_complete_donors}, found {len(complete_donors)}."
        )
    complete_set = set(complete_donors)
    selected = tuple(
        position for position in eligible if donors[position] in complete_set
    )
    if len(selected) != expected_cells:
        raise ValueError(
            f"Cell count mismatch: expected {expected_cells}, found {len(selected)}."
        )

    sample_to_pair: dict[str, tuple[str, str]] = {}
    pair_to_sample: dict[tuple[str, str], str] = {}
    for position in selected:
        pair = (donors[position], timepoints[position])
        sample = samples[position]
        if sample in sample_to_pair and sample_to_pair[sample] != pair:
            raise ValueError(
                f"Sample {sample!r} maps to more than one donor-timepoint pair."
            )
        if pair in pair_to_sample and pair_to_sample[pair] != sample:
            raise ValueError(
                f"Donor-timepoint pair {pair!r} maps to multiple samples."
            )
        sample_to_pair[sample] = pair
        pair_to_sample[pair] = sample
    expected_pairs = {
        (donor, timepoint)
        for donor in complete_donors
        for timepoint in HUMAN_TIMEPOINTS
    }
    if set(pair_to_sample) != expected_pairs:
        raise ValueError("Selected cells do not cover every complete donor pair.")

    selected_ids = tuple(cell_ids[position] for position in selected)
    ordered_digest = sha256_lines(selected_ids)
    if (
        expected_ordered_cell_sha256 is not None
        and ordered_digest != expected_ordered_cell_sha256
    ):
        raise ValueError(
            "Ordered-cell SHA-256 mismatch: expected "
            f"{expected_ordered_cell_sha256}, found {ordered_digest}."
        )

    return HumanCellUniverse(
        cell_ids=selected_ids,
        joint_positions=selected,
        parent_positions=tuple(
            parent_lookup[cell_id] for cell_id in selected_ids
        ),
        donors=tuple(donors[position] for position in selected),
        timepoints=tuple(timepoints[position] for position in selected),
        samples=tuple(samples[position] for position in selected),
        complete_donors=complete_donors,
        ordered_cell_sha256=ordered_digest,
    )


def make_within_sample_hash_split(
    *,
    cell_ids: Sequence[str],
    samples: Sequence[str],
    train_fraction: float,
    salt: str,
) -> HumanHashSplit:
    """Assign a fixed fraction within every sample by stable SHA-256 rank."""
    ids = _string_tuple("cell_ids", cell_ids)
    _unique("cell_ids", ids)
    sample_ids = _string_tuple("samples", samples, expected=len(ids))
    if (
        isinstance(train_fraction, bool)
        or not isinstance(train_fraction, (int, float))
        or not math.isfinite(train_fraction)
        or not 0.0 < train_fraction < 1.0
    ):
        raise ValueError("train_fraction must be finite and inside (0, 1).")
    if not isinstance(salt, str) or not salt or "\n" in salt or "\r" in salt:
        raise ValueError("salt must be a non-empty single-line string.")

    scores = tuple(
        hashlib.sha256(
            f"{salt}\0{sample}\0{cell_id}".encode()
        ).hexdigest()
        for cell_id, sample in zip(ids, sample_ids, strict=True)
    )
    by_sample: dict[str, list[int]] = {}
    for position, sample in enumerate(sample_ids):
        by_sample.setdefault(sample, []).append(position)

    assignments = ["heldout"] * len(ids)
    for sample, positions in by_sample.items():
        if len(positions) < 2:
            raise ValueError(
                f"Sample {sample!r} needs at least two cells for a split."
            )
        ordered = sorted(
            positions,
            key=lambda position: (scores[position], ids[position]),
        )
        n_train = math.floor(len(ordered) * float(train_fraction))
        n_train = min(max(n_train, 1), len(ordered) - 1)
        for position in ordered[:n_train]:
            assignments[position] = "train"

    canonical_lines = sorted(
        f"{sample}\t{cell_id}\t{assignment}\t{score}"
        for cell_id, sample, assignment, score in zip(
            ids,
            sample_ids,
            assignments,
            scores,
            strict=True,
        )
    )
    return HumanHashSplit(
        cell_ids=ids,
        samples=sample_ids,
        assignments=tuple(assignments),
        scores=scores,
        train_fraction=float(train_fraction),
        salt=salt,
        assignment_sha256=sha256_lines(canonical_lines),
    )


def _validate_count_matrix(
    counts: sp.spmatrix | np.ndarray,
) -> sp.csr_matrix:
    if sp.issparse(counts):
        matrix = counts.tocsr(copy=False)
    else:
        array = np.asarray(counts)
        if array.ndim != 2:
            raise ValueError("counts must be a two-dimensional matrix.")
        matrix = sp.csr_matrix(array)
    values = matrix.data
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("counts must be finite and nonnegative.")
    if not np.array_equal(values, np.rint(values)):
        raise ValueError("counts must be integer-valued.")
    return matrix


def select_pearson_residual_hvgs(
    *,
    counts: sp.spmatrix | np.ndarray,
    gene_names: Sequence[str],
    training_mask: Sequence[bool],
    n_top_genes: int,
    theta: float = 100.0,
    clip: float | None = None,
    chunk_size: int = 128,
) -> PearsonResidualHVGSelection:
    """Rank genes by clipped Pearson-residual variance on training cells."""
    matrix = _validate_count_matrix(counts)
    genes = _string_tuple(
        "gene_names",
        gene_names,
        expected=matrix.shape[1],
    )
    _unique("gene_names", genes)
    mask_values = tuple(training_mask)
    if len(mask_values) != matrix.shape[0] or any(
        not isinstance(value, (bool, np.bool_)) for value in mask_values
    ):
        raise ValueError(
            "training_mask must be a boolean vector aligned to count rows."
        )
    mask = np.asarray(mask_values, dtype=bool)
    n_training = int(mask.sum())
    if n_training < 2:
        raise ValueError("At least two training cells are required.")
    if (
        isinstance(n_top_genes, bool)
        or not isinstance(n_top_genes, int)
        or not 0 < n_top_genes <= matrix.shape[1]
    ):
        raise ValueError("n_top_genes must be inside [1, n_genes].")
    if (
        isinstance(theta, bool)
        or not isinstance(theta, (int, float))
        or not math.isfinite(theta)
        or theta <= 0
    ):
        raise ValueError("theta must be finite and positive.")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")
    effective_clip = math.sqrt(n_training) if clip is None else clip
    if (
        isinstance(effective_clip, bool)
        or not isinstance(effective_clip, (int, float))
        or not math.isfinite(effective_clip)
        or effective_clip < 0
    ):
        raise ValueError("clip must be finite and nonnegative.")

    training = matrix[mask].tocsr()
    cell_sums = np.asarray(training.sum(axis=1), dtype=np.float64).ravel()
    if np.any(cell_sums <= 0):
        raise ValueError("Every training cell must have a positive library size.")
    total = float(cell_sums.sum())
    residual_variances = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, matrix.shape[1], chunk_size):
        stop = min(start + chunk_size, matrix.shape[1])
        dense = training[:, start:stop].toarray().astype(
            np.float64,
            copy=False,
        )
        gene_sums = dense.sum(axis=0, dtype=np.float64)
        nonzero = gene_sums > 0
        if not np.any(nonzero):
            continue
        expected = np.outer(
            cell_sums,
            gene_sums[nonzero],
        )
        expected /= total
        denominator = np.sqrt(expected + expected**2 / float(theta))
        residuals = (dense[:, nonzero] - expected) / denominator
        np.clip(
            residuals,
            -float(effective_clip),
            float(effective_clip),
            out=residuals,
        )
        block_variances = np.zeros(stop - start, dtype=np.float64)
        block_variances[nonzero] = residuals.var(axis=0, ddof=0)
        residual_variances[start:stop] = block_variances

    positions = np.arange(matrix.shape[1], dtype=np.int64)
    ranking = np.lexsort((positions, -residual_variances))
    selected = ranking[:n_top_genes]
    selected_names = tuple(genes[position] for position in selected)
    return PearsonResidualHVGSelection(
        gene_names=genes,
        selected_gene_names=selected_names,
        selected_positions=tuple(int(position) for position in selected),
        ranking_positions=tuple(int(position) for position in ranking),
        residual_variances=tuple(
            float(value) for value in residual_variances
        ),
        training_cell_count=n_training,
        theta=float(theta),
        clip=float(effective_clip),
        ordered_gene_sha256=sha256_lines(selected_names),
    )


def select_biological_proteins(
    *,
    panel_features: Sequence[str],
    panel_is_control: Sequence[bool],
    joint_protein_names: Sequence[str],
    parent_protein_names: Sequence[str],
    expected_count: int,
    expected_ordered_protein_sha256: str | None = None,
) -> BiologicalProteinSelection:
    """Freeze the panel-ordered non-control proteins shared by both sources."""
    features = _string_tuple("panel_features", panel_features)
    _unique("panel_features", features)
    control_values = tuple(panel_is_control)
    if len(control_values) != len(features) or any(
        not isinstance(value, (bool, np.bool_)) for value in control_values
    ):
        raise ValueError(
            "panel_is_control must be a boolean vector aligned to features."
        )
    joint_names = _string_tuple(
        "joint_protein_names",
        joint_protein_names,
        expected=len(features),
    )
    parent_names = _string_tuple(
        "parent_protein_names",
        parent_protein_names,
        expected=len(features),
    )
    if joint_names != features or parent_names != features:
        raise ValueError(
            "Panel, joint, and parent source protein orders must match exactly."
        )
    selected_positions = tuple(
        position
        for position, is_control in enumerate(control_values)
        if not is_control
    )
    if len(selected_positions) != expected_count:
        raise ValueError(
            f"Protein count mismatch: expected {expected_count}, "
            f"found {len(selected_positions)}."
        )
    names = tuple(features[position] for position in selected_positions)
    digest = sha256_lines(names)
    if (
        expected_ordered_protein_sha256 is not None
        and digest != expected_ordered_protein_sha256
    ):
        raise ValueError(
            "Ordered-protein SHA-256 mismatch: expected "
            f"{expected_ordered_protein_sha256}, found {digest}."
        )
    return BiologicalProteinSelection(
        names=names,
        positions=selected_positions,
        ordered_protein_sha256=digest,
    )


def _require_exact_keys(
    name: str,
    payload: dict,
    expected: set[str],
) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a mapping.")
    missing = expected - set(payload)
    unknown = set(payload) - expected
    if missing:
        raise ValueError(f"{name} fields missing: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")


def _validate_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


def _validate_source_file_records(
    records: object,
    *,
    expected_source_sha256: dict[str, str],
) -> None:
    if not isinstance(records, list):
        raise ValueError("source_files must be a list.")
    roles = []
    for record in records:
        _require_exact_keys(
            "source file record",
            record,
            {"role", "path", "sha256", "bytes"},
        )
        role = record["role"]
        roles.append(role)
        if role not in expected_source_sha256:
            raise ValueError(f"Unknown source file role {role!r}.")
        path = record["path"]
        if not isinstance(path, str) or not PurePosixPath(path).is_absolute():
            raise ValueError("Source file paths must be absolute.")
        _validate_sha256("source file sha256", record["sha256"])
        if record["sha256"] != expected_source_sha256[role]:
            raise ValueError(f"Source hash drift for role {role!r}.")
        if (
            isinstance(record["bytes"], bool)
            or not isinstance(record["bytes"], int)
            or record["bytes"] <= 0
        ):
            raise ValueError("Source file bytes must be a positive integer.")
    if roles != list(expected_source_sha256):
        raise ValueError(
            "source_files must contain the exact ordered source roles."
        )


def _validate_code_file_records(records: object) -> None:
    if not isinstance(records, list):
        raise ValueError("code_files must be a list.")
    expected_paths = [
        "benchmarks/mrtotalvi/human_lineage.py",
        "benchmarks/mrtotalvi/freeze_human_lineage.py",
    ]
    observed_paths = []
    for record in records:
        _require_exact_keys(
            "code file record",
            record,
            {"path", "sha256", "bytes"},
        )
        path = record["path"]
        if (
            not isinstance(path, str)
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
        ):
            raise ValueError("Code file paths must be safe and relative.")
        observed_paths.append(path)
        _validate_sha256("code file sha256", record["sha256"])
        if (
            isinstance(record["bytes"], bool)
            or not isinstance(record["bytes"], int)
            or record["bytes"] <= 0
        ):
            raise ValueError("Code file bytes must be a positive integer.")
    if observed_paths != expected_paths:
        raise ValueError("code_files must contain the exact derivation files.")


def validate_human_lineage_manifest(payload: dict) -> dict:
    """Validate a complete RDX-01 domain manifest without schema drift."""
    expected_fields = {
        "schema_version",
        "run_id",
        "created_at",
        "status",
        "base_commit",
        "source_repository_commit",
        "contract",
        "source_files",
        "code_files",
        "source_dimensions",
        "selected_dimensions",
        "covariate_levels",
        "digests",
        "count_verification",
        "shared_metadata_verification",
        "factual_human_da",
    }
    _require_exact_keys("human-lineage manifest", payload, expected_fields)
    schema_version = payload["schema_version"]
    if schema_version == "mrtotalvi-human-lineage-v1":
        contract = human_lineage_contract()
    elif schema_version == "mrtotalvi-human-lineage-v2":
        contract = human_lineage_contract_v2()
    else:
        raise ValueError("Unsupported human-lineage manifest schema.")
    if (
        not isinstance(payload["run_id"], str)
        or not _RUN_ID_PATTERN.fullmatch(payload["run_id"])
    ):
        raise ValueError("Human-lineage run_id has an invalid format.")
    try:
        created_at = datetime.fromisoformat(payload["created_at"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Human-lineage created_at must be ISO-8601."
        ) from error
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("Human-lineage created_at must be timezone-aware.")
    if payload["status"] != "complete":
        raise ValueError("A derived human-lineage manifest must be complete.")
    if payload["base_commit"] != _BASE_COMMIT:
        raise ValueError("Human-lineage base commit drifted.")
    if (
        not isinstance(payload["source_repository_commit"], str)
        or not _COMMIT_PATTERN.fullmatch(
            payload["source_repository_commit"]
        )
    ):
        raise ValueError(
            "source_repository_commit must be a lowercase Git digest."
        )
    if payload["contract"] != contract:
        raise ValueError("Human-lineage contract does not match the frozen contract.")
    _validate_source_file_records(
        payload["source_files"],
        expected_source_sha256=contract["source_sha256"],
    )
    _validate_code_file_records(payload["code_files"])

    source_dimensions = payload["source_dimensions"]
    expected_source_dimensions = {
        "joint_cells": 86_681,
        "parent_cells": 126_299,
        "rna_features": 36_601,
        "source_proteins": 137,
    }
    _require_exact_keys(
        "source_dimensions",
        source_dimensions,
        set(expected_source_dimensions),
    )
    if source_dimensions != expected_source_dimensions:
        raise ValueError("Human-lineage source dimensions drifted.")

    selected_dimensions = payload["selected_dimensions"]
    expected_selected_keys = {
        "cells",
        "complete_donors",
        "samples",
        "training_cells",
        "heldout_cells",
        "genes",
        "proteins",
    }
    _require_exact_keys(
        "selected_dimensions",
        selected_dimensions,
        expected_selected_keys,
    )
    fixed_selected = {
        "cells": 46_817,
        "complete_donors": 10,
        "samples": 20,
        "genes": 5_000,
        "proteins": 130,
    }
    if any(
        selected_dimensions[name] != value
        for name, value in fixed_selected.items()
    ):
        raise ValueError("Human-lineage selected dimensions drifted.")
    training_cells = selected_dimensions["training_cells"]
    heldout_cells = selected_dimensions["heldout_cells"]
    if (
        isinstance(training_cells, bool)
        or not isinstance(training_cells, int)
        or isinstance(heldout_cells, bool)
        or not isinstance(heldout_cells, int)
        or training_cells <= 0
        or heldout_cells <= 0
        or training_cells + heldout_cells != 46_817
    ):
        raise ValueError("Training and held-out cell counts are inconsistent.")

    levels = payload["covariate_levels"]
    _require_exact_keys(
        "covariate_levels",
        levels,
        set(COVARIATE_LEVEL_COLUMNS),
    )
    for column in COVARIATE_LEVEL_COLUMNS:
        values = levels[column]
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, str)
                or not value
                or "\n" in value
                or "\r" in value
                for value in values
            )
            or values != sorted(set(values))
        ):
            raise ValueError(
                f"Covariate levels for {column!r} must be sorted unique strings."
            )
    if (
        len(levels["donor"]) != 10
        or len(levels["donor_timepoint"]) != 20
        or levels["species"] != ["human"]
        or levels["timepoint"] != list(HUMAN_TIMEPOINTS)
    ):
        raise ValueError("Human-lineage covariate levels drifted.")

    digests = payload["digests"]
    _require_exact_keys("digests", digests, set(_DIGEST_NAMES))
    for name in _DIGEST_NAMES:
        _validate_sha256(name, digests[name])
    expected = contract["expected"]
    if digests["ordered_cells"] != expected["ordered_cell_sha256"]:
        raise ValueError("Ordered-cell digest drifted.")
    if (
        digests["ordered_selected_proteins"]
        != expected["ordered_protein_sha256"]
    ):
        raise ValueError("Ordered-protein digest drifted.")
    if payload["count_verification"] != contract["count_verification"]:
        raise ValueError("Count-verification claim drifted.")
    if payload["shared_metadata_verification"] != (
        "joint_parent_exact_on_selected_cells_for_"
        "batch_donor_species_timepoint"
    ):
        raise ValueError("Shared-metadata verification claim drifted.")
    if payload["factual_human_da"] != "locked_not_computed_or_inspected":
        raise ValueError("Factual human DA must remain locked in RDX-01.")
    return payload
