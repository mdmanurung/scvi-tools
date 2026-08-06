"""Derive and atomically seal the RDX-01 human development cohort."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .governance import _rename_no_replace
from .human_lineage import (
    SHARED_PARENT_METADATA,
    covariate_levels,
    derive_human_cell_universe,
    human_lineage_contract_digest_v2,
    human_lineage_contract_v2,
    make_within_sample_hash_split,
    select_biological_proteins,
    select_pearson_residual_hvgs,
    sha256_csr_matrix,
    sha256_lines,
    validate_human_lineage_manifest,
    verify_shared_metadata,
)
from .manifest import (
    ArtifactRecord,
    RunManifest,
    make_run_id,
    sha256_file,
    verify_run_manifest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/exports/para-lipg-hpc/mdmanurung/schisto_citeseq")
DEFAULT_SOURCES = {
    "harmonized_joint": (
        SOURCE_ROOT
        / "analysis/harmonized_integration/outputs/human_immune_joint.h5ad"
    ),
    "final_parent": (
        SOURCE_ROOT / "analysis/final-assembly/outputs/schisto_human.h5ad"
    ),
    "source_registry": SOURCE_ROOT / "analysis/authoritative_h5ads.tsv",
    "protein_panel": (
        SOURCE_ROOT / "analysis/adt-denoising/outputs/adt_panel_map.tsv"
    ),
    "migration_inventory": (
        SOURCE_ROOT / "workflow/migration/baseline/input-inventory.tsv"
    ),
    "selected_run_state": SOURCE_ROOT / "workflow/state/selected-run.json",
    "migration_inventory_supersession": (
        SOURCE_ROOT
        / "workflow/migration/corrections/"
        "S06E-s01-baseline-inventory-rebinding-completion.json"
    ),
}
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / ".scratch/mrtotalvi-v2-redesign/human-lineage-runs"
)
LINEAGE_AMENDMENT = (
    REPOSITORY_ROOT
    / ".scratch/mrtotalvi-v2-redesign/human-lineage-amendment-v2.md"
)
CODE_PATHS = (
    "benchmarks/mrtotalvi/human_lineage.py",
    "benchmarks/mrtotalvi/freeze_human_lineage.py",
)
DERIVED_H5AD = "human-w00-w22.h5ad"


def _active_human_lineage_contract() -> dict:
    """Return the explicitly approved contract for new lineage freezes."""
    return human_lineage_contract_v2()


def _active_human_lineage_version() -> dict:
    """Return the schema, contract, and digest bound to new freezes."""
    return {
        "manifest_schema_version": "mrtotalvi-human-lineage-v2",
        "contract": _active_human_lineage_contract(),
        "contract_digest": human_lineage_contract_digest_v2(),
    }


REQUIRED_METADATA = tuple(
    _active_human_lineage_contract()["required_metadata"]
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_json_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _decode_strings(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values)
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in flattened
        ],
        dtype=str,
    )


def _read_h5_vector(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        values = np.asarray(node[:])
        if values.dtype.kind in {"O", "S", "U"}:
            return _decode_strings(values)
        return values
    if node.attrs.get("encoding-type") != "categorical":
        raise ValueError(f"Unsupported H5AD vector encoding at {node.name!r}.")
    categories = _decode_strings(np.asarray(node["categories"][:]))
    codes = np.asarray(node["codes"][:], dtype=np.int64)
    if np.any(codes < 0) or np.any(codes >= categories.size):
        raise ValueError(f"Categorical field {node.name!r} has missing codes.")
    return categories[codes]


def _read_h5_vector_at_positions(
    node: h5py.Dataset | h5py.Group,
    positions: np.ndarray,
) -> np.ndarray:
    """Read selected H5AD values while rejecting selected missing codes."""
    selected = np.asarray(positions, dtype=np.int64)
    if (
        selected.ndim != 1
        or selected.size == 0
        or np.any(selected < 0)
        or len(np.unique(selected)) != selected.size
    ):
        raise ValueError("Selected H5AD positions must be nonnegative and unique.")
    if isinstance(node, h5py.Dataset):
        values = _read_h5_vector(node)
        if np.any(selected >= len(values)):
            raise ValueError("Selected H5AD positions are out of range.")
        return values[selected]
    if node.attrs.get("encoding-type") != "categorical":
        raise ValueError(f"Unsupported H5AD vector encoding at {node.name!r}.")
    categories = _decode_strings(np.asarray(node["categories"][:]))
    all_codes = np.asarray(node["codes"][:], dtype=np.int64)
    if np.any(selected >= len(all_codes)):
        raise ValueError("Selected H5AD positions are out of range.")
    codes = all_codes[selected]
    if np.any(codes < 0) or np.any(codes >= categories.size):
        raise ValueError(
            f"Categorical field {node.name!r} has selected missing codes."
        )
    return categories[codes]


def _read_h5_boolean(node: h5py.Dataset | h5py.Group) -> np.ndarray:
    """Read native or exact FALSE/TRUE categorical H5AD booleans."""
    values = np.asarray(_read_h5_vector(node))
    if values.dtype.kind == "b":
        return values.astype(bool, copy=False)
    if values.dtype.kind not in {"O", "S", "U"}:
        raise ValueError(
            f"Boolean field {node.name!r} has unsupported storage."
        )
    labels = np.char.upper(_decode_strings(values))
    observed = set(labels.tolist())
    if not observed or not observed.issubset({"FALSE", "TRUE"}):
        raise ValueError(
            f"Boolean field {node.name!r} has categories {sorted(observed)}."
        )
    return labels == "TRUE"


def _sparse_shape(group: h5py.Group, encoding: str) -> tuple[int, int]:
    if group.attrs.get("encoding-type") != encoding:
        raise ValueError(f"{group.name!r} must use {encoding!r} encoding.")
    shape = tuple(int(value) for value in group.attrs["shape"])
    if len(shape) != 2:
        raise ValueError(f"{group.name!r} must be two-dimensional.")
    return shape


def _read_csr_rows(
    group: h5py.Group,
    rows: np.ndarray,
    *,
    output_chunk_rows: int = 512,
) -> sp.csr_matrix:
    shape = _sparse_shape(group, "csr_matrix")
    rows = np.asarray(rows, dtype=np.int64)
    if (
        rows.ndim != 1
        or rows.size == 0
        or np.any(rows < 0)
        or np.any(rows >= shape[0])
        or np.any(np.diff(rows) <= 0)
    ):
        raise ValueError("CSR row positions must be non-empty, sorted, and unique.")
    indptr = np.asarray(group["indptr"][:], dtype=np.int64)
    row_nnz = indptr[rows + 1] - indptr[rows]
    output_indptr = np.empty(rows.size + 1, dtype=np.int64)
    output_indptr[0] = 0
    np.cumsum(row_nnz, out=output_indptr[1:])
    output_indices = np.empty(
        int(output_indptr[-1]),
        dtype=group["indices"].dtype,
    )
    output_data = np.empty(
        int(output_indptr[-1]),
        dtype=group["data"].dtype,
    )
    for output_start in range(0, rows.size, output_chunk_rows):
        output_stop = min(output_start + output_chunk_rows, rows.size)
        source_rows = rows[output_start:output_stop]
        source_data_start = int(indptr[source_rows[0]])
        source_data_stop = int(indptr[source_rows[-1] + 1])
        block_indices = np.asarray(
            group["indices"][source_data_start:source_data_stop]
        )
        block_data = np.asarray(
            group["data"][source_data_start:source_data_stop]
        )
        for offset, source_row in enumerate(source_rows):
            output_row = output_start + offset
            source_start = int(indptr[source_row]) - source_data_start
            source_stop = int(indptr[source_row + 1]) - source_data_start
            destination_start = int(output_indptr[output_row])
            destination_stop = int(output_indptr[output_row + 1])
            output_indices[destination_start:destination_stop] = (
                block_indices[source_start:source_stop]
            )
            output_data[destination_start:destination_stop] = block_data[
                source_start:source_stop
            ]
    matrix = sp.csr_matrix(
        (output_data, output_indices, output_indptr),
        shape=(rows.size, shape[1]),
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _canonical_integer_sparse(
    matrix: sp.spmatrix,
    *,
    output_format: str,
) -> sp.spmatrix:
    values = matrix.data
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Raw counts must be finite and nonnegative.")
    if not np.array_equal(values, np.rint(values)):
        raise ValueError("Raw counts must be integer-valued.")
    normalized = matrix.astype(np.int64)
    normalized.sum_duplicates()
    normalized.eliminate_zeros()
    normalized.sort_indices()
    if output_format == "csr":
        normalized = normalized.tocsr()
    elif output_format == "csc":
        normalized = normalized.tocsc()
    else:
        raise ValueError(f"Unsupported sparse output format {output_format!r}.")
    normalized.sort_indices()
    return normalized


def _read_parent_csc_block(
    group: h5py.Group,
    *,
    source_indptr: np.ndarray,
    row_lookup: np.ndarray,
    column_start: int,
    column_stop: int,
) -> sp.csc_matrix:
    _sparse_shape(group, "csc_matrix")
    data_start = int(source_indptr[column_start])
    data_stop = int(source_indptr[column_stop])
    source_rows = np.asarray(
        group["indices"][data_start:data_stop],
        dtype=np.int64,
    )
    source_data = np.asarray(group["data"][data_start:data_stop])
    mapped_rows = row_lookup[source_rows]
    keep = mapped_rows >= 0
    local_indptr = (
        source_indptr[column_start : column_stop + 1] - data_start
    )
    counts = np.empty(column_stop - column_start, dtype=np.int64)
    for offset in range(column_stop - column_start):
        counts[offset] = np.count_nonzero(
            keep[local_indptr[offset] : local_indptr[offset + 1]]
        )
    output_indptr = np.empty(counts.size + 1, dtype=np.int64)
    output_indptr[0] = 0
    np.cumsum(counts, out=output_indptr[1:])
    selected_rows = mapped_rows[keep]
    selected_data = source_data[keep]
    block = sp.csc_matrix(
        (selected_data, selected_rows, output_indptr),
        shape=(int(np.count_nonzero(row_lookup >= 0)), column_stop - column_start),
    )
    return _canonical_integer_sparse(block, output_format="csc")


def _verify_all_rna_counts(
    *,
    joint_counts: sp.csr_matrix,
    parent_group: h5py.Group,
    parent_positions: tuple[int, ...],
    parent_cells: int,
    column_chunk_size: int = 256,
) -> sp.csc_matrix:
    parent_shape = _sparse_shape(parent_group, "csc_matrix")
    if parent_shape != (parent_cells, joint_counts.shape[1]):
        raise ValueError("Parent raw-count matrix dimensions drifted.")
    row_lookup = np.full(parent_cells, -1, dtype=np.int64)
    parent_rows = np.asarray(parent_positions, dtype=np.int64)
    if len(np.unique(parent_rows)) != parent_rows.size:
        raise ValueError("Selected parent positions must be unique.")
    row_lookup[parent_rows] = np.arange(parent_rows.size, dtype=np.int64)
    parent_indptr = np.asarray(parent_group["indptr"][:], dtype=np.int64)
    joint_csc = _canonical_integer_sparse(
        joint_counts.tocsc(),
        output_format="csc",
    )
    for start in range(0, joint_csc.shape[1], column_chunk_size):
        stop = min(start + column_chunk_size, joint_csc.shape[1])
        parent_block = _read_parent_csc_block(
            parent_group,
            source_indptr=parent_indptr,
            row_lookup=row_lookup,
            column_start=start,
            column_stop=stop,
        )
        joint_block = joint_csc[:, start:stop].copy()
        joint_block.sort_indices()
        if (
            not np.array_equal(joint_block.indptr, parent_block.indptr)
            or not np.array_equal(joint_block.indices, parent_block.indices)
            or not np.array_equal(joint_block.data, parent_block.data)
        ):
            raise ValueError(
                "Joint and parent RNA counts disagree in source columns "
                f"[{start}, {stop})."
            )
        if start % 2_560 == 0:
            print(
                f"verified RNA columns {start:,}-{stop:,}",
                flush=True,
            )
    return joint_csc


def _read_dense_rows(
    dataset: h5py.Dataset,
    rows: tuple[int, ...],
) -> np.ndarray:
    positions = np.asarray(rows, dtype=np.int64)
    if (
        positions.ndim != 1
        or positions.size == 0
        or len(np.unique(positions)) != positions.size
    ):
        raise ValueError("Dense row positions must be non-empty and unique.")
    order = np.argsort(positions, kind="stable")
    sorted_positions = positions[order]
    sorted_values = np.asarray(dataset[sorted_positions, :])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return sorted_values[inverse]


def _canonical_dense_counts(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("Protein counts must be a two-dimensional matrix.")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Protein counts must be finite and nonnegative.")
    if not np.array_equal(array, np.rint(array)):
        raise ValueError("Protein counts must be integer-valued.")
    return np.rint(array).astype(np.int64)


def _sha256_dense_counts(values: np.ndarray) -> str:
    counts = _canonical_dense_counts(values)
    digest = hashlib.sha256(b"mrtotalvi-dense-counts-v1\0")
    digest.update(np.asarray(counts.shape, dtype="<i8").tobytes())
    digest.update(counts.astype("<i8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _parse_panel_controls(values: pd.Series) -> tuple[bool, ...]:
    parsed = []
    for value in values.tolist():
        normalized = str(value).strip().lower()
        if normalized in {"true", "1"}:
            parsed.append(True)
        elif normalized in {"false", "0"}:
            parsed.append(False)
        else:
            raise ValueError(f"Unrecognized panel is_control value {value!r}.")
    return tuple(parsed)


def _source_records(
    *,
    contract: dict,
    sources: dict[str, Path] | None = None,
) -> list[dict]:
    source_paths = DEFAULT_SOURCES if sources is None else sources
    expected_roles = list(contract["source_sha256"])
    if list(source_paths) != expected_roles:
        raise ValueError(
            "Source paths must contain the exact ordered contract roles."
        )
    records = []
    for role, expected_digest in contract["source_sha256"].items():
        path = source_paths[role].resolve()
        if not path.is_file():
            raise ValueError(f"Required source is missing: {path}.")
        observed = sha256_file(path)
        if observed != expected_digest:
            raise ValueError(
                f"Source hash drift for {role!r}: expected "
                f"{expected_digest}, found {observed}."
            )
        records.append(
            {
                "role": role,
                "path": str(path),
                "sha256": observed,
                "bytes": path.stat().st_size,
            }
        )
        print(f"verified source {role}: {observed}", flush=True)
    return records


def _validate_inventory_supersession(
    *,
    contract: dict,
    sources: dict[str, Path],
) -> dict:
    """Validate the exact reviewed meaning of the inventory correction."""
    supersedes = contract.get("supersedes")
    if not isinstance(supersedes, dict):
        raise ValueError("The active lineage contract lacks supersession authority.")
    authority_role = supersedes.get("authority_role")
    authority_path = sources.get(authority_role)
    if authority_path is None:
        raise ValueError("The inventory supersession authority source is missing.")
    try:
        payload = json.loads(authority_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "The inventory supersession authority is unreadable."
        ) from error
    expected_rebindings = [
        {
            "corrected_sha256": supersedes["replacement_sha256"],
            "path": "workflow/migration/baseline/input-inventory.tsv",
            "superseded_sha256": supersedes["prior_sha256"],
        }
    ]
    scope = payload.get("scope_verified")
    if (
        payload.get("schema_version") != 1
        or payload.get("correction_id")
        != supersedes["authority_correction_id"]
        or payload.get("completes")
        != "S06E-s01-baseline-cellbouncer-config-refreeze"
        or payload.get("input_id") != "config_cellbouncer_parallel"
        or payload.get("baseline_rebindings") != expected_rebindings
        or not isinstance(scope, dict)
        or scope.get("changed_rows") != 1
        or scope.get("row_identity_matches_refreeze_record") is not True
    ):
        raise ValueError(
            "Inventory supersession authority semantics do not match v2."
        )
    return payload


def _verify_sources_unchanged(records: list[dict]) -> None:
    for record in records:
        path = Path(record["path"])
        if (
            path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(
                f"Source drifted during derivation: {record['role']!r}."
            )


def _code_records(repository_root: Path) -> list[dict]:
    records = []
    for relative_path in CODE_PATHS:
        path = repository_root / relative_path
        records.append(
            {
                "path": relative_path,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def _source_repository_commit() -> str:
    completed = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("Upstream source repository did not return a Git digest.")
    return commit


def _required_paths_present(
    source: h5py.File,
    paths: set[str],
    *,
    source_name: str,
) -> None:
    missing = sorted(path for path in paths if path not in source)
    if missing:
        raise ValueError(f"{source_name} is missing H5AD paths {missing}.")


def _derive_in_memory() -> dict:
    contract = _active_human_lineage_contract()
    source_records = _source_records(contract=contract)
    _validate_inventory_supersession(
        contract=contract,
        sources=DEFAULT_SOURCES,
    )
    expected = contract["expected"]
    panel = pd.read_csv(DEFAULT_SOURCES["protein_panel"], sep="\t")
    if not {"feature", "is_control"}.issubset(panel.columns):
        raise ValueError("Protein panel requires feature and is_control columns.")

    joint_path = DEFAULT_SOURCES["harmonized_joint"]
    parent_path = DEFAULT_SOURCES["final_parent"]
    with h5py.File(joint_path, "r") as joint, h5py.File(parent_path, "r") as parent:
        _required_paths_present(
            joint,
            {
                "obs/_index",
                *(f"obs/{column}" for column in REQUIRED_METADATA),
                "var/_index",
                "layers/counts",
                "obsm/protein",
                "uns/protein_names",
            },
            source_name="harmonized joint",
        )
        _required_paths_present(
            parent,
            {
                "obs/_index",
                "obs/pass_qc",
                *(
                    f"obs/{column}"
                    for column in SHARED_PARENT_METADATA
                ),
                "var/_index",
                "X",
                "obsm/protein",
                "uns/protein_names",
            },
            source_name="final parent",
        )
        joint_shape = _sparse_shape(joint["layers/counts"], "csr_matrix")
        parent_shape = _sparse_shape(parent["X"], "csc_matrix")
        if joint_shape != (
            expected["joint_cells"],
            expected["rna_features"],
        ):
            raise ValueError(f"Joint source dimensions drifted: {joint_shape}.")
        if parent_shape != (
            expected["parent_cells"],
            expected["rna_features"],
        ):
            raise ValueError(f"Parent source dimensions drifted: {parent_shape}.")

        joint_ids = _decode_strings(np.asarray(joint["obs/_index"][:]))
        parent_ids = _decode_strings(np.asarray(parent["obs/_index"][:]))
        parent_pass_qc = _read_h5_boolean(parent["obs/pass_qc"])
        metadata = {
            column: _read_h5_vector(joint[f"obs/{column}"])
            for column in REQUIRED_METADATA
        }
        universe = derive_human_cell_universe(
            joint_cell_ids=joint_ids,
            joint_donors=_decode_strings(metadata["donor"]),
            joint_timepoints=_decode_strings(metadata["timepoint"]),
            joint_samples=_decode_strings(metadata["donor_timepoint"]),
            parent_cell_ids=parent_ids,
            parent_pass_qc=parent_pass_qc,
            expected_cells=expected["cells"],
            expected_complete_donors=expected["complete_donors"],
            expected_ordered_cell_sha256=expected[
                "ordered_cell_sha256"
            ],
        )
        split_contract = contract["split"]
        split = make_within_sample_hash_split(
            cell_ids=universe.cell_ids,
            samples=universe.samples,
            train_fraction=split_contract["train_fraction"],
            salt=split_contract["salt"],
        )
        training_mask = np.asarray(
            [assignment == "train" for assignment in split.assignments],
            dtype=bool,
        )
        if len(set(universe.samples)) != expected["samples"]:
            raise ValueError("Selected donor-timepoint sample count drifted.")

        joint_positions = np.asarray(
            universe.joint_positions,
            dtype=np.int64,
        )
        parent_positions = np.asarray(
            universe.parent_positions,
            dtype=np.int64,
        )
        selected_metadata = {
            column: np.asarray(values)[joint_positions]
            for column, values in metadata.items()
        }
        selected_parent_metadata = {
            column: _read_h5_vector_at_positions(
                parent[f"obs/{column}"],
                parent_positions,
            )
            for column in SHARED_PARENT_METADATA
        }
        shared_metadata_digest = verify_shared_metadata(
            cell_ids=universe.cell_ids,
            joint_metadata=selected_metadata,
            parent_metadata=selected_parent_metadata,
        )
        selected_covariate_levels = covariate_levels(selected_metadata)

        joint_genes = _decode_strings(np.asarray(joint["var/_index"][:]))
        parent_genes = _decode_strings(np.asarray(parent["var/_index"][:]))
        if (
            len(np.unique(joint_genes)) != joint_genes.size
            or not np.array_equal(joint_genes, parent_genes)
        ):
            raise ValueError("Joint and parent ordered RNA features differ.")

        joint_protein_names = _decode_strings(
            np.asarray(joint["uns/protein_names"][:])
        )
        parent_protein_names = _decode_strings(
            np.asarray(parent["uns/protein_names"][:])
        )
        protein_selection = select_biological_proteins(
            panel_features=tuple(panel["feature"].astype(str)),
            panel_is_control=_parse_panel_controls(panel["is_control"]),
            joint_protein_names=tuple(joint_protein_names),
            parent_protein_names=tuple(parent_protein_names),
            expected_count=expected["selected_proteins"],
            expected_ordered_protein_sha256=expected[
                "ordered_protein_sha256"
            ],
        )

        print("reading selected joint RNA rows", flush=True)
        joint_counts = _read_csr_rows(
            joint["layers/counts"],
            np.asarray(universe.joint_positions, dtype=np.int64),
        )
        joint_counts = _canonical_integer_sparse(
            joint_counts,
            output_format="csr",
        )
        hvg_contract = contract["hvg"]
        print("computing training-only Pearson-residual HVGs", flush=True)
        hvg = select_pearson_residual_hvgs(
            counts=joint_counts,
            gene_names=tuple(joint_genes),
            training_mask=training_mask,
            n_top_genes=hvg_contract["n_top_genes"],
            theta=hvg_contract["theta"],
            chunk_size=hvg_contract["chunk_size"],
        )
        print("verifying all joint/parent RNA counts", flush=True)
        joint_csc = _verify_all_rna_counts(
            joint_counts=joint_counts,
            parent_group=parent["X"],
            parent_positions=universe.parent_positions,
            parent_cells=expected["parent_cells"],
        )
        del joint_counts
        selected_counts = joint_csc[
            :, np.asarray(hvg.selected_positions, dtype=np.int64)
        ].tocsr()
        selected_counts = _canonical_integer_sparse(
            selected_counts,
            output_format="csr",
        )
        del joint_csc

        joint_proteins = _canonical_dense_counts(
            _read_dense_rows(
                joint["obsm/protein"],
                universe.joint_positions,
            )
        )
        parent_proteins = _canonical_dense_counts(
            _read_dense_rows(
                parent["obsm/protein"],
                universe.parent_positions,
            )
        )
        if not np.array_equal(joint_proteins, parent_proteins):
            raise ValueError("Joint and parent protein counts disagree.")
        selected_proteins = joint_proteins[
            :, np.asarray(protein_selection.positions, dtype=np.int64)
        ]

    _verify_sources_unchanged(source_records)
    return {
        "source_records": source_records,
        "source_repository_commit": _source_repository_commit(),
        "universe": universe,
        "split": split,
        "training_mask": training_mask,
        "joint_genes": tuple(joint_genes),
        "hvg": hvg,
        "protein_selection": protein_selection,
        "joint_protein_names": tuple(joint_protein_names),
        "selected_counts": selected_counts,
        "selected_proteins": selected_proteins,
        "metadata": selected_metadata,
        "shared_metadata_digest": shared_metadata_digest,
        "covariate_levels": selected_covariate_levels,
    }


def _environment_payload() -> dict:
    packages = {}
    for package in (
        "anndata",
        "h5py",
        "numpy",
        "pandas",
        "scipy",
        "scvi-tools",
        "torch",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _write_split_table(path: Path, state: dict) -> None:
    universe = state["universe"]
    split = state["split"]
    lines = [
        "cell_id\tdonor\ttimepoint\tdonor_timepoint\tassignment\tselection_sha256\n"
    ]
    for values in zip(
        universe.cell_ids,
        universe.donors,
        universe.timepoints,
        universe.samples,
        split.assignments,
        split.scores,
        strict=True,
    ):
        lines.append("\t".join(values) + "\n")
    path.write_text("".join(lines), encoding="utf-8")


def _write_hvg_table(path: Path, state: dict) -> None:
    hvg = state["hvg"]
    ranking_by_position = np.empty(
        len(hvg.ranking_positions),
        dtype=np.int64,
    )
    ranking_by_position[
        np.asarray(hvg.ranking_positions, dtype=np.int64)
    ] = np.arange(len(hvg.ranking_positions), dtype=np.int64)
    selected = set(hvg.selected_positions)
    lines = [
        "gene\tsource_position\tpearson_residual_rank\t"
        "pearson_residual_variance\tselected\n"
    ]
    for position, (gene, variance) in enumerate(
        zip(hvg.gene_names, hvg.residual_variances, strict=True)
    ):
        lines.append(
            f"{gene}\t{position}\t{ranking_by_position[position]}\t"
            f"{variance:.17g}\t{str(position in selected).lower()}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _validate_split_table_evidence(
    *,
    table: pd.DataFrame,
    obs: pd.DataFrame,
    split_contract: dict,
) -> dict[str, str]:
    expected_columns = [
        "cell_id",
        "donor",
        "timepoint",
        "donor_timepoint",
        "assignment",
        "selection_sha256",
    ]
    if list(table.columns) != expected_columns or len(table) != len(obs):
        raise ValueError("The split table has an invalid shape or schema.")
    cell_ids = tuple(obs.index.astype(str))
    samples = tuple(obs["donor_timepoint"].astype(str))
    recomputed = make_within_sample_hash_split(
        cell_ids=cell_ids,
        samples=samples,
        train_fraction=split_contract["train_fraction"],
        salt=split_contract["salt"],
    )
    expected_values = {
        "cell_id": cell_ids,
        "donor": tuple(obs["donor"].astype(str)),
        "timepoint": tuple(obs["timepoint"].astype(str)),
        "donor_timepoint": samples,
        "assignment": recomputed.assignments,
        "selection_sha256": recomputed.scores,
    }
    for column, expected in expected_values.items():
        observed = tuple(table[column].astype(str))
        if observed != expected:
            raise ValueError(
                f"The split table disagrees with cohort column {column!r}."
            )
    if tuple(obs["lineage_split"].astype(str)) != recomputed.assignments:
        raise ValueError("The cohort split assignments fail the hash rule.")
    if (
        tuple(obs["lineage_split_sha256"].astype(str))
        != recomputed.scores
    ):
        raise ValueError("The cohort split scores fail the hash rule.")
    training_cells = tuple(
        cell_id
        for cell_id, assignment in zip(
            cell_ids,
            recomputed.assignments,
            strict=True,
        )
        if assignment == "train"
    )
    heldout_cells = tuple(
        cell_id
        for cell_id, assignment in zip(
            cell_ids,
            recomputed.assignments,
            strict=True,
        )
        if assignment == "heldout"
    )
    return {
        "split_assignments": recomputed.assignment_sha256,
        "ordered_training_cells": sha256_lines(training_cells),
        "ordered_heldout_cells": sha256_lines(heldout_cells),
    }


def _validate_hvg_table_evidence(
    *,
    table: pd.DataFrame,
    cohort_genes: tuple[str, ...],
    expected_source_genes: int,
    expected_selected_genes: int,
) -> dict[str, str]:
    expected_columns = [
        "gene",
        "source_position",
        "pearson_residual_rank",
        "pearson_residual_variance",
        "selected",
    ]
    if (
        list(table.columns) != expected_columns
        or len(table) != expected_source_genes
    ):
        raise ValueError("The HVG table has an invalid shape or schema.")
    genes = tuple(table["gene"].astype(str))
    if (
        len(set(genes)) != expected_source_genes
        or any(not gene for gene in genes)
    ):
        raise ValueError("The HVG table gene identities are invalid.")
    try:
        positions = pd.to_numeric(
            table["source_position"],
            errors="raise",
        ).to_numpy(dtype=np.int64)
        ranks = pd.to_numeric(
            table["pearson_residual_rank"],
            errors="raise",
        ).to_numpy(dtype=np.int64)
        variances = pd.to_numeric(
            table["pearson_residual_variance"],
            errors="raise",
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("The HVG table has nonnumeric ranking fields.") from error
    expected_positions = np.arange(expected_source_genes, dtype=np.int64)
    if not np.array_equal(positions, expected_positions):
        raise ValueError("The HVG table source positions are not complete.")
    expected_ranking = np.lexsort((positions, -variances))
    expected_ranks = np.empty(expected_source_genes, dtype=np.int64)
    expected_ranks[expected_ranking] = expected_positions
    if (
        not np.isfinite(variances).all()
        or not np.array_equal(ranks, expected_ranks)
    ):
        raise ValueError("The HVG table ranking is inconsistent.")
    selected_labels = tuple(
        value.strip().lower() for value in table["selected"].astype(str)
    )
    if not set(selected_labels).issubset({"true", "false"}):
        raise ValueError("The HVG table selected field is not boolean.")
    selected = np.asarray(
        [value == "true" for value in selected_labels],
        dtype=bool,
    )
    if not np.array_equal(selected, ranks < expected_selected_genes):
        raise ValueError("The HVG table selected mask disagrees with rank.")
    ranked_selected = tuple(
        genes[position]
        for position in expected_ranking[:expected_selected_genes]
    )
    if ranked_selected != cohort_genes:
        raise ValueError("The HVG selected ranking disagrees with cohort genes.")
    return {
        "ordered_source_genes": sha256_lines(genes),
        "ordered_selected_genes": sha256_lines(ranked_selected),
    }


def _make_obs(state: dict) -> pd.DataFrame:
    universe = state["universe"]
    metadata = state["metadata"]
    obs = pd.DataFrame(
        {
            column: np.asarray(values)
            for column, values in metadata.items()
        },
        index=pd.Index(universe.cell_ids, name="cell_id"),
    )
    obs["final_parent_pass_qc"] = True
    obs["lineage_split"] = pd.Categorical(
        state["split"].assignments,
        categories=["train", "heldout"],
        ordered=True,
    )
    obs["lineage_split_sha256"] = state["split"].scores
    return obs


def _make_var(state: dict) -> pd.DataFrame:
    hvg = state["hvg"]
    positions = np.asarray(hvg.selected_positions, dtype=np.int64)
    variances = np.asarray(hvg.residual_variances, dtype=np.float64)
    return pd.DataFrame(
        {
            "source_position": positions,
            "pearson_residual_variance": variances[positions],
            "pearson_residual_rank": np.arange(
                positions.size,
                dtype=np.int64,
            ),
            "selection_training_only": True,
        },
        index=pd.Index(hvg.selected_gene_names, name="gene"),
    )


def _digest_payload(state: dict) -> dict:
    universe = state["universe"]
    split = state["split"]
    training_cells = tuple(
        cell_id
        for cell_id, assignment in zip(
            universe.cell_ids,
            split.assignments,
            strict=True,
        )
        if assignment == "train"
    )
    heldout_cells = tuple(
        cell_id
        for cell_id, assignment in zip(
            universe.cell_ids,
            split.assignments,
            strict=True,
        )
        if assignment == "heldout"
    )
    return {
        "ordered_cells": universe.ordered_cell_sha256,
        "split_assignments": split.assignment_sha256,
        "ordered_training_cells": sha256_lines(training_cells),
        "ordered_heldout_cells": sha256_lines(heldout_cells),
        "ordered_source_genes": sha256_lines(state["joint_genes"]),
        "ordered_selected_genes": state["hvg"].ordered_gene_sha256,
        "ordered_source_proteins": sha256_lines(
            state["joint_protein_names"]
        ),
        "ordered_selected_proteins": (
            state["protein_selection"].ordered_protein_sha256
        ),
        "selected_rna_counts": sha256_csr_matrix(
            state["selected_counts"]
        ),
        "selected_protein_counts": _sha256_dense_counts(
            state["selected_proteins"]
        ),
        "shared_parent_metadata": state["shared_metadata_digest"],
    }


def _write_complete_run(output_root: Path, state: dict) -> Path:
    created_at = datetime.now(UTC)
    version = _active_human_lineage_version()
    contract = version["contract"]
    code_records = _code_records(REPOSITORY_ROOT)
    digests = _digest_payload(state)
    universe = state["universe"]
    split = state["split"]
    training_cells = sum(
        assignment == "train" for assignment in split.assignments
    )
    selected_dimensions = {
        "cells": len(universe.cell_ids),
        "complete_donors": len(universe.complete_donors),
        "samples": len(set(universe.samples)),
        "training_cells": training_cells,
        "heldout_cells": len(universe.cell_ids) - training_cells,
        "genes": len(state["hvg"].selected_gene_names),
        "proteins": len(state["protein_selection"].names),
    }
    source_dimensions = {
        "joint_cells": 86_681,
        "parent_cells": 126_299,
        "rna_features": len(state["joint_genes"]),
        "source_proteins": 137,
    }
    code_digest = _canonical_json_digest(code_records)
    config_digest = version["contract_digest"]
    data_digest = _canonical_json_digest(
        {
            "sources": state["source_records"],
            "source_dimensions": source_dimensions,
            "selected_dimensions": selected_dimensions,
            "covariate_levels": state["covariate_levels"],
            "digests": digests,
        }
    )
    run_id = make_run_id(
        timestamp=created_at,
        code_digest=code_digest,
        config_digest=config_digest,
        data_digest=data_digest,
    )
    lineage_manifest = {
        "schema_version": version["manifest_schema_version"],
        "run_id": run_id,
        "created_at": created_at.isoformat(),
        "status": "complete",
        "base_commit": "d8c8e997a67997a53f55923eb3ab14e6cf06f94c",
        "source_repository_commit": state["source_repository_commit"],
        "contract": contract,
        "source_files": state["source_records"],
        "code_files": code_records,
        "source_dimensions": source_dimensions,
        "selected_dimensions": selected_dimensions,
        "covariate_levels": {
            column: list(values)
            for column, values in state["covariate_levels"].items()
        },
        "digests": digests,
        "count_verification": contract["count_verification"],
        "shared_metadata_verification": (
            "joint_parent_exact_on_selected_cells_for_"
            "batch_donor_species_timepoint"
        ),
        "factual_human_da": "locked_not_computed_or_inspected",
    }
    validate_human_lineage_manifest(lineage_manifest)

    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / run_id
    if final.exists() or final.is_symlink():
        raise FileExistsError(f"Human-lineage run already exists: {final}.")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output_root)
    )
    try:
        counts = state["selected_counts"].astype(np.int32)
        proteins = state["selected_proteins"].astype(np.int32)
        cohort = ad.AnnData(
            X=counts.copy(),
            obs=_make_obs(state),
            var=_make_var(state),
        )
        cohort.layers["counts"] = counts
        cohort.obsm["protein_expression"] = proteins
        cohort.uns["protein_names"] = np.asarray(
            state["protein_selection"].names,
            dtype=object,
        )
        cohort.uns["mrtotalvi_human_lineage"] = {
            "run_id": run_id,
            "contract_sha256": config_digest,
            "ordered_cell_sha256": digests["ordered_cells"],
            "split_assignment_sha256": digests["split_assignments"],
            "ordered_gene_sha256": digests["ordered_selected_genes"],
            "ordered_protein_sha256": digests[
                "ordered_selected_proteins"
            ],
            "factual_human_da": "locked_not_computed_or_inspected",
        }
        cohort.write_h5ad(temporary / DERIVED_H5AD, compression="gzip")
        _write_split_table(temporary / "split.tsv", state)
        _write_hvg_table(temporary / "hvg-ranking.tsv", state)
        _write_json(temporary / "environment.json", _environment_payload())
        (temporary / "source-hashes.sha256").write_text(
            "".join(
                f"{record['sha256']}  {record['role']}\t{record['path']}\n"
                for record in state["source_records"]
            ),
            encoding="utf-8",
        )
        shutil.copyfile(
            LINEAGE_AMENDMENT,
            temporary / "lineage-amendment.md",
        )
        _write_json(
            temporary / "lineage-manifest.json",
            lineage_manifest,
        )
        checksum_names = (
            DERIVED_H5AD,
            "environment.json",
            "hvg-ranking.tsv",
            "lineage-amendment.md",
            "lineage-manifest.json",
            "source-hashes.sha256",
            "split.tsv",
        )
        (temporary / "checksums.sha256").write_text(
            "".join(
                f"{sha256_file(temporary / name)}  {name}\n"
                for name in checksum_names
            ),
            encoding="utf-8",
        )
        artifact_names = (*checksum_names, "checksums.sha256")
        artifacts = tuple(
            ArtifactRecord(
                path=name,
                sha256=sha256_file(temporary / name),
                bytes=(temporary / name).stat().st_size,
            )
            for name in artifact_names
        )
        run_manifest = RunManifest(
            schema_version="mrtotalvi-benchmark-run-v1",
            run_id=run_id,
            created_at=created_at.isoformat(),
            code_digest=code_digest,
            config_digest=config_digest,
            data_digest=data_digest,
            evidence_tier="pilot_cache",
            scientific_scope=(
                "human development lineage only; factual W22-minus-W00 DA "
                "locked and not computed or inspected"
            ),
            status="complete",
            artifacts=artifacts,
        )
        _write_json(
            temporary / "run-manifest.json",
            run_manifest.to_dict(),
        )
        _rename_no_replace(temporary, final)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return final


def _verify_checksum_inventory(path: Path) -> None:
    expected = {
        DERIVED_H5AD,
        "environment.json",
        "hvg-ranking.tsv",
        "lineage-amendment.md",
        "lineage-manifest.json",
        "source-hashes.sha256",
        "split.tsv",
    }
    observed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative_path = line.partition("  ")
        if not separator or relative_path in observed:
            raise ValueError("Malformed or duplicate lineage checksum entry.")
        observed.add(relative_path)
        artifact = path.parent / relative_path
        if sha256_file(artifact) != digest:
            raise ValueError(f"Checksum mismatch for {relative_path!r}.")
    if observed != expected:
        raise ValueError("Lineage checksum inventory is incomplete.")


def _verify_source_metadata(
    *,
    lineage: dict,
    obs: pd.DataFrame,
) -> tuple[str, dict[str, tuple[str, ...]]]:
    source_paths = {
        record["role"]: Path(record["path"])
        for record in lineage["source_files"]
    }
    expected = lineage["contract"]["expected"]
    with (
        h5py.File(source_paths["harmonized_joint"], "r") as joint,
        h5py.File(source_paths["final_parent"], "r") as parent,
    ):
        joint_ids = _decode_strings(np.asarray(joint["obs/_index"][:]))
        parent_ids = _decode_strings(np.asarray(parent["obs/_index"][:]))
        parent_pass_qc = _read_h5_boolean(parent["obs/pass_qc"])
        joint_metadata = {
            column: _read_h5_vector(joint[f"obs/{column}"])
            for column in REQUIRED_METADATA
        }
        universe = derive_human_cell_universe(
            joint_cell_ids=joint_ids,
            joint_donors=_decode_strings(joint_metadata["donor"]),
            joint_timepoints=_decode_strings(joint_metadata["timepoint"]),
            joint_samples=_decode_strings(
                joint_metadata["donor_timepoint"]
            ),
            parent_cell_ids=parent_ids,
            parent_pass_qc=parent_pass_qc,
            expected_cells=expected["cells"],
            expected_complete_donors=expected["complete_donors"],
            expected_ordered_cell_sha256=expected[
                "ordered_cell_sha256"
            ],
        )
        if tuple(obs.index.astype(str)) != universe.cell_ids:
            raise ValueError("Derived cohort is not the recomputed source universe.")
        joint_positions = np.asarray(
            universe.joint_positions,
            dtype=np.int64,
        )
        parent_positions = np.asarray(
            universe.parent_positions,
            dtype=np.int64,
        )
        selected_joint = {
            column: np.asarray(values)[joint_positions]
            for column, values in joint_metadata.items()
        }
        selected_parent = {
            column: _read_h5_vector_at_positions(
                parent[f"obs/{column}"],
                parent_positions,
            )
            for column in SHARED_PARENT_METADATA
        }
        shared_digest = verify_shared_metadata(
            cell_ids=universe.cell_ids,
            joint_metadata=selected_joint,
            parent_metadata=selected_parent,
        )
        if not np.all(parent_pass_qc[parent_positions]):
            raise ValueError("Derived cohort retains a parent-QC failure.")
    required_obs = {
        *REQUIRED_METADATA,
        "final_parent_pass_qc",
        "lineage_split",
        "lineage_split_sha256",
    }
    if not required_obs.issubset(obs.columns):
        raise ValueError("Derived cohort is missing required metadata columns.")
    if not np.asarray(obs["final_parent_pass_qc"], dtype=bool).all():
        raise ValueError("Derived cohort parent-QC flags are not all true.")
    for column in REQUIRED_METADATA:
        if tuple(obs[column].astype(str)) != tuple(
            _decode_strings(selected_joint[column])
        ):
            raise ValueError(
                f"Derived metadata {column!r} disagrees with the joint source."
            )
    return shared_digest, covariate_levels(selected_joint)


def verify_human_lineage_run(
    run_dir: str | Path,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    verify_sources: bool = True,
) -> dict:
    """Independently re-read every RDX-01 manifest and cohort identity."""
    root = Path(run_dir).resolve()
    expected_files = {
        DERIVED_H5AD,
        "checksums.sha256",
        "environment.json",
        "hvg-ranking.tsv",
        "lineage-amendment.md",
        "lineage-manifest.json",
        "run-manifest.json",
        "source-hashes.sha256",
        "split.tsv",
    }
    actual_files = {
        child.name for child in root.iterdir() if child.is_file()
    }
    if actual_files != expected_files or any(
        child.is_dir() for child in root.iterdir()
    ):
        raise ValueError("Human-lineage run has an unexpected file inventory.")
    if (root.parent / "latest").exists() or (root.parent / "latest").is_symlink():
        raise ValueError("Human-lineage output root cannot contain latest.")
    run_manifest = verify_run_manifest(root / "run-manifest.json")
    _verify_checksum_inventory(root / "checksums.sha256")
    lineage_payload = json.loads(
        (root / "lineage-manifest.json").read_text(encoding="utf-8")
    )
    lineage = validate_human_lineage_manifest(lineage_payload)
    if lineage["run_id"] != run_manifest.run_id:
        raise ValueError("Run and lineage manifest identities differ.")

    repository = Path(repository_root).resolve()
    for record in lineage["code_files"]:
        path = (repository / record["path"]).resolve()
        if (
            not path.is_relative_to(repository)
            or sha256_file(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"Derivation code drift for {record['path']!r}.")
    if verify_sources:
        _verify_sources_unchanged(lineage["source_files"])

    cohort = ad.read_h5ad(root / DERIVED_H5AD, backed="r")
    try:
        dimensions = lineage["selected_dimensions"]
        if cohort.shape != (dimensions["cells"], dimensions["genes"]):
            raise ValueError("Derived cohort dimensions disagree with manifest.")
        cell_ids = tuple(cohort.obs_names.astype(str))
        genes = tuple(cohort.var_names.astype(str))
        obs = cohort.obs.copy()
        if sha256_lines(cell_ids) != lineage["digests"]["ordered_cells"]:
            raise ValueError("Derived cohort cell order disagrees with manifest.")
        if sha256_lines(genes) != lineage["digests"]["ordered_selected_genes"]:
            raise ValueError("Derived cohort gene order disagrees with manifest.")
        shared_digest, observed_levels = _verify_source_metadata(
            lineage=lineage,
            obs=obs,
        )
        if shared_digest != lineage["digests"]["shared_parent_metadata"]:
            raise ValueError("Shared source metadata digest disagrees.")
        if {
            column: list(values)
            for column, values in observed_levels.items()
        } != lineage["covariate_levels"]:
            raise ValueError("Derived cohort covariate levels disagree.")

        split_table = pd.read_csv(
            root / "split.tsv",
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )
        split_evidence = _validate_split_table_evidence(
            table=split_table,
            obs=obs,
            split_contract=lineage["contract"]["split"],
        )
        for name, digest in split_evidence.items():
            if digest != lineage["digests"][name]:
                raise ValueError(f"Split digest {name!r} disagrees.")
        assignments = tuple(obs["lineage_split"].astype(str))
        if assignments.count("train") != dimensions["training_cells"]:
            raise ValueError("Derived cohort training count drifted.")
        if assignments.count("heldout") != dimensions["heldout_cells"]:
            raise ValueError("Derived cohort held-out count drifted.")

        hvg_table = pd.read_csv(
            root / "hvg-ranking.tsv",
            sep="\t",
            dtype={
                "gene": str,
                "selected": str,
            },
            keep_default_na=False,
        )
        hvg_evidence = _validate_hvg_table_evidence(
            table=hvg_table,
            cohort_genes=genes,
            expected_source_genes=lineage["source_dimensions"][
                "rna_features"
            ],
            expected_selected_genes=dimensions["genes"],
        )
        for name, digest in hvg_evidence.items():
            if digest != lineage["digests"][name]:
                raise ValueError(f"HVG digest {name!r} disagrees.")

        counts = cohort.layers["counts"][:]
        if sha256_csr_matrix(counts) != lineage["digests"][
            "selected_rna_counts"
        ]:
            raise ValueError("Derived cohort RNA counts disagree with manifest.")
        if sha256_csr_matrix(cohort.X[:]) != lineage["digests"][
            "selected_rna_counts"
        ]:
            raise ValueError("Derived cohort X and counts layer differ.")
        proteins = np.asarray(cohort.obsm["protein_expression"])
        if _sha256_dense_counts(proteins) != lineage["digests"][
            "selected_protein_counts"
        ]:
            raise ValueError(
                "Derived cohort protein counts disagree with manifest."
            )
        protein_names = tuple(
            str(value) for value in cohort.uns["protein_names"]
        )
        if sha256_lines(protein_names) != lineage["digests"][
            "ordered_selected_proteins"
        ]:
            raise ValueError("Derived cohort protein order drifted.")
        embedded = cohort.uns.get("mrtotalvi_human_lineage")
        contract_digest = _canonical_json_digest(lineage["contract"])
        expected_embedded = {
            "run_id": lineage["run_id"],
            "contract_sha256": contract_digest,
            "ordered_cell_sha256": lineage["digests"]["ordered_cells"],
            "split_assignment_sha256": lineage["digests"][
                "split_assignments"
            ],
            "ordered_gene_sha256": lineage["digests"][
                "ordered_selected_genes"
            ],
            "ordered_protein_sha256": lineage["digests"][
                "ordered_selected_proteins"
            ],
            "factual_human_da": "locked_not_computed_or_inspected",
        }
        if embedded != expected_embedded:
            raise ValueError("Embedded lineage and factual-DA lock drifted.")
    finally:
        cohort.file.close()
    return lineage


def _write_blocked_run(output_root: Path, error: Exception) -> Path:
    created_at = datetime.now(UTC)
    error_payload = {
        "schema_version": "mrtotalvi-human-lineage-blocked-v1",
        "created_at": created_at.isoformat(),
        "status": "blocked",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "factual_human_da": "locked_not_computed_or_inspected",
        "fallback_universe": "not_created",
    }
    digest = _canonical_json_digest(error_payload)
    run_id = (
        created_at.strftime("%Y%m%dT%H%M%SZ")
        + f"-blocked-{digest[:12]}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / run_id
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output_root)
    )
    try:
        _write_json(temporary / "blocked-report.json", error_payload)
        (temporary / "checksums.sha256").write_text(
            f"{sha256_file(temporary / 'blocked-report.json')}  "
            "blocked-report.json\n",
            encoding="utf-8",
        )
        _rename_no_replace(temporary, final)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return final


def main() -> int:
    """Derive, seal, and independently verify one new lineage run."""
    args = _parse_args()
    output_root = args.output_root.resolve()
    try:
        state = _derive_in_memory()
        run_dir = _write_complete_run(output_root, state)
        verify_human_lineage_run(run_dir)
    except Exception as error:
        blocked_dir = _write_blocked_run(output_root, error)
        print(
            f"RDX-01 blocked; evidence sealed at {blocked_dir}",
            file=sys.stderr,
        )
        raise
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
