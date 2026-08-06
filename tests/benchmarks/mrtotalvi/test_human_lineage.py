"""Fail-closed human-lineage contracts for the MrTotalVI redesign."""

from __future__ import annotations

import copy
import hashlib
import json

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from benchmarks.mrtotalvi.freeze_human_lineage import (
    DEFAULT_SOURCES,
    LINEAGE_AMENDMENT,
    _active_human_lineage_contract,
    _active_human_lineage_version,
    _derive_in_memory,
    _read_csr_rows,
    _read_h5_boolean,
    _read_h5_vector_at_positions,
    _source_records,
    _validate_hvg_table_evidence,
    _validate_inventory_supersession,
    _validate_split_table_evidence,
    _verify_all_rna_counts,
)
from benchmarks.mrtotalvi.human_lineage import (
    covariate_levels,
    derive_human_cell_universe,
    human_lineage_contract,
    human_lineage_contract_digest,
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


def test_cell_universe_preserves_harmonized_order_after_parent_qc_and_pairing():
    """Parent QC and complete-pair filtering must never reorder joint cells."""
    joint_cell_ids = np.asarray(
        ["cell-c", "cell-a", "cell-e", "cell-b", "cell-d", "cell-f"]
    )
    joint_donors = np.asarray(["D1", "D1", "D2", "D1", "D1", "D2"])
    joint_timepoints = np.asarray(["W22", "W00", "W00", "W22", "W00", "W22"])
    joint_samples = np.asarray(
        ["D1_W22", "D1_W00", "D2_W00", "D1_W22", "D1_W00", "D2_W22"]
    )
    parent_cell_ids = np.asarray(
        ["cell-f", "cell-b", "cell-d", "cell-a", "cell-e", "cell-c"]
    )
    parent_pass_qc = np.asarray([False, True, True, True, True, True])

    universe = derive_human_cell_universe(
        joint_cell_ids=joint_cell_ids,
        joint_donors=joint_donors,
        joint_timepoints=joint_timepoints,
        joint_samples=joint_samples,
        parent_cell_ids=parent_cell_ids,
        parent_pass_qc=parent_pass_qc,
        expected_cells=4,
        expected_complete_donors=1,
    )

    assert universe.cell_ids == ("cell-c", "cell-a", "cell-b", "cell-d")
    assert universe.joint_positions == (0, 1, 3, 4)
    assert universe.parent_positions == (5, 3, 1, 2)
    assert universe.donors == ("D1", "D1", "D1", "D1")
    assert universe.timepoints == ("W22", "W00", "W22", "W00")
    assert universe.samples == ("D1_W22", "D1_W00", "D1_W22", "D1_W00")
    assert universe.complete_donors == ("D1",)


def test_cell_universe_fails_closed_on_mismatch_or_incomplete_pairs():
    """Missing parent IDs and count drift cannot produce a fallback universe."""
    common = {
        "joint_cell_ids": ["a", "b", "c", "d"],
        "joint_donors": ["D1", "D1", "D2", "D2"],
        "joint_timepoints": ["W00", "W22", "W00", "W22"],
        "joint_samples": ["D1_W00", "D1_W22", "D2_W00", "D2_W22"],
        "parent_pass_qc": [True, True, True, False],
        "expected_cells": 2,
        "expected_complete_donors": 1,
    }
    with pytest.raises(ValueError, match="Every harmonized cell"):
        derive_human_cell_universe(
            parent_cell_ids=["a", "b", "c", "missing"],
            **common,
        )
    with pytest.raises(ValueError, match="Cell count mismatch"):
        derive_human_cell_universe(
            parent_cell_ids=["a", "b", "c", "d"],
            **{**common, "expected_cells": 3},
        )
    with pytest.raises(ValueError, match="Ordered-cell SHA-256 mismatch"):
        derive_human_cell_universe(
            parent_cell_ids=["a", "b", "c", "d"],
            expected_ordered_cell_sha256="0" * 64,
            **common,
        )


def test_shared_metadata_must_match_after_exact_cell_alignment():
    """A shared parent covariate mismatch blocks an otherwise valid cohort."""
    cell_ids = ("c2", "c1")
    joint = {
        "batch": ("B2", "B1"),
        "donor": ("D2", "D1"),
        "species": ("human", "human"),
        "timepoint": ("W22", "W00"),
    }
    parent = dict(joint)

    digest = verify_shared_metadata(
        cell_ids=cell_ids,
        joint_metadata=joint,
        parent_metadata=parent,
    )
    assert len(digest) == 64
    mismatched = dict(parent)
    mismatched["donor"] = ("D2", "wrong")
    with pytest.raises(ValueError, match="shared metadata"):
        verify_shared_metadata(
            cell_ids=cell_ids,
            joint_metadata=joint,
            parent_metadata=mismatched,
        )


def test_within_sample_hash_split_is_deterministic_and_order_invariant():
    """Split membership is based on cell identity within sample, not row order."""
    cell_ids = tuple(f"cell-{index}" for index in range(10))
    samples = ("S1",) * 5 + ("S2",) * 5
    split = make_within_sample_hash_split(
        cell_ids=cell_ids,
        samples=samples,
        train_fraction=0.6,
        salt="fixture-salt",
    )
    reverse = make_within_sample_hash_split(
        cell_ids=cell_ids[::-1],
        samples=samples[::-1],
        train_fraction=0.6,
        salt="fixture-salt",
    )

    membership = dict(zip(split.cell_ids, split.assignments, strict=True))
    reverse_membership = dict(
        zip(reverse.cell_ids, reverse.assignments, strict=True)
    )
    assert membership == reverse_membership
    assert sum(
        assignment == "train"
        for assignment, sample in zip(
            split.assignments,
            split.samples,
            strict=True,
        )
        if sample == "S1"
    ) == 3
    assert sum(
        assignment == "train"
        for assignment, sample in zip(
            split.assignments,
            split.samples,
            strict=True,
        )
        if sample == "S2"
    ) == 3
    assert split.assignment_sha256 == reverse.assignment_sha256


def test_pearson_residual_hvgs_use_training_cells_only():
    """Held-out count changes must not alter the frozen HVG ranking."""
    counts = np.asarray(
        [
            [9, 0, 2, 1],
            [8, 1, 2, 1],
            [0, 8, 2, 1],
            [1, 9, 2, 1],
            [2, 2, 30, 0],
            [2, 2, 40, 0],
        ],
        dtype=np.int64,
    )
    training_mask = np.asarray([True, True, True, True, False, False])
    first = select_pearson_residual_hvgs(
        counts=sp.csr_matrix(counts),
        gene_names=("g0", "g1", "g2", "g3"),
        training_mask=training_mask,
        n_top_genes=2,
        theta=100.0,
        chunk_size=2,
    )
    changed = counts.copy()
    changed[~training_mask] = np.asarray(
        [[0, 0, 0, 10_000], [0, 0, 0, 20_000]]
    )
    second = select_pearson_residual_hvgs(
        counts=sp.csr_matrix(changed),
        gene_names=("g0", "g1", "g2", "g3"),
        training_mask=training_mask,
        n_top_genes=2,
        theta=100.0,
        chunk_size=3,
    )

    assert first.selected_gene_names == second.selected_gene_names
    np.testing.assert_array_equal(
        first.residual_variances,
        second.residual_variances,
    )
    assert first.selected_gene_names == ("g0", "g1")
    assert first.training_cell_count == 4
    assert first.clip == 2.0


def test_pearson_residual_hvgs_reject_fractional_counts():
    """The raw-count contract cannot be satisfied by rounded input."""
    counts = sp.csr_matrix(
        np.asarray([[1.0, 0.5], [0.0, 2.0], [1.0, 1.0]])
    )
    with pytest.raises(ValueError, match="integer-valued"):
        select_pearson_residual_hvgs(
            counts=counts,
            gene_names=("g0", "g1"),
            training_mask=np.asarray([True, True, False]),
            n_top_genes=1,
        )


def test_biological_protein_order_is_panel_order_and_hash_locked():
    """Control exclusion preserves exact panel/source order."""
    selection = select_biological_proteins(
        panel_features=("CD3", "Isotype-1", "CD4", "CD8"),
        panel_is_control=(False, True, False, False),
        joint_protein_names=("CD3", "Isotype-1", "CD4", "CD8"),
        parent_protein_names=("CD3", "Isotype-1", "CD4", "CD8"),
        expected_count=3,
    )

    assert selection.names == ("CD3", "CD4", "CD8")
    assert selection.positions == (0, 2, 3)
    with pytest.raises(ValueError, match="source protein orders"):
        select_biological_proteins(
            panel_features=("CD3", "Isotype-1", "CD4", "CD8"),
            panel_is_control=(False, True, False, False),
            joint_protein_names=("CD3", "CD4", "Isotype-1", "CD8"),
            parent_protein_names=("CD3", "Isotype-1", "CD4", "CD8"),
            expected_count=3,
        )


def test_human_lineage_contract_freezes_derivation_and_factual_da_lock():
    """Every executor choice affecting the human cohort is machine-frozen."""
    contract = human_lineage_contract()

    assert contract["expected"]["cells"] == 46_817
    assert contract["expected"]["complete_donors"] == 10
    assert contract["split"] == {
        "group_key": "donor_timepoint",
        "method": "sha256_rank_within_group",
        "salt": "mrtotalvi-v2-redesign-human-lineage-v1",
        "train_fraction": 0.8,
    }
    assert contract["hvg"] == {
        "batch_key": None,
        "chunk_size": 128,
        "clip": "sqrt_training_cells",
        "flavor": "pearson_residuals",
        "n_top_genes": 5_000,
        "theta": 100.0,
        "training_cells_only": True,
    }
    assert contract["shared_parent_metadata"] == [
        "batch",
        "donor",
        "species",
        "timepoint",
    ]
    assert contract["covariate_level_columns"] == [
        "batch",
        "donor",
        "donor_timepoint",
        "species",
        "timepoint",
    ]
    assert contract["factual_human_da"] == "locked_not_computed_or_inspected"
    assert human_lineage_contract_digest() == (
        "607f16dd8080a5e5057ee87888a67a6cdb27a318bfd3374ab424416f555ef2e4"
    )


def test_human_lineage_v2_pins_the_reviewed_inventory_supersession():
    """The refreeze changes only the documented inventory authority chain."""
    legacy = human_lineage_contract()
    prospective = human_lineage_contract_v2()

    assert prospective["schema_version"] == (
        "mrtotalvi-human-lineage-contract-v2"
    )
    assert prospective["supersedes"] == {
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
    assert prospective["source_sha256"] == {
        **legacy["source_sha256"],
        "migration_inventory": (
            "a321d71d15c7f256de22f32dd1d8742b9ca4df64c8dd0a3f5bbcee2cdb418481"
        ),
        "migration_inventory_supersession": (
            "3817dbe8b683af9aa0dceda2ed4daf791cc4e79ba433ac9d5fad3ea4458afcc7"
        ),
    }
    unchanged_keys = set(legacy) - {"schema_version", "source_sha256"}
    assert {
        key: prospective[key]
        for key in unchanged_keys
    } == {
        key: legacy[key]
        for key in unchanged_keys
    }
    assert human_lineage_contract_digest_v2() == (
        "b50f4e3a4322b0c2f16337e40508f8eb3cd2b910963cd36096868e383a6bba78"
    )


def _human_lineage_manifest_payload(
    *,
    contract: dict,
    schema_version: str,
) -> dict:
    return {
        "schema_version": schema_version,
        "run_id": "20260726T120000Z-" + "a" * 8 + "-" + "b" * 8 + "-" + "c" * 8,
        "created_at": "2026-07-26T12:00:00+00:00",
        "status": "complete",
        "base_commit": "d8c8e997a67997a53f55923eb3ab14e6cf06f94c",
        "source_repository_commit": "2e1e9ef708724314ef509d2fe4ede19341d55d6c",
        "contract": contract,
        "source_files": [
            {
                "role": role,
                "path": f"/source/{role}",
                "sha256": contract["source_sha256"][role],
                "bytes": 1,
            }
            for role in contract["source_sha256"]
        ],
        "code_files": [
            {
                "path": "benchmarks/mrtotalvi/human_lineage.py",
                "sha256": "7" * 64,
                "bytes": 1,
            },
            {
                "path": "benchmarks/mrtotalvi/freeze_human_lineage.py",
                "sha256": "8" * 64,
                "bytes": 1,
            },
        ],
        "source_dimensions": {
            "joint_cells": 86_681,
            "parent_cells": 126_299,
            "rna_features": 36_601,
            "source_proteins": 137,
        },
        "selected_dimensions": {
            "cells": 46_817,
            "complete_donors": 10,
            "samples": 20,
            "training_cells": 37_443,
            "heldout_cells": 9_374,
            "genes": 5_000,
            "proteins": 130,
        },
        "covariate_levels": {
            "batch": ["B1", "B2"],
            "donor": [f"D{index}" for index in range(10)],
            "donor_timepoint": [
                f"D{index}_{timepoint}"
                for index in range(10)
                for timepoint in ("W00", "W22")
            ],
            "species": ["human"],
            "timepoint": ["W00", "W22"],
        },
        "digests": {
            name: (
                contract["expected"]["ordered_cell_sha256"]
                if name == "ordered_cells"
                else contract["expected"]["ordered_protein_sha256"]
                if name == "ordered_selected_proteins"
                else character * 64
            )
            for name, character in (
                ("ordered_cells", "a"),
                ("split_assignments", "b"),
                ("ordered_training_cells", "c"),
                ("ordered_heldout_cells", "d"),
                ("ordered_source_genes", "e"),
                ("ordered_selected_genes", "f"),
                ("ordered_source_proteins", "1"),
                ("ordered_selected_proteins", "2"),
                ("selected_rna_counts", "3"),
                ("selected_protein_counts", "4"),
                ("shared_parent_metadata", "5"),
            )
        },
        "count_verification": (
            "joint_parent_exact_on_selected_cells_all_36601_rna_and_137_proteins"
        ),
        "shared_metadata_verification": (
            "joint_parent_exact_on_selected_cells_for_batch_donor_species_timepoint"
        ),
        "factual_human_da": "locked_not_computed_or_inspected",
    }


def test_human_lineage_manifest_rejects_schema_or_contract_drift():
    """The sealed domain manifest cannot accept undeclared executor choices."""
    payload = _human_lineage_manifest_payload(
        contract=human_lineage_contract(),
        schema_version="mrtotalvi-human-lineage-v1",
    )
    assert validate_human_lineage_manifest(payload) == payload

    changed = copy.deepcopy(payload)
    changed["contract"]["split"]["train_fraction"] = 0.75
    with pytest.raises(ValueError, match="contract"):
        validate_human_lineage_manifest(changed)
    changed = copy.deepcopy(payload)
    changed["undeclared"] = True
    with pytest.raises(ValueError, match="Unknown human-lineage manifest"):
        validate_human_lineage_manifest(changed)
    changed = copy.deepcopy(payload)
    changed["covariate_levels"]["donor"].append("extra")
    with pytest.raises(ValueError, match="covariate levels"):
        validate_human_lineage_manifest(changed)


def test_human_lineage_manifest_dispatches_v1_and_v2_without_substitution():
    """Each lineage schema accepts only its exact versioned contract."""
    prospective = _human_lineage_manifest_payload(
        contract=human_lineage_contract_v2(),
        schema_version="mrtotalvi-human-lineage-v2",
    )
    assert validate_human_lineage_manifest(prospective) == prospective

    changed = copy.deepcopy(prospective)
    changed["schema_version"] = "mrtotalvi-human-lineage-v1"
    with pytest.raises(ValueError, match="contract"):
        validate_human_lineage_manifest(changed)

    changed = copy.deepcopy(prospective)
    changed["contract"] = human_lineage_contract()
    with pytest.raises(ValueError, match="contract"):
        validate_human_lineage_manifest(changed)


def test_source_records_are_selected_by_the_explicit_contract(tmp_path):
    """The refreeze hashes exactly the sources declared by its v2 contract."""
    sources = {
        "data": tmp_path / "data.bin",
        "authority": tmp_path / "authority.json",
    }
    sources["data"].write_bytes(b"data")
    sources["authority"].write_bytes(b"authority")
    contract = {
        "source_sha256": {
            role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in sources.items()
        }
    }

    records = _source_records(contract=contract, sources=sources)

    assert [record["role"] for record in records] == ["data", "authority"]
    assert [record["path"] for record in records] == [
        str(path.resolve()) for path in sources.values()
    ]
    assert [record["bytes"] for record in records] == [4, 9]

    changed = copy.deepcopy(contract)
    changed["source_sha256"]["authority"] = "0" * 64
    with pytest.raises(ValueError, match="Source hash drift"):
        _source_records(contract=changed, sources=sources)


def test_refreeze_executor_selects_v2_contract_and_amendment():
    """The approved refreeze binds the correction and never rewrites v1."""
    assert _active_human_lineage_contract() == human_lineage_contract_v2()
    assert list(DEFAULT_SOURCES)[-1] == "migration_inventory_supersession"
    assert DEFAULT_SOURCES["migration_inventory_supersession"].name == (
        "S06E-s01-baseline-inventory-rebinding-completion.json"
    )
    assert LINEAGE_AMENDMENT.name == "human-lineage-amendment-v2.md"
    amendment = LINEAGE_AMENDMENT.read_text(encoding="utf-8")
    assert (
        "a321d71d15c7f256de22f32dd1d8742b9ca4df64c8dd0a3f5bbcee2cdb418481"
        in amendment
    )
    assert (
        "3817dbe8b683af9aa0dceda2ed4daf791cc4e79ba433ac9d5fad3ea4458afcc7"
        in amendment
    )


def test_refreeze_writer_binds_the_active_v2_schema_and_digest():
    """Every newly sealed lineage payload identifies the approved contract."""
    assert _active_human_lineage_version() == {
        "manifest_schema_version": "mrtotalvi-human-lineage-v2",
        "contract": human_lineage_contract_v2(),
        "contract_digest": (
            "b50f4e3a4322b0c2f16337e40508f8eb3cd2b910963cd36096868e383a6bba78"
        ),
    }


def test_refreeze_derivation_uses_the_active_v2_contract(monkeypatch):
    """Source preflight must receive v2 before any cohort data are read."""
    observed = {}

    def record_contract(*, contract, sources=None):
        observed["contract"] = contract
        return []

    def stop_after_source_preflight(*args, **kwargs):
        raise RuntimeError("stop after source preflight")

    monkeypatch.setattr(
        "benchmarks.mrtotalvi.freeze_human_lineage._source_records",
        record_contract,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.freeze_human_lineage.pd.read_csv",
        stop_after_source_preflight,
    )

    with pytest.raises(RuntimeError, match="stop after source preflight"):
        _derive_in_memory()

    assert observed["contract"] == human_lineage_contract_v2()


def test_inventory_supersession_authority_is_semantically_validated(tmp_path):
    """A hash-matched but contradictory correction cannot authorize refreeze."""
    contract = human_lineage_contract_v2()
    payload = _validate_inventory_supersession(
        contract=contract,
        sources=DEFAULT_SOURCES,
    )
    assert payload["correction_id"] == (
        "S06E-s01-baseline-inventory-rebinding-completion"
    )
    assert payload["baseline_rebindings"] == [
        {
            "corrected_sha256": contract["supersedes"][
                "replacement_sha256"
            ],
            "path": "workflow/migration/baseline/input-inventory.tsv",
            "superseded_sha256": contract["supersedes"]["prior_sha256"],
        }
    ]

    changed = copy.deepcopy(payload)
    changed["scope_verified"]["changed_rows"] = 2
    changed_path = tmp_path / "changed-correction.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    changed_sources = {
        **DEFAULT_SOURCES,
        "migration_inventory_supersession": changed_path,
    }
    with pytest.raises(ValueError, match="semantics"):
        _validate_inventory_supersession(
            contract=contract,
            sources=changed_sources,
        )


def test_covariate_levels_are_sorted_unique_and_contract_complete():
    """Registered biological and technical levels are explicit manifest data."""
    levels = covariate_levels(
        {
            "batch": ("B2", "B1", "B2"),
            "donor": ("D2", "D1", "D2"),
            "donor_timepoint": ("D2_W22", "D1_W00", "D2_W22"),
            "species": ("human", "human", "human"),
            "timepoint": ("W22", "W00", "W22"),
        }
    )

    assert levels == {
        "batch": ("B1", "B2"),
        "donor": ("D1", "D2"),
        "donor_timepoint": ("D1_W00", "D2_W22"),
        "species": ("human",),
        "timepoint": ("W00", "W22"),
    }


def test_sparse_count_digest_is_canonical():
    """Storage ordering and explicit zeros cannot change count identity."""
    first = sp.csr_matrix(
        (
            np.asarray([2, 1, 0], dtype=np.int32),
            np.asarray([1, 0, 1], dtype=np.int32),
            np.asarray([0, 3, 3], dtype=np.int32),
        ),
        shape=(2, 3),
    )
    second = sp.csr_matrix(np.asarray([[1, 2, 0], [0, 0, 0]], dtype=np.int32))

    assert sha256_csr_matrix(first) == sha256_csr_matrix(second)


def _write_sparse_group(parent, name, matrix, encoding):
    group = parent.create_group(name)
    group.attrs["encoding-type"] = encoding
    group.attrs["encoding-version"] = "0.1.0"
    group.attrs["shape"] = matrix.shape
    group.create_dataset("data", data=matrix.data)
    group.create_dataset("indices", data=matrix.indices)
    group.create_dataset("indptr", data=matrix.indptr)
    return group


def test_parent_qc_reader_accepts_exact_h5ad_false_true_categorical(tmp_path):
    """Boolean semantics are preserved across H5AD categorical storage."""
    path = tmp_path / "boolean.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("pass_qc")
        group.attrs["encoding-type"] = "categorical"
        group.create_dataset(
            "categories",
            data=np.asarray(["FALSE", "TRUE"], dtype=object),
            dtype=h5py.string_dtype(),
        )
        group.create_dataset(
            "codes",
            data=np.asarray([1, 0, 1], dtype=np.int8),
        )
    with h5py.File(path, "r") as handle:
        values = _read_h5_boolean(handle["pass_qc"])

    np.testing.assert_array_equal(values, [True, False, True])


def test_selected_categorical_reader_ignores_only_unselected_missing_codes(
    tmp_path,
):
    """Missing parent metadata outside the selected cohort is not imputed."""
    path = tmp_path / "categorical.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("donor")
        group.attrs["encoding-type"] = "categorical"
        group.create_dataset(
            "categories",
            data=np.asarray(["D1", "D2"], dtype=object),
            dtype=h5py.string_dtype(),
        )
        group.create_dataset(
            "codes",
            data=np.asarray([-1, 0, 1], dtype=np.int8),
        )
    with h5py.File(path, "r") as handle:
        values = _read_h5_vector_at_positions(
            handle["donor"],
            np.asarray([1, 2], dtype=np.int64),
        )
        np.testing.assert_array_equal(values, ["D1", "D2"])
        with pytest.raises(ValueError, match="selected missing codes"):
            _read_h5_vector_at_positions(
                handle["donor"],
                np.asarray([0], dtype=np.int64),
            )


def test_h5_count_verifier_matches_csr_joint_to_reordered_csc_parent(tmp_path):
    """All-feature equality is checked after exact cell-ID row alignment."""
    joint_values = np.asarray(
        [[1, 0, 2, 0], [0, 3, 0, 0], [4, 0, 0, 5]],
        dtype=np.int64,
    )
    parent_values = np.asarray(
        [
            joint_values[2],
            [9, 9, 9, 9],
            joint_values[0],
            joint_values[1],
        ],
        dtype=np.int64,
    )
    path = tmp_path / "counts.h5"
    with h5py.File(path, "w") as handle:
        joint_group = _write_sparse_group(
            handle,
            "joint",
            sp.csr_matrix(joint_values),
            "csr_matrix",
        )
        parent_group = _write_sparse_group(
            handle,
            "parent",
            sp.csc_matrix(parent_values),
            "csc_matrix",
        )
        selected_joint = _read_csr_rows(
            joint_group,
            np.asarray([0, 2], dtype=np.int64),
            output_chunk_rows=1,
        )
        verified = _verify_all_rna_counts(
            joint_counts=selected_joint,
            parent_group=parent_group,
            parent_positions=(2, 0),
            parent_cells=4,
            column_chunk_size=2,
        )
        np.testing.assert_array_equal(
            verified.toarray(),
            joint_values[[0, 2]],
        )

    with h5py.File(path, "r+") as handle:
        handle["parent/data"][0] += 1
    with h5py.File(path, "r") as handle:
        selected_joint = _read_csr_rows(
            handle["joint"],
            np.asarray([0, 2], dtype=np.int64),
        )
        with pytest.raises(ValueError, match="RNA counts disagree"):
            _verify_all_rna_counts(
                joint_counts=selected_joint,
                parent_group=handle["parent"],
                parent_positions=(2, 0),
                parent_cells=4,
                column_chunk_size=2,
            )


def test_split_evidence_links_table_membership_scores_and_cohort_obs():
    """Independent split verification catches score-only tampering."""
    cell_ids = tuple(f"c{index}" for index in range(6))
    samples = ("S1",) * 3 + ("S2",) * 3
    split_contract = {
        "group_key": "donor_timepoint",
        "method": "sha256_rank_within_group",
        "salt": "fixture-salt",
        "train_fraction": 0.67,
    }
    split = make_within_sample_hash_split(
        cell_ids=cell_ids,
        samples=samples,
        salt=split_contract["salt"],
        train_fraction=split_contract["train_fraction"],
    )
    obs = pd.DataFrame(
        {
            "donor": ("D1",) * 3 + ("D2",) * 3,
            "timepoint": ("W00",) * 3 + ("W22",) * 3,
            "donor_timepoint": samples,
            "lineage_split": split.assignments,
            "lineage_split_sha256": split.scores,
        },
        index=pd.Index(cell_ids),
    )
    table = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "donor": obs["donor"].to_numpy(),
            "timepoint": obs["timepoint"].to_numpy(),
            "donor_timepoint": samples,
            "assignment": split.assignments,
            "selection_sha256": split.scores,
        }
    )

    evidence = _validate_split_table_evidence(
        table=table,
        obs=obs,
        split_contract=split_contract,
    )
    assert evidence["split_assignments"] == split.assignment_sha256
    tampered = table.copy()
    tampered.loc[0, "selection_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="split table"):
        _validate_split_table_evidence(
            table=tampered,
            obs=obs,
            split_contract=split_contract,
        )


def test_hvg_evidence_links_complete_ranking_to_cohort_gene_order():
    """Independent HVG verification catches a rank-preserving row swap."""
    table = pd.DataFrame(
        {
            "gene": ["g0", "g1", "g2", "g3"],
            "source_position": [0, 1, 2, 3],
            "pearson_residual_rank": [0, 1, 2, 3],
            "pearson_residual_variance": [4.0, 3.0, 2.0, 1.0],
            "selected": ["true", "true", "false", "false"],
        }
    )

    evidence = _validate_hvg_table_evidence(
        table=table,
        cohort_genes=("g0", "g1"),
        expected_source_genes=4,
        expected_selected_genes=2,
    )
    assert evidence["ordered_selected_genes"] == sha256_lines(("g0", "g1"))
    tampered = table.copy()
    tampered.loc[[0, 1], "gene"] = ["g1", "g0"]
    with pytest.raises(ValueError, match="HVG"):
        _validate_hvg_table_evidence(
            table=tampered,
            cohort_genes=("g0", "g1"),
            expected_source_genes=4,
            expected_selected_genes=2,
        )
