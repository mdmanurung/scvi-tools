"""Tests for prepare_paired_cytoanvi (Plan A)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scvi.external.cytovi.marker_harmonization import DEFAULT_GENE_TO_PROTEIN
from scvi.external.cytovi.paired_cytoanvi import prepare_paired_cytoanvi


def _synthetic_pair(
    *,
    rna_samples: list[str],
    cy_samples: list[str],
    n_rna: int = 40,
    n_cy: int = 60,
    seed: int = 0,
) -> tuple[AnnData, AnnData]:
    rng = np.random.default_rng(seed)
    protein_markers = ["CD3", "CD4", "CD8a", "CD19"]
    genes = []
    for p in protein_markers:
        gene = next((g for g, pr in DEFAULT_GENE_TO_PROTEIN.items() if pr == p), p)
        genes.append(gene)

    rna = AnnData(
        X=rng.normal(size=(n_rna, len(genes))).astype(np.float32),
        obs={
            "celltype": rng.choice(["T", "B"], size=n_rna).astype(str),
            "sample_id": rng.choice(rna_samples, size=n_rna).astype(str),
        },
        var=pd.DataFrame(index=genes),
    )
    cy_x = rng.uniform(0, 1, size=(n_cy, len(protein_markers))).astype(np.float32)
    cytof = AnnData(
        X=cy_x,
        obs={
            "celltype": rng.choice(["T", "B"], size=n_cy).astype(str),
            "sample_id": rng.choice(cy_samples, size=n_cy).astype(str),
        },
        var=pd.DataFrame(index=protein_markers),
    )
    cytof.layers["scaled"] = cytof.X.copy()
    return rna, cytof


def test_missing_sample_id_raises():
    rna, cy = _synthetic_pair(rna_samples=["d1"], cy_samples=["d1"])
    del rna.obs["sample_id"]
    with pytest.raises(ValueError, match="sample_id"):
        prepare_paired_cytoanvi(rna, cy, nn_count=5, npcs=3)


def test_missing_labels_key_raises():
    rna, cy = _synthetic_pair(rna_samples=["d1"], cy_samples=["d1"])
    del cy.obs["celltype"]
    with pytest.raises(ValueError, match="celltype"):
        prepare_paired_cytoanvi(rna, cy, nn_count=5, npcs=3)


def test_no_sample_overlap_raises():
    rna, cy = _synthetic_pair(rna_samples=["d1"], cy_samples=["d2"])
    with pytest.raises(ValueError, match="No shared"):
        prepare_paired_cytoanvi(rna, cy, nn_count=5, npcs=3)


def test_partial_pairing_ok():
    rna, cy = _synthetic_pair(
        rna_samples=["shared_a", "shared_b", "rna_only"],
        cy_samples=["shared_a", "shared_b", "cy_only"],
        n_rna=90,
        n_cy=90,
    )
    merged, markers = prepare_paired_cytoanvi(rna, cy, nn_count=5, npcs=3)
    assert merged.n_obs == rna.n_obs + cy.n_obs
    assert set(merged.obs["modality"]) == {"RNA", "CyTOF"}
    shared = {"shared_a", "shared_b"}
    assert shared <= set(merged.obs["sample_id"])
    assert "rna_only" in set(merged.obs.loc[merged.obs["modality"] == "RNA", "sample_id"])
    assert "cy_only" in set(merged.obs.loc[merged.obs["modality"] == "CyTOF", "sample_id"])
    assert len(markers) >= 4
    assert "scaled" in merged.layers


def test_cytof_missing_scaled_raises():
    rna, cy = _synthetic_pair(rna_samples=["d1"], cy_samples=["d1"])
    del cy.layers["scaled"]
    with pytest.raises(ValueError, match="scaled"):
        prepare_paired_cytoanvi(rna, cy, nn_count=5, npcs=3)


def test_rna_labels_masked_unknown():
    rna, cy = _synthetic_pair(rna_samples=["d1", "d2"], cy_samples=["d1", "d2"])
    merged, _ = prepare_paired_cytoanvi(rna, cy, nn_count=5, npcs=3)
    assert set(merged.obs.loc[merged.obs["modality"] == "RNA", "celltype"]) == {"Unknown"}
    assert "eval_celltype" in merged.obs.columns
