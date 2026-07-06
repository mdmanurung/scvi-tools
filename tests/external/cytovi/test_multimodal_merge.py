"""Tests for RNA + cytometry merge utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scvi.external.cytovi.multimodal_merge import (
    build_multimodal_anndata,
    merge_rna_cytof_expression,
    merged_to_anndata,
)
from scvi.external.cytovi.scennep import scennep


def _make_adata(
    n_obs: int,
    markers: list[str],
    rng: np.random.Generator,
    *,
    sample_ids: list[str] | None = None,
) -> AnnData:
    if sample_ids is None:
        sample_ids = ["donor_1", "donor_2"]
    obs = {
        "celltype": rng.choice(["T", "B"], size=n_obs).astype(str),
        "sample_id": rng.choice(sample_ids, size=n_obs).astype(str),
    }
    return AnnData(
        X=rng.normal(size=(n_obs, len(markers))).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=markers),
    )


@pytest.fixture()
def paired_modalities():
    markers = ["CD3", "CD4", "CD8a", "CD19"]
    rng = np.random.default_rng(1)
    rna = _make_adata(30, markers, rng)
    rna = scennep(rna, markers=markers, nn_count=5, npcs=3, copy=True)
    cy = _make_adata(50, markers, rng)
    cy.layers["scaled"] = cy.X.copy()
    return rna, cy, markers


def test_merge_rna_cytof(paired_modalities):
    rna, cy, markers = paired_modalities
    merged = merge_rna_cytof_expression(
        rna,
        cy,
        markers,
        rna_layer="scennep",
        cytof_layer="scaled",
    )
    assert len(merged) == rna.n_obs + cy.n_obs
    assert set(merged["modality"]) == {"RNA", "CyTOF"}
    assert "sample_id" in merged.columns
    for m in markers:
        assert m in merged.columns


def test_merged_to_anndata(paired_modalities):
    rna, cy, markers = paired_modalities
    merged = merge_rna_cytof_expression(rna, cy, markers)
    adata = merged_to_anndata(merged, markers)
    assert adata.n_obs == len(merged)
    assert list(adata.var_names) == markers
    assert "modality" in adata.obs.columns
    assert "sample_id" in adata.obs.columns


def test_build_multimodal_anndata_preserves_sample_id(paired_modalities):
    rna, cy, markers = paired_modalities
    merged = build_multimodal_anndata(rna, cy, markers)
    assert set(merged.obs["modality"]) == {"RNA", "CyTOF"}
    assert "sample_id" in merged.obs.columns
    rna_samples = set(rna.obs["sample_id"])
    cy_samples = set(cy.obs["sample_id"])
    assert set(merged.obs.loc[merged.obs["modality"] == "RNA", "sample_id"]) <= rna_samples
    assert set(merged.obs.loc[merged.obs["modality"] == "CyTOF", "sample_id"]) <= cy_samples
    assert "scaled" in merged.layers
    assert set(merged.obs.loc[merged.obs["modality"] == "RNA", "celltype"]) == {"Unknown"}
