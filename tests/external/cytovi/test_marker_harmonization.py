"""Tests for protein ↔ gene marker harmonization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scvi.external.cytovi.marker_harmonization import (
    DEFAULT_PROTEIN_GENE_MAP,
    collapse_cd45_markers,
    harmonize_marker_intersection,
    rename_rna_to_protein_names,
    shared_markers,
)


def _make_adata(genes: list[str], n_obs: int = 20) -> AnnData:
    x = np.random.default_rng(0).normal(size=(n_obs, len(genes))).astype(np.float32)
    return AnnData(X=x, var=pd.DataFrame(index=genes))


@pytest.fixture()
def rna_adata():
    return _make_adata(["PTPRC", "MS4A1", "CD3E", "CD4", "CD8A", "GAPDH"])


def test_shared_markers(rna_adata):
    cytof = ["CD45", "CD20", "CD3", "CD4", "CD8a", "CD19"]
    got = shared_markers(rna_adata.var_names, cytof)
    assert "CD45" in got
    assert "CD20" in got
    assert "GAPDH" not in got


def test_shared_markers_fail_empty():
    with pytest.raises(ValueError, match="No shared markers"):
        shared_markers(["GAPDH"], ["CD45"])


def test_rename_rna_to_protein(rna_adata):
    from scvi.external.cytovi.marker_harmonization import DEFAULT_GENE_TO_PROTEIN

    out = rename_rna_to_protein_names(rna_adata, DEFAULT_GENE_TO_PROTEIN, inplace=False)
    assert "CD45" in out.var_names
    assert "CD20" in out.var_names
    assert "GAPDH" in out.var_names


def test_harmonize_marker_intersection(rna_adata):
    cytof = ["CD45", "CD20", "CD3", "CD4", "CD8a"]
    rna, markers = harmonize_marker_intersection(rna_adata, cytof)
    assert set(markers).issubset(set(cytof))
    assert set(markers).issubset(set(rna.var_names))


def test_collapse_cd45():
    df = pd.DataFrame(
        {
            "CD45RA": [1.0, 2.0],
            "CD45RO": [3.0, 1.0],
            "CD3": [0.5, 0.5],
        }
    )
    out = collapse_cd45_markers(df)
    assert "CD45" in out.columns
    assert "CD45RA" not in out.columns
    assert out["CD45"].tolist() == [3.0, 2.0]
