"""Protein ↔ gene marker harmonization for RNA + cytometry integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

# Protein name → Ensembl/gene symbol (cyCombine cocluster vignette).
DEFAULT_PROTEIN_GENE_MAP: dict[str, str] = {
    "CD45": "PTPRC",
    "CD11b": "ITGAM",
    "CD79b": "CD79B",
    "CD11c": "ITGAX",
    "IgD": "IGHD",
    "CD16": "FCGR3A",
    "CD25": "IL2RA",
    "CD43": "SPN",
    "CD103": "ITGAE",
    "Tbet": "TBX21",
    "FoxP3": "FOXP3",
    "CD161": "KLRB1",
    "CD8a": "CD8A",
    "CD49a": "ITGA1",
    "CD3": "CD3E",
    "CD20": "MS4A1",
    "HLA-DR": "HLA-DRA",  # HLADR alias dropped; inverse dict kept unambiguous
    "CD56": "NCAM1",
    "CD4": "CD4",
    "CD19": "CD19",
    "CD14": "CD14",
    "CD68": "CD68",
    "CD27": "CD27",
    "CD38": "CD38",
    "CD69": "CD69",
    "CD86": "CD86",
    "CD33": "CD33",
}

# Gene symbol → protein name for renaming RNA ``var_names`` (one protein per gene).
DEFAULT_GENE_TO_PROTEIN: dict[str, str] = {
    gene: protein for protein, gene in DEFAULT_PROTEIN_GENE_MAP.items()
}


def validate_mapping(mapping: dict[str, str]) -> dict[str, str]:
    if not mapping:
        raise ValueError("mapping must be a non-empty dict of protein_name -> gene_symbol.")
    bad = {k: v for k, v in mapping.items() if not k or not v}
    if bad:
        raise ValueError(f"Invalid empty keys/values in mapping: {bad}")
    return dict(mapping)


def validate_gene_to_protein(mapping: dict[str, str]) -> dict[str, str]:
    if not mapping:
        raise ValueError("gene_to_protein must be a non-empty dict.")
    return dict(mapping)


def rename_rna_to_protein_names(
    adata: AnnData,
    gene_to_protein: dict[str, str] | None = None,
    *,
    inplace: bool = False,
) -> AnnData:
    """Rename ``adata.var_names`` from gene symbols to protein names."""
    gene_to_protein = validate_gene_to_protein(gene_to_protein or DEFAULT_GENE_TO_PROTEIN)
    out = adata if inplace else adata.copy()
    new_names = []
    for name in out.var_names:
        if name in gene_to_protein:
            new_names.append(gene_to_protein[name])
        else:
            new_names.append(name)
    if len(set(new_names)) != len(new_names):
        dupes = pd.Series(new_names).value_counts()
        dupes = dupes[dupes > 1].index.tolist()
        raise ValueError(
            f"Renaming genes to protein names produced duplicate var_names: {dupes}. "
            "Collapse isoforms (e.g. CD45RA/CD45RO → CD45) before renaming."
        )
    out.var_names = new_names
    return out


def collapse_cd45_markers(
    expr_df: pd.DataFrame,
    *,
    cd45_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Collapse CD45RA/CD45RO into a single CD45 column (element-wise max).

    .. warning::
        CD45RA and CD45RO are T-cell differentiation markers that distinguish naive
        (CD45RA+) from memory (CD45RO+) T cells.  Collapsing them into a single CD45 max
        column **permanently loses that distinction** in the resulting expression matrix.
        Only call this function when cross-panel harmonization requires a shared CD45
        backbone and the downstream analysis does not need to resolve naive vs memory T
        cells.  If in doubt, keep CD45RA and CD45RO as separate columns.
    """
    cols = cd45_columns or [c for c in ("CD45", "CD45RA", "CD45RO") if c in expr_df.columns]
    if not cols:
        return expr_df
    out = expr_df.copy()
    cd45 = out[cols].max(axis=1)
    out = out.drop(columns=[c for c in cols if c != "CD45"], errors="ignore")
    out["CD45"] = cd45
    return out


def shared_markers(
    rna_features: list[str] | np.ndarray,
    cytof_features: list[str] | np.ndarray,
    mapping: dict[str, str] | None = None,
    gene_to_protein: dict[str, str] | None = None,
) -> list[str]:
    """Return protein-named markers present in both modalities after gene→protein mapping."""
    mapping = validate_mapping(mapping or DEFAULT_PROTEIN_GENE_MAP)
    gene_to_protein = validate_gene_to_protein(gene_to_protein or DEFAULT_GENE_TO_PROTEIN)
    rna_set = set(rna_features)
    rna_as_protein = {gene_to_protein.get(g, g) for g in rna_set}
    cytof_set = set(cytof_features)
    shared = sorted(rna_as_protein & cytof_set)
    if not shared:
        raise ValueError(
            "No shared markers between RNA (after gene→protein mapping) and cytometry. "
            f"RNA mapped sample: {sorted(rna_as_protein)[:15]}… "
            f"Cytometry sample: {sorted(cytof_set)[:15]}…"
        )
    return shared


def harmonize_marker_intersection(
    rna_adata: AnnData,
    cytof_markers: list[str],
    gene_to_protein: dict[str, str] | None = None,
    mapping: dict[str, str] | None = None,
) -> tuple[AnnData, list[str]]:
    """Rename RNA genes to protein names and return shared marker list.

    Raises if the intersection is empty or renaming creates duplicates.
    """
    gene_to_protein = validate_gene_to_protein(gene_to_protein or DEFAULT_GENE_TO_PROTEIN)
    rna = rename_rna_to_protein_names(rna_adata, gene_to_protein, inplace=False)
    markers = shared_markers(
        rna.var_names,
        cytof_markers,
        mapping=mapping,
        gene_to_protein=gene_to_protein,
    )
    missing_in_rna = [m for m in markers if m not in rna.var_names]
    if missing_in_rna:
        raise ValueError(f"Shared markers missing from RNA after rename: {missing_in_rna}")
    return rna, markers
