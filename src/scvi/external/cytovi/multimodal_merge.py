"""Merge scennep-smoothed RNA with cytometry for joint integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData


def _as_marker_frame(
    adata: AnnData,
    markers: list[str],
    *,
    layer: str,
    modality_value: str,
    sample_key: str,
    celltype_key: str | None,
) -> pd.DataFrame:
    if layer not in adata.layers and layer != "X":
        raise ValueError(
            f"Layer {layer!r} not found in adata.layers. Available: {list(adata.layers.keys())}"
        )
    if sample_key not in adata.obs:
        raise ValueError(
            f"sample_key={sample_key!r} not in adata.obs. Available: {list(adata.obs.columns)}"
        )
    sub = adata[:, markers]
    if layer == "X":
        mat = sub.X
    else:
        mat = sub.layers[layer]
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    df = pd.DataFrame(np.asarray(mat), index=sub.obs_names, columns=markers)
    df["modality"] = modality_value
    df[sample_key] = adata.obs.loc[sub.obs_names, sample_key].astype(str).values
    if celltype_key is not None:
        if celltype_key not in adata.obs:
            raise ValueError(
                f"celltype_key={celltype_key!r} not in adata.obs. "
                f"Available: {list(adata.obs.columns)}"
            )
        df["celltype"] = adata.obs.loc[sub.obs_names, celltype_key].astype(str).values
    df["id"] = np.arange(len(df), dtype=np.int64)
    return df


def merge_rna_cytof_expression(
    rna_adata: AnnData,
    cytof_adata: AnnData,
    markers: list[str],
    *,
    sample_key: str = "sample_id",
    rna_layer: str = "scennep",
    cytof_layer: str = "scaled",
    rna_modality: str = "RNA",
    cytof_modality: str = "CyTOF",
    rna_celltype_key: str | None = "celltype",
    cytof_celltype_key: str | None = "celltype",
) -> pd.DataFrame:
    """Stack RNA and cytometry cells in shared marker space (long table for export/debug).

    Parameters
    ----------
    rna_adata
        RNA AnnData with scennep-smoothed expression in ``rna_layer``.
    cytof_adata
        Cytometry AnnData with scaled protein intensities.
    markers
        Shared protein-named marker list (required).
    sample_key
        Column in ``obs`` with paired sample / donor identifiers.

    Returns
    -------
    DataFrame with marker columns, ``modality``, ``sample_key``, ``celltype``, ``id``.
    """
    if not markers:
        raise ValueError("markers must be a non-empty list.")
    for name, adata, layer in (
        ("RNA", rna_adata, rna_layer),
        ("CyTOF", cytof_adata, cytof_layer),
    ):
        missing = [m for m in markers if m not in adata.var_names]
        if missing:
            raise ValueError(f"{name} adata missing markers: {missing}")

    rna_df = _as_marker_frame(
        rna_adata,
        markers,
        layer=rna_layer,
        modality_value=rna_modality,
        sample_key=sample_key,
        celltype_key=rna_celltype_key,
    )
    cy_df = _as_marker_frame(
        cytof_adata,
        markers,
        layer=cytof_layer,
        modality_value=cytof_modality,
        sample_key=sample_key,
        celltype_key=cytof_celltype_key,
    )
    cy_df["id"] = np.arange(len(rna_df), len(rna_df) + len(cy_df), dtype=np.int64)
    return pd.concat([rna_df, cy_df], axis=0, ignore_index=True)


def build_multimodal_anndata(
    rna_adata: AnnData,
    cytof_adata: AnnData,
    markers: list[str],
    *,
    sample_key: str = "sample_id",
    rna_layer: str = "scennep",
    cytof_layer: str = "scaled",
    rna_modality: str = "RNA",
    cytof_modality: str = "CyTOF",
    scaled_layer: str = "scaled",
    labels_key: str = "celltype",
    rna_unlabeled: str = "Unknown",
) -> AnnData:
    """Build a single AnnData for CytoVI/CytoANVI from scennep RNA + cytometry.

    RNA cells receive ``labels_key=rna_unlabeled``; cytometry keeps observed labels.
    ``obs["modality"]`` is the technology batch key; ``obs[sample_key]`` is preserved.
    """
    import anndata as ad

    if sample_key not in rna_adata.obs:
        raise ValueError(
            f"RNA adata missing obs[{sample_key!r}]. Available: {list(rna_adata.obs.columns)}"
        )
    if sample_key not in cytof_adata.obs:
        raise ValueError(
            f"CyTOF adata missing obs[{sample_key!r}]. Available: {list(cytof_adata.obs.columns)}"
        )
    if rna_layer not in rna_adata.layers:
        raise ValueError(
            f"RNA layer {rna_layer!r} missing. Run scennep first. "
            f"Available: {list(rna_adata.layers.keys())}"
        )
    if cytof_layer not in cytof_adata.layers:
        raise ValueError(
            f"CyTOF layer {cytof_layer!r} missing. Available: {list(cytof_adata.layers.keys())}"
        )

    rna_sub = rna_adata[:, markers].copy()
    cy_sub = cytof_adata[:, markers].copy()
    rna_sub.layers[scaled_layer] = np.asarray(rna_sub.layers[rna_layer], dtype=np.float32)
    if labels_key not in rna_sub.obs:
        raise ValueError(f"RNA adata must have obs[{labels_key!r}] for evaluation metrics.")
    rna_sub.obs["eval_celltype"] = rna_sub.obs[labels_key].astype(str)
    rna_sub.obs[labels_key] = rna_unlabeled
    rna_sub.obs["modality"] = rna_modality
    rna_sub.obs[sample_key] = rna_adata.obs.loc[rna_sub.obs_names, sample_key].astype(str)

    cy_sub.layers[scaled_layer] = np.asarray(cy_sub.layers[cytof_layer], dtype=np.float32)
    if labels_key not in cy_sub.obs:
        raise ValueError(f"CyTOF adata must have obs[{labels_key!r}] for semi-supervised training.")
    cy_sub.obs["eval_celltype"] = cy_sub.obs[labels_key].astype(str)
    cy_sub.obs["modality"] = cytof_modality
    cy_sub.obs[sample_key] = cytof_adata.obs.loc[cy_sub.obs_names, sample_key].astype(str)

    merged = ad.concat([rna_sub, cy_sub], join="inner", index_unique="-")
    merged.obs["modality"] = merged.obs["modality"].astype(str)
    merged.obs[sample_key] = merged.obs[sample_key].astype(str)
    merged.obs["eval_celltype"] = merged.obs["eval_celltype"].astype(str)
    return merged


def merged_to_anndata(
    merged: pd.DataFrame,
    markers: list[str],
    *,
    sample_key: str = "sample_id",
) -> AnnData:
    """Convert a merged expression table to AnnData for CytoVI / CytoANVI training."""
    import anndata as ad

    meta_cols = {"modality", sample_key, "celltype", "id"}
    extra = [c for c in merged.columns if c not in markers and c not in meta_cols]
    if extra:
        raise ValueError(f"Unexpected non-marker columns in merged table: {extra}")
    x = merged[markers].to_numpy(dtype=np.float32)
    obs_cols = [c for c in ("modality", sample_key, "celltype", "id") if c in merged.columns]
    obs = merged[obs_cols].copy()
    return ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=markers))
