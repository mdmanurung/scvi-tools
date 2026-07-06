"""Paired scRNA-seq + CyTOF preprocessing for CytoANVI (Plan A)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

from .marker_harmonization import harmonize_marker_intersection
from .multimodal_merge import build_multimodal_anndata
from .scennep import scennep

ScennepDistance = Literal["cosine", "euclidean"]


def _scale_rna_minmax(rna: AnnData, markers: list[str], layer: str = "scaled") -> None:
    """Per-marker min-max scale RNA to [0, 1] on shared markers (in-place)."""
    mat = rna[:, markers].X
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    x = np.asarray(mat, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("RNA expression contains NaN or inf before scaling.")
    xmin = x.min(axis=0)
    xmax = x.max(axis=0)
    span = xmax - xmin
    if np.any(span <= 0):
        bad = [markers[i] for i in np.where(span <= 0)[0]]
        raise ValueError(f"RNA markers with zero variance after subsetting: {bad}")
    scaled = ((x - xmin) / span).astype(np.float32)
    rna.layers[layer] = scaled


def prepare_paired_cytoanvi(
    rna_adata: AnnData,
    cytof_adata: AnnData,
    *,
    sample_key: str = "sample_id",
    labels_key: str = "celltype",
    unlabeled_category: str = "Unknown",
    markers: list[str] | None = None,
    nn_count: int = 20,
    scennep_distance: ScennepDistance = "cosine",
    npcs: int | None = None,
) -> tuple[AnnData, list[str]]:
    """Prepare paired scRNA + CyTOF for CytoANVI training.

    Harmonizes markers, scales RNA, runs scennep on RNA, and merges with cytometry.
    Requires ``obs[sample_key]`` on both objects and at least one shared sample ID.
    CyTOF must have ``layers['scaled']`` (arcsinh + min-max via ``cytovi`` preprocessing).

    Parameters
    ----------
    rna_adata
        scRNA-seq AnnData (gene symbols or pre-renamed protein names).
    cytof_adata
        CyTOF AnnData with protein marker names and ``layers['scaled']``.
    sample_key
        Column with paired donor / draw identifiers.
    labels_key
        Cell-type column; RNA values are masked to ``unlabeled_category`` for training.
    unlabeled_category
        Label for RNA cells during semi-supervised training.
    markers
        Explicit shared **protein-named** markers present in both objects; if ``None``,
        computed via gene→protein harmonization.
    nn_count
        scennep neighborhood size.
    scennep_distance
        scennep graph distance metric.
    npcs
        Explicit PC count for scennep; ``None`` selects from variance explained.

    Returns
    -------
    merged AnnData ready for ``CytoANVI.setup_anndata(..., batch_key='modality', sample_key=sample_key)``
    and the shared marker list.
    """
    if sample_key not in rna_adata.obs:
        raise ValueError(
            f"RNA adata missing obs[{sample_key!r}]. Available: {list(rna_adata.obs.columns)}"
        )
    if sample_key not in cytof_adata.obs:
        raise ValueError(
            f"CyTOF adata missing obs[{sample_key!r}]. Available: {list(cytof_adata.obs.columns)}"
        )
    if "scaled" not in cytof_adata.layers:
        raise ValueError(
            "CyTOF adata must have layers['scaled'] (arcsinh + min-max). "
            f"Available layers: {list(cytof_adata.layers.keys())}"
        )
    if labels_key not in rna_adata.obs:
        raise ValueError(
            f"RNA adata missing obs[{labels_key!r}] for evaluation metrics. "
            f"Available: {list(rna_adata.obs.columns)}"
        )
    if labels_key not in cytof_adata.obs:
        raise ValueError(
            f"CyTOF adata missing obs[{labels_key!r}] for semi-supervised training. "
            f"Available: {list(cytof_adata.obs.columns)}"
        )

    rna_samples = set(rna_adata.obs[sample_key].astype(str).unique())
    cy_samples = set(cytof_adata.obs[sample_key].astype(str).unique())
    shared_samples = rna_samples & cy_samples
    if not shared_samples:
        raise ValueError(
            f"No shared {sample_key} between RNA and CyTOF. "
            f"RNA samples: {sorted(rna_samples)[:10]}… CyTOF samples: {sorted(cy_samples)[:10]}…"
        )

    if markers is None:
        rna_h, markers = harmonize_marker_intersection(rna_adata, list(cytof_adata.var_names))
    else:
        rna_h = rna_adata.copy()
        missing_rna = [m for m in markers if m not in rna_h.var_names]
        missing_cy = [m for m in markers if m not in cytof_adata.var_names]
        if missing_rna:
            raise ValueError(f"markers missing from RNA: {missing_rna}")
        if missing_cy:
            raise ValueError(f"markers missing from CyTOF: {missing_cy}")

    rna_work = rna_h[:, markers].copy()
    _scale_rna_minmax(rna_work, markers, layer="scaled")

    scennep_kw: dict = {
        "markers": markers,
        "layer": "scaled",
        "output_layer": "scennep",
        "nn_count": nn_count,
        "distance": scennep_distance,
        "copy": False,
    }
    if npcs is not None:
        scennep_kw["npcs"] = npcs
    scennep(rna_work, **scennep_kw)

    merged = build_multimodal_anndata(
        rna_work,
        cytof_adata,
        markers,
        sample_key=sample_key,
        rna_layer="scennep",
        cytof_layer="scaled",
        labels_key=labels_key,
        rna_unlabeled=unlabeled_category,
    )
    return merged, markers
