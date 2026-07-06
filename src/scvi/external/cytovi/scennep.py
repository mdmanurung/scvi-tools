"""Single-cell nearest-neighbor pseudobulking (scennep).

Python port of the R package ``scennep`` (https://github.com/shdam/scennep). For each cell,
aggregates expression with its shared-nearest-neighbor (SNN) graph neighbors using SNN edge
weights — intended to smooth scRNA-seq dropouts before assimilating RNA to cytometry data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import logging

import numpy as np
import scanpy as sc
from scipy import sparse

if TYPE_CHECKING:
    from anndata import AnnData

from ._utils import validate_layer_key

logger = logging.getLogger(__name__)

DistanceMetric = Literal["cosine", "euclidean", "manhattan"]
NormalizationFlavor = Literal["lognorm", "none"]


def _resolve_markers(adata: AnnData, markers: list[str]) -> list[str]:
    if not markers:
        raise ValueError("markers must be a non-empty list of feature names.")
    missing = [m for m in markers if m not in adata.var_names]
    if missing:
        raise ValueError(
            f"markers not found in adata.var_names: {missing}. "
            f"Available (first 20): {list(adata.var_names[:20])}"
        )
    return list(markers)


def _select_npcs(adata: AnnData, pc_explained: float, npcs: int | None) -> int:
    if npcs is not None:
        if npcs < 1:
            raise ValueError(f"npcs must be >= 1, got {npcs}.")
        return int(npcs)
    if "pca" not in adata.uns:
        raise ValueError("PCA not found in adata.uns; run scennep with valid expression first.")
    variance_ratio = np.asarray(adata.uns["pca"]["variance_ratio"])
    if variance_ratio.size == 0:
        raise ValueError("PCA variance_ratio is empty; cannot select npcs automatically.")
    cumulative = np.cumsum(variance_ratio)
    hits = np.where(cumulative >= pc_explained)[0]
    if hits.size == 0:
        raise ValueError(
            f"PCA cannot explain pc_explained={pc_explained} of variance "
            f"(max cumulative={float(cumulative[-1]):.4f}). Pass explicit npcs=."
        )
    return int(hits[0] + 1)


def _prepare_expression(
    adata: AnnData,
    *,
    layer: str | None,
    flavor: NormalizationFlavor,
    markers: list[str],
) -> AnnData:
    """Return a copy with normalized expression in ``adata.X`` for PCA."""
    work = adata[:, markers].copy()
    if flavor == "lognorm":
        if layer is not None:
            raise ValueError("Pass layer=None when flavor='lognorm'.")
        sc.pp.normalize_total(work, target_sum=1e4)
        sc.pp.log1p(work)
    elif flavor == "none":
        if layer is not None:
            validate_layer_key(work, layer)
            work.X = work.layers[layer]
    else:
        raise ValueError(f"Unknown flavor={flavor!r}; expected 'lognorm' or 'none'.")
    return work


def _build_snn_graph(
    adata: AnnData,
    *,
    nn_count: int,
    npcs: int,
    distance: DistanceMetric,
) -> sparse.csr_matrix:
    if nn_count < 1:
        raise ValueError(f"nn_count must be >= 1, got {nn_count}.")
    if adata.n_obs < 2:
        raise ValueError(f"scennep requires at least 2 cells, got n_obs={adata.n_obs}.")
    if npcs >= adata.n_obs:
        raise ValueError(
            f"npcs={npcs} must be < n_obs={adata.n_obs} for neighborhood graph construction."
        )
    metric = "cosine" if distance == "cosine" else distance
    sc.pp.neighbors(
        adata,
        n_neighbors=nn_count,
        n_pcs=npcs,
        use_rep="X_pca",
        metric=metric,
    )
    graph = adata.obsp["connectivities"]
    if not sparse.issparse(graph):
        graph = sparse.csr_matrix(graph)
    return graph.tocsr()


def _pseudobulk_expression(
    expr: np.ndarray,
    snn_graph: sparse.csr_matrix,
    *,
    nn_top: int,
    nn_cutoff: float,
) -> np.ndarray:
    """Weighted neighbor aggregation (cells × genes)."""
    n_cells, n_genes = expr.shape
    out = np.empty((n_cells, n_genes), dtype=np.float64)
    for cell_id in range(n_cells):
        row = snn_graph.getrow(cell_id)
        mask = row.data > nn_cutoff
        if not np.any(mask):
            raise ValueError(
                f"Cell index {cell_id} has no SNN neighbors above nn_cutoff={nn_cutoff}. "
                "Lower nn_cutoff or increase nn_count."
            )
        neighbor_idx = row.indices[mask]
        weights = row.data[mask].astype(np.float64)
        order = np.argsort(-weights)
        take = min(nn_top, neighbor_idx.size)
        neighbor_idx = neighbor_idx[order[:take]]
        weights = weights[order[:take]]
        if neighbor_idx.size == 1:
            out[cell_id] = expr[neighbor_idx[0]]
            continue
        neighbor_expr = expr[neighbor_idx]
        out[cell_id] = (neighbor_expr * weights[:, None]).sum(axis=0) / weights.sum()
    return out


def scennep(
    adata: AnnData,
    *,
    markers: list[str],
    layer: str | None = None,
    output_layer: str = "scennep",
    nn_count: int = 20,
    nn_top: int | None = None,
    nn_cutoff: float = 1 / 5,
    pc_explained: float = 0.90,
    npcs: int | None = None,
    distance: DistanceMetric = "cosine",
    flavor: NormalizationFlavor = "none",
    copy: bool = False,
) -> AnnData:
    """Pseudobulk each cell with its SNN neighbors (scennep).

    Parameters
    ----------
    adata
        Annotated data matrix (cells × genes).
    markers
        Feature names to retain for pseudobulking (required; no auto-selection).
    layer
        Layer with normalized expression when ``flavor='none'``. If ``None``, uses ``adata.X``.
    output_layer
        Layer key where pseudobulked expression is stored.
    nn_count
        ``k`` for the neighborhood graph (matches R ``nn_count``).
    nn_top
        Number of top SNN neighbors to aggregate. Defaults to ``nn_count``.
    nn_cutoff
        Minimum SNN edge weight to include a neighbor.
    pc_explained
        Fraction of variance for automatic PC selection when ``npcs`` is ``None``.
    npcs
        Explicit number of PCs. Required when automatic selection cannot reach ``pc_explained``.
    distance
        Distance metric for the neighbor search (``cosine`` or ``euclidean``).
    flavor
        ``'none'`` uses ``layer``/``X`` as-is; ``'lognorm'`` applies normalize_total + log1p.
    copy
        If ``True``, return a copy; otherwise modify ``adata`` in place.

    Returns
    -------
    AnnData with ``adata.layers[output_layer]`` set to pseudobulked expression (cells × markers).
    """
    if nn_top is None:
        nn_top = nn_count
    if nn_top < 1:
        raise ValueError(f"nn_top must be >= 1, got {nn_top}.")

    markers = _resolve_markers(adata, markers)
    out = adata.copy() if copy else adata

    work = _prepare_expression(out, layer=layer, flavor=flavor, markers=markers)
    max_comps = min(work.n_obs - 1, work.n_vars - 1, 50)
    if max_comps < 1:
        raise ValueError(
            f"Cannot run PCA: n_obs={work.n_obs}, n_vars={work.n_vars} — need both >= 2."
        )

    # Save raw expression before z-scoring — sc.pp.scale modifies work.X in-place, so
    # pseudobulk must use the pre-scale values, not the z-scores used for PCA/SNN.
    _raw = work.X
    orig_expr = np.asarray(_raw.todense() if sparse.issparse(_raw) else _raw, dtype=np.float64)

    sc.pp.scale(work, zero_center=True, max_value=10)
    sc.tl.pca(work, n_comps=max_comps, svd_solver="arpack")
    selected_npcs = min(_select_npcs(work, pc_explained, npcs), max_comps)

    snn_graph = _build_snn_graph(
        work,
        nn_count=nn_count,
        npcs=selected_npcs,
        distance=distance,
    )

    expr = orig_expr

    pseudobulk = _pseudobulk_expression(
        expr,
        snn_graph,
        nn_top=nn_top,
        nn_cutoff=nn_cutoff,
    )
    out.layers[output_layer] = pseudobulk
    logger.info(
        "scennep: pseudobulked %d cells × %d markers (k=%d, npcs=%d, distance=%s)",
        out.n_obs,
        len(markers),
        nn_count,
        selected_npcs,
        distance,
    )
    return out
