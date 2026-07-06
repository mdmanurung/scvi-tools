"""Integration baselines for Track A (paper Figure 2E).

cyCombinePy (https://github.com/mdmanurung/cyCombinePy) is used for **batch correction only**.
It does not impute missing markers — A3 imputation baselines are CytoVI + KNN only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from benchmarks.common.preprocessing import (
    BENCHMARK_LAYER,
    apply_preproc_scheme,
    set_expression_from_layer,
)
from benchmarks.common.scib import LATENT_OBSM, pca_embedding

if TYPE_CHECKING:
    from benchmarks.common.preprocessing import PreprocScheme

CYCOMBINE_LAYER = "cycombine_corrected"

# Map paper preprocessing names to cyCombinePy ``norm_method``.
_CYCOMBINE_NORM: dict[PreprocScheme, str] = {
    "minmax": "scale",
    "zscore": "scale",
    "rank": "rank",
}


def run_harmony(
    adata,
    *,
    batch_key: str,
    scheme: PreprocScheme,
    source_layer: str = "scaled",
    max_iter_harmony: int = 20,
    n_pcs: int = 50,
):
    """Harmony on PCA of preprocessed expression; embedding in ``obsm[X_benchmark]``."""
    import harmonypy as hm

    a = adata.copy()
    apply_preproc_scheme(a, scheme, source_layer=source_layer, out_layer=BENCHMARK_LAYER)
    set_expression_from_layer(a, BENCHMARK_LAYER)
    a = pca_embedding(a, n_comps=n_pcs, layer=None, obsm_key="X_pca")
    meta = a.obs[[batch_key]].copy()
    ho = hm.run_harmony(a.obsm["X_pca"], meta, batch_key, max_iter_harmony=max_iter_harmony)
    a.obsm[LATENT_OBSM] = np.asarray(ho.Z_corr)
    return a


def run_cycombinepy(
    adata,
    *,
    batch_key: str,
    scheme: PreprocScheme,
    source_layer: str = "scaled",
    n_pcs: int = 50,
    seed: int = 473,
):
    """CyCombinePy batch correction (Python port of cyCombine); PCA embedding for scib.

    Requires ``pip install cycombinepy`` (see benchmarks/cytovi/README.md).
    """
    import cycombinepy as cc

    a = adata.copy()
    apply_preproc_scheme(a, scheme, source_layer=source_layer, out_layer=BENCHMARK_LAYER)
    set_expression_from_layer(a, BENCHMARK_LAYER)

    norm_method = _CYCOMBINE_NORM[scheme]
    cc.batch_correct(
        a,
        batch_key=batch_key,
        norm_method=norm_method,
        seed=seed,
        out_layer=CYCOMBINE_LAYER,
    )
    a = pca_embedding(a, n_comps=n_pcs, layer=CYCOMBINE_LAYER, obsm_key=LATENT_OBSM)
    return a


def cycombinepy_available() -> bool:
    """Return whether cyCombinePy can be imported in the current environment."""
    try:
        import cycombinepy  # noqa: F401

        return True
    except ImportError:
        return False
