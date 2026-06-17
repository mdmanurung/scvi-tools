"""Paper-faithful cytometry preprocessing for benchmarks."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PreprocScheme = Literal["minmax", "zscore", "rank"]

ARCSINH_COFACTORS = {
    "nunez": 2000,
    "roider": 500,
    "kreutmair": 2000,
    "mass_cyt": 10,
    "cite": 5,
    "glass_bcell": 5,
}

BENCHMARK_LAYER = "benchmark_input"


def _as_dense(adata, layer: str | None) -> np.ndarray:
    if layer is None:
        return np.asarray(adata.X)
    return np.asarray(adata.layers[layer])


def apply_preproc_scheme(
    adata,
    scheme: PreprocScheme,
    *,
    source_layer: str = "scaled",
    out_layer: str = BENCHMARK_LAYER,
    feature_range: tuple[float, float] = (0.0, 1.0),
):
    """Apply min-max, z-score, or rank scaling on top of arcsinh-transformed expression.

    The paper evaluates CytoVI, Harmony, and cyCombine under three post-arcsinh schemes.
    """
    x = _as_dense(adata, source_layer).copy()
    if scheme == "minmax":
        x = MinMaxScaler(feature_range=feature_range).fit_transform(x)
    elif scheme == "zscore":
        x = StandardScaler().fit_transform(x)
    elif scheme == "rank":
        x = np.apply_along_axis(lambda col: rankdata(col, method="average"), 0, x)
        x = MinMaxScaler(feature_range=feature_range).fit_transform(x)
    else:
        raise ValueError(scheme)
    adata.layers[out_layer] = x
    return adata


def set_expression_from_layer(adata, layer: str):
    """Copy a layer into ``adata.X`` (dense) for methods that read ``X`` directly."""
    adata.X = _as_dense(adata, layer).copy()
    return adata
