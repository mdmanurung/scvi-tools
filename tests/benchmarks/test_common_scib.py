"""Smoke tests for shared benchmark infrastructure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData

from benchmarks.common.scib import LATENT_OBSM, _finite_dense_matrix, run_scib_benchmark
from benchmarks.common.training import NAN_LAYER, resolve_nan_layer
from benchmarks.cytoanvi.data import holdout_safe_name


def _tiny_adata(n: int = 80, n_batches: int = 2, n_labels: int = 3) -> AnnData:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 10)).astype(np.float32)
    adata = AnnData(x)
    adata.obs["batch"] = [f"b{i % n_batches}" for i in range(n)]
    adata.obs["labels"] = [f"L{i % n_labels}" for i in range(n)]
    adata.obsm[LATENT_OBSM] = rng.normal(size=(n, 5)).astype(np.float32)
    return adata


def test_run_scib_benchmark_returns_aggregates():
    result = run_scib_benchmark(
        _tiny_adata(),
        batch_key="batch",
        label_key="labels",
        subsample_per_batch=200,
        seed=0,
    )
    for key in ("batch_correction", "bio_conservation", "total"):
        assert key in result
        assert np.isfinite(result[key])


def test_run_scib_benchmark_accepts_nan_x_with_finite_embedding():
    adata = _tiny_adata()
    adata.X[0, 0] = np.nan

    result = run_scib_benchmark(
        adata,
        batch_key="batch",
        label_key="labels",
        subsample_per_batch=200,
        seed=0,
    )

    assert np.isfinite(result["total"])


def test_finite_dense_matrix_guards_large_sparse_densification():
    from scipy import sparse

    x = sparse.csr_matrix((20, 20), dtype=np.float32)

    with pytest.raises(MemoryError, match="Refusing to densify sparse matrix"):
        _finite_dense_matrix(x, max_dense_elements=100)


def test_resolve_nan_layer_only_when_present():
    adata = _tiny_adata()

    assert resolve_nan_layer(adata) is None

    adata.layers[NAN_LAYER] = np.ones(adata.shape, dtype=np.float32)
    assert resolve_nan_layer(adata) == NAN_LAYER
    assert resolve_nan_layer(adata, None) is None


@pytest.mark.parametrize(
    ("holdout", "expected"),
    [
        ("Naive CD4 T", "Naive_CD4_T"),
        ("Treg CD69+", "Treg_CD69_plus"),
        ("Treg CD69-", "Treg_CD69_minus"),
    ],
)
def test_holdout_safe_name(holdout, expected):
    assert holdout_safe_name(holdout) == expected


def test_validate_vignette_roider_ok():
    from benchmarks.common.fetch_data import validate_vignette

    data_dir = Path("data")
    if not (data_dir / "Roider_et_al_BNHL_panel1.h5ad").exists():
        pytest.skip("vignette roider data not present")
    report = validate_vignette(data_dir)
    assert report["Roider_et_al_BNHL_panel1.h5ad"]["ok"]
    assert report["Roider_et_al_BNHL_panel2.h5ad"]["ok"]
