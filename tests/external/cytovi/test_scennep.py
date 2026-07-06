"""Tests for scennep pseudobulking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scvi.external.cytovi.scennep import scennep


def _make_adata(x: np.ndarray, var_names: list[str], **obs_kw) -> AnnData:
    return AnnData(
        X=x,
        obs=obs_kw or None,
        var=pd.DataFrame(index=var_names),
    )


@pytest.fixture()
def tiny_rna():
    """Deterministic 40-cell × 8-gene matrix for scennep regression."""
    rng = np.random.default_rng(42)
    n_obs, n_vars = 40, 8
    x = rng.normal(size=(n_obs, n_vars)).astype(np.float32)
    genes = [f"CD{i}" for i in range(n_vars)]
    return _make_adata(
        x,
        genes,
        celltype=rng.choice(["A", "B", "C"], size=n_obs).astype(str),
    )


def test_scennep_output_shape(tiny_rna):
    markers = list(tiny_rna.var_names)
    out = scennep(tiny_rna, markers=markers, nn_count=5, npcs=5, copy=True)
    assert "scennep" in out.layers
    assert out.layers["scennep"].shape == (tiny_rna.n_obs, len(markers))


def test_scennep_changes_expression(tiny_rna):
    markers = list(tiny_rna.var_names)
    out = scennep(tiny_rna, markers=markers, nn_count=5, npcs=5, copy=True)
    raw = np.asarray(tiny_rna.X)
    pb = np.asarray(out.layers["scennep"])
    assert not np.allclose(raw, pb)


def test_scennep_reproducible(tiny_rna):
    markers = list(tiny_rna.var_names)
    kw = dict(markers=markers, nn_count=5, npcs=5, distance="cosine", copy=True)
    a = scennep(tiny_rna, **kw)
    b = scennep(tiny_rna, **kw)
    np.testing.assert_allclose(a.layers["scennep"], b.layers["scennep"], rtol=1e-5)


def test_scennep_smoothing_invariants(tiny_rna):
    """k-NN smoothed output must be finite and lie within the input value range.

    scennep computes a weighted average of neighbour values (positive weights summing to 1),
    so each output value is a convex combination of inputs — it cannot fall outside
    [min_input, max_input] for any marker.  This test is platform-independent because it
    checks mathematical invariants of the smoothing operation, not floating-point bitwise
    equivalence (which depends on the LAPACK backend used for PCA).
    """
    markers = list(tiny_rna.var_names)
    out = scennep(tiny_rna, markers=markers, nn_count=5, npcs=5, distance="cosine", copy=True)
    pb = np.asarray(out.layers["scennep"])
    raw = np.asarray(tiny_rna.X)
    assert np.all(np.isfinite(pb)), "scennep output contains non-finite values"
    assert pb.min() >= raw.min() - 1e-4, "smoothed values below input minimum — violates convex smoothing"
    assert pb.max() <= raw.max() + 1e-4, "smoothed values above input maximum — violates convex smoothing"


def test_scennep_fail_empty_markers(tiny_rna):
    with pytest.raises(ValueError, match="non-empty"):
        scennep(tiny_rna, markers=[], copy=True)


def test_scennep_fail_missing_markers(tiny_rna):
    with pytest.raises(ValueError, match="not found"):
        scennep(tiny_rna, markers=["NOT_A_MARKER"], copy=True)


def test_scennep_fail_too_few_cells():
    genes = [f"G{i}" for i in range(5)]
    adata = _make_adata(np.ones((1, 5)), genes)
    with pytest.raises(ValueError, match="Cannot run PCA"):
        scennep(adata, markers=genes, copy=True)


def test_scennep_fail_lognorm_with_layer(tiny_rna):
    with pytest.raises(ValueError, match="layer=None"):
        scennep(tiny_rna, markers=list(tiny_rna.var_names), flavor="lognorm", layer="X", copy=True)
