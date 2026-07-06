from __future__ import annotations

import sys
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from benchmarks.common.training import SCALED_LAYER
from benchmarks.cytoanvi import baselines


def _label_transfer_adata() -> ad.AnnData:
    X = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [9.0, 9.0],
            [9.1, 9.0],
            [0.2, 0.1],
            [9.2, 9.1],
        ],
        dtype=np.float32,
    )
    obs = pd.DataFrame(
        {"labels": ["A", "A", "B", "B", "Unknown", "Unknown"]},
        index=[f"cell_{i}" for i in range(X.shape[0])],
    )
    adata = ad.AnnData(X=X.copy(), obs=obs)
    adata.layers[SCALED_LAYER] = X.copy()
    return adata


def test_flowsom_knn_uses_flowsom_python_package(monkeypatch):
    adata = _label_transfer_adata()
    calls = {}

    class FakeFlowSOM:
        def __init__(self, inp, *, n_clusters, xdim, ydim, seed, **kwargs):
            calls["n_clusters"] = n_clusters
            calls["xdim"] = xdim
            calls["ydim"] = ydim
            calls["seed"] = seed
            calls["shape"] = inp.X.shape
            self._cell_data = SimpleNamespace(
                obs=pd.DataFrame({"metaclustering": [0, 0, 1, 1, 0, 1]})
            )

        def get_cell_data(self):
            return self._cell_data

    monkeypatch.setitem(sys.modules, "flowsom", SimpleNamespace(FlowSOM=FakeFlowSOM))

    pred, X, unlabeled = baselines.flowsom_knn(
        adata,
        "labels",
        "Unknown",
        xdim=3,
        ydim=2,
        n_metaclusters=4,
        seed=7,
    )

    assert pred.tolist() == ["A", "B"]
    assert unlabeled.tolist() == [False, False, False, False, True, True]
    assert X.dtype == np.float32
    assert calls == {"n_clusters": 4, "xdim": 3, "ydim": 2, "seed": 7, "shape": (6, 2)}


def test_rapids_graph_knn_uses_rapids_singlecell_leiden(monkeypatch):
    adata = _label_transfer_adata()
    calls = {}

    def fake_neighbors(graph_adata, *, n_neighbors, use_rep, **kwargs):
        calls["neighbors"] = {
            "n_neighbors": n_neighbors,
            "use_rep": use_rep,
            "shape": graph_adata.X.shape,
        }
        assert use_rep in graph_adata.obsm

    def fake_leiden(graph_adata, *, key_added, resolution, random_state, **kwargs):
        calls["leiden"] = {
            "key_added": key_added,
            "resolution": resolution,
            "random_state": random_state,
        }
        graph_adata.obs[key_added] = ["0", "0", "1", "1", "0", "1"]

    fake_rsc = SimpleNamespace(
        pp=SimpleNamespace(neighbors=fake_neighbors),
        tl=SimpleNamespace(leiden=fake_leiden),
    )
    monkeypatch.setitem(sys.modules, "rapids_singlecell", fake_rsc)

    pred, X, unlabeled = baselines.rapids_graph_knn(
        adata,
        "labels",
        "Unknown",
        k=4,
        resolution=0.5,
        seed=11,
    )

    assert pred.tolist() == ["A", "B"]
    assert unlabeled.tolist() == [False, False, False, False, True, True]
    assert X.dtype == np.float32
    assert calls["neighbors"] == {"n_neighbors": 4, "use_rep": "X_baseline", "shape": (6, 2)}
    assert calls["leiden"] == {
        "key_added": "rapids_graph_clusters",
        "resolution": 0.5,
        "random_state": 11,
    }


def test_rapids_graph_knn_real_rapids_singlecell_smoke():
    try:
        import rapids_singlecell  # noqa: F401
    except Exception as err:
        pytest.skip(f"rapids_singlecell is not importable: {err}")
    cp = pytest.importorskip("cupy")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as err:
        pytest.skip(f"CUDA device is not available: {err}")
    if device_count < 1:
        pytest.skip("CUDA device is not available")

    adata = _label_transfer_adata()
    pred, X, unlabeled = baselines.rapids_graph_knn(
        adata,
        "labels",
        "Unknown",
        k=2,
        resolution=1.0,
        seed=3,
    )

    assert pred.shape == (2,)
    assert set(pred).issubset({"A", "B"})
    assert X.shape == (6, 2)
    assert unlabeled.tolist() == [False, False, False, False, True, True]
