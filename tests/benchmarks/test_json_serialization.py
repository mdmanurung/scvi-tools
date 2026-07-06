"""JSON serialization regressions for benchmark outputs."""

from __future__ import annotations

import json

import numpy as np

from benchmarks.common.seeds import save_json


def test_save_json_normalizes_numpy_scalars_and_non_finite_values(tmp_path):
    out = tmp_path / "result.json"

    save_json(
        out,
        {
            "np_int": np.int64(7),
            "np_float": np.float32(1.25),
            "nan": np.float64(np.nan),
            "posinf": float("inf"),
            "nested": [np.float64("-inf"), {"ok": np.bool_(True)}],
        },
    )

    text = out.read_text()
    assert "NaN" not in text
    assert "Infinity" not in text

    loaded = json.loads(text)
    assert loaded == {
        "np_int": 7,
        "np_float": 1.25,
        "nan": None,
        "posinf": None,
        "nested": [None, {"ok": True}],
    }
