"""Multiseed benchmark runners and JSON aggregation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def run_multiseed(
    fn: Callable[[int], dict[str, Any]],
    seeds: list[int] | tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    """Run ``fn(seed)`` for each seed; return per-seed results and numeric mean ± sd."""
    per_seed = {str(s): fn(s) for s in seeds}
    return {"seeds": list(seeds), "per_seed": per_seed, "summary": _summarize_numeric(per_seed)}


def _summarize_numeric(per_seed: dict[str, dict]) -> dict:
    """Flatten numeric leaves across seeds and compute mean/std."""
    flat: dict[str, list[float]] = {}

    def _walk(obj, prefix: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                _walk(v, key)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            flat.setdefault(prefix, []).append(float(obj))

    for run in per_seed.values():
        _walk(run)

    out = {}
    for k, vals in flat.items():
        arr = np.asarray(vals)
        out[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)), "n": len(vals)}
    return out


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
