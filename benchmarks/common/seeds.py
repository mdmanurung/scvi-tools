"""Multiseed benchmark runners and JSON aggregation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any


def run_multiseed(
    fn: Callable[[int], dict[str, Any]],
    seeds: list[int] | tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    """Run ``fn(seed)`` for each seed; return per-seed results and numeric mean ± sd."""
    per_seed = {str(s): fn(s) for s in seeds}
    return {"seeds": list(seeds), "per_seed": per_seed, "summary": _summarize_numeric(per_seed)}


def _summarize_numeric(per_seed: dict[str, dict]) -> dict:
    """Flatten numeric leaves across seeds and compute mean/std.

    # NOTE (inferential-stats limitation): this function reports mean ± std over the
    # available seeds only (typically 3).  With n=3, std is a poor variance estimate and
    # no paired significance tests or bootstrap CIs are computed here.  Treat the
    # reported ± values as descriptive only — they do NOT constitute inferential
    # statistical evidence.  For an optional bootstrap 95% CI helper that is NOT wired
    # into task output schemas, see :func:`bootstrap_ci` below.
    """
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


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap percentile CI for the mean of *values*.

    Standalone helper — NOT wired into :func:`_summarize_numeric` or any task output
    schema.  Call this explicitly when a single-metric cross-seed CI is needed for
    reporting, e.g.::

        from benchmarks.common.seeds import bootstrap_ci
        ci = bootstrap_ci([0.46, 0.38, 0.53], ci=0.95)
        # {"mean": 0.457, "ci_low": 0.39, "ci_high": 0.52, ...}

    Parameters
    ----------
    values : list[float]
        Observed values (e.g. per-seed macro-F1 or mean_auroc).
    n_boot : int
        Number of bootstrap resamples (default 2 000).
    ci : float
        Coverage probability (default 0.95 → 95%).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``mean``, ``ci_low``, ``ci_high``, ``ci_level``, ``n``, ``n_boot``.
    """
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot_means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(boot_means, 100 * alpha)),
        "ci_high": float(np.percentile(boot_means, 100 * (1.0 - alpha))),
        "ci_level": ci,
        "n": len(arr),
        "n_boot": n_boot,
    }


def to_jsonable(obj: Any) -> Any:
    """Convert benchmark payloads to strict JSON-compatible Python values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, tuple | list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def save_json(path: str | Path, payload: dict) -> None:
    """Write a strict JSON benchmark payload to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(to_jsonable(payload), fh, indent=2, allow_nan=False)
