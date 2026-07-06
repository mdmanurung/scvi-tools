"""Aggregate B1 label-transfer seed JSONs (Roider full-cohort) into a multiseed summary.

Usage:
    python .scratch/cytoanvi-benchmark/aggregate_b1_roider_multiseed.py

Or with explicit seed files:
    python .scratch/cytoanvi-benchmark/aggregate_b1_roider_multiseed.py \
        --out results/roider_full_b1_multiseed.json \
        results/roider_full_b1_s0.json \
        results/roider_full_b1_s1.json \
        results/roider_full_b1_s2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.common.seeds import _summarize_numeric, save_json

HERE = Path(__file__).parent
RESULTS = HERE / "results"

SEED_FILES = {
    0: RESULTS / "roider_full_b1_s0.json",
    1: RESULTS / "roider_full_b1_s1.json",
    2: RESULTS / "roider_full_b1_s2.json",
}

OUT = RESULTS / "roider_full_b1_multiseed.json"


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (HERE / path)


def _parse_args(argv: list[str] | None = None) -> None:
    global SEED_FILES, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seed_files", nargs="*", help="per-seed B1 JSONs, in seed order (0,1,2,...)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args(argv)
    if args.seed_files:
        SEED_FILES = {i: _resolve(p) for i, p in enumerate(args.seed_files)}
    if args.out is not None:
        OUT = _resolve(args.out)


def _macro_f1(b1: dict, key: str) -> float:
    return b1.get(key, {}).get("macro_f1", float("nan"))


def main() -> None:
    import numpy as np

    per_seed: dict[str, dict] = {}
    dataset: str | None = None

    for s, path in SEED_FILES.items():
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        data = json.loads(path.read_text())
        b1 = data.get("b1", data)
        per_seed[str(s)] = b1
        if dataset is None:
            dataset = data.get("dataset")
        f1_anvi = _macro_f1(b1, "cytoanvi")
        f1_knn = _macro_f1(b1, "cytovi_knn")
        print(f"  seed {s}: cytoanvi={f1_anvi:.4f}  cytovi_knn={f1_knn:.4f}  delta={f1_anvi - f1_knn:+.4f}")

    if not per_seed:
        print("No seed files found.")
        return

    found = sorted(int(k) for k in per_seed)
    print(f"\nAggregating seeds: {found}")

    summary = _summarize_numeric(per_seed)

    def _vals(key: str, subkey: str = "macro_f1") -> list[float]:
        return [
            per_seed[str(s)].get(key, {}).get(subkey, float("nan"))
            for s in found
            if str(s) in per_seed
        ]

    cytoanvi_f1s = _vals("cytoanvi")
    cytovi_f1s = _vals("cytovi_knn")
    deltas = [a - b for a, b in zip(cytoanvi_f1s, cytovi_f1s)]
    xgb_f1s = [v for v in _vals("xgboost") if not np.isnan(v)]
    flowsom_f1s = [v for v in _vals("flowsom") if not np.isnan(v)]

    payload = {
        "seeds": found,
        "dataset": dataset,
        "per_seed": per_seed,
        "summary": summary,
        "headline": {
            "cytoanvi_macro_f1_mean": float(np.mean(cytoanvi_f1s)),
            "cytoanvi_macro_f1_std": float(np.std(cytoanvi_f1s, ddof=1)) if len(cytoanvi_f1s) > 1 else 0.0,
            "cytovi_knn_macro_f1_mean": float(np.mean(cytovi_f1s)),
            "cytovi_knn_macro_f1_std": float(np.std(cytovi_f1s, ddof=1)) if len(cytovi_f1s) > 1 else 0.0,
            "delta_macro_f1_mean": float(np.mean(deltas)),
            "delta_macro_f1_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "xgboost_macro_f1_mean": float(np.mean(xgb_f1s)) if xgb_f1s else float("nan"),
            "flowsom_macro_f1_mean": float(np.mean(flowsom_f1s)) if flowsom_f1s else float("nan"),
        },
    }

    save_json(OUT, payload)
    print(f"\nWrote {OUT}")
    hl = payload["headline"]
    print(f"  CytoANVI:   {hl['cytoanvi_macro_f1_mean']:.4f} ± {hl['cytoanvi_macro_f1_std']:.4f}")
    print(f"  CytoVI-kNN: {hl['cytovi_knn_macro_f1_mean']:.4f} ± {hl['cytovi_knn_macro_f1_std']:.4f}")
    print(f"  Δ:          {hl['delta_macro_f1_mean']:+.4f} ± {hl['delta_macro_f1_std']:.4f}")
    if not np.isnan(hl["xgboost_macro_f1_mean"]):
        print(f"  XGBoost:    {hl['xgboost_macro_f1_mean']:.4f}")

    gate = hl["delta_macro_f1_mean"] >= 0.03
    print(f"\n  Gate (Δ ≥ +0.03): {'✅ PASS' if gate else '❌ FAIL'}")


if __name__ == "__main__":
    _parse_args()
    main()
