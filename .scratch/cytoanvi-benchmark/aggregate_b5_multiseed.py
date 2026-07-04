"""Aggregate B5 holdout-sweep seed JSONs into a multiseed summary.

Usage (default = legacy e1000 seed set):
    python .scratch/cytoanvi-benchmark/aggregate_b5_multiseed.py

Usage (roider-full 3-seed set, as submitted 2026-07-03):
    python .scratch/cytoanvi-benchmark/aggregate_b5_multiseed.py \
        --out results/roider_full_b5_sweep_multiseed.json \
        results/roider_full_b5_sweep_s0.json \
        results/roider_full_b5_sweep_s1.json \
        results/roider_full_b5_sweep_s2.json

Positional seed files are mapped to seeds 0,1,2 in the order given. Paths may be absolute or
relative to this script's directory. With no positional args the legacy e1000 paths are used.
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
RESULTS = HERE / "results" / "e1000"

# Legacy default (e1000 sweep) — used when no positional seed files are passed.
SEED_FILES = {
    0: RESULTS / "roider_e1000_b5_sweep.json",
    1: RESULTS / "roider_e1000_b5_sweep_s1.json",
    2: RESULTS / "roider_e1000_b5_sweep_s2.json",
}

OUT = RESULTS / "roider_e1000_b5_multiseed.json"


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (HERE / path)


def _parse_args(argv: list[str] | None = None) -> None:
    """Override the module-level SEED_FILES / OUT from CLI args when provided."""
    global SEED_FILES, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seed_files", nargs="*", help="per-seed B5 JSONs, in seed order (0,1,2,...)")
    ap.add_argument("--out", type=str, default=None, help="output multiseed JSON path")
    args = ap.parse_args(argv)
    if args.seed_files:
        SEED_FILES = {i: _resolve(p) for i, p in enumerate(args.seed_files)}
    if args.out is not None:
        OUT = _resolve(args.out)


def main() -> None:
    per_seed: dict[str, dict] = {}
    for s, path in SEED_FILES.items():
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        data = json.loads(path.read_text())
        # Each seed file has top-level keys: seed, max_epochs, b5, dataset
        # The b5 sub-dict is the task_b5_holdout_sweep result.
        b5 = data.get("b5", data)  # fallback: top-level is the task result
        per_seed[str(s)] = b5
        n_types = len(b5.get("per_type", {}))
        mean_a = b5.get("mean_auroc", float("nan"))
        best_a = b5.get("best_auroc", float("nan"))
        print(f"  seed {s}: {n_types} types | mean_auroc={mean_a:.3f} [PRIMARY] | best_auroc={best_a:.3f} [max/secondary]")

    if not per_seed:
        print("No seed files found.")
        return

    found = sorted(int(k) for k in per_seed)
    print(f"\nAggregating seeds: {found}")

    # Numeric summary across seeds (mean ± std for all numeric leaves).
    summary = _summarize_numeric(per_seed)

    import numpy as np

    # Headline metrics across seeds
    def _seed_vals(key, default=float("nan")):
        return [per_seed[str(s)].get(key, default) for s in found if str(s) in per_seed]

    best_aurocs = _seed_vals("best_auroc")
    mean_aurocs = _seed_vals("mean_auroc")
    n_fdr_sigs = _seed_vals("n_fdr_significant", 0)
    # CytoVI OOD baseline (present when the sweep was run with --b5-cytovi-baseline). The
    # comparison of mean_auroc vs cytovi_mean_auroc is the point of the redesigned B5.
    cytovi_aurocs = [v for v in _seed_vals("cytovi_mean_auroc") if not np.isnan(v)]

    payload = {
        "seeds": found,
        "per_seed": per_seed,
        "summary": summary,
        # REPORTING NOTE: mean_auroc is the PRIMARY headline — unweighted mean over ALL
        # held-out cell types.  best_auroc is the MAX over cell types (one cherry-picked
        # type) and must NOT be presented as the summary statistic for novelty detection.
        "headline": {
            # PRIMARY METRIC — report this in the paper.
            "mean_auroc_mean": float(np.mean(mean_aurocs)),
            "mean_auroc_std": float(np.std(mean_aurocs, ddof=1)) if len(mean_aurocs) > 1 else 0.0,
            "n_fdr_significant_mean": float(np.mean(n_fdr_sigs)),
            # BASELINE — CytoANVI's mean_auroc is only useful if it beats this.
            "cytovi_mean_auroc_mean": (
                float(np.mean(cytovi_aurocs)) if cytovi_aurocs else float("nan")
            ),
            "cytovi_mean_auroc_std": (
                float(np.std(cytovi_aurocs, ddof=1)) if len(cytovi_aurocs) > 1 else 0.0
            ),
            # SECONDARY — max over types; not a summary statistic.
            "best_auroc_mean": float(np.mean(best_aurocs)),
            "best_auroc_std": float(np.std(best_aurocs, ddof=1)) if len(best_aurocs) > 1 else 0.0,
        },
    }

    save_json(OUT, payload)
    print(f"\nWrote {OUT}")
    print(f"  [PRIMARY] mean_auroc:        {payload['headline']['mean_auroc_mean']:.3f} ± {payload['headline']['mean_auroc_std']:.3f}")
    print(f"  [BASELINE] cytovi_mean_auroc:{payload['headline']['cytovi_mean_auroc_mean']:.3f} ± {payload['headline']['cytovi_mean_auroc_std']:.3f}")
    print(f"  n_fdr_sig:                   {payload['headline']['n_fdr_significant_mean']:.1f}")
    print(f"  [secondary] best_auroc (max over types, NOT summary): {payload['headline']['best_auroc_mean']:.3f} ± {payload['headline']['best_auroc_std']:.3f}")

    # Per-type AUROC summary
    all_types = sorted({t for s in per_seed.values() for t in s.get("per_type", {})})
    print("\nPer-type AUROC (mean ± std across seeds):")
    for t in all_types:
        vals = [per_seed[str(s)]["per_type"][t]["auroc"] for s in found if t in per_seed.get(str(s), {}).get("per_type", {})]
        if vals:
            m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            sig_marker = "**" if m > 0.7 else "  "
            print(f"  {sig_marker}{t:<30} {m:.3f} ± {sd:.3f}  (n={len(vals)})")


if __name__ == "__main__":
    _parse_args()
    main()
