"""CLI for the CytoANVI benchmark.

Examples
--------
# smoke test (synthetic, no download) — proves the harness end-to-end
python -m benchmarks.cytoanvi.run --dataset synthetic --task all --max-epochs 3

# inspect a real dataset's obs columns / layers (to set --labels-key etc.)
python -m benchmarks.cytoanvi.run --dataset roider --inspect --data-dir benchmarks/cytoanvi/data

# real run once data is present
python -m benchmarks.cytoanvi.run --dataset roider --task all \
    --data-dir benchmarks/cytoanvi/data --labels-key cell_type --batch-key batch --out results.json

Run from the repo root with the env on the path, e.g.
  PYTHONPATH=src LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python -m benchmarks.cytoanvi.run ...
"""

from __future__ import annotations

import argparse
import json

import scvi

from . import data as data_mod
from . import tasks as task_mod


def _load(dataset, data_dir):
    if dataset == "synthetic":
        return data_mod.make_synthetic_panels()
    if dataset == "roider":
        return data_mod.load_roider(data_dir)
    if dataset == "nunez":
        merged = data_mod.load_nunez(data_dir)
        return merged, merged, None  # single panel; p1 == merged, no second panel
    raise ValueError(dataset)


def _inspect(p1, p2):
    print("=== panel 1 ===")
    print("obs:", list(p1.obs.columns))
    print("layers:", list(p1.layers.keys()))
    print("n_obs x n_vars:", p1.shape)
    for c in p1.obs.columns:
        nun = p1.obs[c].nunique()
        if nun <= 30:
            print(f"  {c}: {nun} -> {sorted(map(str, p1.obs[c].unique()))[:12]}")
    if p2 is not None:
        print("=== panel 2 ===  vars:", p2.shape, "obs:", list(p2.obs.columns))


def main():
    """Parse args, load the dataset, run the requested benchmark task(s), print/save metrics."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "roider", "nunez"], required=True)
    ap.add_argument("--task", choices=["b1", "b2", "b3", "b5", "all"], default="all")
    ap.add_argument("--data-dir", default="benchmarks/cytoanvi/data")
    ap.add_argument("--labels-key", default="labels")
    ap.add_argument("--batch-key", default="batch")
    ap.add_argument("--sample-key", default=None)
    ap.add_argument("--unlabeled", default="Unknown")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scvi.settings.seed = args.seed
    merged, p1, p2 = _load(args.dataset, args.data_dir)

    if args.inspect:
        _inspect(p1, p2)
        return

    # synthetic uses the unlabeled category "label_0" (an existing synthetic label); real data
    # uses --unlabeled. For single-panel tasks we use p1 (the labelled panel).
    unlab = "label_0" if args.dataset == "synthetic" else args.unlabeled
    kw = {
        "labels_key": args.labels_key,
        "unlabeled_category": unlab,
        "batch_key": args.batch_key,
        "sample_key": args.sample_key,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
    }

    results = {"dataset": args.dataset, "seed": args.seed}
    tasks = ["b1", "b2", "b3", "b5"] if args.task == "all" else [args.task]
    for t in tasks:
        print(f"\n=== running {t} ===")
        if t == "b1":
            results["b1"] = task_mod.task_b1_label_transfer(p1, **kw)
        elif t == "b2":
            results["b2"] = task_mod.task_b2_integration(p1, **kw)
        elif t == "b3":
            if p2 is None:
                print("  skipped (single-panel dataset has no panel 2)")
                continue
            results["b3"] = task_mod.task_b3_panel_divergent(
                p1,
                p2,
                labels_key=args.labels_key,
                unlabeled_category=unlab,
                batch_key=args.batch_key,
                sample_key=args.sample_key,
                seed=args.seed,
                max_epochs=args.max_epochs,
            )
        elif t == "b5":
            results["b5"] = task_mod.task_b5_novelty(p1, **kw)
        print(json.dumps(results[t], indent=2))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
