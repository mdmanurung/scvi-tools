"""CLI for the CytoANVI benchmark (Track B).

Examples
--------
# smoke test (synthetic, no download)
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset synthetic --task all --max-epochs 3 --subsample-per-batch 200

# real run
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset roider --task all --max-epochs 1000 --seeds 0,1,2 \
  --labels-key cell_type --batch-key batch --out results.json
"""

from __future__ import annotations

import argparse
import json

import scvi

from benchmarks.common.seeds import run_multiseed, save_json

from . import data as data_mod
from . import tasks as task_mod


def _load(dataset, data_dir, leiden_resolution=0.05, max_cells=100_000, roider_max_patients=None):
    if dataset == "synthetic":
        return data_mod.make_synthetic_panels()
    if dataset == "roider":
        return data_mod.load_roider(data_dir)
    if dataset == "roider-full":
        return data_mod.load_roider_full(data_dir, max_patients=roider_max_patients)
    if dataset == "nunez":
        merged = data_mod.load_nunez(
            data_dir, leiden_resolution=leiden_resolution, max_cells=max_cells
        )
        return merged, merged, None
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


def _run_tasks(args, p1, p2, unlab, seed):
    scvi.settings.seed = seed
    kw = {
        "labels_key": args.labels_key,
        "unlabeled_category": unlab,
        "batch_key": args.batch_key,
        "sample_key": args.sample_key,
        "seed": seed,
        "max_epochs": args.max_epochs,
    }
    b2_kw = {**kw, "subsample_per_batch": args.subsample_per_batch}
    results = {"seed": seed, "max_epochs": args.max_epochs}
    tasks = ["b1", "b2", "b3", "b5"] if args.task == "all" else [args.task]

    for t in tasks:
        print(f"\n=== running {t} (seed={seed}) ===")
        if t == "b1":
            results["b1"] = task_mod.task_b1_label_transfer(p1, **kw)
        elif t == "b2":
            results["b2"] = task_mod.task_b2_integration(p1, **b2_kw)
        elif t == "b3":
            if p2 is None:
                print("  skipped (single-panel dataset)")
                continue
            results["b3"] = task_mod.task_b3_panel_divergent(
                p1,
                p2,
                labels_key=args.labels_key,
                unlabeled_category=unlab,
                batch_key=args.batch_key,
                sample_key=args.sample_key,
                seed=seed,
                max_epochs=args.max_epochs,
            )
        elif t == "b5":
            b5_kw = {k: v for k, v in kw.items() if k != "subsample_per_batch"}
            if args.holdout_sweep:
                results["b5"] = task_mod.task_b5_holdout_sweep(p1, **b5_kw)
            else:
                results["b5"] = task_mod.task_b5_novelty(
                    p1, holdout_type=args.holdout_type, **b5_kw
                )
        print(json.dumps(results.get(t), indent=2))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "roider", "roider-full", "nunez"], required=True)
    ap.add_argument("--task", choices=["b1", "b2", "b3", "b5", "all"], default="all")
    ap.add_argument("--data-dir", default="benchmarks/cytoanvi/data")
    ap.add_argument("--labels-key", default="labels")
    ap.add_argument("--batch-key", default="batch")
    ap.add_argument("--sample-key", default=None)
    ap.add_argument("--unlabeled", default="Unknown")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 0,1,2")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--subsample-per-batch", type=int, default=10_000)
    ap.add_argument("--holdout-type", default=None, help="B5: cell type held out as novel")
    ap.add_argument("--holdout-sweep", action="store_true", help="B5: sweep all cell types")
    ap.add_argument("--leiden-resolution", type=float, default=0.05, help="Nuñez: Leiden resolution")
    ap.add_argument("--max-cells", type=int, default=100_000, help="Nuñez: subsample cap")
    ap.add_argument(
        "--roider-max-patients",
        type=int,
        default=None,
        help="roider-full: cap patients while ingest matures (default: all paired)",
    )
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    merged, p1, p2 = _load(
        args.dataset,
        args.data_dir,
        leiden_resolution=args.leiden_resolution,
        max_cells=args.max_cells,
        roider_max_patients=args.roider_max_patients,
    )

    if args.inspect:
        _inspect(p1, p2)
        return

    unlab = "label_0" if args.dataset == "synthetic" else args.unlabeled
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    if len(seeds) > 1:
        payload = run_multiseed(lambda s: _run_tasks(args, p1, p2, unlab, s), seeds=seeds)
    else:
        payload = _run_tasks(args, p1, p2, unlab, seeds[0])

    payload["dataset"] = args.dataset
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        save_json(args.out, payload)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
