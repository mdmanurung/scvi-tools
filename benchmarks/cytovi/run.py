"""CLI for Track A — CYTOVI paper benchmarks."""

from __future__ import annotations

import argparse
import json

import scvi

from benchmarks.common.baselines import cycombinepy_available
from benchmarks.common.seeds import run_multiseed, save_json
from benchmarks.cytovi import data as data_mod
from benchmarks.cytovi import tasks_imputation as imp_mod
from benchmarks.cytovi import tasks_integration as task_mod


def _load(dataset: str, data_dir: str):
    if dataset == "synthetic":
        merged, p1, _ = data_mod.make_synthetic_panels()
        return p1
    if dataset == "nunez":
        return data_mod.load_nunez(data_dir)
    if dataset == "roider":
        _, p1, _ = data_mod.load_roider(data_dir)
        return p1
    raise ValueError(dataset)


def main():
    ap = argparse.ArgumentParser(description="CYTOVI paper benchmarks (Track A)")
    ap.add_argument("--dataset", choices=["synthetic", "nunez", "roider"], required=True)
    ap.add_argument("--task", choices=["a2", "a3"], default="a2")
    ap.add_argument("--data-dir", default="benchmarks/cytoanvi/data")
    ap.add_argument("--labels-key", default="labels")
    ap.add_argument("--batch-key", default="batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds, e.g. 0,1,2")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--subsample-per-batch", type=int, default=10_000)
    ap.add_argument("--max-markers", type=int, default=None, help="A3: limit markers (smoke)")
    ap.add_argument("--max-cells", type=int, default=50_000, help="A3: subsample cells")
    ap.add_argument("--no-cycombinepy", action="store_true")
    ap.add_argument("--no-harmony", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dataset == "synthetic" and args.max_epochs == 1000 and not args.inspect:
        print("note: use --max-epochs 3 for synthetic smoke tests")

    adata = _load(args.dataset, args.data_dir)

    if args.inspect:
        print("obs:", list(adata.obs.columns))
        print("layers:", list(adata.layers.keys()))
        print("shape:", adata.shape)
        print("cycombinepy installed:", cycombinepy_available())
        return

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    def _run(seed: int):
        scvi.settings.seed = seed
        if args.task == "a2":
            return task_mod.task_a2_integration(
                adata,
                labels_key=args.labels_key,
                batch_key=args.batch_key,
                max_epochs=args.max_epochs,
                seed=seed,
                subsample_per_batch=args.subsample_per_batch,
                include_cycombinepy=not args.no_cycombinepy,
                include_harmony=not args.no_harmony,
            )
        if args.task == "a3":
            markers = None
            if args.max_markers is not None:
                markers = list(adata.var_names[: args.max_markers])
            return imp_mod.task_a3_imputation(
                adata,
                batch_key=args.batch_key,
                max_epochs=args.max_epochs,
                max_cells=args.max_cells,
                seed=seed,
                markers=markers,
            )
        raise ValueError(args.task)

    payload = (
        run_multiseed(_run, seeds=seeds)
        if len(seeds) > 1
        else {"seeds": seeds, "per_seed": {str(seeds[0]): _run(seeds[0])}}
    )
    payload["dataset"] = args.dataset
    payload["task"] = args.task

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        save_json(args.out, payload)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
