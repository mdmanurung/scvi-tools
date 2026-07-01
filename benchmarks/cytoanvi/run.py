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
from benchmarks.common.training import NAN_LAYER

from . import data as data_mod
from . import tasks as task_mod
from .paired_rna_cytof import task_b7_multimodal_integration


def _load(
    dataset,
    data_dir,
    leiden_resolution=None,
    max_cells=100_000,
    roider_max_patients=None,
    require_annotated_nunez=False,
):
    if dataset == "synthetic":
        return data_mod.make_synthetic_panels()
    if dataset == "roider":
        return data_mod.load_roider(data_dir)
    if dataset == "roider-full":
        res = 1.0 if leiden_resolution is None else leiden_resolution
        return data_mod.load_roider_full(
            data_dir,
            max_patients=roider_max_patients,
            leiden_labels=True,
            leiden_resolution=res,
        )
    if dataset == "nunez":
        res = 0.05 if leiden_resolution is None else leiden_resolution
        if require_annotated_nunez:
            import os

            h5ad = data_mod._resolve_file(data_dir, "nunez_annotated.h5ad")
            if not (os.path.exists(h5ad) and os.path.getsize(h5ad) > 0):
                raise FileNotFoundError(
                    f"Annotated Nuñez file missing at {h5ad}. Run "
                    "`python -m benchmarks.cytoanvi.annotate_nunez` first."
                )
        merged = data_mod.load_nunez(
            data_dir,
            leiden_resolution=res,
            max_cells=max_cells,
            annotate=not require_annotated_nunez,
        )
        return merged, merged, None
    if dataset == "paired-rna-cytof":
        from .paired_rna_cytof import make_synthetic_paired_rna_cytof

        rna, cy, _markers = make_synthetic_paired_rna_cytof(seed=0)
        return rna, cy, None
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
        "nan_layer": NAN_LAYER,
        "seed": seed,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
    }
    b1_kw = {
        **kw,
        "n_samples_per_label": args.n_samples_per_label,
        "reduce_lr_on_plateau": args.reduce_lr_on_plateau,
    }
    b2_kw = {**kw, "subsample_per_batch": args.subsample_per_batch}
    results = {"seed": seed, "max_epochs": args.max_epochs}
    all_tasks = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9"]
    tasks = all_tasks if args.task == "all" else [args.task]

    for t in tasks:
        print(f"\n=== running {t} (seed={seed}) ===")
        if t == "b1":
            results["b1"] = task_mod.task_b1_label_transfer(p1, **b1_kw)
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
                batch_size=args.batch_size,
            )
        elif t == "b5":
            b5_kw = {k: v for k, v in kw.items() if k != "subsample_per_batch"}
            if args.holdout_sweep:
                results["b5"] = task_mod.task_b5_holdout_sweep(p1, **b5_kw)
            else:
                results["b5"] = task_mod.task_b5_novelty(
                    p1, holdout_type=args.holdout_type, **b5_kw
                )
        elif t == "b4":
            results["b4"] = task_mod.task_b4_continual(p1, **kw)
        elif t == "b6":
            lambdas = None
            if args.ewc_lambdas:
                lambdas = [float(x) for x in args.ewc_lambdas.split(",")]
            results["b6"] = task_mod.task_b6_lambda_sweep(
                p1,
                labels_key=args.labels_key,
                unlabeled_category=unlab,
                batch_key=args.batch_key,
                sample_key=args.sample_key,
                nan_layer=NAN_LAYER,
                seed=seed,
                max_epochs=args.max_epochs,
                lambdas=lambdas,
                batch_size=args.batch_size,
            )
        elif t == "b7":
            if args.dataset != "paired-rna-cytof":
                print("  skipped (requires --dataset paired-rna-cytof)")
                continue
            print(
                "  B7 uses batch_key=modality, sample_key=sample_id, labels_key=celltype "
                "(ignores --batch-key / --labels-key / --sample-key)"
            )
            results["b7"] = task_b7_multimodal_integration(
                p1,
                p2,
                labels_key="celltype",
                unlabeled_category=unlab,
                batch_key="modality",
                sample_key="sample_id",
                seed=seed,
                max_epochs=args.max_epochs,
            )
        elif t == "b8":
            b8_kw = {**kw}
            if args.hierarchy_edges is not None:
                with open(args.hierarchy_edges, encoding="utf-8") as fh:
                    b8_kw["hierarchy_edges"] = json.load(fh)
            results["b8"] = task_mod.task_b8_hce_label_transfer(p1, **b8_kw)
        elif t == "b9":
            run_mapqc = args.mapqc_run or args.dataset != "synthetic"
            results["b9"] = task_mod.task_b9_mapqc(
                p1,
                labels_key=args.labels_key,
                unlabeled_category=unlab,
                batch_key=args.batch_key,
                sample_key=args.sample_key,
                nan_layer=NAN_LAYER,
                seed=seed,
                max_epochs=args.max_epochs,
                n_nhoods=args.mapqc_n_nhoods,
                k_min=args.mapqc_k_min,
                k_max=args.mapqc_k_max,
                run_mapqc=run_mapqc,
                batch_size=args.batch_size,
            )
        print(json.dumps(results.get(t), indent=2))
    return results


def main():
    """CLI entry point — parse arguments and dispatch benchmark tasks."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        choices=["synthetic", "roider", "roider-full", "nunez", "paired-rna-cytof"],
        required=True,
    )
    ap.add_argument(
        "--task",
        choices=["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "all"],
        default="all",
    )
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
    ap.add_argument(
        "--ewc-lambdas",
        default=None,
        help="B6: comma-separated ewc_importance values (default: 0,1,10,100,1000)",
    )
    ap.add_argument(
        "--require-annotated-nunez",
        action="store_true",
        help="Nuñez: fail if nunez_annotated.h5ad is missing (skip Leiden proxy labels)",
    )
    ap.add_argument(
        "--hierarchy-edges",
        default=None,
        help="B8: JSON file with parent→children hierarchy edges (observed labels only)",
    )
    ap.add_argument("--mapqc-n-nhoods", type=int, default=3, help="B9: mapQC neighborhoods")
    ap.add_argument("--mapqc-k-min", type=int, default=5, help="B9: mapQC min neighborhood size")
    ap.add_argument("--mapqc-k-max", type=int, default=15, help="B9: mapQC max neighborhood size")
    ap.add_argument(
        "--mapqc-run",
        action="store_true",
        help="B9: force mapQC scoring on synthetic (default: plumbing-only on synthetic)",
    )
    ap.add_argument(
        "--leiden-resolution",
        type=float,
        default=None,
        help="Leiden r (default: 1.0 roider-full, 0.05 nunez)",
    )
    ap.add_argument("--max-cells", type=int, default=100_000, help="Nuñez: subsample cap")
    ap.add_argument(
        "--roider-max-patients",
        type=int,
        default=None,
        help="roider-full: cap patients while ingest matures (default: all paired)",
    )
    ap.add_argument(
        "--n-samples-per-label",
        type=int,
        default=None,
        help="B1: balanced labeled-cell sampling per class per minibatch (CytoANVI only)",
    )
    ap.add_argument(
        "--reduce-lr-on-plateau",
        action="store_true",
        default=False,
        help="B1: enable ReduceLROnPlateau for CytoANVI classifier-head stability",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Mini-batch size for CytoVI/CytoANVI training (default: scvi default 128). "
            "Set to 8192 for large cohorts such as roider-full to avoid NaN divergence "
            "and reduce per-epoch wall time."
        ),
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
        require_annotated_nunez=args.require_annotated_nunez,
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
