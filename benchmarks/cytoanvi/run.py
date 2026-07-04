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
from benchmarks.common.annbatch import AnnBatchConfig
from benchmarks.common.seeds import run_multiseed, save_json, to_jsonable
from benchmarks.common.training import NAN_LAYER, resolve_nan_layer

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


def _annbatch_config_from_args(args):
    if not getattr(args, "annbatch", False):
        return None
    cache_mode = getattr(args, "annbatch_cache_mode", "temporary")
    cache_key = getattr(args, "annbatch_cache_key", None)
    if cache_mode == "reuse" and not cache_key:
        raise ValueError("--annbatch-cache-key is required when --annbatch-cache-mode=reuse.")
    return AnnBatchConfig(
        enabled=True,
        cache_dir=args.annbatch_cache_dir,
        chunk_size=args.annbatch_chunk_size,
        preload_nchunks=args.annbatch_preload_nchunks,
        cache_mode=cache_mode,
        cache_key=cache_key,
    )


def _csv_values(value):
    if value is None or value == "":
        return None
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _case_control_config_from_args(args) -> dict:
    key = getattr(args, "case_control_key", None)
    control = _csv_values(getattr(args, "control_values", None))
    case = _csv_values(getattr(args, "case_values", None))
    provided = [key is not None, control is not None, case is not None]
    if any(provided) and not all(provided):
        raise ValueError(
            "Real case/control split requires --case-control-key, --control-values, "
            "and --case-values together."
        )
    return {"case_control_key": key, "control_values": control, "case_values": case}


def _annbatch_payload(config):
    if config is None:
        return None
    return {
        "enabled": bool(config.enabled),
        "cache_dir": str(config.cache_dir) if config.cache_dir is not None else None,
        "chunk_size": int(config.chunk_size),
        "preload_nchunks": int(config.preload_nchunks),
        "cache_mode": config.cache_mode,
        "cache_key": config.cache_key,
    }


def _cytoanvi_training_config_from_args(args) -> dict:
    recipe = getattr(args, "cytoanvi_recipe", "default")
    if recipe not in {"default", "balanced", "publication"}:
        raise ValueError(f"Unknown CytoANVI recipe: {recipe!r}.")
    config = {
        "y_prior": "uniform",
        "class_weighting": "none",
        "class_weight_clip": float(getattr(args, "class_weight_clip", 10.0)),
        "classification_ratio": getattr(args, "classification_ratio", None),
        "reduce_lr_on_plateau": bool(getattr(args, "reduce_lr_on_plateau", False)),
        "learning_rate": getattr(args, "learning_rate", None),
        "gradient_clip_val": getattr(args, "gradient_clip_val", None),
    }
    if recipe in {"balanced", "publication"}:
        config.update(
            {
                "y_prior": "empirical",
                "class_weighting": "sqrt_inverse_frequency",
                "reduce_lr_on_plateau": True,
            }
        )
    if recipe == "publication":
        if config["learning_rate"] is None:
            config["learning_rate"] = 5e-4
        if config["gradient_clip_val"] is None:
            config["gradient_clip_val"] = 1.0
    if getattr(args, "y_prior", None) is not None:
        config["y_prior"] = args.y_prior
    if getattr(args, "class_weighting", None) is not None:
        config["class_weighting"] = args.class_weighting
    return {key: value for key, value in config.items() if value is not None}


def _run_tasks(args, p1, p2, unlab, seed):
    scvi.settings.seed = seed
    nan_layer = resolve_nan_layer(p1, NAN_LAYER)
    annbatch_config = _annbatch_config_from_args(args)
    cytoanvi_training_config = _cytoanvi_training_config_from_args(args)
    case_control_config = _case_control_config_from_args(args)
    kw = {
        "labels_key": args.labels_key,
        "unlabeled_category": unlab,
        "batch_key": args.batch_key,
        "sample_key": args.sample_key,
        "nan_layer": nan_layer,
        "seed": seed,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "annbatch_config": annbatch_config,
        "cytoanvi_training_config": cytoanvi_training_config,
    }
    b1_kw = {
        **kw,
        "n_samples_per_label": args.n_samples_per_label,
        "reduce_lr_on_plateau": args.reduce_lr_on_plateau,
        "b1_baselines": getattr(args, "b1_baselines", "all"),
        "accelerator": getattr(args, "training_accelerator", "auto"),
        "devices": getattr(args, "training_devices", "auto"),
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
                annbatch_config=annbatch_config,
                cytoanvi_training_config=cytoanvi_training_config,
            )
        elif t == "b5":
            b5_kw = {k: v for k, v in kw.items() if k != "subsample_per_batch"}
            b5_compute_logit = not getattr(args, "b5_no_logit", False)
            b5_cytovi = getattr(args, "b5_cytovi_baseline", False)
            if args.holdout_sweep:
                results["b5"] = task_mod.task_b5_holdout_sweep(
                    p1,
                    b5_mode=getattr(args, "b5_mode", "transductive"),
                    max_holdout_types=getattr(args, "b5_max_holdout_types", None),
                    checkpoint_path=getattr(args, "b5_checkpoint", None),
                    compute_logit=b5_compute_logit,
                    cytovi_baseline=b5_cytovi,
                    **b5_kw,
                )
            else:
                results["b5"] = task_mod.task_b5_novelty(
                    p1,
                    holdout_type=args.holdout_type,
                    b5_mode=getattr(args, "b5_mode", "transductive"),
                    compute_logit=b5_compute_logit,
                    cytovi_baseline=b5_cytovi,
                    **b5_kw,
                )
        elif t == "b4":
            results["b4"] = task_mod.task_b4_continual(
                p1,
                **kw,
                **case_control_config,
            )
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
                nan_layer=nan_layer,
                seed=seed,
                max_epochs=args.max_epochs,
                lambdas=lambdas,
                batch_size=args.batch_size,
                annbatch_config=annbatch_config,
                cytoanvi_training_config=cytoanvi_training_config,
                **case_control_config,
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
                batch_size=args.batch_size,
                annbatch_config=annbatch_config,
                cytoanvi_training_config=cytoanvi_training_config,
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
                nan_layer=nan_layer,
                seed=seed,
                max_epochs=args.max_epochs,
                n_nhoods=args.mapqc_n_nhoods,
                k_min=args.mapqc_k_min,
                k_max=args.mapqc_k_max,
                run_mapqc=run_mapqc,
                batch_size=args.batch_size,
                annbatch_config=annbatch_config,
                cytoanvi_training_config=cytoanvi_training_config,
            )
        print(json.dumps(to_jsonable(results.get(t)), indent=2, allow_nan=False))
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
        "--b5-max-holdout-types",
        type=int,
        default=None,
        help="B5 sweep: hold out only the N most populous cell types (feasibility on many-cluster "
        "label sets). Default: all types.",
    )
    ap.add_argument(
        "--b5-checkpoint",
        default=None,
        help="B5 sweep: write partial results to this path after each held-out type (survive "
        "timeouts).",
    )
    ap.add_argument(
        "--b5-cytovi-baseline",
        action="store_true",
        help="B5 (inductive): also fit an unsupervised CytoVI OOD baseline per type and report "
        "its AUROC for comparison.",
    )
    ap.add_argument(
        "--b5-no-logit",
        action="store_true",
        help="B5: skip the secondary logit-space TTA uncertainty pass (roughly halves per-holdout "
        "evaluation cost).",
    )
    ap.add_argument(
        "--b5-mode",
        choices=["transductive", "inductive"],
        default="transductive",
        help="B5: transductive scoring on the merged object or inductive calibrated scoring.",
    )
    ap.add_argument(
        "--ewc-lambdas",
        default=None,
        help="B6: comma-separated ewc_importance values (default: 0,1,10,100,1000)",
    )
    ap.add_argument(
        "--case-control-key",
        default=None,
        help="B4/B6: obs column defining real reference/query case-control groups.",
    )
    ap.add_argument(
        "--control-values",
        default=None,
        help="B4/B6: comma-separated values in --case-control-key to use as reference controls.",
    )
    ap.add_argument(
        "--case-values",
        default=None,
        help="B4/B6: comma-separated values in --case-control-key to use as query cases.",
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
        "--b1-baselines",
        default="all",
        help=(
            "B1 optional baselines to run. Use all, fast, none, or comma-separated "
            "values from harmony,xgboost,rapids-graph,flowsom. Core CytoVI/raw kNN "
            "baselines always run."
        ),
    )
    ap.add_argument(
        "--training-accelerator",
        default="auto",
        help="B1: Lightning accelerator for CytoANVI/CytoVI training (for example auto or cpu).",
    )
    ap.add_argument(
        "--training-devices",
        default="auto",
        help="B1: Lightning devices for CytoANVI/CytoVI training.",
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
    ap.add_argument(
        "--annbatch",
        action="store_true",
        help="Use the experimental AnnBatch benchmark backend for CytoANVI training.",
    )
    ap.add_argument(
        "--annbatch-cache-dir",
        default=".scratch/cytoanvi-benchmark/annbatch-cache",
        help="Directory for temporary AnnBatch benchmark stores.",
    )
    ap.add_argument(
        "--annbatch-chunk-size",
        type=int,
        default=512,
        help="AnnBatch contiguous obs chunk size.",
    )
    ap.add_argument(
        "--annbatch-preload-nchunks",
        type=int,
        default=32,
        help="Number of AnnBatch chunks to preload.",
    )
    ap.add_argument(
        "--annbatch-cache-mode",
        choices=["temporary", "reuse"],
        default="temporary",
        help="AnnBatch cache behavior: create per-run stores or reuse a stable cache key.",
    )
    ap.add_argument(
        "--annbatch-cache-key",
        default=None,
        help="Stable cache key used when --annbatch-cache-mode=reuse.",
    )
    ap.add_argument(
        "--cytoanvi-recipe",
        choices=["default", "balanced", "publication"],
        default="default",
        help="CytoANVI benchmark training recipe.",
    )
    ap.add_argument(
        "--class-weighting",
        choices=["none", "inverse_frequency", "sqrt_inverse_frequency"],
        default=None,
        help="Override CytoANVI classifier-loss class weighting.",
    )
    ap.add_argument(
        "--class-weight-clip",
        type=float,
        default=10.0,
        help="Maximum computed CytoANVI class weight before mean normalization.",
    )
    ap.add_argument(
        "--y-prior",
        choices=["uniform", "empirical"],
        default=None,
        help="Override CytoANVI label prior.",
    )
    ap.add_argument(
        "--classification-ratio",
        type=float,
        default=None,
        help="Override SemiSupervisedTrainingPlan classification_ratio.",
    )
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override CytoANVI SemiSupervisedTrainingPlan learning rate.",
    )
    ap.add_argument(
        "--gradient-clip-val",
        type=float,
        default=None,
        help="Override Lightning gradient_clip_val for CytoANVI training.",
    )
    ap.add_argument(
        "--warm-start-cytovi",
        action="store_true",
        help="Reserved for future B1 CytoVI warm-start reuse; not enabled by recipes.",
    )
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
    annbatch_info = _annbatch_payload(_annbatch_config_from_args(args))
    if annbatch_info is not None:
        payload["annbatch"] = annbatch_info
    payload = to_jsonable(payload)
    text = json.dumps(payload, indent=2, allow_nan=False)
    print(text)
    if args.out:
        save_json(args.out, payload)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
