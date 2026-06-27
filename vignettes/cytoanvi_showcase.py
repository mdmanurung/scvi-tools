"""CytoANVI showcase vignette — end-to-end demonstration on real cytometry data.

Runs the full CytoANVI story on the two CytoVI vignette datasets:

  D2 Nuñez PBMC  — same antibody panel, two batches, 11 manual cell types
                   → semi-supervised label transfer, integration, novelty, continual update
  D1 Roider BNHL — two divergent panels (shared backbone + panel-specific markers)
                   → panel-aware query mapping via scArches surgery

Usage (from repo root, scvi-tools >= 1.4.3 env with GPU recommended):

    PYTHONPATH=src:. python vignettes/cytoanvi_showcase.py --max-epochs 100

    # quick smoke (synthetic only, ~2 min):
    PYTHONPATH=src:. python vignettes/cytoanvi_showcase.py --smoke

Outputs land in ``.scratch/cytoanvi-vignette/`` (JSON summary + UMAP figures).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import scvi

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "benchmarks" / "cytoanvi" / "data"
OUT_DIR = REPO / ".scratch" / "cytoanvi-vignette"

UNLABELED = "Unknown"
LABELS_KEY = "cell_type"
BATCH_KEY = "batch"
LATENT_KEY = "X_CytoANVI"


def _banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}", flush=True)


def _umap_figure(adata, color_keys: list[str], out_path: Path, basis: str = LATENT_KEY) -> None:
    """Neighbors + UMAP on the CytoANVI latent; save a multi-panel figure."""
    import matplotlib.pyplot as plt
    import scanpy as sc

    a = adata.copy()
    if basis not in a.obsm:
        return
    sc.pp.neighbors(a, use_rep=basis, n_neighbors=15)
    sc.tl.umap(a)
    keys = [k for k in color_keys if k in a.obs.columns]
    if not keys:
        return
    sc.pl.umap(a, color=keys, ncols=min(3, len(keys)), show=False, wspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved figure → {out_path}", flush=True)


def _holdout_mask(labels, unlabeled_category, frac, seed):
    """Stratified holdout (mirrors benchmarks.cytoanvi.tasks._holdout)."""
    from benchmarks.cytoanvi.tasks import _holdout

    return _holdout(labels, unlabeled_category, frac, seed)


def section_nunez(max_epochs: int, subsample: int, seed: int) -> dict:
    """Nuñez PBMC: label transfer, integration, novelty, continual update."""
    from benchmarks.common.scib import LATENT_OBSM, run_scib_benchmark
    from benchmarks.common.training import latent_obsm, train_cytoanvi, train_cytovi
    from benchmarks.cytoanvi import data as data_mod
    from benchmarks.cytoanvi import metrics as bench_metrics
    from benchmarks.cytoanvi import tasks as task_mod
    from benchmarks.cytoanvi.baselines import cytovi_latent_and_knn

    _banner("Section A — Nuñez PBMC (same panel, batch correction + label transfer)")
    print(
        textwrap.dedent(
            """
            Dataset: 200k PBMCs, 41 markers, 2 batches (CytoVI batch-correction tutorial).
            Labels: 11 manually annotated subsets (nunez_annotated.h5ad).
            CytoANVI exercise: hold out 20% of labels → semi-supervised training → predict().
            Baseline: CytoVI latent + k-NN (the CytoVI vignette label-transfer method).
            """
        ).strip(),
        flush=True,
    )

    adata = data_mod.load_nunez(
        str(REPO / "data"),
        auto_download=False,
        max_cells=subsample,
        annotate=False,
        seed=seed,
    )
    print(
        f"  loaded {adata.n_obs:,} cells × {adata.n_vars} markers, "
        f"{adata.obs[LABELS_KEY].nunique()} cell types",
        flush=True,
    )

    scvi.settings.seed = seed
    kw = dict(
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        batch_key=BATCH_KEY,
        seed=seed,
        max_epochs=max_epochs,
    )

    # --- B1 + B2: one CytoANVI and one CytoVI train each (no redundant retrains) ---
    true = np.asarray(adata.obs[LABELS_KEY].astype(str))
    held = _holdout_mask(true, UNLABELED, 0.2, seed)
    work = adata.copy()
    masked = true.copy()
    masked[held] = UNLABELED
    work.obs[LABELS_KEY] = masked

    print("\n  → training CytoANVI (semi-supervised) …", flush=True)
    anvi_model, anvi_adata = train_cytoanvi(
        work,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        batch_key=BATCH_KEY,
        max_epochs=max_epochs,
    )
    cytoanvi_pred = np.asarray(anvi_model.predict())
    b1_cytoanvi = bench_metrics.label_transfer_metrics(true[held], cytoanvi_pred[held])

    print("  → training CytoVI + k-NN baseline …", flush=True)
    knn_pred_unlab, cytovi_latent, unlab_mask = cytovi_latent_and_knn(
        work,
        LABELS_KEY,
        UNLABELED,
        batch_key=BATCH_KEY,
        max_epochs=max_epochs,
    )
    knn_full = masked.copy()
    knn_full[unlab_mask] = knn_pred_unlab
    b1_knn = bench_metrics.label_transfer_metrics(true[held], knn_full[held])

    b1 = {
        "task": "b1_label_transfer",
        "seed": seed,
        "max_epochs": max_epochs,
        "holdout_frac": 0.2,
        "n_held": int(held.sum()),
        "cytoanvi": b1_cytoanvi,
        "cytovi_knn": b1_knn,
    }
    print(
        f"     CytoANVI  macro-F1={b1_cytoanvi['macro_f1']:.3f}  acc={b1_cytoanvi['accuracy']:.3f}  "
        f"(n_held={b1['n_held']})",
        flush=True,
    )
    print(
        f"     CytoVI kNN macro-F1={b1_knn['macro_f1']:.3f}  acc={b1_knn['accuracy']:.3f}  "
        f"ΔF1={b1_cytoanvi['macro_f1'] - b1_knn['macro_f1']:+.3f}",
        flush=True,
    )

    print("\n  → B2 integration (scib-metrics on shared latents) …", flush=True)
    latent_obsm(anvi_adata, anvi_model, obsm_key=LATENT_OBSM)
    vi_adata = adata.copy()
    vi_adata.obsm[LATENT_OBSM] = cytovi_latent
    keep = true != UNLABELED
    subsample_per_batch = min(10_000, subsample // 2)
    b2 = {
        "task": "b2_integration",
        "seed": seed,
        "max_epochs": max_epochs,
        "subsample_per_batch": subsample_per_batch,
        "cytoanvi": run_scib_benchmark(
            anvi_adata[keep].copy(),
            batch_key=BATCH_KEY,
            label_key=LABELS_KEY,
            embedding_obsm_key=LATENT_OBSM,
            subsample_per_batch=subsample_per_batch,
            seed=seed,
        ),
        "cytovi": run_scib_benchmark(
            vi_adata[keep].copy(),
            batch_key=BATCH_KEY,
            label_key=LABELS_KEY,
            embedding_obsm_key=LATENT_OBSM,
            subsample_per_batch=subsample_per_batch,
            seed=seed,
        ),
    }
    anvi_scib, cyt_scib = b2["cytoanvi"], b2["cytovi"]
    print(
        f"     CytoANVI  total={anvi_scib['total']:.3f}  "
        f"batch={anvi_scib['batch_correction']:.3f}  bio={anvi_scib['bio_conservation']:.3f}",
        flush=True,
    )
    print(
        f"     CytoVI    total={cyt_scib['total']:.3f}  "
        f"batch={cyt_scib['batch_correction']:.3f}  bio={cyt_scib['bio_conservation']:.3f}",
        flush=True,
    )

    anvi_adata.obsm[LATENT_KEY] = anvi_adata.obsm[LATENT_OBSM]
    anvi_adata.obs["pred"] = cytoanvi_pred
    _umap_figure(anvi_adata, [BATCH_KEY, LABELS_KEY, "pred"], OUT_DIR / "nunez_umap.png")

    print("\n  → B5 novelty detection (hold out one cell type) …", flush=True)
    holdout = "Classical monocytes"
    b5 = task_mod.task_b5_novelty(adata, holdout_type=holdout, **kw)
    print(f"     holdout={holdout!r}  AUROC={b5['auroc']:.3f}  n_novel={b5['n_novel']}", flush=True)

    print("\n  → B4 continual update vs plain surgery …", flush=True)
    b4 = task_mod.task_b4_continual(adata, **kw)
    print(
        f"     plain surgery control drift={b4['plain_surgery']['control_latent_drift']:.4f}  "
        f"query acc={b4['plain_surgery']['query_label_transfer']['accuracy']:.3f}",
        flush=True,
    )
    print(
        f"     continual+replay drift={b4['continual_update']['control_latent_drift']:.4f}  "
        f"query acc={b4['continual_update']['query_label_transfer']['accuracy']:.3f}",
        flush=True,
    )

    return {"dataset": "nunez", "b1": b1, "b2": b2, "b5": b5, "b4": b4}


def section_roider(max_epochs: int, seed: int) -> dict:
    """Roider BNHL: panel-1 reference → panel-2 query mapping."""
    from benchmarks.common.training import train_cytoanvi
    from benchmarks.cytoanvi import data as data_mod
    from benchmarks.cytoanvi import tasks as task_mod
    from scvi.external import cytovi

    _banner("Section B — Roider BNHL (panel-divergent query mapping)")
    print(
        textwrap.dedent(
            """
            Dataset: B-cell lymphoma cohort, two antibody panels (CytoVI advanced tutorial).
            Panel 1 carries cell-type labels; panel 2 is unlabelled in the vignette.
            CytoANVI exercise: train on merged panels with nan-mask → map panel-2 cells via
            prepare_query_anndata + load_query_data + predict().
            """
        ).strip(),
        flush=True,
    )

    merged, p1, p2 = data_mod.load_roider(str(DATA_DIR), auto_download=False)
    print(
        f"  panel 1: {p1.n_obs:,} cells  panel 2: {p2.n_obs:,} cells  "
        f"merged backbone via nan_layer",
        flush=True,
    )

    scvi.settings.seed = seed
    kw = dict(
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        batch_key=BATCH_KEY,
        seed=seed,
        max_epochs=max_epochs,
    )

    print("\n  → B3 panel-divergent mapping …", flush=True)
    b3 = task_mod.task_b3_panel_divergent(p1, p2, **kw)
    p1_ho = b3["p1_holdout"]
    conc = b3["p2_concordance_vs_knn"]
    print(
        f"     p1 holdout macro-F1={p1_ho['macro_f1']:.3f}  "
        f"p2 concordance vs CytoVI kNN={conc['agreement']:.3f}  "
        f"(n_p2={b3['n_p2']})",
        flush=True,
    )

    m = cytovi.merge_batches([p1.copy(), p2.copy()], batch_key="panel_batch")
    labels = np.asarray(m.obs[LABELS_KEY].astype(str))
    is_p2 = np.isin(m.obs_names, p2.obs_names)
    masked = labels.copy()
    masked[is_p2] = UNLABELED
    m.obs[LABELS_KEY] = masked

    model, a_viz = train_cytoanvi(
        m,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        batch_key=BATCH_KEY,
        nan_layer="_nan_mask",
        max_epochs=max_epochs,
    )
    a_viz.obsm[LATENT_KEY] = model.get_latent_representation()
    a_viz.obs["pred"] = model.predict()
    a_viz.obs["panel"] = np.where(is_p2, "panel2", "panel1")
    _umap_figure(a_viz, ["panel", LABELS_KEY, "pred"], OUT_DIR / "roider_umap.png")

    return {"dataset": "roider", "b3": b3}


def section_warmstart(max_epochs: int = 15) -> dict:
    """Quick warm-start demo: CytoVI → CytoANVI on synthetic data."""
    from scvi.data import synthetic_iid
    from scvi.external import CYTOVI, CytoANVI
    from scvi.external import cytovi as cytovi_pp

    _banner("Section C — Warm-start from CytoVI (synthetic cytometry)")
    print(
        "  Train unsupervised CytoVI, then CytoANVI.from_cytovi_model() + fine-tune classifier.",
        flush=True,
    )

    adata = synthetic_iid(
        batch_size=256, n_genes=20, n_proteins=0, n_batches=2, n_labels=4, rna_dist="normal"
    )
    adata.layers["raw"] = adata.X.copy()
    cytovi_pp.transform_arcsinh(adata)
    cytovi_pp.scale(adata)
    adata.obs[LABELS_KEY] = adata.obs["labels"].astype(str)
    adata.obs.loc[adata.obs[LABELS_KEY] == "label_0", LABELS_KEY] = UNLABELED

    CYTOVI.setup_anndata(adata, layer="scaled", batch_key="batch", labels_key=LABELS_KEY)
    cv = CYTOVI(adata, n_latent=8)
    cv.train(max_epochs=max_epochs)

    anvi = CytoANVI.from_cytovi_model(cv, unlabeled_category=UNLABELED, labels_key=LABELS_KEY)
    anvi.train(max_epochs=max_epochs)
    labeled = adata.obs[LABELS_KEY] != UNLABELED
    true = adata.obs["labels"].astype(str)
    pred = anvi.predict()
    acc = float((pred[labeled] == true[labeled]).mean())
    print(f"  warm-start label accuracy on labeled cells: {acc:.3f}", flush=True)
    return {"dataset": "synthetic_warmstart", "accuracy_labeled": acc, "max_epochs": max_epochs}


def section_synthetic_smoke(max_epochs: int = 10) -> dict:
    """Minimal synthetic run mirroring example_reference_query.py."""
    _banner("Section D — Synthetic smoke (example_reference_query)")
    example_path = (
        REPO / "docs" / "tutorials" / "notebooks" / "cytometry" / "cytoanvi_example_reference_query.py"
    )
    spec = importlib.util.spec_from_file_location("cytoanvi_example_reference_query", example_path)
    example_reference_query = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Could not load CytoANVI example from {example_path}.")
    spec.loader.exec_module(example_reference_query)
    example_reference_query.main(max_epochs=max_epochs)
    return {"dataset": "synthetic_smoke", "max_epochs": max_epochs, "status": "ok"}


def print_plan() -> None:
    _banner("CytoANVI showcase vignette — plan")
    print(
        textwrap.dedent(
            """
            CytoANVI extends CytoVI with a scANVI-style classifier for antibody cytometry.
            This vignette demonstrates every major API surface on real + synthetic data:

              1. Semi-supervised reference training (partial labels, y_prior, predict)
              2. Label transfer vs CytoVI k-NN baseline (macro-F1 on held-out cells)
              3. Latent integration quality (scib-metrics: batch vs bio tradeoff)
              4. Panel-divergent query mapping (prepare_query_anndata, nan_layer, surgery)
              5. Uncertainty scoring for novel/ambiguous cells (get_uncertainty, Bregman info)
              6. Continual reference update (load_query_data_with_replay, EWC + replay buffer)
              7. Warm-start from a trained CytoVI model (from_cytovi_model)

            Datasets: Nuñez PBMC (D2), Roider BNHL (D1), plus synthetic sanity checks.
            """
        ).strip(),
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-epochs", type=int, default=100, help="training epochs per model")
    ap.add_argument("--subsample", type=int, default=50_000, help="Nuñez cell cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="synthetic-only quick run")
    ap.add_argument("--skip-roider", action="store_true")
    ap.add_argument("--skip-nunez", action="store_true")
    args = ap.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print_plan()
    t0 = time.time()
    results: dict = {
        "max_epochs": args.max_epochs,
        "seed": args.seed,
        "subsample": args.subsample,
        "sections": {},
    }

    if args.smoke:
        results["sections"]["smoke"] = section_synthetic_smoke(max_epochs=min(10, args.max_epochs))
    else:
        if not args.skip_nunez:
            results["sections"]["nunez"] = section_nunez(args.max_epochs, args.subsample, args.seed)
        if not args.skip_roider:
            results["sections"]["roider"] = section_roider(args.max_epochs, args.seed)
        results["sections"]["warmstart"] = section_warmstart(max_epochs=min(20, args.max_epochs))

    results["elapsed_s"] = round(time.time() - t0, 1)
    out_json = OUT_DIR / "showcase_summary.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)

    _banner("Vignette complete")
    print(f"  elapsed: {results['elapsed_s']}s", flush=True)
    print(f"  summary → {out_json}", flush=True)
    if (OUT_DIR / "nunez_umap.png").exists():
        print(f"  figures → {OUT_DIR}/*.png", flush=True)


if __name__ == "__main__":
    main()
