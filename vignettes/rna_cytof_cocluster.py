#!/usr/bin/env python3
"""Paired scRNA + CyTOF label transfer with CytoANVI (Plan A).

Flow: prepare_paired_cytoanvi → CytoANVI.train → predict → optional Leiden on latent.

Usage (from repo root, synthetic smoke — no data download):

    PYTHONPATH=src:. python vignettes/rna_cytof_cocluster.py --smoke

Real h5ad inputs:

    PYTHONPATH=src:. python vignettes/rna_cytof_cocluster.py \\
        --rna path/to/rna.h5ad --cytof path/to/cytof.h5ad \\
        --out-dir .scratch/paired_cytoanvi
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DEFAULT = REPO / ".scratch" / "paired_cytoanvi"


def _setup_path() -> None:
    os.chdir(REPO)
    for p in (REPO / "src", REPO):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def run_pipeline(
    rna,
    cytof,
    *,
    out_dir: Path,
    seed: int,
    max_epochs: int,
    nn_count: int,
    npcs: int | None,
    leiden: bool,
) -> dict:
    import scanpy as sc

    from cytoanvi import CytoANVI
    from scvi.external.cytovi.paired_cytoanvi import prepare_paired_cytoanvi

    out_dir.mkdir(parents=True, exist_ok=True)
    merged, markers = prepare_paired_cytoanvi(
        rna,
        cytof,
        nn_count=nn_count,
        npcs=npcs,
    )

    CytoANVI.setup_anndata(
        merged,
        layer="scaled",
        batch_key="modality",
        sample_key="sample_id",
        labels_key="celltype",
        unlabeled_category="Unknown",
    )
    model = CytoANVI(merged, y_prior="empirical")
    model.train(max_epochs=max_epochs)

    merged.obsm["X_CytoANVI"] = model.get_latent_representation()
    preds = model.predict()
    merged.obs["pred"] = preds
    merged.write_h5ad(out_dir / "merged.h5ad")

    if leiden:
        sc.pp.neighbors(merged, use_rep="X_CytoANVI", n_neighbors=15)
        sc.tl.leiden(merged, resolution=0.5, key_added="leiden")

    from benchmarks.cytoanvi.metrics import rna_macro_f1_paired

    summary = {
        "seed": seed,
        "max_epochs": max_epochs,
        "n_markers": len(markers),
        "markers": markers,
        "n_cells": int(merged.n_obs),
        "n_rna": int((merged.obs["modality"] == "RNA").sum()),
        "n_cytof": int((merged.obs["modality"] == "CyTOF").sum()),
        "rna_macro_f1_paired": rna_macro_f1_paired(merged, preds),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def run_smoke(out_dir: Path, seed: int, max_epochs: int, nn_count: int, leiden: bool) -> dict:
    from benchmarks.cytoanvi.paired_rna_cytof import make_synthetic_paired_rna_cytof

    rna, cytof, _ = make_synthetic_paired_rna_cytof(seed=seed)
    summary = run_pipeline(
        rna,
        cytof,
        out_dir=out_dir,
        seed=seed,
        max_epochs=max_epochs,
        nn_count=nn_count,
        npcs=5,
        leiden=leiden,
    )
    summary["mode"] = "smoke"
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def run_real(
    rna_path: Path,
    cytof_path: Path,
    out_dir: Path,
    seed: int,
    max_epochs: int,
    nn_count: int,
    leiden: bool,
) -> dict:
    import anndata as ad

    if not rna_path.exists():
        raise FileNotFoundError(f"RNA h5ad not found: {rna_path}")
    if not cytof_path.exists():
        raise FileNotFoundError(f"CyTOF h5ad not found: {cytof_path}")

    rna = ad.read_h5ad(rna_path)
    cytof = ad.read_h5ad(cytof_path)
    summary = run_pipeline(
        rna,
        cytof,
        out_dir=out_dir,
        seed=seed,
        max_epochs=max_epochs,
        nn_count=nn_count,
        npcs=None,
        leiden=leiden,
    )
    summary["mode"] = "real"
    summary["rna_path"] = str(rna_path)
    summary["cytof_path"] = str(cytof_path)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="run on synthetic paired data")
    ap.add_argument("--rna", type=Path, default=None, help="scRNA-seq h5ad with sample_id")
    ap.add_argument("--cytof", type=Path, default=None, help="CyTOF h5ad with scaled layer + sample_id")
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=447)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--nn-count", type=int, default=20)
    ap.add_argument("--leiden", action="store_true", help="cluster on X_CytoANVI latent")
    args = ap.parse_args()

    _setup_path()
    np.random.seed(args.seed)

    if args.smoke:
        summary = run_smoke(args.out_dir, args.seed, args.max_epochs, args.nn_count, args.leiden)
    elif args.rna and args.cytof:
        summary = run_real(
            args.rna, args.cytof, args.out_dir, args.seed, args.max_epochs, args.nn_count, args.leiden
        )
    else:
        ap.error("Pass --smoke or both --rna and --cytof")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
