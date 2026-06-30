r"""Annotate Nuñez vignette PBMCs with CytoVI-tutorial manual cell types.

The Nuñez FCS files ship **without** cell-type labels. The CytoVI batch-correction tutorial
trains CytoVI, clusters the latent space (Leiden, resolution 0.4), then maps clusters to eleven
PBMC subsets via a fixed dictionary.

This script reproduces that workflow and writes a **new** AnnData file (default:
``data/nunez_annotated.h5ad``). It does not modify source FCS files or benchmark result JSON.

Example::

    PYTHONPATH=src:. python -m benchmarks.cytoanvi.annotate_nunez \
        --data-dir data --out data/nunez_annotated.h5ad --max-epochs 100

Reference: https://docs.scvi-tools.org/en/latest/tutorials/notebooks/cytometry/CytoVI_batch_correction_tutorial.html
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scanpy as sc

from benchmarks.cytoanvi.data import SCALED_LAYER, _resolve_file

# Leiden cluster id (str) → cell type name from the CytoVI batch-correction tutorial.
NUNEZ_TUTORIAL_ANNOTATION: dict[str, str] = {
    "0": "B cells",
    "1": "Naive CD4 T cells",
    "2": "Memory CD4 T cells",
    "3": "Dendritic cells",
    "4": "Classical monocytes",
    "5": "Non-classical monocytes",
    "6": "Natural killer cells",
    "7": "Memory CD8 T cells",
    "8": "Naive CD8 T cells",
    "9": "Regulatory T cells",
    "10": "Plasmacytoid dendritic cells",
}


def _merge_extra_leiden_clusters(
    adata,
    leiden_key: str,
    latent_key: str,
    *,
    max_cluster: int = 10,
) -> list[str]:
    """Merge Leiden clusters above ``max_cluster`` into nearest major cluster (latent centroid)."""
    clusters = adata.obs[leiden_key].astype(str)
    latent = np.asarray(adata.obsm[latent_key])
    majors = sorted({c for c in clusters.unique() if int(c) <= max_cluster}, key=int)
    extras = sorted({c for c in clusters.unique() if int(c) > max_cluster}, key=int)
    if not extras:
        return []

    centroids = {
        c: latent[(clusters == c).to_numpy()].mean(axis=0) for c in majors
    }
    merged_from: list[str] = []
    for c in extras:
        mask = (clusters == c).to_numpy()
        extra_centroid = latent[mask].mean(axis=0)
        nearest = min(
            majors,
            key=lambda major: float(np.linalg.norm(centroids[major] - extra_centroid)),
        )
        adata.obs.loc[mask, leiden_key] = nearest
        merged_from.append(f"{c}->{nearest}")
    return merged_from


def load_nunez_merged(data_dir: str):
    """Read both Nuñez FCS batches, preprocess, and merge (no labels)."""
    from scvi.external import cytovi

    paths = {}
    for name in ("Nunez_PBMCs_batch1.fcs", "Nunez_PBMCs_batch2.fcs"):
        p = _resolve_file(data_dir, name)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            raise FileNotFoundError(
                f"{p} missing or empty. Place vignette FCS in {data_dir} or repo data/."
            )
        paths[name] = p

    b1 = cytovi.read_fcs(paths["Nunez_PBMCs_batch1.fcs"], remove_markers=["Time", "LD", "-"])
    b2 = cytovi.read_fcs(paths["Nunez_PBMCs_batch2.fcs"], remove_markers=["Time", "LD", "-"])
    for adata in (b1, b2):
        cytovi.transform_arcsinh(adata)
        cytovi.scale(adata)
    merged = cytovi.merge_batches([b1, b2])
    merged.obs_names_make_unique()
    return merged


def annotate_with_cytovi_tutorial(
    adata,
    *,
    max_epochs: int = 100,
    leiden_resolution: float = 0.4,
    leiden_key: str = "leiden_CytoVI",
    labels_key: str = "cell_type",
    latent_key: str = "X_CytoVI",
    seed: int = 0,
    annotation_map: dict[str, str] | None = None,
    merge_extra_clusters: bool = True,
    model_checkpoint: str | None = None,
):
    """Train CytoVI, cluster latent, apply tutorial manual labels."""
    import scvi
    from scvi.external import CYTOVI

    scvi.settings.seed = seed
    annotation_map = annotation_map or NUNEZ_TUTORIAL_ANNOTATION

    CYTOVI.setup_anndata(adata, layer=SCALED_LAYER, batch_key="batch")
    if model_checkpoint and os.path.isdir(model_checkpoint):
        model = CYTOVI.load(model_checkpoint, adata=adata)
    else:
        model = CYTOVI(adata)
        model.train(max_epochs=max_epochs)
        if model_checkpoint:
            os.makedirs(model_checkpoint, exist_ok=True)
            model.save(model_checkpoint, overwrite=True)
    model.module.eval()

    adata.obsm[latent_key] = model.get_latent_representation()
    sc.pp.neighbors(adata, use_rep=latent_key)
    sc.tl.leiden(
        adata,
        resolution=leiden_resolution,
        key_added=leiden_key,
        flavor="igraph",
        directed=False,
    )

    merged = []
    if merge_extra_clusters:
        merged = _merge_extra_leiden_clusters(
            adata, leiden_key, latent_key, max_cluster=max(map(int, annotation_map))
        )

    clusters = adata.obs[leiden_key].astype(str)
    unmapped = sorted(set(clusters) - set(annotation_map))
    if unmapped:
        raise ValueError(
            f"Leiden clusters {unmapped} have no entry in the tutorial annotation dict. "
            "Re-run with a different --leiden-resolution or extend NUNEZ_TUTORIAL_ANNOTATION."
        )

    adata.obs[labels_key] = clusters.map(annotation_map).astype("category")
    if merged:
        adata.uns.setdefault("nunez_annotation", {})["merged_leiden_clusters"] = merged
    return adata, model


def main():
    """CLI entry point — parse arguments and run annotation."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data", help="Directory with Nuñez FCS files")
    ap.add_argument(
        "--out",
        default="data/nunez_annotated.h5ad",
        help="Output path (created; does not overwrite FCS inputs)",
    )
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--leiden-resolution", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--model-checkpoint",
        default=None,
        help="Directory to save/load CytoVI (skips training if present)",
    )
    ap.add_argument(
        "--no-merge-extra-clusters",
        action="store_true",
        help="Fail instead of merging Leiden clusters >10 into nearest 0–10",
    )
    ap.add_argument("--metadata-out", default=None, help="Optional JSON sidecar with label counts")
    args = ap.parse_args()

    adata = load_nunez_merged(args.data_dir)
    adata, _model = annotate_with_cytovi_tutorial(
        adata,
        max_epochs=args.max_epochs,
        leiden_resolution=args.leiden_resolution,
        seed=args.seed,
        merge_extra_clusters=not args.no_merge_extra_clusters,
        model_checkpoint=args.model_checkpoint,
    )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    adata.write_h5ad(out_path)

    summary = {
        "out": out_path,
        "n_cells": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "labels_key": "cell_type",
        "batch_key": "batch",
        "leiden_key": "leiden_CytoVI",
        "latent_key": "X_CytoVI",
        "max_epochs": args.max_epochs,
        "leiden_resolution": args.leiden_resolution,
        "cell_type_counts": adata.obs["cell_type"].value_counts().astype(int).to_dict(),
    }
    print(json.dumps(summary, indent=2))

    if args.metadata_out:
        meta_path = os.path.abspath(args.metadata_out)
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {meta_path}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
