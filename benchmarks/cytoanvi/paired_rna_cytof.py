"""Synthetic paired RNA + CyTOF fixtures and B7 multimodal integration benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from benchmarks.common.scib import LATENT_OBSM, run_scib_benchmark
from benchmarks.common.training import SCALED_LAYER, latent_obsm, train_cytoanvi
from benchmarks.cytoanvi.metrics import rna_macro_f1_paired
from scvi.external import cytovi
from scvi.external.cytovi.marker_harmonization import (
    DEFAULT_GENE_TO_PROTEIN,
    harmonize_marker_intersection,
)
from scvi.external.cytovi.paired_cytoanvi import prepare_paired_cytoanvi


def make_synthetic_paired_rna_cytof(
    seed: int = 0,
    n_rna: int = 200,
    n_cytof: int = 300,
    n_labels: int = 3,
    n_samples: int = 3,
) -> tuple[AnnData, AnnData, list[str]]:
    """Paired RNA + CyTOF with overlapping immune markers (no download).

    RNA uses gene symbols; cytometry uses protein names. Donors are drawn from a shared pool of
    ``n_samples`` IDs (random assignment; overlap across modalities is expected). Returns
    ``(rna, cytof, markers)``.
    """
    rng = np.random.default_rng(seed)
    protein_markers = ["CD3", "CD4", "CD8a", "CD19", "CD14", "CD56"]
    genes = []
    for p in protein_markers:
        gene = next((g for g, pr in DEFAULT_GENE_TO_PROTEIN.items() if pr == p), p)
        genes.append(gene)

    labels = [f"type_{i}" for i in range(n_labels)]
    sample_ids = [f"donor_{i}" for i in range(n_samples)]
    rna_labels = rng.choice(labels, size=n_rna)
    cy_labels = rng.choice(labels, size=n_cytof)
    rna_samples = rng.choice(sample_ids, size=n_rna)
    cy_samples = rng.choice(sample_ids, size=n_cytof)

    rna_x = rng.normal(size=(n_rna, len(genes))).astype(np.float32)
    for i, lab in enumerate(labels):
        mask = rna_labels == lab
        rna_x[mask] += (i + 1) * 0.5

    rna = AnnData(
        X=rna_x,
        obs={
            "celltype": rna_labels.astype(str),
            "sample_id": rna_samples.astype(str),
        },
        var=pd.DataFrame(index=genes),
    )

    cy_x = rng.uniform(0, 1, size=(n_cytof, len(protein_markers))).astype(np.float32)
    for i, lab in enumerate(labels):
        mask = cy_labels == lab
        cy_x[mask] += (i + 1) * 0.1
    cy_x = np.clip(cy_x, 0, 1)

    cytof = AnnData(
        X=cy_x,
        obs={
            "celltype": cy_labels.astype(str),
            "sample_id": cy_samples.astype(str),
        },
        var=pd.DataFrame(index=protein_markers),
    )
    cytof.layers["raw"] = cytof.X.copy()
    cytovi.transform_arcsinh(cytof, global_scaling_factor=5)
    cytovi.scale(cytof)

    rna_h, markers = harmonize_marker_intersection(rna, protein_markers)
    return rna_h, cytof, markers


def task_b7_multimodal_integration(
    rna_adata: AnnData | None = None,
    cytof_adata: AnnData | None = None,
    markers: list[str] | None = None,
    *,
    labels_key: str = "celltype",
    unlabeled_category: str = "Unknown",
    batch_key: str = "modality",
    sample_key: str = "sample_id",
    seed: int = 0,
    max_epochs: int = 50,
    nn_count: int = 10,
    npcs: int | None = 5,
    subsample_per_batch: int = 500,
) -> dict:
    """B7: CytoANVI on scennep-smoothed paired RNA + CyTOF (Plan A).

    Primary metric: RNA macro-F1 on paired-sample RNA cells.
    Secondary: scib batch mixing on latent.
    """
    if rna_adata is None or cytof_adata is None:
        rna_adata, cytof_adata, markers = make_synthetic_paired_rna_cytof(seed=seed)

    shared_samples = set(rna_adata.obs[sample_key].astype(str)) & set(
        cytof_adata.obs[sample_key].astype(str)
    )

    merged, markers = prepare_paired_cytoanvi(
        rna_adata,
        cytof_adata,
        sample_key=sample_key,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        markers=markers,
        nn_count=nn_count,
        npcs=npcs,
    )

    model, anvi_adata = train_cytoanvi(
        merged,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        layer=SCALED_LAYER,
        max_epochs=max_epochs,
    )
    latent_obsm(anvi_adata, model, obsm_key=LATENT_OBSM)
    preds = model.predict()

    eval_label_key = "eval_celltype" if "eval_celltype" in anvi_adata.obs else labels_key
    rna_macro_f1 = rna_macro_f1_paired(
        anvi_adata,
        preds,
        batch_key=batch_key,
        sample_key=sample_key,
        eval_label_key=eval_label_key,
        shared_samples=shared_samples,
    )

    cytoanvi_scib = run_scib_benchmark(
        anvi_adata,
        batch_key=batch_key,
        label_key=eval_label_key,
        embedding_obsm_key=LATENT_OBSM,
        subsample_per_batch=subsample_per_batch,
        seed=seed,
    )

    return {
        "task": "b7_multimodal_integration",
        "seed": seed,
        "max_epochs": max_epochs,
        "n_rna": int((merged.obs[batch_key] == "RNA").sum()),
        "n_cytof": int((merged.obs[batch_key] == "CyTOF").sum()),
        "n_shared_samples": len(shared_samples),
        "n_markers": len(markers),
        "markers": markers,
        "rna_macro_f1_paired": rna_macro_f1,
        "cytoanvi": cytoanvi_scib,
    }
