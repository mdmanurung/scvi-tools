"""Runnable CytoANVI example: build a labeled reference, map a query, transfer labels.

Mirrors the structure of the scANVI scArches surgery tutorial, but on antibody-intensity
cytometry data (CytoVI's domain). Run with an scvi-tools >= 1.4.3 environment:

    python -m scvi.external.cytoanvi.example_reference_query

Stages: reference train -> query surgery -> label transfer -> evaluation.
"""

from __future__ import annotations

import numpy as np

from scvi.data import synthetic_iid
from scvi.external import CytoANVI
from scvi.external import cytovi as cytovi_pp

LAYER = "scaled"
BATCH_KEY = "batch"
LABELS_KEY = "labels"
UNLABELED = "Unknown"


def _make_cytometry_adata(seed: int, n_genes: int = 30, n_batches: int = 2, n_labels: int = 5):
    """Synthetic antibody-intensity AnnData (continuous, not counts) with cell-type labels."""
    rng = np.random.default_rng(seed)
    adata = synthetic_iid(
        batch_size=256,
        n_genes=n_genes,
        n_proteins=0,
        n_regions=0,
        n_batches=n_batches,
        n_labels=n_labels,
        rna_dist="normal",
    )
    adata.obs_names = [f"s{seed}_{n}" for n in adata.obs_names]
    adata.layers["raw"] = adata.X.copy()
    cytovi_pp.transform_arcsinh(adata)
    cytovi_pp.scale(adata)
    # keep the ground-truth labels around for evaluation
    adata.obs["true_label"] = adata.obs[LABELS_KEY].astype(str).to_numpy()
    _ = rng  # reserved for future label-masking variations
    return adata


def main(max_epochs: int = 20) -> None:
    """Run the reference -> query -> label-transfer pipeline on synthetic cytometry data."""
    # --- Reference: hold out one label as "unlabeled" to exercise the semi-supervised path ---
    ref = _make_cytometry_adata(seed=0)
    held_out = "label_0"
    ref.obs[LABELS_KEY] = ref.obs[LABELS_KEY].astype(str)
    ref.obs.loc[ref.obs[LABELS_KEY] == held_out, LABELS_KEY] = UNLABELED

    CytoANVI.setup_anndata(
        ref,
        layer=LAYER,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
    )
    model = CytoANVI(ref, n_latent=10)
    model.train(max_epochs=max_epochs)
    ref.obsm["X_CytoANVI"] = model.get_latent_representation()
    ref.obs["pred"] = model.predict()

    ref_labeled = ref.obs[LABELS_KEY] != UNLABELED
    ref_acc = float((ref.obs["pred"][ref_labeled] == ref.obs["true_label"][ref_labeled]).mean())
    print(f"[reference] label-transfer accuracy on labeled cells: {ref_acc:.3f}")

    # --- Query: an unannotated dataset mapped onto the reference via scArches surgery ---
    query = _make_cytometry_adata(seed=1)
    query.obs[LABELS_KEY] = UNLABELED  # all query cells unlabeled

    q_model = CytoANVI.load_query_data(query, model)
    q_model.train(max_epochs=max_epochs, plan_kwargs={"weight_decay": 0.0})
    query.obsm["X_CytoANVI"] = q_model.get_latent_representation()
    query.obs["pred"] = q_model.predict()

    # evaluate against held-back ground truth (the held-out reference label never appears
    # in the query here, so accuracy is computed over the labels the model can predict)
    eval_mask = query.obs["true_label"] != held_out
    q_acc = float((query.obs["pred"][eval_mask] == query.obs["true_label"][eval_mask]).mean())
    print(f"[query] label-transfer accuracy on mappable cells: {q_acc:.3f}")
    print("[done] reference -> surgery -> label transfer complete.")


if __name__ == "__main__":
    main()
