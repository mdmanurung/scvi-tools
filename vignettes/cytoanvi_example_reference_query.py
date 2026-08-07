"""Runnable CytoANVI example: build a labeled reference, map a query, transfer labels.

Mirrors the structure of the scANVI scArches surgery tutorial, but on antibody-intensity
cytometry data (CytoVI's domain). Run with the cytoanvi 0.2.0 environment:

    PYTHONPATH=src:. python vignettes/cytoanvi_example_reference_query.py

Stages:
  1. same-panel: reference train -> query surgery -> label transfer -> evaluation.
  2. panel-divergent: reference with a backbone/panel-specific split -> a query measured on the
     backbone panel only -> ``prepare_query_anndata`` (pad + mask absent markers) -> surgery ->
     label transfer.
  3. continual engineering path: an explicitly predeclared reference replay subset plus an
     external matched control. TTA novelty and uncertainty-selected replay remain unsupported;
     see ``docs/usage_readiness.md``.
"""

from __future__ import annotations

import numpy as np

from cytoanvi import CytoANVI
from scvi.data import synthetic_iid
from scvi.external import cytovi as cytovi_pp

LAYER = "scaled"
NAN_LAYER = "_nan_mask"
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
    print("[done] same-panel reference -> surgery -> label transfer complete.\n")

    panel_divergent_query(max_epochs=max_epochs)
    continual_update(max_epochs=max_epochs)


def continual_update(max_epochs: int = 20) -> None:
    """Continual update with replay declared independently of model uncertainty."""
    ref = _make_cytometry_adata(seed=4)
    ref.obs[LABELS_KEY] = ref.obs[LABELS_KEY].astype(str)
    ref.obs.loc[ref.obs[LABELS_KEY] == "label_0", LABELS_KEY] = UNLABELED

    CytoANVI.setup_anndata(
        ref,
        layer=LAYER,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
    )
    model = CytoANVI(ref, n_latent=10)
    model.train(max_epochs=max_epochs)

    # This fixed selection rule is declared before query construction and does not inspect model
    # uncertainty or outcomes. Scientific continual-learning use remains a P2 no-go.
    replay = ref[: max(1, ref.n_obs // 5)].copy()
    query = _make_cytometry_adata(seed=5)
    query.obs[LABELS_KEY] = UNLABELED
    control = query[:64].copy()

    updated = CytoANVI.load_query_data_with_replay(
        query, model, replay_adata=replay, control_adata=control
    )
    updated.train(max_epochs=max_epochs, plan_kwargs={"ewc_importance": 1.0})
    print(f"[continual] query predictions: {updated.predict()[:5]}")
    print("[done] predeclared-replay continual engineering path complete.")


def panel_divergent_query(max_epochs: int = 20) -> None:
    """Map a query measured with a *different antibody panel* than the reference.

    The reference carries a backbone (markers shared across the cohort) plus panel-specific
    markers measured only in a sub-cohort; CytoVI encodes on the backbone and reconstructs the
    full panel. The query was measured on the backbone panel only. ``prepare_query_anndata`` pads
    the absent panel-specific markers and masks them via the ``nan_layer`` (so they are *not* read
    as observed-zero intensities) before scArches surgery.
    """
    # --- Reference with a genuine backbone / panel-specific split ---
    ref = _make_cytometry_adata(seed=2)
    ref_vars = ref.var_names
    backbone = list(ref_vars[:25])
    panel_specific = list(ref_vars[25:])

    # panel-specific markers were only measured in part of the reference cohort: mask them there
    # so they fall outside the backbone (the set of markers present in *all* cells).
    nan_mask = np.ones_like(ref.layers[LAYER])
    sub = ref.n_obs // 2
    nan_mask[:sub, 25:] = 0.0
    ref.layers[LAYER][:sub, 25:] = 0.0
    ref.layers[NAN_LAYER] = nan_mask

    held_out = "label_0"
    ref.obs[LABELS_KEY] = ref.obs[LABELS_KEY].astype(str)
    ref.obs.loc[ref.obs[LABELS_KEY] == held_out, LABELS_KEY] = UNLABELED

    CytoANVI.setup_anndata(
        ref,
        layer=LAYER,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        nan_layer=NAN_LAYER,
    )
    model = CytoANVI(ref, n_latent=10)
    model.train(max_epochs=max_epochs)
    n_backbone = int(model.module.encoder_marker_mask.sum())
    print(
        f"[panel] reference backbone = {n_backbone} markers, {len(panel_specific)} panel-specific"
    )

    # --- Query measured on the backbone panel only (panel-specific markers absent) ---
    query = _make_cytometry_adata(seed=3)[:, backbone].copy()
    query.obs[LABELS_KEY] = UNLABELED

    # pad the absent panel-specific markers and mask them (vs. the gene-style zero-fill)
    CytoANVI.prepare_query_anndata(query, model)
    assert list(query.var_names) == list(ref_vars)
    masked = np.asarray(query.layers[NAN_LAYER])[:, query.var_names.get_indexer(panel_specific)]
    print(
        f"[panel] query padded to {query.n_vars} markers; "
        f"absent markers masked = {bool((masked == 0).all())}"
    )

    q_model = CytoANVI.load_query_data(query, model)
    q_model.train(max_epochs=max_epochs, plan_kwargs={"weight_decay": 0.0})
    query.obs["pred"] = q_model.predict()

    eval_mask = query.obs["true_label"] != held_out
    q_acc = float((query.obs["pred"][eval_mask] == query.obs["true_label"][eval_mask]).mean())
    print(f"[panel] panel-divergent label-transfer accuracy on mappable cells: {q_acc:.3f}")
    print("[done] panel-divergent prepare_query -> surgery -> label transfer complete.")


if __name__ == "__main__":
    main()
