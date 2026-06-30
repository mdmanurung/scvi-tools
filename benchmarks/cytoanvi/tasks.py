"""Benchmark tasks B1, B2, B3, B4, B5, B6.

Each task returns a plain dict of metrics; run.py serializes them to JSON.
"""

from __future__ import annotations

import warnings

import numpy as np

from benchmarks.common.scib import LATENT_OBSM, run_scib_benchmark
from benchmarks.common.training import (
    _GRAD_CLIP,
    NAN_LAYER,
    latent_obsm,
    train_cytoanvi,
)

from . import metrics
from .baselines import (
    cytovi_latent_and_knn,
    flowsom_knn,
    harmony_latent_and_knn,
    phenograph_knn,
    raw_marker_knn,
    xgboost_classifier,
)


def _holdout(labels, unlabeled_category, frac, seed):
    """Stratified mask of cells to blank to the unlabeled category (the holdout to score)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(str)
    held = np.zeros(len(labels), dtype=bool)
    for lab in set(labels):
        if lab == unlabeled_category:
            continue
        idx = np.where(labels == lab)[0]
        k = max(1, int(round(frac * len(idx))))
        held[rng.choice(idx, size=k, replace=False)] = True
    return held


def task_b1_label_transfer(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    holdout_frac=0.2,
    seed=0,
    max_epochs=1000,
    n_latent=None,
    n_samples_per_label=None,
    reduce_lr_on_plateau=False,
    batch_size=None,
):
    """B1: CytoANVI classifier vs CytoVI k-NN at transferring labels to held-out cells."""
    true = np.asarray(adata.obs[labels_key].astype(str))
    held = _holdout(true, unlabeled_category, holdout_frac, seed)

    work = adata.copy()
    masked = true.copy()
    masked[held] = unlabeled_category
    work.obs[labels_key] = masked

    model, a = train_cytoanvi(
        work,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
        n_samples_per_label=n_samples_per_label,
        reduce_lr_on_plateau=reduce_lr_on_plateau,
        batch_size=batch_size,
    )
    cytoanvi_pred = np.asarray(model.predict())

    knn_pred_unlab, _, unlab_mask = cytovi_latent_and_knn(
        work,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    knn_full = masked.copy()
    knn_full[unlab_mask] = knn_pred_unlab

    raw_pred_unlab, _, raw_unlab_mask = raw_marker_knn(
        work,
        labels_key,
        unlabeled_category,
    )
    raw_full = masked.copy()
    raw_full[raw_unlab_mask] = raw_pred_unlab

    try:
        hm_pred_unlab, _, hm_unlab_mask = harmony_latent_and_knn(
            work,
            labels_key,
            unlabeled_category,
            batch_key=batch_key,
            seed=seed,
        )
        hm_full = masked.copy()
        hm_full[hm_unlab_mask] = hm_pred_unlab
        harmony_result = metrics.label_transfer_metrics(true[held], hm_full[held])
    except (ImportError, ValueError, KeyError) as e:
        # Harmony is an optional baseline; missing package or config errors should not abort B1.
        # Any other exception is a code bug and should propagate.
        import traceback; traceback.print_exc()
        harmony_result = {"error": str(e)}

    try:
        xgb_pred_unlab, _, xgb_unlab_mask = xgboost_classifier(work, labels_key, unlabeled_category, seed=seed)
        xgb_full = masked.copy()
        xgb_full[xgb_unlab_mask] = xgb_pred_unlab
        xgboost_result = metrics.label_transfer_metrics(true[held], xgb_full[held])
    except (ImportError, ValueError, KeyError) as e:
        import traceback; traceback.print_exc()
        xgboost_result = {"error": str(e)}

    try:
        pg_pred_unlab, _, pg_unlab_mask = phenograph_knn(work, labels_key, unlabeled_category, seed=seed)
        pg_full = masked.copy()
        pg_full[pg_unlab_mask] = pg_pred_unlab
        phenograph_result = metrics.label_transfer_metrics(true[held], pg_full[held])
    except (ImportError, ValueError, KeyError) as e:
        import traceback; traceback.print_exc()
        phenograph_result = {"error": str(e)}

    try:
        fsom_pred_unlab, _, fsom_unlab_mask = flowsom_knn(work, labels_key, unlabeled_category, seed=seed)
        fsom_full = masked.copy()
        fsom_full[fsom_unlab_mask] = fsom_pred_unlab
        flowsom_result = metrics.label_transfer_metrics(true[held], fsom_full[held])
    except (ImportError, ValueError, KeyError) as e:
        import traceback; traceback.print_exc()
        flowsom_result = {"error": str(e)}

    return {
        "task": "b1_label_transfer",
        "seed": seed,
        "max_epochs": max_epochs,
        "holdout_frac": holdout_frac,
        "n_held": int(held.sum()),
        "cytoanvi": metrics.label_transfer_metrics(true[held], cytoanvi_pred[held]),
        "cytovi_knn": metrics.label_transfer_metrics(true[held], knn_full[held]),
        "raw_marker_knn": metrics.label_transfer_metrics(true[held], raw_full[held]),
        "harmony_knn": harmony_result,
        "xgboost": xgboost_result,
        "phenograph": phenograph_result,
        "flowsom": flowsom_result,
    }


def task_b2_integration(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    seed=0,
    max_epochs=1000,
    n_latent=None,
    subsample_per_batch=10_000,
    batch_size=None,
):
    """B2: scib-metrics on CytoANVI vs CytoVI latents."""
    labels = np.asarray(adata.obs[labels_key].astype(str))
    keep = labels != unlabeled_category

    model, a = train_cytoanvi(
        adata,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    anvi_adata = a.copy()
    latent_obsm(anvi_adata, model, obsm_key=LATENT_OBSM)

    _, cytovi_latent, _ = cytovi_latent_and_knn(
        adata,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    vi_adata = adata.copy()
    vi_adata.obsm[LATENT_OBSM] = cytovi_latent

    # scib over labelled cells only (exclude unlabeled category)
    anvi_sub = anvi_adata[keep].copy()
    vi_sub = vi_adata[keep].copy()

    out = {
        "task": "b2_integration",
        "seed": seed,
        "max_epochs": max_epochs,
        "subsample_per_batch": subsample_per_batch,
        "cytoanvi": run_scib_benchmark(
            anvi_sub,
            batch_key=batch_key,
            label_key=labels_key,
            embedding_obsm_key=LATENT_OBSM,
            subsample_per_batch=subsample_per_batch,
            seed=seed,
        ),
        "cytovi": run_scib_benchmark(
            vi_sub,
            batch_key=batch_key,
            label_key=labels_key,
            embedding_obsm_key=LATENT_OBSM,
            subsample_per_batch=subsample_per_batch,
            seed=seed,
        ),
    }
    return out


def task_b3_panel_divergent(
    p1,
    p2,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    holdout_frac=0.2,
    seed=0,
    max_epochs=1000,
    n_latent=None,
    batch_size=None,
):
    """B3: panel-1 reference -> panel-2 query via panel-aware prep + scArches surgery."""
    from scvi.external import cytovi

    merged = cytovi.merge_batches([p1.copy(), p2.copy()], batch_key="panel_batch")
    labels = np.asarray(merged.obs[labels_key].astype(str))
    is_p2 = np.isin(merged.obs_names, p2.obs_names)
    # Compute holdout on p1 cells only so the RNG state is independent of p2 contents.
    p1_idx = np.where(~is_p2)[0]
    p1_held = _holdout(labels[p1_idx], unlabeled_category, holdout_frac, seed)
    held = np.zeros(len(labels), dtype=bool)
    held[p1_idx] = p1_held
    masked = labels.copy()
    masked[held | is_p2] = unlabeled_category
    merged.obs[labels_key] = masked

    model, a = train_cytoanvi(
        merged,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=NAN_LAYER,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    pred = np.asarray(model.predict())

    out = {
        "task": "b3_panel_divergent",
        "seed": seed,
        "max_epochs": max_epochs,
        "n_p2": int(is_p2.sum()),
        "p1_holdout": metrics.label_transfer_metrics(labels[held], pred[held]),
    }

    knn_pred_unlab, _, unlab_mask = cytovi_latent_and_knn(
        merged,
        labels_key,
        unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=NAN_LAYER,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    knn_full = masked.copy()
    knn_full[unlab_mask] = knn_pred_unlab
    out["p2_inter_method_agreement_vs_knn"] = metrics.concordance(pred[is_p2], knn_full[is_p2])
    return out


def task_b5_novelty(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    holdout_type=None,
    seed=0,
    max_epochs=1000,
    n_latent=None,
    batch_size=None,
):
    """B5: does get_uncertainty flag a cell type held out of the reference entirely?"""
    labels = np.asarray(adata.obs[labels_key].astype(str))
    types = sorted(t for t in set(labels) if t != unlabeled_category)
    if holdout_type is None:
        holdout_type = types[0]
    is_novel = labels == holdout_type

    work = adata.copy()
    masked = labels.copy()
    masked[is_novel] = unlabeled_category
    work.obs[labels_key] = masked
    model, a = train_cytoanvi(
        work,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    unc_latent = model.get_uncertainty(mode="latent")
    unc_logit = model.get_uncertainty(mode="logit")
    return {
        "task": "b5_novelty",
        "seed": seed,
        "max_epochs": max_epochs,
        "holdout_type": holdout_type,
        # uncertainty scored on the full adata (novel cells included) — calibration is
        # transductive: the training set did not contain the held-out type, but uncertainty
        # is measured on the same merged object.  Flag so consumers know the evaluation mode.
        "b5_evaluation_mode": "calibration_transductive",
        "latent": metrics.novelty_auroc(unc_latent, is_novel),
        "logit": metrics.novelty_auroc(unc_logit, is_novel),
        **metrics.novelty_auroc(unc_latent, is_novel),  # top-level for backward compat
    }


def task_b5_holdout_sweep(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    seed=0,
    max_epochs=1000,
    n_latent=None,
    batch_size=None,
):
    """B5 sweep: AUROC for each cell type held out as novel."""
    labels = np.asarray(adata.obs[labels_key].astype(str))
    types = sorted(t for t in set(labels) if t != unlabeled_category)
    per_type = {}
    for ht in types:
        per_type[ht] = task_b5_novelty(
            adata,
            labels_key=labels_key,
            unlabeled_category=unlabeled_category,
            batch_key=batch_key,
            sample_key=sample_key,
            nan_layer=nan_layer,
            holdout_type=ht,
            seed=seed,
            max_epochs=max_epochs,
            n_latent=n_latent,
            batch_size=batch_size,
        )
    valid_types = [ht for ht, v in per_type.items() if not np.isnan(v.get("auroc", float("nan")))]
    aurocs = [per_type[ht]["auroc"] for ht in valid_types]

    # BH FDR over per-type AUROCs: normal approximation z = (AUROC - 0.5) / se,
    # se = sqrt((n_novel + n_ref + 1) / (12 * n_novel * n_ref))  # Wilcoxon AUC SE.
    # One-sided sf because we only care about AUROC > 0.5 (detectable novelty).
    fdr_fields: dict = {}
    if len(aurocs) >= 2:
        from scipy.stats import norm
        from statsmodels.stats.multitest import multipletests

        all_labels = np.asarray(adata.obs[labels_key].astype(str))
        n_novel_arr = np.array([(all_labels == ht).sum() for ht in valid_types])
        n_ref_arr = adata.n_obs - n_novel_arr
        se = np.sqrt((n_novel_arr + n_ref_arr + 1) / np.maximum(12 * n_novel_arr * n_ref_arr, 1))
        z = (np.array(aurocs) - 0.5) / se
        p_vals = norm.sf(z)
        _, fdr_q, _, _ = multipletests(p_vals, method="fdr_bh")
        sig_types = [ht for ht, q in zip(valid_types, fdr_q) if q < 0.05]
        fdr_fields = {
            "auroc_pvalues": {ht: float(p) for ht, p in zip(valid_types, p_vals)},
            "auroc_fdr_q": {ht: float(q) for ht, q in zip(valid_types, fdr_q)},
            "n_fdr_significant": len(sig_types),
            "mean_auroc_fdr_sig": float(np.mean([per_type[ht]["auroc"] for ht in sig_types])) if sig_types else float("nan"),
        }

    return {
        "task": "b5_holdout_sweep",
        "seed": seed,
        "max_epochs": max_epochs,
        # Each per-type novelty score is transductive: uncertainty is measured on
        # the full merged object (ref + novel), so the model has never seen the held-out
        # type during training but the uncertainty threshold is not cross-validated.
        # Consumers comparing across seeds should treat absolute AUROC values as
        # optimistic; only relative rankings across cell types are robust.
        "calibration_note": "transductive — uncertainty thresholds not cross-validated",
        "per_type": per_type,
        "mean_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "n_fdr_significant": fdr_fields.get("n_fdr_significant", 0),
        "mean_auroc_fdr_sig": fdr_fields.get("mean_auroc_fdr_sig", float("nan")),
        "best_auroc": float(max(aurocs)) if aurocs else float("nan"),
        **fdr_fields,
    }


def _split_reference_query(
    adata,
    *,
    batch_key: str,
    query_batch_values: list[str],
    seed: int = 0,
):
    """Pseudo case/control split: query batches vs reference batches (plumbing validation).

    Returns ``(ref, query, fallback_used)`` where ``fallback_used`` is ``True`` when the
    batch-based split was too small and a random 70/30 permutation was used instead.
    """
    batch = np.asarray(adata.obs[batch_key].astype(str))
    is_query = np.isin(batch, query_batch_values)
    ref = adata[~is_query].copy()
    query = adata[is_query].copy()
    if ref.n_obs < 64 or query.n_obs < 64:
        warnings.warn(
            f"_split_reference_query: batch-based split yielded ref={ref.n_obs} / "
            f"query={query.n_obs} cells — falling back to random 70/30 permutation. "
            "Results are NOT a real case-control test.",
            UserWarning,
            stacklevel=2,
        )
        rng = np.random.default_rng(seed)
        perm = rng.permutation(adata.n_obs)
        cut = int(0.7 * adata.n_obs)
        ref = adata[perm[:cut]].copy()
        query = adata[perm[cut:]].copy()
        return ref, query, True
    return ref, query, False


def _replay_latent_drift(ref_model, updated_model, replay_adata):
    """Mean per-cell L2 drift of replay/reference latents after query surgery."""
    z_ref = ref_model.get_latent_representation(replay_adata)
    z_new = updated_model.get_latent_representation(replay_adata)
    return float(np.linalg.norm(z_ref - z_new, axis=1).mean())


def _b4_setup(
    adata,
    *,
    labels_key: str = "labels",
    unlabeled_category: str = "Unknown",
    batch_key: str = "batch",
    sample_key=None,
    nan_layer=None,
    seed: int = 0,
    max_epochs: int = 1000,
    n_latent=None,
    control_frac: float = 0.1,
    replay_frac: float = 0.2,
    query_batch_values=None,
    batch_size=None,
) -> dict:
    """Train reference and select replay/control — shared setup for B4 and B6.

    Returns a dict consumed by :func:`task_b4_continual`.  B6 calls this once and passes
    the result to each per-λ surgery call, avoiding redundant reference retraining.
    """
    from cytoanvi import CytoANVI

    if query_batch_values is None:
        batches = sorted(adata.obs[batch_key].astype(str).unique())
        query_batch_values = [batches[-1]]
    query_batch_values = list(query_batch_values)

    labels = np.asarray(adata.obs[labels_key].astype(str))
    ref_adata, query_adata, _fallback_split = _split_reference_query(
        adata, batch_key=batch_key, query_batch_values=query_batch_values, seed=seed
    )
    # Build true_query in query_adata.obs_names order to handle the permuted-fallback branch
    # of _split_reference_query (np.isin returns adata order, not query order).
    label_by_obs = dict(zip(adata.obs_names, labels, strict=True))
    true_query = np.asarray([label_by_obs[n] for n in query_adata.obs_names])
    query_adata = query_adata.copy()

    ref_model, _ = train_cytoanvi(
        ref_adata,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )

    # Sample control before blanking labels so control cells retain their true labels.
    # Cap n_ctrl so it never exceeds the query population (avoids ValueError from
    # rng.choice(replace=False) when max(16, frac*n) > n for small query sets).
    n_ctrl = min(max(16, int(control_frac * query_adata.n_obs)), query_adata.n_obs)
    rng = np.random.default_rng(seed)
    ctrl_idx = rng.choice(query_adata.n_obs, n_ctrl, replace=False)
    control = query_adata[ctrl_idx].copy()
    query_adata.obs[labels_key] = unlabeled_category

    replay = CytoANVI.select_replay_by_uncertainty(ref_model, ref_adata, fraction=replay_frac)

    train_extra: dict = {}
    if batch_size is not None:
        train_extra["batch_size"] = batch_size

    return {
        "ref_model": ref_model,
        "ref_adata": ref_adata,
        "query_adata": query_adata,
        "true_query": true_query,
        "control": control,
        "replay": replay,
        "query_batch_values": query_batch_values,
        "train_extra": train_extra,
        "_fallback_split": _fallback_split,
    }


def task_b4_continual(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    seed=0,
    max_epochs=1000,
    n_latent=None,
    ewc_importance=1.0,
    control_frac=0.1,
    replay_frac=0.2,
    query_batch_values=None,
    batch_size=None,
    _setup=None,
):
    """B4: continual update vs plain surgery on a pseudo case/control batch split.

    Uses one batch as the query cohort and the other as reference (plumbing validation until a
    real case/control cytometry axis is available). Compares ``load_query_data`` vs
    ``load_query_data_with_replay`` on replay/reference latent drift and query label transfer.

    Pass ``_setup`` (from :func:`_b4_setup`) to reuse a pre-built reference model and replay
    buffer — used by :func:`task_b6_lambda_sweep` to avoid retraining across λ values.
    """
    from cytoanvi import CytoANVI

    if _setup is None:
        _setup = _b4_setup(
            adata,
            labels_key=labels_key,
            unlabeled_category=unlabeled_category,
            batch_key=batch_key,
            sample_key=sample_key,
            nan_layer=nan_layer,
            seed=seed,
            max_epochs=max_epochs,
            n_latent=n_latent,
            control_frac=control_frac,
            replay_frac=replay_frac,
            query_batch_values=query_batch_values,
            batch_size=batch_size,
        )

    ref_model = _setup["ref_model"]
    ref_adata = _setup["ref_adata"]
    query_adata = _setup["query_adata"]
    true_query = _setup["true_query"]
    control = _setup["control"]
    replay = _setup["replay"]
    train_extra = _setup["train_extra"]

    # Use .copy() so surgery calls do not mutate the shared query_adata in the setup dict
    # (load_query_data runs setup_anndata in-place; B6 reuses the same setup across λ).
    plain = CytoANVI.load_query_data(query_adata.copy(), ref_model)
    # Plain surgery uses automatic optimization → gradient_clip_val via Trainer.
    plain.train(
        max_epochs=min(50, max_epochs),
        gradient_clip_val=_GRAD_CLIP,
        plan_kwargs={"weight_decay": 0.0},
        **train_extra,
    )

    continual = CytoANVI.load_query_data_with_replay(
        query_adata.copy(),
        ref_model,
        replay_adata=replay,
        control_adata=control,
    )
    # Continual uses manual optimization → clip via plan_kwargs["gradient_clip_norm"].
    continual.train(
        max_epochs=min(50, max_epochs),
        plan_kwargs={
            "ewc_importance": ewc_importance,
            "weight_decay": 0.0,
            "gradient_clip_norm": _GRAD_CLIP,
        },
        **train_extra,
    )

    plain_pred = np.asarray(plain.predict())
    cont_pred = np.asarray(continual.predict())

    return {
        "task": "b4_continual",
        "seed": seed,
        "max_epochs": max_epochs,
        "ewc_importance": ewc_importance,
        "n_reference": int(ref_adata.n_obs),
        "n_query": int(query_adata.n_obs),
        "n_replay": int(replay.n_obs),
        "n_control": int(control.n_obs),
        "query_batch_values": list(_setup["query_batch_values"]),
        "_fallback_split": _setup["_fallback_split"],
        "note": "Pseudo case/control via batch split — validates plumbing, not biology.",
        "plain_surgery": {
            "replay_latent_drift": _replay_latent_drift(ref_model, plain, replay),
            "query_label_transfer": metrics.label_transfer_metrics(true_query, plain_pred),
        },
        "continual_update": {
            "replay_latent_drift": _replay_latent_drift(ref_model, continual, replay),
            "query_label_transfer": metrics.label_transfer_metrics(true_query, cont_pred),
        },
    }


def default_synthetic_hierarchy_edges():
    """Parent→children edges for synthetic_iid observed labels (label_1 .. label_4)."""
    return {
        "label_1": ["label_2", "label_3"],
        "label_2": ["label_4"],
        "label_3": [],
    }


def task_b8_hce_label_transfer(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    holdout_frac=0.2,
    seed=0,
    max_epochs=1000,
    n_latent=None,
    hierarchy_edges=None,
    batch_size=None,
):
    """B8: flat CE vs HCE when a user ontology matches observed model labels.

    Trains two CytoANVI models on the same stratified label holdout: default flat CE, then HCE via
    ``set_hierarchy`` before training. Also scores ``predict_hierarchical(leaf_only=True)`` on the
    HCE model.
    """
    if hierarchy_edges is None:
        hierarchy_edges = default_synthetic_hierarchy_edges()

    true = np.asarray(adata.obs[labels_key].astype(str))
    held = _holdout(true, unlabeled_category, holdout_frac, seed)

    work = adata.copy()
    masked = true.copy()
    masked[held] = unlabeled_category
    work.obs[labels_key] = masked

    flat_model, _ = train_cytoanvi(
        work,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    flat_pred = np.asarray(flat_model.predict())

    hce_model, _ = train_cytoanvi(
        work,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
        hierarchy_edges=hierarchy_edges,
    )
    hce_pred = np.asarray(hce_model.predict())
    hier_pred = np.asarray(hce_model.predict_hierarchical(leaf_only=True))

    flat_metrics = metrics.label_transfer_metrics(true[held], flat_pred[held])
    hce_metrics = metrics.label_transfer_metrics(true[held], hce_pred[held])

    # predict_hierarchical(leaf_only=True) can never emit internal-node labels, so evaluate
    # hier_pred only on held cells whose true label is a leaf for a fair comparison.
    internal_labels = {p for p, ch in hierarchy_edges.items() if ch}
    leaf_held = held & ~np.isin(true, list(internal_labels))
    hier_metrics = metrics.label_transfer_metrics(true[leaf_held], hier_pred[leaf_held])
    flat_leaf_metrics = metrics.label_transfer_metrics(true[leaf_held], flat_pred[leaf_held])

    return {
        "task": "b8_hce_label_transfer",
        "seed": seed,
        "max_epochs": max_epochs,
        "holdout_frac": holdout_frac,
        "n_held": int(held.sum()),
        "n_leaf_held": int(leaf_held.sum()),
        "hierarchy_edges": hierarchy_edges,
        "flat_ce": flat_metrics,
        "hce_flat_predict": hce_metrics,
        "hce_hierarchical_predict": hier_metrics,
        "delta_hce_vs_flat_macro_f1": float(hce_metrics["macro_f1"] - flat_metrics["macro_f1"]),
        # delta uses leaf-only subset on both sides: internal-label cells can never be correct
        # under leaf_only=True, so including them would unfairly penalise hierarchical decoding.
        "delta_hierarchical_vs_flat_macro_f1": float(
            hier_metrics["macro_f1"] - flat_leaf_metrics["macro_f1"]
        ),
    }


def task_b6_lambda_sweep(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    seed=0,
    max_epochs=1000,
    n_latent=None,
    lambdas=None,
    query_batch_values=None,
    batch_size=None,
    control_frac=0.1,
    replay_frac=0.2,
):
    """B6: sweep ``ewc_importance`` (λ) for continual update; report replay drift vs query F1.

    The reference model and replay buffer are built once (via :func:`_b4_setup`) and shared
    across all λ values to avoid redundant reference retraining (~N× savings for N lambdas).
    """
    if lambdas is None:
        lambdas = [0.0, 1.0, 10.0, 100.0, 1000.0]
    setup = _b4_setup(
        adata,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        seed=seed,
        max_epochs=max_epochs,
        n_latent=n_latent,
        control_frac=control_frac,
        replay_frac=replay_frac,
        query_batch_values=query_batch_values,
        batch_size=batch_size,
    )
    per_lambda = {}
    for lam in lambdas:
        per_lambda[str(lam)] = task_b4_continual(
            adata,
            labels_key=labels_key,
            unlabeled_category=unlabeled_category,
            batch_key=batch_key,
            sample_key=sample_key,
            nan_layer=nan_layer,
            seed=seed,
            max_epochs=max_epochs,
            n_latent=n_latent,
            ewc_importance=lam,
            query_batch_values=setup["query_batch_values"],
            batch_size=batch_size,
            _setup=setup,
        )["continual_update"]
    drifts = np.asarray([v["replay_latent_drift"] for v in per_lambda.values()], dtype=float)
    f1s = [v["query_label_transfer"]["macro_f1"] for v in per_lambda.values()]
    out = {
        "task": "b6_lambda_sweep",
        "seed": seed,
        "max_epochs": max_epochs,
        "lambdas": list(lambdas),
        "_fallback_split": setup["_fallback_split"],
        "per_lambda": per_lambda,
        "note": "Reports the full λ table; recommend only when replay drift has a unique minimum.",
    }
    finite = np.isfinite(drifts)
    if not finite.any():
        out["recommendation_status"] = "no_recommendation"
        out["recommendation_reason"] = "Replay latent drift is non-finite for all λ values."
        return out
    min_drift = float(np.min(drifts[finite]))
    best_indices = np.flatnonzero(np.isclose(drifts, min_drift, rtol=1e-6, atol=1e-8))
    if len(best_indices) != 1:
        out["recommendation_status"] = "no_recommendation"
        out["recommendation_reason"] = (
            "Replay latent drift is tied across λ values; no recommended_lambda emitted."
        )
        return out
    best_idx = int(best_indices[0])
    out.update(
        {
            "recommendation_status": "recommended",
            "recommended_lambda": float(lambdas[best_idx]),
            "recommended_replay_latent_drift": float(drifts[best_idx]),
            "recommended_query_macro_f1": float(f1s[best_idx]),
        }
    )
    return out


MAPQC_SAMPLE_KEY = "mapqc_sample"
MAPQC_STATUS_KEY = "mapqc_status"


def _assign_mapqc_pseudo_samples(adata, batch_key: str, seed: int):
    """Split one batch → reference, the other → query; assign ≥3 ref / ≥2 query sample IDs."""
    adata = adata.copy()
    batches = sorted(adata.obs[batch_key].astype(str).unique())
    if len(batches) < 2:
        raise ValueError("B9 requires at least two batches in adata.")
    ref_batch = batches[0]
    is_ref = adata.obs[batch_key].astype(str) == ref_batch
    rng = np.random.default_rng(seed)
    ref_samples = [f"ref_s{i}" for i in range(4)]
    query_samples = [f"query_s{i}" for i in range(2)]

    ref_idx = np.where(is_ref.to_numpy())[0]
    query_idx = np.where((~is_ref).to_numpy())[0]

    # Build full object-dtype columns and assign whole-column to avoid pandas>=2.0
    # copy-on-write silent no-ops from chained iloc assignments.
    # Sample IDs use deterministic round-robin (cell order within a batch is stable);
    # rng is reserved exclusively for the subsequent randomised status assignment.
    sample_col = np.empty(adata.n_obs, dtype=object)
    sample_col[ref_idx] = np.take(ref_samples, np.arange(len(ref_idx)) % len(ref_samples))
    sample_col[query_idx] = np.take(
        query_samples, np.arange(len(query_idx)) % len(query_samples)
    )
    adata.obs[MAPQC_SAMPLE_KEY] = sample_col

    # Preserve rng.choice call order: status is drawn after sample assignment.
    status_choices = rng.choice(["control", "case"], size=len(query_idx), p=[0.5, 0.5])
    status_col = np.empty(adata.n_obs, dtype=object)
    status_col[ref_idx] = np.nan
    status_col[query_idx] = status_choices
    adata.obs[MAPQC_STATUS_KEY] = status_col
    return adata, is_ref.to_numpy()


def task_b9_mapqc(
    adata,
    labels_key="labels",
    unlabeled_category="Unknown",
    batch_key="batch",
    sample_key=None,
    nan_layer=None,
    seed=0,
    max_epochs=1000,
    n_latent=None,
    n_nhoods=3,
    k_min=5,
    k_max=15,
    run_mapqc=True,
    batch_size=None,
):
    """B9: mapQC on CytoANVI latents after query surgery (pseudo batch ref/query split).

    Set ``run_mapqc=False`` for plumbing-only checks (joint latent build) on small synthetic data.
    """
    from cytoanvi import CytoANVI, mapping_qc

    if run_mapqc:
        try:
            mapping_qc._require_mapqc()
        except ImportError as err:
            return {
                "task": "b9_mapqc",
                "seed": seed,
                "max_epochs": max_epochs,
                "status": "blocked",
                "blocked_reason": str(err),
                "run_mapqc": run_mapqc,
            }

    work, is_ref = _assign_mapqc_pseudo_samples(adata, batch_key=batch_key, seed=seed)
    ref_adata = work[is_ref].copy()
    query_adata = work[~is_ref].copy()
    true_query = np.asarray(work.obs[labels_key].astype(str))[~is_ref]

    ref_model, _ = train_cytoanvi(
        ref_adata,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )

    query_train = query_adata.copy()
    query_train.obs[labels_key] = unlabeled_category
    query_model = CytoANVI.load_query_data(query_train, ref_model)
    query_train_kw: dict = {"plan_kwargs": {"weight_decay": 0.0}}
    if batch_size is not None:
        query_train_kw["batch_size"] = batch_size
    query_model.train(max_epochs=min(50, max_epochs), **query_train_kw)

    joint = mapping_qc.build_mapqc_anndata(
        query_model,
        ref_adata,
        query_adata,
        sample_key=MAPQC_SAMPLE_KEY,
    )
    out = {
        "task": "b9_mapqc",
        "seed": seed,
        "max_epochs": max_epochs,
        "n_reference": int(ref_adata.n_obs),
        "n_query": int(query_adata.n_obs),
        "joint_n_obs": int(joint.n_obs),
        "emb_key": mapping_qc.DEFAULT_EMB_KEY,
        "run_mapqc": run_mapqc,
        "query_label_transfer": metrics.label_transfer_metrics(
            true_query, np.asarray(query_model.predict())
        ),
    }
    if not run_mapqc:
        out["status"] = "plumbing_only"
        out["note"] = (
            "Joint latent AnnData built; mapQC skipped "
            "(use run_mapqc=True on real case/control data)."
        )
        return out

    mapping_qc.run_mapqc_on_joint(
        joint,
        sample_key=MAPQC_SAMPLE_KEY,
        n_nhoods=n_nhoods,
        k_min=k_min,
        k_max=k_max,
        grouping_key=labels_key,
        seed=seed,
        verbose=False,
        min_n_cells=5,
    )
    out.update(
        {
            "status": "mapqc_complete",
            "n_nhoods": n_nhoods,
            "k_min": k_min,
            "k_max": k_max,
            "query_control_mapqc": mapping_qc.query_control_mapqc_rate(
                joint,
                control_value="control",
                case_control_key=MAPQC_STATUS_KEY,
            ),
            "note": "Requires pip install scvi-tools[cytoanvi-mapping-qc].",
        }
    )
    return out
