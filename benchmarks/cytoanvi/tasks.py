"""Benchmark tasks B1, B2, B3, B4, B5, B6.

Each task returns a plain dict of metrics; run.py serializes them to JSON.
"""

from __future__ import annotations

import numpy as np

from benchmarks.common.scib import LATENT_OBSM, run_scib_benchmark
from benchmarks.common.training import NAN_LAYER, SCALED_LAYER, latent_obsm, train_cytoanvi

from . import metrics
from .baselines import cytovi_latent_and_knn


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
    )
    knn_full = masked.copy()
    knn_full[unlab_mask] = knn_pred_unlab

    return {
        "task": "b1_label_transfer",
        "seed": seed,
        "max_epochs": max_epochs,
        "holdout_frac": holdout_frac,
        "n_held": int(held.sum()),
        "cytoanvi": metrics.label_transfer_metrics(true[held], cytoanvi_pred[held]),
        "cytovi_knn": metrics.label_transfer_metrics(true[held], knn_full[held]),
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
):
    """B3: panel-1 reference -> panel-2 query via panel-aware prep + scArches surgery."""
    from scvi.external import cytovi

    merged = cytovi.merge_batches([p1.copy(), p2.copy()], batch_key="panel_batch")
    labels = np.asarray(merged.obs[labels_key].astype(str))
    is_p2 = np.isin(merged.obs_names, p2.obs_names)
    held = _holdout(labels, unlabeled_category, holdout_frac, seed) & ~is_p2
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
    )
    knn_full = masked.copy()
    knn_full[unlab_mask] = knn_pred_unlab
    out["p2_concordance_vs_knn"] = metrics.concordance(pred[is_p2], knn_full[is_p2])
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
    )
    unc = model.get_uncertainty()
    return {
        "task": "b5_novelty",
        "seed": seed,
        "max_epochs": max_epochs,
        "holdout_type": holdout_type,
        **metrics.novelty_auroc(unc, is_novel),
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
        )
    aurocs = [v["auroc"] for v in per_type.values() if not np.isnan(v["auroc"])]
    return {
        "task": "b5_holdout_sweep",
        "seed": seed,
        "max_epochs": max_epochs,
        "per_type": per_type,
        "best_auroc": float(max(aurocs)) if aurocs else float("nan"),
        "mean_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
    }


def _split_reference_query(
    adata,
    *,
    batch_key: str,
    query_batch_values: list[str],
    seed: int = 0,
):
    """Pseudo case/control split: query batches vs reference batches (plumbing validation)."""
    batch = np.asarray(adata.obs[batch_key].astype(str))
    is_query = np.isin(batch, query_batch_values)
    ref = adata[~is_query].copy()
    query = adata[is_query].copy()
    if ref.n_obs < 64 or query.n_obs < 64:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(adata.n_obs)
        cut = int(0.7 * adata.n_obs)
        ref = adata[perm[:cut]].copy()
        query = adata[perm[cut:]].copy()
    return ref, query


def _control_latent_drift(ref_model, updated_model, control_adata):
    """Mean per-cell L2 drift of control latents between reference and updated models."""
    z_ref = ref_model.get_latent_representation(control_adata)
    z_new = updated_model.get_latent_representation(control_adata)
    return float(np.linalg.norm(z_ref - z_new, axis=1).mean())


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
):
    """B4: continual update vs plain surgery on a pseudo case/control batch split.

    Uses one batch as the query cohort and the other as reference (plumbing validation until a
    real case/control cytometry axis is available). Compares ``load_query_data`` vs
    ``load_query_data_with_replay`` on control-latent drift and query label transfer.
    """
    from scvi.external import CytoANVI

    if query_batch_values is None:
        batches = sorted(adata.obs[batch_key].astype(str).unique())
        query_batch_values = [batches[-1]]

    labels = np.asarray(adata.obs[labels_key].astype(str))
    ref_adata, query_adata = _split_reference_query(
        adata, batch_key=batch_key, query_batch_values=query_batch_values, seed=seed
    )
    true_query = labels[np.isin(adata.obs_names, query_adata.obs_names)]
    query_adata = query_adata.copy()
    query_adata.obs[labels_key] = unlabeled_category

    ref_model, _ = train_cytoanvi(
        ref_adata,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
    )

    n_ctrl = max(16, int(control_frac * query_adata.n_obs))
    n_replay = max(16, int(replay_frac * ref_adata.n_obs))
    control = query_adata[:n_ctrl].copy()
    replay = CytoANVI.select_replay_by_uncertainty(ref_model, ref_adata, fraction=replay_frac)

    plain = CytoANVI.load_query_data(query_adata, ref_model)
    plain.train(max_epochs=min(50, max_epochs), plan_kwargs={"weight_decay": 0.0})

    continual = CytoANVI.load_query_data_with_replay(
        query_adata.copy(),
        ref_model,
        replay_adata=replay,
        control_adata=control,
    )
    continual.train(
        max_epochs=min(50, max_epochs),
        plan_kwargs={"ewc_importance": ewc_importance, "weight_decay": 0.0},
    )

    true_query = labels[np.isin(adata.obs_names, query_adata.obs_names)]
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
        "query_batch_values": list(query_batch_values),
        "note": "Pseudo case/control via batch split — validates plumbing, not biology.",
        "plain_surgery": {
            "control_latent_drift": _control_latent_drift(ref_model, plain, control),
            "query_label_transfer": metrics.label_transfer_metrics(true_query, plain_pred),
        },
        "continual_update": {
            "control_latent_drift": _control_latent_drift(ref_model, continual, control),
            "query_label_transfer": metrics.label_transfer_metrics(true_query, cont_pred),
        },
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
):
    """B6: sweep ``ewc_importance`` (λ) for continual update; report control drift vs query F1."""
    if lambdas is None:
        lambdas = [0.0, 1.0, 10.0, 100.0, 1000.0]
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
            query_batch_values=query_batch_values,
        )["continual_update"]
    drifts = [v["control_latent_drift"] for v in per_lambda.values()]
    f1s = [v["query_label_transfer"]["macro_f1"] for v in per_lambda.values()]
    best_idx = int(np.argmin(drifts))
    return {
        "task": "b6_lambda_sweep",
        "seed": seed,
        "max_epochs": max_epochs,
        "lambdas": list(lambdas),
        "per_lambda": per_lambda,
        "recommended_lambda": float(lambdas[best_idx]),
        "recommended_control_drift": float(drifts[best_idx]),
        "recommended_query_macro_f1": float(f1s[best_idx]),
        "note": "Heuristic: λ with lowest control-latent drift; retune on real case/control data.",
    }
