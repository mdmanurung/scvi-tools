"""Benchmark tasks B1, B2, B3, B4, B5, B6.

Each task returns a plain dict of metrics; run.py serializes them to JSON.
"""

from __future__ import annotations

import gc
import time
import warnings

import numpy as np
import torch as _torch

from benchmarks.common.scib import LATENT_OBSM, run_scib_benchmark
from benchmarks.common.training import (
    _GRAD_CLIP,
    NAN_LAYER,
    annbatch_train_kwargs,
    latent_obsm,
    resolve_nan_layer,
    train_cytoanvi,
)

from . import metrics
from .baselines import (
    cytovi_latent_and_knn,
    cytovi_novelty_score,
    flowsom_knn,
    harmony_latent_and_knn,
    knn_distance_novelty,
    rapids_graph_knn,
    raw_marker_knn,
    xgboost_classifier,
)

_OPTIONAL_BASELINE_ERRORS = (ImportError, ValueError, KeyError)
_HARMONY_BASELINE_ERRORS = (*_OPTIONAL_BASELINE_ERRORS, IndexError)
_B1_OPTIONAL_BASELINES = ("harmony_knn", "xgboost", "rapids_graph", "flowsom")
_B1_FAST_OPTIONAL_BASELINES = ("harmony_knn", "xgboost")
_B1_BASELINE_ALIASES = {
    "harmony": "harmony_knn",
    "harmony_knn": "harmony_knn",
    "xgboost": "xgboost",
    "rapids": "rapids_graph",
    "rapids_graph": "rapids_graph",
    "rapids-graph": "rapids_graph",
    "flowsom": "flowsom",
}


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


def _label_transfer_metrics_from_unlabeled(true, masked, held, pred_unlab, unlab_mask):
    full = masked.copy()
    full[unlab_mask] = pred_unlab
    return metrics.label_transfer_metrics(true[held], full[held])


def _optional_label_transfer_metrics(
    baseline_func,
    work,
    labels_key,
    unlabeled_category,
    *,
    true,
    masked,
    held,
    seed,
    error_types=_OPTIONAL_BASELINE_ERRORS,
    **baseline_kwargs,
):
    try:
        pred_unlab, _, unlab_mask = baseline_func(
            work,
            labels_key,
            unlabeled_category,
            seed=seed,
            **baseline_kwargs,
        )
        return _label_transfer_metrics_from_unlabeled(true, masked, held, pred_unlab, unlab_mask)
    except error_types as e:
        # Optional baselines should be recorded in JSON without aborting B1.
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


def _resolve_b1_optional_baselines(value="all") -> set[str]:
    if value is None or value == "all":
        return set(_B1_OPTIONAL_BASELINES)
    if value == "fast":
        return set(_B1_FAST_OPTIONAL_BASELINES)
    if value == "none":
        return set()
    if isinstance(value, str):
        requested = [v.strip() for v in value.split(",") if v.strip()]
    else:
        requested = list(value)
    selected = set()
    invalid = []
    for name in requested:
        key = str(name).strip().lower().replace("-", "_")
        resolved = _B1_BASELINE_ALIASES.get(key)
        if resolved is None:
            invalid.append(str(name))
        else:
            selected.add(resolved)
    if invalid:
        valid = ", ".join(("all", "fast", "none", *_B1_BASELINE_ALIASES))
        raise ValueError(f"Unknown B1 optional baseline(s): {invalid}. Valid values: {valid}.")
    return selected


def _skipped_b1_baseline():
    return {"skipped": "not requested"}


def _training_config(cytoanvi_training_config=None, **overrides) -> dict:
    """Merge benchmark-level CytoANVI training config with task-specific overrides."""
    config = dict(cytoanvi_training_config or {})
    for key, value in overrides.items():
        if isinstance(value, bool):
            # bool=False means "use base config default"; only True explicitly enables a flag.
            if value:
                config[key] = value
        elif value is not None:
            config[key] = value
    return config


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
    annbatch_config=None,
    cytoanvi_training_config=None,
    b1_baselines="all",
    accelerator: str = "auto",
    devices: int | list[int] | str = "auto",
):
    """B1: CytoANVI classifier vs CytoVI k-NN at transferring labels to held-out cells."""
    true = np.asarray(adata.obs[labels_key].astype(str))
    held = _holdout(true, unlabeled_category, holdout_frac, seed)

    work = adata.copy()
    masked = true.copy()
    masked[held] = unlabeled_category
    work.obs[labels_key] = masked
    training_config = _training_config(
        cytoanvi_training_config,
        reduce_lr_on_plateau=reduce_lr_on_plateau,
    )
    optional_baselines = _resolve_b1_optional_baselines(b1_baselines)

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
        batch_size=batch_size,
        annbatch_config=annbatch_config,
        accelerator=accelerator,
        devices=devices,
        **training_config,
    )
    cytoanvi_pred = np.asarray(model.predict())
    train_mask = masked != unlabeled_category
    cytoanvi_diagnostics = metrics.label_transfer_diagnostics(
        masked[train_mask],
        true[held],
        cytoanvi_pred[held],
    )

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
        accelerator=accelerator,
        devices=devices,
    )
    knn_result = _label_transfer_metrics_from_unlabeled(
        true, masked, held, knn_pred_unlab, unlab_mask
    )

    raw_pred_unlab, _, raw_unlab_mask = raw_marker_knn(
        work,
        labels_key,
        unlabeled_category,
    )
    raw_result = _label_transfer_metrics_from_unlabeled(
        true, masked, held, raw_pred_unlab, raw_unlab_mask
    )
    harmony_result = (
        _optional_label_transfer_metrics(
            harmony_latent_and_knn,
            work,
            labels_key,
            unlabeled_category,
            true=true,
            masked=masked,
            held=held,
            seed=seed,
            error_types=_HARMONY_BASELINE_ERRORS,
            batch_key=batch_key,
        )
        if "harmony_knn" in optional_baselines
        else _skipped_b1_baseline()
    )
    xgboost_result = (
        _optional_label_transfer_metrics(
            xgboost_classifier,
            work,
            labels_key,
            unlabeled_category,
            true=true,
            masked=masked,
            held=held,
            seed=seed,
        )
        if "xgboost" in optional_baselines
        else _skipped_b1_baseline()
    )
    rapids_graph_result = (
        _optional_label_transfer_metrics(
            rapids_graph_knn,
            work,
            labels_key,
            unlabeled_category,
            true=true,
            masked=masked,
            held=held,
            seed=seed,
        )
        if "rapids_graph" in optional_baselines
        else _skipped_b1_baseline()
    )
    flowsom_result = (
        _optional_label_transfer_metrics(
            flowsom_knn,
            work,
            labels_key,
            unlabeled_category,
            true=true,
            masked=masked,
            held=held,
            seed=seed,
        )
        if "flowsom" in optional_baselines
        else _skipped_b1_baseline()
    )

    return {
        "task": "b1_label_transfer",
        "seed": seed,
        "max_epochs": max_epochs,
        "holdout_frac": holdout_frac,
        "n_held": int(held.sum()),
        "cytoanvi": metrics.label_transfer_metrics(true[held], cytoanvi_pred[held]),
        "cytoanvi_diagnostics": cytoanvi_diagnostics,
        "cytovi_knn": knn_result,
        "raw_marker_knn": raw_result,
        "harmony_knn": harmony_result,
        "xgboost": xgboost_result,
        "rapids_graph": rapids_graph_result,
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
    annbatch_config=None,
    cytoanvi_training_config=None,
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
        annbatch_config=annbatch_config,
        **_training_config(cytoanvi_training_config),
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
    annbatch_config=None,
    cytoanvi_training_config=None,
):
    """B3: panel-1 reference -> panel-2 query via panel-aware prep + scArches surgery."""
    from scvi.external import cytovi

    merged = cytovi.merge_batches([p1.copy(), p2.copy()], batch_key="panel_batch")
    nan_layer = resolve_nan_layer(merged, NAN_LAYER)
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
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
        annbatch_config=annbatch_config,
        **_training_config(cytoanvi_training_config),
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
        nan_layer=nan_layer,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    knn_full = masked.copy()
    knn_full[unlab_mask] = knn_pred_unlab
    # IMPORTANT: p2_inter_method_agreement_vs_knn is the fraction of panel-2 cells
    # where CytoANVI and CytoVI-kNN assign the same label.  This measures
    # INTER-METHOD CONCORDANCE between two predictors that share the CytoVI encoder
    # backbone — it is NOT ground-truth accuracy.  There are no independent
    # manually-gated panel-2 labels, so cross-panel correctness CANNOT be validated
    # from this metric alone.  Do not report this as "accuracy".
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
    annbatch_config=None,
    cytoanvi_training_config=None,
    b5_mode="transductive",
    specificity=0.95,
    compute_logit=True,
    cytovi_baseline=False,
    cytovi_n_neighbors=15,
):
    """B5: does get_uncertainty flag a cell type held out of the reference entirely?

    ``compute_logit`` controls whether the second (``mode="logit"``) TTA uncertainty pass runs;
    the headline metric uses the ``latent`` pass, so skip logit (``compute_logit=False``) to
    roughly halve the per-holdout evaluation cost. ``cytovi_baseline=True`` (inductive mode) also
    fits an unsupervised CytoVI novelty baseline (latent kNN-distance OOD score) on the seen cells
    reports its AUROC alongside CytoANVI's, turning B5 into a comparative benchmark.
    """
    if b5_mode not in {"transductive", "inductive"}:
        raise ValueError("b5_mode must be one of {'transductive', 'inductive'}.")
    labels = np.asarray(adata.obs[labels_key].astype(str))
    types = sorted(t for t in set(labels) if t != unlabeled_category)
    if holdout_type is None:
        holdout_type = types[0]
    is_novel = labels == holdout_type

    def _finish(model, unc_latent, unc_logit, eval_is_novel, *, extra=None, mode_note=""):
        latent_result = metrics.novelty_auroc(unc_latent, eval_is_novel)
        logit_result = metrics.novelty_auroc(unc_logit, eval_is_novel)
        del model
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        out = {
            "task": "b5_novelty",
            "seed": seed,
            "max_epochs": max_epochs,
            "holdout_type": holdout_type,
            "b5_evaluation_mode": mode_note,
            "latent": latent_result,
            "logit": logit_result,
            **latent_result,  # top-level for backward compat
        }
        if extra:
            out.update(extra)
        return out

    if b5_mode == "inductive":
        seen_idx = np.flatnonzero((~is_novel) & (labels != unlabeled_category))
        novel_idx = np.flatnonzero(is_novel)
        if len(novel_idx) == 0:
            raise ValueError(f"holdout_type {holdout_type!r} has no cells.")
        rng = np.random.default_rng(seed)
        train_seen_parts = []
        calib_seen_parts = []
        for lab in sorted(set(labels[seen_idx])):
            lab_idx = seen_idx[labels[seen_idx] == lab].copy()
            rng.shuffle(lab_idx)
            if len(lab_idx) < 5:
                raise ValueError(
                    f"B5 inductive mode needs at least 5 seen cells per label; "
                    f"label {lab!r} has {len(lab_idx)}."
                )
            cut = max(1, int(0.8 * len(lab_idx)))
            cut = min(cut, len(lab_idx) - 1)
            train_seen_parts.append(lab_idx[:cut])
            calib_seen_parts.append(lab_idx[cut:])
        train_seen = np.concatenate(train_seen_parts)
        calib_seen = np.concatenate(calib_seen_parts)
        eval_idx = np.concatenate([calib_seen, novel_idx])

        def _timing(phase, since):
            # Per-phase wall-clock breakdown, printed to the job log so the B5 bottleneck
            # (data prep vs training vs TTA vs baseline) is observable per held-out type.
            secs = time.perf_counter() - since
            print(f"[B5-TIMING] holdout={holdout_type} phase={phase} secs={secs:.1f}", flush=True)
            return time.perf_counter()

        _t = time.perf_counter()
        work = adata[train_seen].copy()
        eval_adata = adata[eval_idx].copy()
        eval_labels = labels[eval_idx]
        eval_is_novel = eval_labels == holdout_type
        eval_adata.obs[labels_key] = unlabeled_category
        _t = _timing("data_prep", _t)
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
            annbatch_config=annbatch_config,
            **_training_config(cytoanvi_training_config),
        )
        _t = _timing("cytoanvi_train", _t)
        calib_mask = ~eval_is_novel
        unc_latent = model.get_uncertainty(eval_adata, mode="latent", batch_size=batch_size)
        _t = _timing("latent_tta", _t)
        latent_extra = metrics.precision_at_specificity(
            unc_latent,
            eval_is_novel,
            specificity=specificity,
            uncertainty_ref=unc_latent[calib_mask],
        )
        latent_result = metrics.novelty_auroc(unc_latent, eval_is_novel)
        latent_result.update(latent_extra)

        # Secondary logit-space uncertainty pass — a second full TTA sweep, so it is skipped when
        # compute_logit is False to roughly halve the per-holdout evaluation cost.
        logit_result = None
        if compute_logit:
            unc_logit = model.get_uncertainty(eval_adata, mode="logit", batch_size=batch_size)
            logit_result = metrics.novelty_auroc(unc_logit, eval_is_novel)
            logit_result.update(
                metrics.precision_at_specificity(
                    unc_logit,
                    eval_is_novel,
                    specificity=specificity,
                    uncertainty_ref=unc_logit[calib_mask],
                )
            )
            _t = _timing("logit_tta", _t)

        # CytoANVI-latent kNN-distance OOD diagnostic: same procedure as cytovi_novelty_score
        # but using CytoANVI's own latent space.  Runs when cytovi_baseline=True so the two
        # scores are always produced together.  The comparison CytoANVI-kNN vs CytoVI-kNN
        # isolates "bad TTA method" (if CytoANVI-kNN ≈ CytoVI-kNN) from "bad latent space"
        # (if CytoANVI-kNN also fails).  Must be extracted before del model.
        cytoanvi_knn_result = None
        if cytovi_baseline:
            try:
                z_ref_ca = np.asarray(model.get_latent_representation(work))
                z_eval_ca = np.asarray(model.get_latent_representation(eval_adata))
                cytoanvi_knn_score = knn_distance_novelty(z_ref_ca, z_eval_ca, cytovi_n_neighbors)
                cytoanvi_knn_result = metrics.novelty_auroc(cytoanvi_knn_score, eval_is_novel)
                _t = _timing("cytoanvi_knn_baseline", _t)
            except Exception as exc:  # noqa: BLE001 - diagnostic must not abort the main result
                cytoanvi_knn_result = {"auroc": float("nan"), "error": repr(exc)}

        del model
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        # Unsupervised CytoVI novelty baseline on the SAME seen cells (fair holdout): reports an
        # independent AUROC so B5 answers "is CytoANVI's uncertainty better than a plain CytoVI
        # OOD detector at flagging the novel population?" rather than a bare, baseline-free number.
        cytovi_result = None
        if cytovi_baseline:
            try:
                cytovi_score = cytovi_novelty_score(
                    work,
                    eval_adata,
                    batch_key=batch_key,
                    sample_key=sample_key,
                    nan_layer=nan_layer,
                    n_neighbors=cytovi_n_neighbors,
                    max_epochs=max_epochs,
                    n_latent=n_latent,
                    batch_size=batch_size,
                )
                cytovi_result = metrics.novelty_auroc(cytovi_score, eval_is_novel)
                gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                _t = _timing("cytovi_baseline", _t)
            except Exception as exc:  # noqa: BLE001 - baseline must never abort the CytoANVI result
                cytovi_result = {"auroc": float("nan"), "error": repr(exc)}

        return {
            "task": "b5_novelty",
            "seed": seed,
            "max_epochs": max_epochs,
            "holdout_type": holdout_type,
            "b5_evaluation_mode": "inductive_calibrated",
            "specificity": specificity,
            "n_train_seen": int(len(train_seen)),
            "n_calibration_seen": int(len(calib_seen)),
            "n_eval": int(len(eval_idx)),
            "latent": latent_result,
            "logit": logit_result,
            "cytovi_baseline": cytovi_result,
            "cytoanvi_knn_baseline": cytoanvi_knn_result,
            **metrics.novelty_auroc(unc_latent, eval_is_novel),
        }

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
        annbatch_config=annbatch_config,
        **_training_config(cytoanvi_training_config),
    )
    unc_latent = model.get_uncertainty(mode="latent")
    unc_logit = model.get_uncertainty(mode="logit")
    return _finish(
        model,
        unc_latent,
        unc_logit,
        is_novel,
        mode_note="calibration_transductive",
    )


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
    annbatch_config=None,
    cytoanvi_training_config=None,
    b5_mode="transductive",
    specificity=0.95,
    max_holdout_types=None,
    checkpoint_path=None,
    compute_logit=True,
    cytovi_baseline=False,
    cytovi_n_neighbors=15,
):
    """B5 sweep: novelty-detection AUROC for each cell type held out in turn.

    PRIMARY HEADLINE METRIC: ``mean_auroc`` — the unweighted mean AUROC across all held-out
    cell types.  On Roider-e1000 this is ~0.46 (near chance), meaning the model does not
    reliably flag novel cells across the full label set.

    ``best_auroc`` — the MAX over cell types — is also retained for secondary analysis but
    is a cherry-picked single-type result and MUST NOT be presented as the summary statistic
    for novelty detection.  Use ``mean_auroc`` as the headline.

    Scaling / comparison options:

    - ``max_holdout_types`` — if set, sweep only the N **most populous** cell types instead of all
      of them. On roider-full the label set is ~47 Leiden clusters and a full sweep (one training
      per type) is infeasible in a 48h job; limiting to the major populations (e.g. 11) is both
      tractable and more interpretable than holding out tiny arbitrary sub-clusters.
    - ``checkpoint_path`` — if set, the accumulated per-type results are written after **each**
      type, so a job killed mid-sweep still yields partial results (the sweep otherwise only
      returns at the end).
    - ``compute_logit`` / ``cytovi_baseline`` — forwarded to :func:`task_b5_novelty`; the latter
      adds an unsupervised CytoVI OOD baseline AUROC per type (inductive mode) for comparison.
    """
    labels = np.asarray(adata.obs[labels_key].astype(str))
    types = sorted(t for t in set(labels) if t != unlabeled_category)
    if max_holdout_types is not None and len(types) > max_holdout_types:
        freq = {t: int((labels == t).sum()) for t in types}
        types = sorted(sorted(types, key=lambda t: freq[t], reverse=True)[:max_holdout_types])
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
            annbatch_config=annbatch_config,
            cytoanvi_training_config=cytoanvi_training_config,
            b5_mode=b5_mode,
            specificity=specificity,
            compute_logit=compute_logit,
            cytovi_baseline=cytovi_baseline,
            cytovi_n_neighbors=cytovi_n_neighbors,
        )
        if checkpoint_path is not None:
            from benchmarks.common.seeds import save_json

            save_json(
                checkpoint_path,
                {
                    "task": "b5_holdout_sweep_partial",
                    "seed": seed,
                    "swept_types": list(types),
                    "completed_types": list(per_type),
                    "per_type": per_type,
                },
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
        sig_types = [ht for ht, q in zip(valid_types, fdr_q, strict=True) if q < 0.05]
        fdr_fields = {
            "auroc_pvalues": {ht: float(p) for ht, p in zip(valid_types, p_vals, strict=True)},
            "auroc_fdr_q": {ht: float(q) for ht, q in zip(valid_types, fdr_q, strict=True)},
            "n_fdr_significant": len(sig_types),
            "mean_auroc_fdr_sig": (
                float(np.mean([per_type[ht]["auroc"] for ht in sig_types]))
                if sig_types
                else float("nan")
            ),
        }

    calibration_note = (
        "inductive - uncertainty thresholds calibrated on held-out seen cells"
        if b5_mode == "inductive"
        else "transductive - uncertainty thresholds not cross-validated"
    )

    return {
        "task": "b5_holdout_sweep",
        "seed": seed,
        "max_epochs": max_epochs,
        # Each per-type novelty score is transductive: uncertainty is measured on
        # the full merged object (ref + novel), so the model has never seen the held-out
        # type during training but the uncertainty threshold is not cross-validated.
        # Consumers comparing across seeds should treat absolute AUROC values as
        # optimistic; only relative rankings across cell types are robust.
        "b5_mode": b5_mode,
        "calibration_note": calibration_note,
        # REPORTING NOTE: mean_auroc is the PRIMARY headline metric — it is the
        # unweighted mean AUROC across ALL held-out cell types.  best_auroc is the
        # MAX over types (a single cherry-picked type) and must NOT be presented as
        # the summary statistic for novelty detection across the full label set.
        "reporting_note": (
            "PRIMARY metric: mean_auroc (mean over all cell types). "
            "best_auroc is max over types (single cherry-picked type; NOT a summary statistic)."
        ),
        "per_type": per_type,
        "mean_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "n_fdr_significant": fdr_fields.get("n_fdr_significant", 0),
        "mean_auroc_fdr_sig": fdr_fields.get("mean_auroc_fdr_sig", float("nan")),
        # best_auroc: MAX over cell types — retained for secondary analysis only.
        # This is the AUROC of the single best-detected novel type, not the mean.
        "best_auroc": float(max(aurocs)) if aurocs else float("nan"),
        "n_holdout_types": len(types),
        # CytoVI OOD baseline: mean AUROC across types (when cytovi_baseline=True). Compare
        # against mean_auroc above — CytoANVI's uncertainty is only useful if it beats this.
        "cytovi_mean_auroc": _mean_baseline_auroc(per_type, "cytovi_baseline"),
        # CytoANVI-latent kNN-distance OOD diagnostic: same kNN-distance procedure as
        # cytovi_mean_auroc but using CytoANVI's own latent.  Compare against cytovi_mean_auroc
        # to determine whether the B5 negative result is due to a bad TTA method or a bad latent.
        "cytoanvi_knn_mean_auroc": _mean_baseline_auroc(per_type, "cytoanvi_knn_baseline"),
        **fdr_fields,
    }


def _mean_baseline_auroc(per_type: dict, key: str) -> float:
    """Mean AUROC across held-out types for a given per-type baseline key (NaN if not run)."""
    vals = [
        v[key]["auroc"]
        for v in per_type.values()
        if isinstance(v.get(key), dict)
        and not np.isnan(v[key].get("auroc", float("nan")))
    ]
    return float(np.mean(vals)) if vals else float("nan")


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


def _split_reference_query_by_case_control(
    adata,
    *,
    case_control_key: str,
    control_values: list[str],
    case_values: list[str],
):
    """Real case/control split for continual-update evaluation."""
    if case_control_key not in adata.obs:
        raise KeyError(f"case_control_key {case_control_key!r} is not present in adata.obs.")
    control_set = {str(v) for v in control_values}
    case_set = {str(v) for v in case_values}
    overlap = control_set & case_set
    if overlap:
        raise ValueError(f"control_values and case_values overlap: {sorted(overlap)!r}.")
    status = adata.obs[case_control_key].astype(str)
    is_control = status.isin(control_set).to_numpy()
    is_case = status.isin(case_set).to_numpy()
    ref = adata[is_control].copy()
    query = adata[is_case].copy()
    if ref.n_obs < 64 or query.n_obs < 64:
        raise ValueError(
            "Real case/control split is too small for B4/B6: "
            f"reference={ref.n_obs}, query={query.n_obs}. Need at least 64 cells each."
        )
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
    case_control_key=None,
    control_values=None,
    case_values=None,
    batch_size=None,
    annbatch_config=None,
    cytoanvi_training_config=None,
) -> dict:
    """Train reference and select replay/control — shared setup for B4 and B6.

    Returns a dict consumed by :func:`task_b4_continual`.  B6 calls this once and passes
    the result to each per-λ surgery call, avoiding redundant reference retraining.
    """
    from cytoanvi import CytoANVI

    labels = np.asarray(adata.obs[labels_key].astype(str))
    use_real_split = (
        case_control_key is not None and control_values is not None and case_values is not None
    )
    if use_real_split:
        ref_adata, query_adata, _fallback_split = _split_reference_query_by_case_control(
            adata,
            case_control_key=case_control_key,
            control_values=list(control_values),
            case_values=list(case_values),
        )
        case_control_mode = "real"
        query_batch_values = []
    else:
        if query_batch_values is None:
            batches = sorted(adata.obs[batch_key].astype(str).unique())
            query_batch_values = [batches[-1]]
        query_batch_values = list(query_batch_values)
        ref_adata, query_adata, _fallback_split = _split_reference_query(
            adata, batch_key=batch_key, query_batch_values=query_batch_values, seed=seed
        )
        case_control_mode = "pseudo_batch"
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
        annbatch_config=annbatch_config,
        **_training_config(cytoanvi_training_config),
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
        "case_control_mode": case_control_mode,
        "case_control_key": case_control_key,
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
    case_control_key=None,
    control_values=None,
    case_values=None,
    batch_size=None,
    annbatch_config=None,
    cytoanvi_training_config=None,
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
            case_control_key=case_control_key,
            control_values=control_values,
            case_values=case_values,
            batch_size=batch_size,
            annbatch_config=annbatch_config,
            cytoanvi_training_config=cytoanvi_training_config,
        )

    ref_model = _setup["ref_model"]
    ref_adata = _setup["ref_adata"]
    query_adata = _setup["query_adata"]
    true_query = _setup["true_query"]
    control = _setup["control"]
    replay = _setup["replay"]
    train_extra = _setup["train_extra"]
    case_control_mode = _setup.get("case_control_mode", "pseudo_batch")
    note = (
        "Real case/control split - use for biological continual-update evaluation."
        if case_control_mode == "real"
        else "Pseudo case/control via batch split - validates plumbing, not biology."
    )

    # Use .copy() so surgery calls do not mutate the shared query_adata in the setup dict
    # (load_query_data runs setup_anndata in-place; B6 reuses the same setup across λ).
    plain = CytoANVI.load_query_data(query_adata.copy(), ref_model)
    # Plain surgery uses automatic optimization → gradient_clip_val via Trainer.
    plain.train(
        max_epochs=min(50, max_epochs),
        gradient_clip_val=_GRAD_CLIP,
        plan_kwargs={"weight_decay": 0.0},
        **train_extra,
        **annbatch_train_kwargs(plain, annbatch_config, batch_size=batch_size),
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
        **annbatch_train_kwargs(continual, annbatch_config, batch_size=batch_size),
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
        "case_control_mode": case_control_mode,
        "case_control_key": _setup.get("case_control_key"),
        "_fallback_split": _setup["_fallback_split"],
        "note": note,
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
    annbatch_config=None,
    cytoanvi_training_config=None,
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
        annbatch_config=annbatch_config,
        **_training_config(cytoanvi_training_config),
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
        annbatch_config=annbatch_config,
        **_training_config(cytoanvi_training_config),
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
    case_control_key=None,
    control_values=None,
    case_values=None,
    batch_size=None,
    control_frac=0.1,
    replay_frac=0.2,
    annbatch_config=None,
    cytoanvi_training_config=None,
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
        case_control_key=case_control_key,
        control_values=control_values,
        case_values=case_values,
        batch_size=batch_size,
        annbatch_config=annbatch_config,
        cytoanvi_training_config=cytoanvi_training_config,
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
            case_control_key=case_control_key,
            control_values=control_values,
            case_values=case_values,
            batch_size=batch_size,
            annbatch_config=annbatch_config,
            cytoanvi_training_config=cytoanvi_training_config,
            _setup=setup,
        )["continual_update"]
    drifts = np.asarray([v["replay_latent_drift"] for v in per_lambda.values()], dtype=float)
    f1s = [v["query_label_transfer"]["macro_f1"] for v in per_lambda.values()]
    out = {
        "task": "b6_lambda_sweep",
        "seed": seed,
        "max_epochs": max_epochs,
        "lambdas": list(lambdas),
        "case_control_mode": setup.get("case_control_mode", "pseudo_batch"),
        "case_control_key": setup.get("case_control_key"),
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
    sample_col[query_idx] = np.take(query_samples, np.arange(len(query_idx)) % len(query_samples))
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
    annbatch_config=None,
    cytoanvi_training_config=None,
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
    training_config = _training_config(cytoanvi_training_config)

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
        annbatch_config=annbatch_config,
        **training_config,
    )

    query_train = query_adata.copy()
    query_train.obs[labels_key] = unlabeled_category
    query_model = CytoANVI.load_query_data(query_train, ref_model)
    query_plan_kw = {"weight_decay": 0.0}
    if training_config.get("reduce_lr_on_plateau"):
        query_plan_kw["reduce_lr_on_plateau"] = True
    if training_config.get("classification_ratio") is not None:
        query_plan_kw["classification_ratio"] = training_config["classification_ratio"]
    query_train_kw: dict = {"plan_kwargs": query_plan_kw}
    if training_config.get("learning_rate") is not None:
        query_train_kw["lr"] = training_config["learning_rate"]
    if training_config.get("gradient_clip_val") is not None:
        query_train_kw["gradient_clip_val"] = training_config["gradient_clip_val"]
    if batch_size is not None:
        query_train_kw["batch_size"] = batch_size
    query_model.train(
        max_epochs=min(50, max_epochs),
        **query_train_kw,
        **annbatch_train_kwargs(query_model, annbatch_config, batch_size=batch_size),
    )

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
