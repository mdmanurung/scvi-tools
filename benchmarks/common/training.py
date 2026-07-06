"""Paper-faithful CYTOVI / CytoANVI training helpers."""

from __future__ import annotations

import os
import warnings

SCALED_LAYER = "scaled"
NAN_LAYER = "_nan_mask"

# Gradient clipping threshold applied to both CytoVI and CytoANVI training.
# On the full Roider cohort (~1.24 M cells, ~9 700 batches/epoch) the Adam
# optimiser diverges around epoch 60 without clipping: the encoder's loc
# tensor fills with NaN (AddmmBackward0 in the traceback).
#
# CytoVI uses AdversarialTrainingPlan (manual optimization): clipping is
# injected via plan_kwargs={"gradient_clip_norm": _GRAD_CLIP}, which calls
# torch.nn.utils.clip_grad_norm_ inside training_step.
#
# CytoANVI uses SemiSupervisedTrainingPlan (automatic optimization):
# Lightning's Trainer(gradient_clip_val=_GRAD_CLIP) is compatible and used.
#
# Both use the same numeric threshold so the comparison is on equal footing.
_GRAD_CLIP = 1.0


def resolve_nan_layer(adata, nan_layer: str | None = NAN_LAYER) -> str | None:
    """Return the benchmark NaN-mask layer only when it is present."""
    if nan_layer is None:
        return None
    return nan_layer if nan_layer in adata.layers else None


def annbatch_train_kwargs(
    model,
    annbatch_config=None,
    *,
    batch_size: int | None = None,
    n_samples_per_label: int | None = None,
) -> dict:
    """Return ``datamodule`` kwargs for opt-in CytoANVI AnnBatch training."""
    if annbatch_config is None or not getattr(annbatch_config, "enabled", False):
        return {}
    if n_samples_per_label is not None:
        warnings.warn(
            "AnnBatch backend does not support n_samples_per_label; using the standard "
            "scvi AnnDataLoader for this training call.",
            UserWarning,
            stacklevel=2,
        )
        return {}

    from benchmarks.common.annbatch import (
        UnsupportedAnnBatchRegistry,
        make_cytoanvi_annbatch_datamodule,
    )

    try:
        datamodule = make_cytoanvi_annbatch_datamodule(
            model,
            annbatch_config,
            batch_size=batch_size or 4096,
        )
    except UnsupportedAnnBatchRegistry as err:
        warnings.warn(
            f"{err} Falling back to the standard scvi AnnDataLoader.",
            UserWarning,
            stacklevel=2,
        )
        return {}
    return {"datamodule": datamodule}


def train_cytovi(
    adata,
    *,
    batch_key: str,
    labels_key: str | None = None,
    sample_key: str | None = None,
    nan_layer: str | None = None,
    layer: str = SCALED_LAYER,
    n_latent: int | None = None,
    max_epochs: int = 1000,
    batch_size: int | None = None,
    accelerator: str = "auto",
    devices: int | list[int] | str = "auto",
):
    """Train CYTOVI with paper defaults (MoG prior, Gaussian likelihood, latent heuristic).

    Parameters
    ----------
    batch_size
        Mini-batch size passed to the data splitter.  ``None`` preserves CYTOVI's default (128).
        For large cohorts (>100 k cells) set ``batch_size=4096`` or higher to avoid NaN
        divergence and reduce per-epoch wall time (~32× fewer gradient steps on roider-full).
    """
    from scvi.external import CYTOVI

    a = adata.copy()
    setup_kw = {
        "layer": layer,
        "batch_key": batch_key,
        "sample_key": sample_key,
        "nan_layer": nan_layer,
    }
    if labels_key is not None:
        setup_kw["labels_key"] = labels_key
    CYTOVI.setup_anndata(a, **setup_kw)
    model = CYTOVI(a, n_latent=n_latent)
    # AdversarialTrainingPlan uses manual optimization; clip via plan_kwargs.
    train_kw: dict = {
        "accelerator": accelerator,
        "devices": devices,
        "plan_kwargs": {"gradient_clip_norm": _GRAD_CLIP},
    }
    if batch_size is not None:
        train_kw["batch_size"] = batch_size
    model.train(max_epochs=max_epochs, **train_kw)
    model.module.eval()
    return model, a


def train_cytoanvi(
    adata,
    *,
    labels_key: str,
    unlabeled_category: str,
    batch_key: str,
    sample_key: str | None = None,
    nan_layer: str | None = None,
    layer: str = SCALED_LAYER,
    n_latent: int | None = None,
    max_epochs: int = 1000,
    n_samples_per_label: int | None = None,
    reduce_lr_on_plateau: bool = False,
    batch_size: int | None = None,
    accelerator: str = "auto",
    devices: int | list[int] | str = "auto",
    hierarchy_edges: dict | None = None,
    annbatch_config=None,
    y_prior="uniform",
    class_weighting="none",
    class_weight_clip: float = 10.0,
    classification_ratio: float | None = None,
    learning_rate: float | None = None,
    gradient_clip_val: float | None = None,
    precision: str | None = None,
    early_stopping: bool | None = None,
):
    """Train CytoANVI with paper-aligned CYTOVI backbone defaults.

    Parameters
    ----------
    n_samples_per_label
        If set, each minibatch draws this many labeled cells per class,
        balancing representation of rare cell types. Passed to the data
        splitter via ``train()``.
    reduce_lr_on_plateau
        Enable LR reduction on plateau (via ``elbo_validation``). Useful for
        stabilizing the classifier head on difficult seeds.
    batch_size
        Mini-batch size. ``None`` uses CytoANVI's default (4096).
        For very large cohorts (>500 k cells) consider ``batch_size=8192`` to further reduce
        per-epoch wall time.
    """
    from cytoanvi import CytoANVI

    a = adata.copy()
    CytoANVI.setup_anndata(
        a,
        layer=layer,
        batch_key=batch_key,
        labels_key=labels_key,
        unlabeled_category=unlabeled_category,
        sample_key=sample_key,
        nan_layer=nan_layer,
    )
    model = CytoANVI(
        a,
        n_latent=n_latent,
        y_prior=y_prior,
        class_weighting=class_weighting,
        class_weight_clip=class_weight_clip,
    )
    if hierarchy_edges is not None:
        model.set_hierarchy(hierarchy_edges)
    plan_kw = {}
    if reduce_lr_on_plateau:
        plan_kw["reduce_lr_on_plateau"] = True
    if classification_ratio is not None:
        plan_kw["classification_ratio"] = classification_ratio
    if learning_rate is not None:
        plan_kw["lr"] = learning_rate
    train_kw: dict = {
        "accelerator": accelerator,
        "devices": devices,
        "gradient_clip_val": _GRAD_CLIP if gradient_clip_val is None else gradient_clip_val,
        "n_samples_per_label": n_samples_per_label,
        "plan_kwargs": plan_kw or None,
    }
    # Mixed precision (L-044): bf16-mixed is ~27% faster/step — the label-marginalized ELBO is
    # compute-bound and dominated by memory-bound elementwise ops that bf16 accelerates. The
    # explicit ``precision`` arg wins; otherwise fall back to the CYTOANVI_TRAIN_PRECISION env var
    # (e.g. "bf16-mixed"). ``None`` preserves Lightning's fp32 default.
    precision = precision or os.environ.get("CYTOANVI_TRAIN_PRECISION")
    if precision:
        train_kw["precision"] = precision
    if early_stopping is not None:
        train_kw["early_stopping"] = early_stopping
    if batch_size is not None:
        train_kw["batch_size"] = batch_size
    train_kw.update(
        annbatch_train_kwargs(
            model,
            annbatch_config,
            batch_size=batch_size,
            n_samples_per_label=n_samples_per_label,
        )
    )
    model.train(max_epochs=max_epochs, **train_kw)
    model.module.eval()
    return model, a


def latent_obsm(adata, model, obsm_key: str = "X_benchmark"):
    """Store CYTOVI/CytoANVI latent in ``obsm`` for scib-metrics."""
    adata.obsm[obsm_key] = model.get_latent_representation()
    return adata
