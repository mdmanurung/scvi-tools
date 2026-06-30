"""Paper-faithful CYTOVI / CytoANVI training helpers."""

from __future__ import annotations

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
):
    """Train CYTOVI with paper defaults (MoG prior, Gaussian likelihood, latent heuristic).

    Parameters
    ----------
    batch_size
        Mini-batch size passed to the data splitter.  ``None`` preserves scvi's default (128).
        For large cohorts (>100 k cells) set ``batch_size=8192`` to avoid NaN divergence and
        reduce per-epoch wall time (~64× fewer gradient steps on roider-full).
    """
    from scvi.external import CYTOVI

    a = adata.copy()
    setup_kw = dict(
        layer=layer,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
    )
    if labels_key is not None:
        setup_kw["labels_key"] = labels_key
    CYTOVI.setup_anndata(a, **setup_kw)
    model = CYTOVI(a, n_latent=n_latent)
    # AdversarialTrainingPlan uses manual optimization; clip via plan_kwargs.
    train_kw: dict = {
        "accelerator": "auto",
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
    hierarchy_edges: dict | None = None,
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
        Mini-batch size passed to the data splitter.  ``None`` preserves scvi's default (128).
        For large cohorts (>100 k cells) set ``batch_size=8192`` to avoid NaN divergence and
        reduce per-epoch wall time (~64× fewer gradient steps on roider-full).
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
    model = CytoANVI(a, n_latent=n_latent)
    if hierarchy_edges is not None:
        model.set_hierarchy(hierarchy_edges)
    plan_kw = {}
    if reduce_lr_on_plateau:
        plan_kw["reduce_lr_on_plateau"] = True
    train_kw: dict = {
        "accelerator": "auto",
        "gradient_clip_val": _GRAD_CLIP,
        "n_samples_per_label": n_samples_per_label,
        "plan_kwargs": plan_kw or None,
    }
    if batch_size is not None:
        train_kw["batch_size"] = batch_size
    model.train(max_epochs=max_epochs, **train_kw)
    model.module.eval()
    return model, a


def latent_obsm(adata, model, obsm_key: str = "X_benchmark"):
    """Store CYTOVI/CytoANVI latent in ``obsm`` for scib-metrics."""
    adata.obsm[obsm_key] = model.get_latent_representation()
    return adata
