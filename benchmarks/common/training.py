"""Paper-faithful CYTOVI / CytoANVI training helpers."""

from __future__ import annotations

SCALED_LAYER = "scaled"
NAN_LAYER = "_nan_mask"


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
):
    """Train CYTOVI with paper defaults (MoG prior, Gaussian likelihood, latent heuristic)."""
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
    model.train(max_epochs=max_epochs)
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
):
    """Train CytoANVI with paper-aligned CYTOVI backbone defaults."""
    from scvi.external import CytoANVI

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
    model.train(max_epochs=max_epochs)
    model.module.eval()
    return model, a


def latent_obsm(adata, model, obsm_key: str = "X_benchmark"):
    """Store CYTOVI/CytoANVI latent in ``obsm`` for scib-metrics."""
    adata.obsm[obsm_key] = model.get_latent_representation()
    return adata
