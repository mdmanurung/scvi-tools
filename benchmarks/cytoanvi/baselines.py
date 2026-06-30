"""Baselines for CytoANVI B1 label-transfer benchmark."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from benchmarks.common.training import SCALED_LAYER, train_cytovi


def _get_dense(adata, layer: str) -> np.ndarray:
    X = adata.layers[layer] if layer in adata.layers else adata.X
    if sp.issparse(X):
        return np.asarray(X.todense())
    return np.asarray(X)


def cytovi_latent_and_knn(
    adata,
    labels_key: str,
    unlabeled_category: str,
    batch_key: str | None = "batch",
    sample_key: str | None = None,
    nan_layer: str | None = None,
    n_neighbors: int = 20,
    max_epochs: int = 1000,
    n_latent: int | None = None,
    batch_size: int | None = None,
):
    """Train CytoVI, then k-NN-transfer labels in its latent.

    Returns ``(pred_for_unlabeled, latent_all, unlabeled_mask)``.

    Parameters
    ----------
    batch_size
        Forwarded to :func:`~benchmarks.common.training.train_cytovi`.  ``None`` preserves
        scvi's default (128).  Set ``batch_size=8192`` on roider-full to avoid NaN divergence.
    """
    model, a = train_cytovi(
        adata,
        batch_key=batch_key,
        labels_key=labels_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        layer=SCALED_LAYER,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
    )
    latent = model.get_latent_representation()

    labels = np.asarray(a.obs[labels_key].astype(str))
    labelled = labels != unlabeled_category
    knn = KNeighborsClassifier(n_neighbors=min(n_neighbors, int(labelled.sum())))
    knn.fit(latent[labelled], labels[labelled])
    unlabeled_mask = ~labelled
    if unlabeled_mask.any():
        pred_unlabeled = knn.predict(latent[unlabeled_mask])
    else:
        pred_unlabeled = np.array([], dtype=object)
    return pred_unlabeled, latent, unlabeled_mask


def raw_marker_knn(
    adata,
    labels_key: str,
    unlabeled_category: str,
    n_neighbors: int = 20,
    layer: str = SCALED_LAYER,
):
    """KNN directly on arcsinh+scaled markers (no VAE).

    Returns ``(pred_for_unlabeled, X, unlabeled_mask)``.
    """
    X = _get_dense(adata, layer)
    labels = np.asarray(adata.obs[labels_key].astype(str))
    labelled = labels != unlabeled_category
    knn = KNeighborsClassifier(n_neighbors=min(n_neighbors, int(labelled.sum())))
    knn.fit(X[labelled], labels[labelled])
    unlabeled_mask = ~labelled
    if unlabeled_mask.any():
        pred_unlabeled = knn.predict(X[unlabeled_mask])
    else:
        pred_unlabeled = np.array([], dtype=object)
    return pred_unlabeled, X, unlabeled_mask


def harmony_latent_and_knn(
    adata,
    labels_key: str,
    unlabeled_category: str,
    batch_key: str = "batch",
    n_neighbors: int = 20,
    n_pca_components: int = 30,
    layer: str = SCALED_LAYER,
):
    """PCA → Harmony batch correction → kNN label transfer.

    Returns ``(pred_for_unlabeled, Z_harmony, unlabeled_mask)``.
    """
    import harmonypy as hm

    X = _get_dense(adata, layer)
    n_comp = min(n_pca_components, X.shape[1] - 1, X.shape[0] - 1)
    Z_pca = PCA(n_components=n_comp, random_state=0).fit_transform(X)

    meta_df = adata.obs[[batch_key]].copy().astype(str)
    ho = hm.run_harmony(Z_pca, meta_df, batch_key, random_state=0, verbose=False)
    Z_harmony = ho.Z_corr.T  # (n_cells, n_components)

    labels = np.asarray(adata.obs[labels_key].astype(str))
    labelled = labels != unlabeled_category
    knn = KNeighborsClassifier(n_neighbors=min(n_neighbors, int(labelled.sum())))
    knn.fit(Z_harmony[labelled], labels[labelled])
    unlabeled_mask = ~labelled
    if unlabeled_mask.any():
        pred_unlabeled = knn.predict(Z_harmony[unlabeled_mask])
    else:
        pred_unlabeled = np.array([], dtype=object)
    return pred_unlabeled, Z_harmony, unlabeled_mask
