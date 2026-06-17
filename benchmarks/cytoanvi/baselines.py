"""Baseline: CytoVI latent + k-NN label transfer."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from benchmarks.common.training import SCALED_LAYER, train_cytovi


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
):
    """Train CytoVI, then k-NN-transfer labels in its latent.

    Returns ``(pred_for_unlabeled, latent_all, unlabeled_mask)``.
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
