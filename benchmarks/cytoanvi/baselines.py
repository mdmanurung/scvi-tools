"""Baseline: CytoVI latent + k-NN label transfer (the CytoVI vignette's own method).

Equivalent to :meth:`~scvi.external.CYTOVI.impute_categories_from_reference` (k-NN voting in the
CytoVI latent), implemented directly with sklearn so reference and query share one fitted model.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

SCALED_LAYER = "scaled"


def cytovi_latent_and_knn(
    adata,
    labels_key: str,
    unlabeled_category: str,
    batch_key: str | None = "batch",
    sample_key: str | None = None,
    nan_layer: str | None = None,
    n_neighbors: int = 20,
    max_epochs: int = 100,
    n_latent: int = 10,
):
    """Train CytoVI, then k-NN-transfer labels from labelled to unlabeled cells in its latent.

    Returns ``(pred_for_unlabeled, latent_all, unlabeled_mask)``.
    """
    from scvi.external import CYTOVI

    a = adata.copy()
    CYTOVI.setup_anndata(
        a,
        layer=SCALED_LAYER,
        batch_key=batch_key,
        labels_key=labels_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
    )
    model = CYTOVI(a, n_latent=n_latent)
    model.train(max_epochs=max_epochs)
    model.module.eval()
    latent = model.get_latent_representation()

    labels = np.asarray(a.obs[labels_key].astype(str))
    labelled = labels != unlabeled_category
    knn = KNeighborsClassifier(n_neighbors=min(n_neighbors, int(labelled.sum())))
    knn.fit(latent[labelled], labels[labelled])
    pred_unlabeled = knn.predict(latent[~labelled])
    return pred_unlabeled, latent, ~labelled
