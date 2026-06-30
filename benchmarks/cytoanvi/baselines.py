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
    seed: int = 0,
):
    """PCA → Harmony batch correction → kNN label transfer.

    Returns ``(pred_for_unlabeled, Z_harmony, unlabeled_mask)``.
    """
    import harmonypy as hm

    X = _get_dense(adata, layer)
    n_comp = min(n_pca_components, X.shape[1] - 1, X.shape[0] - 1)
    Z_pca = PCA(n_components=n_comp, random_state=seed).fit_transform(X)

    meta_df = adata.obs[[batch_key]].copy().astype(str)
    ho = hm.run_harmony(Z_pca, meta_df, batch_key, random_state=seed, verbose=False)
    # harmonypy ≥0.2.0 returns Z_corr as (n_cells, n_components); older versions as (n_components, n_cells).
    Z_harmony = ho.Z_corr.T if ho.Z_corr.shape[0] == n_comp else ho.Z_corr

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


def xgboost_classifier(
    adata,
    labels_key: str,
    unlabeled_category: str,
    seed: int = 0,
    layer: str = SCALED_LAYER,
):
    """XGBoost supervised classifier directly on scaled markers.

    Returns ``(pred_for_unlabeled, X, unlabeled_mask)``.

    Requires: ``pip install xgboost``
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost is required: pip install xgboost")
    from sklearn.preprocessing import LabelEncoder

    X = _get_dense(adata, layer)
    labels = np.asarray(adata.obs[labels_key].astype(str))
    labelled = labels != unlabeled_category

    le = LabelEncoder().fit(labels[labelled])
    clf = XGBClassifier(random_state=seed, eval_metric="mlogloss", verbosity=0)
    clf.fit(X[labelled], le.transform(labels[labelled]))

    unlabeled_mask = ~labelled
    if unlabeled_mask.any():
        pred_unlabeled = le.inverse_transform(clf.predict(X[unlabeled_mask]))
    else:
        pred_unlabeled = np.array([], dtype=object)
    return pred_unlabeled, X, unlabeled_mask


def phenograph_knn(
    adata,
    labels_key: str,
    unlabeled_category: str,
    k: int = 30,
    seed: int = 0,
    layer: str = SCALED_LAYER,
):
    """Phenograph community detection → majority-vote label assignment.

    Clusters all cells (labeled + unlabeled) jointly, then assigns each
    community the plurality label from labeled cells in that community.
    Orphan communities (no labeled cells) fall back to kNN on markers.

    Returns ``(pred_for_unlabeled, X, unlabeled_mask)``.

    Requires: ``pip install phenograph``
    """
    try:
        import phenograph
    except ImportError:
        raise ImportError("phenograph is required: pip install phenograph")

    X = _get_dense(adata, layer)
    labels = np.asarray(adata.obs[labels_key].astype(str))
    labelled = labels != unlabeled_category

    communities, _, _ = phenograph.cluster(X, k=k, seed=seed)

    fallback_knn = KNeighborsClassifier(n_neighbors=min(5, int(labelled.sum())))
    fallback_knn.fit(X[labelled], labels[labelled])

    comm_to_label: dict[int, str] = {}
    for c in np.unique(communities):
        if c < 0:
            continue
        in_comm = (communities == c) & labelled
        if in_comm.any():
            vals, counts = np.unique(labels[in_comm], return_counts=True)
            comm_to_label[c] = vals[np.argmax(counts)]

    unlabeled_mask = ~labelled
    if unlabeled_mask.any():
        preds = []
        for i in np.where(unlabeled_mask)[0]:
            c = int(communities[i])
            preds.append(comm_to_label[c] if c in comm_to_label else fallback_knn.predict(X[[i]])[0])
        pred_unlabeled = np.array(preds, dtype=object)
    else:
        pred_unlabeled = np.array([], dtype=object)
    return pred_unlabeled, X, unlabeled_mask


def flowsom_knn(
    adata,
    labels_key: str,
    unlabeled_category: str,
    xdim: int = 10,
    ydim: int = 10,
    n_metaclusters: int = 20,
    seed: int = 0,
    layer: str = SCALED_LAYER,
):
    """FlowSOM SOM + hierarchical metaclustering → majority-vote label assignment.

    Builds a self-organizing map on all cells in marker space, metaclusters
    the SOM nodes via agglomerative clustering, then assigns each metacluster
    the plurality label from labeled cells in that metacluster.
    Metaclusters with no labeled cells fall back to kNN on markers.

    Returns ``(pred_for_unlabeled, X, unlabeled_mask)``.

    Requires: ``pip install pyFlowSOM``
    """
    try:
        from pyFlowSOM import map_data_to_nodes, som
    except ImportError:
        raise ImportError("pyFlowSOM is required: pip install pyFlowSOM")
    from sklearn.cluster import AgglomerativeClustering

    X = _get_dense(adata, layer)
    labels = np.asarray(adata.obs[labels_key].astype(str))
    labelled = labels != unlabeled_category

    fsom = som(X, xdim=xdim, ydim=ydim, seed=seed)
    node_per_cell = map_data_to_nodes(X, fsom)

    node_repr = fsom["codes"]
    n_meta = min(n_metaclusters, len(node_repr))
    meta_per_node = AgglomerativeClustering(n_clusters=n_meta).fit_predict(node_repr)
    metacluster_per_cell = meta_per_node[node_per_cell]

    fallback_knn = KNeighborsClassifier(n_neighbors=min(5, int(labelled.sum())))
    fallback_knn.fit(X[labelled], labels[labelled])

    meta_to_label: dict[int, str] = {}
    for c in range(n_meta):
        in_meta = (metacluster_per_cell == c) & labelled
        if in_meta.any():
            vals, counts = np.unique(labels[in_meta], return_counts=True)
            meta_to_label[c] = vals[np.argmax(counts)]

    unlabeled_mask = ~labelled
    if unlabeled_mask.any():
        preds = []
        for i in np.where(unlabeled_mask)[0]:
            c = int(metacluster_per_cell[i])
            preds.append(meta_to_label[c] if c in meta_to_label else fallback_knn.predict(X[[i]])[0])
        pred_unlabeled = np.array(preds, dtype=object)
    else:
        pred_unlabeled = np.array([], dtype=object)
    return pred_unlabeled, X, unlabeled_mask
