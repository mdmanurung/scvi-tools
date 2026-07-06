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
    accelerator: str = "auto",
    devices: int | list[int] | str = "auto",
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
        accelerator=accelerator,
        devices=devices,
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


def knn_distance_novelty(
    z_ref: np.ndarray,
    z_eval: np.ndarray,
    n_neighbors: int = 15,
) -> np.ndarray:
    """Mean Euclidean kNN-distance novelty score in a pre-computed latent space.

    Given reference and eval embeddings, scores each eval cell by the mean Euclidean
    distance to its ``n_neighbors`` nearest reference cells.  Higher = more novel.

    This is the shared distance step used by both the CytoVI OOD baseline and the
    CytoANVI-latent OOD diagnostic so the two scores are identical in methodology and
    differ only in which latent space they operate on.

    Parameters
    ----------
    z_ref
        Latent embeddings of the *seen* (reference) cells, shape ``(n_ref, n_latent)``.
    z_eval
        Latent embeddings of the eval cells (includes both seen calibration and held-out
        novel cells), shape ``(n_eval, n_latent)``.
    n_neighbors
        Number of nearest reference neighbours to average.  Clipped to ``len(z_ref)``.

    Returns
    -------
    np.ndarray of shape ``(n_eval,)`` — per-eval-cell novelty score.
    """
    from sklearn.neighbors import NearestNeighbors

    k = max(1, min(n_neighbors, len(z_ref)))
    nn = NearestNeighbors(n_neighbors=k).fit(z_ref)
    dist, _ = nn.kneighbors(z_eval)
    return dist.mean(axis=1)


def cytovi_novelty_score(
    seen_adata,
    eval_adata,
    *,
    batch_key: str | None = "batch",
    sample_key: str | None = None,
    nan_layer: str | None = None,
    n_neighbors: int = 15,
    max_epochs: int = 1000,
    n_latent: int | None = None,
    batch_size: int | None = None,
    accelerator: str = "auto",
    devices: int | list[int] | str = "auto",
) -> np.ndarray:
    """Unsupervised CytoVI novelty baseline for the B5 holdout sweep.

    Trains CytoVI on the *seen* reference cells (the held-out type is absent from
    ``seen_adata``, so CytoVI never sees it — a fair novelty holdout), then scores each eval cell
    by the mean Euclidean distance to its ``n_neighbors`` nearest reference cells in CytoVI latent.
    Higher distance = further from any seen population = higher novelty score, directly comparable
    to CytoANVI's TTA uncertainty via :func:`~benchmarks.cytoanvi.metrics.novelty_auroc`.

    Returns a per-eval-cell novelty score aligned with ``eval_adata`` (higher = more novel).
    """
    model, _ = train_cytovi(
        seen_adata,
        batch_key=batch_key,
        sample_key=sample_key,
        nan_layer=nan_layer,
        layer=SCALED_LAYER,
        n_latent=n_latent,
        max_epochs=max_epochs,
        batch_size=batch_size,
        accelerator=accelerator,
        devices=devices,
    )
    z_ref = np.asarray(model.get_latent_representation())
    z_eval = np.asarray(model.get_latent_representation(eval_adata))
    return knn_distance_novelty(z_ref, z_eval, n_neighbors)


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
    # harmonypy >=0.2.0 returns Z_corr as (n_cells, n_components);
    # older versions use (n_components, n_cells).
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
    except ImportError as err:
        raise ImportError("xgboost is required: pip install xgboost") from err
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


def _cluster_majority_vote_transfer(
    clusters,
    X: np.ndarray,
    labels: np.ndarray,
    labelled: np.ndarray,
):
    """Assign unlabeled cells from cluster plurality labels with marker-kNN fallback."""
    cluster_labels = np.asarray(clusters).astype(str)
    if cluster_labels.shape[0] != X.shape[0]:
        raise ValueError(
            f"Cluster label count ({cluster_labels.shape[0]}) does not match n_obs ({X.shape[0]})."
        )
    n_labelled = int(labelled.sum())
    if n_labelled == 0:
        raise ValueError("At least one labelled cell is required for cluster label transfer.")

    fallback_knn = KNeighborsClassifier(n_neighbors=min(5, n_labelled))
    fallback_knn.fit(X[labelled], labels[labelled])

    cluster_to_label: dict[str, str] = {}
    for c in np.unique(cluster_labels):
        if c == "-1":
            continue
        in_cluster = (cluster_labels == c) & labelled
        if in_cluster.any():
            vals, counts = np.unique(labels[in_cluster], return_counts=True)
            cluster_to_label[c] = vals[np.argmax(counts)]

    unlabeled_mask = ~labelled
    if unlabeled_mask.any():
        preds = []
        for i in np.where(unlabeled_mask)[0]:
            c = cluster_labels[i]
            preds.append(
                cluster_to_label[c] if c in cluster_to_label else fallback_knn.predict(X[[i]])[0]
            )
        pred_unlabeled = np.array(preds, dtype=object)
    else:
        pred_unlabeled = np.array([], dtype=object)
    return pred_unlabeled, X, unlabeled_mask


def rapids_graph_knn(
    adata,
    labels_key: str,
    unlabeled_category: str,
    k: int = 30,
    resolution: float = 1.0,
    seed: int = 0,
    layer: str = SCALED_LAYER,
):
    """RAPIDS SingleCell graph clustering plus majority-vote label assignment.

    Clusters all cells (labeled + unlabeled) jointly, then assigns each
    Leiden community the plurality label from labeled cells in that community.
    Orphan communities with no labeled cells fall back to kNN on markers.

    Returns ``(pred_for_unlabeled, X, unlabeled_mask)``.

    Requires: ``pip install rapids-singlecell[rapids]``
    """
    try:
        import rapids_singlecell as rsc
    except ImportError as err:
        raise ImportError(
            "rapids-singlecell is required: install scvi-tools[rapids] "
            "or pip install rapids-singlecell[rapids]"
        ) from err
    import anndata as ad

    X = _get_dense(adata, layer).astype(np.float32, copy=False)
    labels = np.asarray(adata.obs[labels_key].astype(str))
    labelled = labels != unlabeled_category
    if X.shape[0] < 2:
        raise ValueError("RAPIDS graph clustering requires at least two cells.")

    graph_adata = ad.AnnData(X=X.copy())
    graph_adata.obsm["X_baseline"] = X
    n_neighbors = min(int(k), X.shape[0] - 1)
    rsc.pp.neighbors(graph_adata, n_neighbors=n_neighbors, use_rep="X_baseline")
    rsc.tl.leiden(
        graph_adata,
        key_added="rapids_graph_clusters",
        resolution=resolution,
        random_state=seed,
    )
    communities = np.asarray(graph_adata.obs["rapids_graph_clusters"])

    return _cluster_majority_vote_transfer(communities, X, labels, labelled)


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
    """FlowSOM Python metaclustering plus majority-vote label assignment.

    Builds a self-organizing map on all cells in marker space, metaclusters
    the SOM nodes, then assigns each metacluster the plurality label from
    labeled cells in that metacluster. Metaclusters with no labeled cells fall
    back to kNN on markers.

    Returns ``(pred_for_unlabeled, X, unlabeled_mask)``.

    Requires: ``pip install flowsom``
    """
    try:
        import flowsom
    except (ImportError, RuntimeError) as err:
        raise ImportError(
            "flowsom is required and must import successfully: pip install flowsom"
        ) from err
    import anndata as ad

    X = _get_dense(adata, layer).astype(np.float32, copy=False)
    labels = np.asarray(adata.obs[labels_key].astype(str))
    labelled = labels != unlabeled_category

    flow_adata = ad.AnnData(X=X.copy())
    if len(adata.var_names) == X.shape[1]:
        flow_adata.var_names = adata.var_names.astype(str)
    n_meta = max(1, min(int(n_metaclusters), X.shape[0], int(xdim) * int(ydim)))
    fsom = flowsom.FlowSOM(
        flow_adata,
        n_clusters=n_meta,
        xdim=xdim,
        ydim=ydim,
        seed=seed,
    )

    if hasattr(fsom, "get_cell_data"):
        cell_data = fsom.get_cell_data()
    else:
        cell_data = fsom.mudata["cell_data"]
    if "metaclustering" not in cell_data.obs:
        raise KeyError("FlowSOM did not write cell-level 'metaclustering' labels.")
    metacluster_per_cell = np.asarray(cell_data.obs["metaclustering"])

    return _cluster_majority_vote_transfer(metacluster_per_cell, X, labels, labelled)
