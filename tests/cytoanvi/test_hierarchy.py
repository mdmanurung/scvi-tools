import numpy as np
import pytest
from conftest import (
    BATCH_KEY,
    LABELS_KEY,
    N_EPOCHS,
    SAMPLE_KEY,
    SCALED_LAYER_KEY,
    UNLABELED,
    MockTreeNode as _MockTreeNode,
    make_adata,
    setup_and_train,
)

from cytoanvi import hierarchy
from cytoanvi._hce import build_reachability_matrix

CELLTYPE_BATCH_KEY = "celltype_batch"


def _mock_schpl_tree():
    """Tree where internal node label_1 is ancestor of label_3 and label_4."""
    leaf_2 = _MockTreeNode("label_2-b1")
    leaf_3 = _MockTreeNode("label_3-b1")
    leaf_4 = _MockTreeNode("label_4-b1")
    internal_1 = _MockTreeNode("label_1", [leaf_3, leaf_4])
    root = _MockTreeNode("root", [internal_1, leaf_2])
    return [root]


def _expected_reachability(label_names, edges):
    return build_reachability_matrix(list(label_names), edges)


def test_reachability_from_schpl_tree_uses_model_labels_not_internal_nodes():
    model_labels = ["label_1", "label_2", "label_3", "label_4"]
    leaf_to_model = {
        "label_2-b1": "label_2",
        "label_3-b1": "label_3",
        "label_4-b1": "label_4",
    }
    matrix = hierarchy.reachability_from_schpl_tree(
        _mock_schpl_tree(),
        model_labels,
        leaf_to_model=leaf_to_model,
    )
    expected_edges = {
        "label_1": ["label_3", "label_4"],
        "label_2": [],
        "label_3": [],
        "label_4": [],
    }
    expected = _expected_reachability(model_labels, expected_edges)
    np.testing.assert_array_equal(matrix, expected)


def test_set_hierarchy_from_schpl_accepts_internal_schpl_nodes():
    adata = make_adata(n_labels=5)
    model = setup_and_train(adata)
    leaf_to_model = {
        "label_2-b1": "label_2",
        "label_3-b1": "label_3",
        "label_4-b1": "label_4",
    }
    hierarchy.set_hierarchy_from_schpl(
        model,
        _mock_schpl_tree(),
        label_map=leaf_to_model,
    )
    assert model.hierarchy_reachability_ is not None
    assert model.hierarchy_reachability_.shape == (4, 4)
    expected_edges = {
        "label_1": ["label_3", "label_4"],
        "label_2": [],
        "label_3": [],
        "label_4": [],
    }
    expected = _expected_reachability(model._observed_label_names(), expected_edges)
    np.testing.assert_array_equal(model.hierarchy_reachability_, expected)


@pytest.mark.optional
def test_learn_hierarchy_on_synthetic_latent():
    pytest.importorskip("scHPL")
    adata = make_adata(n_batches=2, n_labels=5)
    model = setup_and_train(adata)

    latent_adata = hierarchy.latent_to_anndata(model, adata)
    assert latent_adata.n_obs == adata.n_obs
    assert latent_adata.n_vars == model.module.n_latent

    latent_adata.obs[CELLTYPE_BATCH_KEY] = (
        latent_adata.obs[LABELS_KEY].astype(str) + "-" + latent_adata.obs[BATCH_KEY].astype(str)
    )
    batch_order = sorted(latent_adata.obs[BATCH_KEY].astype(str).unique())

    tree, _ = hierarchy.learn_hierarchy(
        latent_adata,
        batch_key=BATCH_KEY,
        batch_order=batch_order,
        cell_type_key=CELLTYPE_BATCH_KEY,
        classifier="knn",
        dimred=False,
        print_conf=False,
    )
    assert tree is not None
