"""scHPL hierarchy helpers tested with a minimal mock TreeNode (no scHPL install)."""

import numpy as np
import pytest

from cytoanvi import CytoANVI, hierarchy
from scvi.data import synthetic_iid
from scvi.external import cytovi as cytovi_pp

from conftest import MockTreeNode, ScalarNameTreeNode  # noqa: F401

SCALED_LAYER_KEY = "scaled"
BATCH_KEY = "batch"
LABELS_KEY = "labels"
SAMPLE_KEY = "sample_key"
UNLABELED = "label_0"


def _mock_schpl_tree():
    cd4 = MockTreeNode(["CD4"])
    cd8 = MockTreeNode(["CD8"])
    internal = MockTreeNode(["T"], [cd4, cd8])
    root = MockTreeNode("root", [internal])
    return root


def _mock_scalar_name_schpl_tree():
    cd4 = ScalarNameTreeNode("CD4")
    cd8 = ScalarNameTreeNode("CD8")
    internal = ScalarNameTreeNode("T", [cd4, cd8])
    root = ScalarNameTreeNode("root", [internal])
    return root


def _mock_schpl_tree_coarse():
    cd4 = MockTreeNode(["CD4"])
    cd8 = MockTreeNode(["CD8"])
    internal = MockTreeNode(["T cells"], [cd4, cd8])
    root = MockTreeNode("root", [internal])
    return root


def _make_adata(n_genes=30, n_batches=2, n_labels=3):
    adata = synthetic_iid(
        batch_size=256,
        n_genes=n_genes,
        n_proteins=0,
        n_regions=0,
        n_batches=n_batches,
        n_labels=n_labels,
        rna_dist="normal",
    )
    adata.obs[SAMPLE_KEY] = np.random.choice(["group_a", "group_b"], size=adata.shape[0])
    adata.layers["raw"] = adata.X.copy()
    cytovi_pp.transform_arcsinh(adata)
    cytovi_pp.scale(adata)
    return adata


def _setup_cytoanvi(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    return CytoANVI(adata, n_latent=10)


@pytest.fixture
def adata_two_labels():
    return _make_adata(n_labels=3)


def test_reachability_from_schpl_tree_mock():
    tree = _mock_schpl_tree()
    model_labels = ["label_1", "label_2"]
    leaf_to_model = {"CD4": "label_1", "CD8": "label_2"}

    reach = hierarchy.reachability_from_schpl_tree(
        tree, model_labels, leaf_to_model=leaf_to_model
    )

    assert reach.shape == (2, 2)
    # Sibling leaves under internal node "T" — neither is an ancestor of the other.
    np.testing.assert_array_equal(reach, np.eye(2, dtype=np.float32))


def test_reachability_from_schpl_tree_accepts_scalar_leaf_names():
    tree = _mock_scalar_name_schpl_tree()
    model_labels = ["label_1", "label_2"]
    leaf_to_model = {"CD4": "label_1", "CD8": "label_2"}

    reach = hierarchy.reachability_from_schpl_tree(
        tree, model_labels, leaf_to_model=leaf_to_model
    )

    np.testing.assert_array_equal(reach, np.eye(2, dtype=np.float32))


def test_reachability_coarse_internal_node():
    tree = _mock_schpl_tree_coarse()
    model_labels = ["T cells", "label_1", "label_2"]
    leaf_to_model = {"CD4": "label_1", "CD8": "label_2"}

    reach = hierarchy.reachability_from_schpl_tree(
        tree, model_labels, leaf_to_model=leaf_to_model
    )

    assert reach.shape == (3, 3)
    assert not np.array_equal(reach, np.eye(3, dtype=np.float32))
    # Coarse "T cells" reaches both fine labels.
    assert reach[0, 1] == 1.0
    assert reach[0, 2] == 1.0
    # Sibling fine labels do not reach each other.
    assert reach[1, 2] == 0.0
    assert reach[2, 1] == 0.0


N_EPOCHS = 2


def test_predict_hierarchical_leaf_only():
    adata = _make_adata(n_labels=4)
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    model.set_hierarchy({"label_1": ["label_2", "label_3"]})
    model.train(max_epochs=N_EPOCHS)

    preds = model.predict_hierarchical(leaf_only=True)
    assert set(np.unique(preds)).issubset({"label_2", "label_3"})
    assert "label_1" not in set(np.unique(preds))


def test_predict_hierarchical_default_returns_leaves():
    """predict_hierarchical() with no leaf_only arg must return leaf labels, not internal/root.

    Pins the corrected default (leaf_only=True). If the default were reverted to False the
    argmax over subtree masses would almost always pick the root, breaking this assertion.
    """
    adata = _make_adata(n_labels=4)
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    # label_1 is internal (parent of label_2, label_3); label_4 is unlabeled/root proxy.
    model.set_hierarchy({"label_1": ["label_2", "label_3"]})
    model.train(max_epochs=N_EPOCHS)

    # Call with NO leaf_only argument — should use the default (True).
    preds = model.predict_hierarchical()
    unique_preds = set(np.unique(preds))
    # With leaf_only=True the argmax is restricted to {label_2, label_3}; label_1 must not appear.
    assert "label_1" not in unique_preds, (
        f"predict_hierarchical() default returned internal node label_1; "
        f"got predictions: {unique_preds}"
    )
    assert unique_preds.issubset({"label_2", "label_3", "label_4"}), (
        f"Unexpected labels in default predict_hierarchical output: {unique_preds}"
    )


def test_set_hierarchy_from_schpl_allows_internal_nodes(adata_two_labels):
    model = _setup_cytoanvi(adata_two_labels)
    tree = _mock_schpl_tree()
    label_map = {"CD4": "label_1", "CD8": "label_2"}

    hierarchy.set_hierarchy_from_schpl(model, tree, label_map=label_map)

    assert model.hierarchy_reachability_ is not None
    assert model.module.reachability_matrix_ is not None
    np.testing.assert_array_equal(
        model.hierarchy_reachability_, np.eye(2, dtype=np.float32)
    )


def test_run_tree_arches_pipeline_learn_mock(monkeypatch):
    adata = _make_adata(n_labels=5)
    model = _setup_cytoanvi(adata)
    model.train(max_epochs=N_EPOCHS, accelerator="cpu")

    mock_tree = _mock_schpl_tree()
    batch_order = sorted(adata.obs[BATCH_KEY].astype(str).unique())
    adata.obs["celltype_batch"] = (
        adata.obs[LABELS_KEY].astype(str) + "-" + adata.obs[BATCH_KEY].astype(str)
    )

    def fake_learn_hierarchy(latent_adata, **kwargs):
        assert latent_adata.n_obs == adata.n_obs
        assert latent_adata.n_vars == model.module.n_latent
        assert kwargs["batch_key"] == BATCH_KEY
        return mock_tree, []

    monkeypatch.setattr(hierarchy, "_require_schpl", lambda: None)
    monkeypatch.setattr(hierarchy, "learn_hierarchy", fake_learn_hierarchy)

    result = hierarchy.run_tree_arches_pipeline(
        reference_model=model,
        batch_key=BATCH_KEY,
        batch_order=batch_order,
        cell_type_key="celltype_batch",
        reference_adata=adata,
        mode="learn",
    )

    assert result["tree"] is mock_tree
    assert result["missing_populations"] == []
    assert result["reference_latent"].n_obs == adata.n_obs
    assert result["reference_latent"].n_vars == model.module.n_latent
