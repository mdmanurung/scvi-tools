import numpy as np
import pytest

from cytoanvi import CytoANVI, hierarchy
from cytoanvi._hce import build_reachability_matrix
from scvi.data import synthetic_iid
from scvi.external import cytovi as cytovi_pp

SCALED_LAYER_KEY = "scaled"
BATCH_KEY = "batch"
LABELS_KEY = "labels"
SAMPLE_KEY = "sample_key"
UNLABELED = "label_0"
N_EPOCHS = 2
CELLTYPE_BATCH_KEY = "celltype_batch"


class _MockTreeNode:
    """Minimal scHPL/newick TreeNode stand-in for hierarchy unit tests."""

    def __init__(
        self,
        name: str | list[str],
        descendants: list["_MockTreeNode"] | None = None,
        ancestor: "_MockTreeNode | None" = None,
    ):
        self.name = [name] if isinstance(name, str) else list(name)
        self.descendants = descendants or []
        self.ancestor = ancestor
        for child in self.descendants:
            child.ancestor = self

    def get_leaves(self):
        if not self.descendants:
            return [self]
        leaves = []
        for child in self.descendants:
            leaves.extend(child.get_leaves())
        return leaves


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
    adata = _make_adata(n_labels=5)
    model = _setup_and_train(adata)
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


def _make_adata(n_genes=30, n_batches=2, n_labels=5):
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


def _setup_and_train(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    model.train(max_epochs=N_EPOCHS, accelerator="cpu")
    return model


@pytest.mark.optional
def test_learn_hierarchy_on_synthetic_latent():
    pytest.importorskip("scHPL")
    adata = _make_adata(n_batches=2, n_labels=5)
    model = _setup_and_train(adata)

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
