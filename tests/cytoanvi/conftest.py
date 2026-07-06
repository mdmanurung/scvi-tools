"""Shared fixtures and helpers for the cytoanvi test suite."""

import numpy as np
import pytest

from cytoanvi import CytoANVI
from scvi.data import synthetic_iid
from scvi.external import cytovi as cytovi_pp

# ---------------------------------------------------------------------------
# Shared constants (used across all cytoanvi test files)
# ---------------------------------------------------------------------------

SCALED_LAYER_KEY = "scaled"
BATCH_KEY = "batch"
LABELS_KEY = "labels"
SAMPLE_KEY = "sample_key"
UNLABELED = "label_0"
N_EPOCHS = 2


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------


def make_adata(n_genes=30, n_batches=2, n_labels=5, batch_size=256):
    adata = synthetic_iid(
        batch_size=batch_size,
        n_genes=n_genes,
        n_proteins=0,
        n_regions=0,
        n_batches=n_batches,
        n_labels=n_labels,
        rna_dist="normal",
    )
    adata.obs[SAMPLE_KEY] = np.random.default_rng(42).choice(
        ["group_a", "group_b"], size=adata.shape[0]
    )
    adata.layers["raw"] = adata.X.copy()
    cytovi_pp.transform_arcsinh(adata)
    cytovi_pp.scale(adata)
    return adata


def setup_cytoanvi(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    return CytoANVI(adata, n_latent=10)


def setup_and_train(adata, max_epochs=N_EPOCHS):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    model.train(max_epochs=max_epochs, accelerator="cpu")
    return model


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def adata_fixture():
    return make_adata()


class MockTreeNode:
    """Minimal stand-in for scHPL TreeNode — no scHPL install required.

    ``name`` is stored as a list (matching scHPL's convention); pass a string or list.
    """

    def __init__(self, name, descendants=None):
        self.name = name if isinstance(name, (list, tuple)) else [name]
        self.descendants = list(descendants or [])
        self.ancestor = None
        for child in self.descendants:
            child.ancestor = self

    def get_leaves(self):
        if not self.descendants:
            return [self]
        leaves = []
        for child in self.descendants:
            leaves.extend(child.get_leaves())
        return leaves


class ScalarNameTreeNode(MockTreeNode):
    """Mock scHPL node variant whose ``name`` is a plain string (not a list)."""

    def __init__(self, name, descendants=None):
        self.name = name
        self.descendants = list(descendants or [])
        self.ancestor = None
        for child in self.descendants:
            child.ancestor = self
