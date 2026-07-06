import os

import numpy as np
import pytest
import torch

from conftest import (
    BATCH_KEY,
    LABELS_KEY,
    N_EPOCHS,
    SAMPLE_KEY,
    SCALED_LAYER_KEY,
    UNLABELED,
    make_adata,
    setup_cytoanvi,
)

from cytoanvi import CytoANVI
from cytoanvi._hce import (
    build_reachability_matrix,
    hierarchical_cross_entropy_loss,
)


@pytest.fixture
def adata():
    return make_adata()


def _toy_dag_edges():
    """A -> B,C; B -> D; C -> E."""
    return {"A": ["B", "C"], "B": ["D"], "C": ["E"]}


def _expected_toy_reachability(label_names):
    """Reachability[i, j] == 1 iff j is reachable from i (j is i or a descendant)."""
    idx = {name: i for i, name in enumerate(label_names)}
    edges = _toy_dag_edges()
    n = len(label_names)
    reach = np.zeros((n, n), dtype=np.float32)

    def descendants(node):
        out = {node}
        for child in edges.get(node, []):
            out |= descendants(child)
        return out

    for i, name in enumerate(label_names):
        for desc in descendants(name):
            reach[i, idx[desc]] = 1.0
    return reach


def test_build_reachability_matrix_toy_dag():
    label_names = ["A", "B", "C", "D", "E"]
    matrix = build_reachability_matrix(label_names, _toy_dag_edges())
    expected = _expected_toy_reachability(label_names)
    assert matrix.shape == (5, 5)
    np.testing.assert_array_equal(matrix, expected)


def test_hierarchical_cross_entropy_loss_finite():
    label_names = ["A", "B", "C", "D", "E"]
    reachability = torch.tensor(
        build_reachability_matrix(label_names, _toy_dag_edges()), dtype=torch.float32
    )
    logits = torch.randn(16, 5)
    targets = torch.randint(0, 5, (16,))
    loss = hierarchical_cross_entropy_loss(logits, targets, reachability)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_hierarchical_cross_entropy_accepts_class_weights():
    label_names = ["A", "B", "C", "D", "E"]
    reachability = torch.tensor(
        build_reachability_matrix(label_names, _toy_dag_edges()), dtype=torch.float32
    )
    logits = torch.randn(16, 5)
    targets = torch.randint(0, 5, (16,))

    unweighted = hierarchical_cross_entropy_loss(logits, targets, reachability)
    weighted = hierarchical_cross_entropy_loss(
        logits, targets, reachability, weight=torch.tensor([1.0, 2.0, 1.0, 1.0, 3.0])
    )

    assert torch.isfinite(weighted)
    assert not torch.allclose(unweighted, weighted)


def test_reachability_matmul_gives_subtree_mass():
    """(probs @ R.T)[i] == sum of probs for i and its descendants (subtree mass).

    This test pins the ``R.T`` convention: R[i,j]=1 iff j is a descendant-or-self of i,
    so (probs @ R.T)[b,i] = sum_j probs[b,j] * R[i,j] = subtree mass of node i.
    A future accidental flip to probs @ R would give ancestor mass instead and this test
    catches it.
    """
    # DAG: A -> B,C; B -> D; C -> E
    # Leaves are D (idx=3) and E (idx=4).
    label_names = ["A", "B", "C", "D", "E"]
    R = torch.tensor(
        build_reachability_matrix(label_names, _toy_dag_edges()), dtype=torch.float32
    )
    # Assign all probability mass to the leaves D and E uniformly.
    probs = torch.zeros(1, 5)
    probs[0, 3] = 0.6  # D
    probs[0, 4] = 0.4  # E

    subtree_mass = torch.matmul(probs, R.T)  # shape (1, 5)

    # Node A (idx=0) has descendants {A,B,C,D,E} — all mass sums to 1.0.
    assert torch.isclose(subtree_mass[0, 0], torch.tensor(1.0)), (
        f"A subtree mass should be 1.0 (got {subtree_mass[0,0]:.4f})"
    )
    # Node B (idx=1) has descendants {B,D} — should hold D's mass 0.6.
    assert torch.isclose(subtree_mass[0, 1], torch.tensor(0.6)), (
        f"B subtree mass should be 0.6 (got {subtree_mass[0,1]:.4f})"
    )
    # Node C (idx=2) has descendants {C,E} — should hold E's mass 0.4.
    assert torch.isclose(subtree_mass[0, 2], torch.tensor(0.4)), (
        f"C subtree mass should be 0.4 (got {subtree_mass[0,2]:.4f})"
    )
    # Leaf D (idx=3): subtree = {D} only → 0.6.
    assert torch.isclose(subtree_mass[0, 3], torch.tensor(0.6))
    # Leaf E (idx=4): subtree = {E} only → 0.4.
    assert torch.isclose(subtree_mass[0, 4], torch.tensor(0.4))


def _synthetic_hierarchy_edges():
    """Hierarchy over synthetic_iid observed labels (label_1 .. label_4)."""
    return {
        "label_1": ["label_2", "label_3"],
        "label_2": ["label_4"],
        "label_3": [],
    }


def test_cytoanvi_set_hierarchy_train_uses_hce(adata):
    model = setup_cytoanvi(adata)
    model.set_hierarchy(_synthetic_hierarchy_edges())
    assert model.hierarchy_reachability_ is not None
    assert model.module.reachability_matrix_ is not None
    model.train(max_epochs=N_EPOCHS)
    assert model.is_trained
    assert model.module.reachability_matrix_ is not None


def test_predict_hierarchical_raises_without_hierarchy(adata):
    model = setup_cytoanvi(adata)
    model.train(max_epochs=N_EPOCHS)
    with pytest.raises(ValueError, match="hierarchy"):
        model.predict_hierarchical()


def test_predict_hierarchical_soft_returns_hierarchy_adjusted_scores(adata):
    model = setup_cytoanvi(adata)
    model.set_hierarchy(_synthetic_hierarchy_edges())
    model.train(max_epochs=N_EPOCHS)

    scores = model.predict_hierarchical(soft=True)
    label_names = model._observed_label_names()
    assert list(scores.columns) == label_names
    assert scores.shape == (adata.n_obs, len(label_names))
    values = np.asarray(scores.values)
    assert np.all(np.isfinite(values))
    assert (values >= 0).all()
    assert np.any(values.sum(axis=1) > 1.0)


def test_set_hierarchy_raises_on_label_mismatch(adata):
    model = setup_cytoanvi(adata)
    bad_edges = {"not_a_real_label": ["also_fake"]}
    with pytest.raises(ValueError):
        model.set_hierarchy(bad_edges)


def test_set_hierarchy_twice(adata):
    model = setup_cytoanvi(adata)
    edges = _synthetic_hierarchy_edges()
    model.set_hierarchy(edges)
    reach_first = model.hierarchy_reachability_.copy()
    model.set_hierarchy(edges)
    np.testing.assert_array_equal(model.hierarchy_reachability_, reach_first)
    assert model.module.reachability_matrix_ is not None


def test_cytoanvi_hierarchy_save_load(adata, tmp_path):
    model = setup_cytoanvi(adata)
    model.set_hierarchy(_synthetic_hierarchy_edges())
    model.train(max_epochs=N_EPOCHS)
    reach_before = model.hierarchy_reachability_.copy()

    model_path = os.path.join(tmp_path, "test_cytoanvi_hierarchy")
    model.save(model_path, save_anndata=True, overwrite=True)
    model2 = CytoANVI.load(model_path)

    assert model2.hierarchy_reachability_ is not None
    np.testing.assert_array_equal(model2.hierarchy_reachability_, reach_before)
    assert model2.module.reachability_matrix_ is not None
    np.testing.assert_allclose(
        model2.module.reachability_matrix_.cpu().numpy(),
        reach_before,
    )


def test_set_hierarchy_twice_save_load(adata, tmp_path):
    """Save/load round-trip after calling set_hierarchy more than once."""
    model = setup_cytoanvi(adata)
    edges = _synthetic_hierarchy_edges()
    model.set_hierarchy(edges)
    model.set_hierarchy(edges)
    model.train(max_epochs=N_EPOCHS)
    reach_before = model.hierarchy_reachability_.copy()

    model_path = os.path.join(tmp_path, "test_cytoanvi_hierarchy_twice")
    model.save(model_path, save_anndata=True, overwrite=True)
    model2 = CytoANVI.load(model_path)

    assert model2.hierarchy_reachability_ is not None
    np.testing.assert_array_equal(model2.hierarchy_reachability_, reach_before)
    assert model2.module.reachability_matrix_ is not None
    np.testing.assert_allclose(
        model2.module.reachability_matrix_.cpu().numpy(),
        reach_before,
    )
