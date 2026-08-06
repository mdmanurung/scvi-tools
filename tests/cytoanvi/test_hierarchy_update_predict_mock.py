"""Coverage for ``hierarchy``'s update/predict paths, which previously had none.

``test_hierarchy_schpl_mock.py`` covers ``run_tree_arches_pipeline`` for ``mode='learn'`` only.
The ``update`` and ``predict`` branches — including every guard clause, the documented
latent-resolution precedence, and the ``retrain=False`` contract that makes "update" an update
rather than a relearn — were untested by anything, mocked or otherwise.

Two techniques used here, both load-bearing rather than stylistic:

1. ``_run_update`` / ``_run_predict`` are called **directly** for guard-clause tests. Their
   validation fires before any scHPL import, so those cases need no mocking at all.
2. For the bodies of ``update_hierarchy`` / ``predict_schpl``, scHPL is faked via
   ``monkeypatch.setitem(sys.modules, ...)``. ``monkeypatch.setattr(hierarchy, "learn_tree", ...)``
   does **not** work: ``learn_tree`` and ``predict_labels`` are imported inside the function
   bodies (``from scHPL.learn import learn_tree``), so they are not module-level attributes of
   ``hierarchy``. Faking the modules also satisfies ``_require_schpl()``'s own ``import scHPL``
   naturally, so it does not need separate patching.
"""

import sys
import types

import numpy as np
import pytest
from anndata import AnnData
from conftest import (
    BATCH_KEY,
    LABELS_KEY,
    N_EPOCHS,
    make_adata,
    setup_cytoanvi,
)

from cytoanvi import hierarchy


class _Tree:
    """Opaque stand-in for an scHPL tree — identity is all these tests need."""


@pytest.fixture
def fake_schpl(monkeypatch):
    """Install fake ``scHPL`` / ``scHPL.learn`` / ``scHPL.predict`` modules with spies.

    Returns a dict that records the call args of ``learn_tree`` / ``predict_labels`` and lets a
    test set their return values.
    """
    spies: dict = {
        "learn_tree_return": (_Tree(), []),
        "predict_labels_return": (np.array(["a", "b"]), np.array([0.9, 0.8])),
    }

    def learn_tree(*args, **kwargs):
        spies["learn_tree_args"] = args
        spies["learn_tree_kwargs"] = kwargs
        return spies["learn_tree_return"]

    def predict_labels(*args, **kwargs):
        spies["predict_labels_args"] = args
        spies["predict_labels_kwargs"] = kwargs
        return spies["predict_labels_return"]

    pkg = types.ModuleType("scHPL")
    learn_mod = types.ModuleType("scHPL.learn")
    predict_mod = types.ModuleType("scHPL.predict")
    learn_mod.learn_tree = learn_tree
    predict_mod.predict_labels = predict_labels
    pkg.learn = learn_mod
    pkg.predict = predict_mod

    monkeypatch.setitem(sys.modules, "scHPL", pkg)
    monkeypatch.setitem(sys.modules, "scHPL.learn", learn_mod)
    monkeypatch.setitem(sys.modules, "scHPL.predict", predict_mod)
    return spies


def _latent_adata(n_obs=12, n_vars=4, batch="b0", cell_type="T"):
    ad = AnnData(X=np.random.default_rng(0).normal(size=(n_obs, n_vars)).astype(np.float32))
    ad.obs[BATCH_KEY] = batch
    ad.obs[LABELS_KEY] = cell_type
    return ad


# ---------------------------------------------------------------------------
# Guard clauses — no scHPL needed, validation fires before any import.
# ---------------------------------------------------------------------------


def test_run_update_requires_query_adata_when_only_reference_given():
    with pytest.raises(ValueError, match="query_adata is required"):
        hierarchy._run_update(
            reference_adata=_latent_adata(),
            query_adata=None,
            combined_adata=None,
            combined_latent=None,
            query_model=object(),
            batch_added=["b1"],
            batch_order=["b0", "b1"],
            batch_key=BATCH_KEY,
            schpl_cell_type_key=LABELS_KEY,
            obs_cols=None,
            tree=_Tree(),
        )


def test_run_update_requires_query_model_when_no_combined_latent():
    with pytest.raises(ValueError, match="query_model is required"):
        hierarchy._run_update(
            reference_adata=None,
            query_adata=_latent_adata(),
            combined_adata=None,
            combined_latent=None,
            query_model=None,
            batch_added=["b1"],
            batch_order=["b0", "b1"],
            batch_key=BATCH_KEY,
            schpl_cell_type_key=LABELS_KEY,
            obs_cols=None,
            tree=_Tree(),
        )


def test_run_update_requires_batch_added():
    """``combined_latent`` is supplied so the two earlier guards pass and this one is reached."""
    with pytest.raises(ValueError, match="batch_added is required"):
        hierarchy._run_update(
            reference_adata=None,
            query_adata=None,
            combined_adata=None,
            combined_latent=_latent_adata(),
            query_model=None,
            batch_added=None,
            batch_order=["b0", "b1"],
            batch_key=BATCH_KEY,
            schpl_cell_type_key=LABELS_KEY,
            obs_cols=None,
            tree=_Tree(),
        )


def test_run_update_requires_some_latent_source():
    """No combined_latent, no combined_adata, and not both reference+query -> final else branch."""
    with pytest.raises(ValueError, match="requires combined_latent, combined_adata, or both"):
        hierarchy._run_update(
            reference_adata=None,
            query_adata=None,
            combined_adata=None,
            combined_latent=None,
            query_model=object(),
            batch_added=["b1"],
            batch_order=["b0", "b1"],
            batch_key=BATCH_KEY,
            schpl_cell_type_key=LABELS_KEY,
            obs_cols=None,
            tree=_Tree(),
        )


def test_run_predict_requires_query_latent():
    with pytest.raises(ValueError, match="query_latent is required"):
        hierarchy._run_predict(query_latent=None, tree=_Tree())


# ---------------------------------------------------------------------------
# update_hierarchy / predict_schpl bodies — require the fake scHPL modules.
# ---------------------------------------------------------------------------


def test_update_hierarchy_rejects_none_tree(fake_schpl):
    with pytest.raises(ValueError, match="tree must not be None"):
        hierarchy.update_hierarchy(
            _latent_adata(),
            tree=None,
            batch_added=["b1"],
            batch_order=["b0", "b1"],
            cell_type_key=LABELS_KEY,
            batch_key=BATCH_KEY,
        )


def test_predict_schpl_rejects_none_tree(fake_schpl):
    with pytest.raises(ValueError, match="tree must not be None"):
        hierarchy.predict_schpl(_latent_adata(), tree=None)


@pytest.mark.parametrize("missing", [BATCH_KEY, LABELS_KEY])
def test_update_hierarchy_rejects_missing_columns(fake_schpl, missing):
    latent = _latent_adata()
    del latent.obs[missing]

    with pytest.raises(ValueError, match=missing):
        hierarchy.update_hierarchy(
            latent,
            tree=_Tree(),
            batch_added=["b1"],
            batch_order=["b0", "b1"],
            cell_type_key=LABELS_KEY,
            batch_key=BATCH_KEY,
        )
    assert "learn_tree_kwargs" not in fake_schpl, "learn_tree must not be reached"


def test_update_hierarchy_passes_retrain_false_and_forwards_batch_added(fake_schpl):
    """The paper-fidelity contract: update the tree, do not relearn it.

    If a refactor ever dropped ``retrain=False`` or stopped forwarding ``batch_added``, scHPL
    would silently relearn the whole hierarchy from scratch. Nothing in the suite noticed that
    before this test.
    """
    tree = _Tree()
    latent = _latent_adata()
    expected_tree, expected_missing = _Tree(), ["some_pop"]
    fake_schpl["learn_tree_return"] = (expected_tree, expected_missing)

    out_tree, missing = hierarchy.update_hierarchy(
        latent,
        tree=tree,
        batch_added=["b1"],
        batch_order=["b0", "b1"],
        cell_type_key=LABELS_KEY,
        batch_key=BATCH_KEY,
    )

    kwargs = fake_schpl["learn_tree_kwargs"]
    assert kwargs["retrain"] is False
    assert kwargs["batch_added"] == ["b1"]
    assert kwargs["tree"] is tree
    assert kwargs["batch_key"] == BATCH_KEY
    assert kwargs["cell_type_key"] == LABELS_KEY
    assert fake_schpl["learn_tree_args"][0] is latent
    # return value is passed through untouched
    assert out_tree is expected_tree
    assert missing == expected_missing


def test_predict_schpl_extracts_X_from_anndata_and_drops_probabilities(fake_schpl):
    latent = _latent_adata()
    expected = np.array(["CD4", "CD8"])
    fake_schpl["predict_labels_return"] = (expected, np.array([0.7, 0.6]))
    tree = _Tree()

    predictions = hierarchy.predict_schpl(latent, tree)

    passed = fake_schpl["predict_labels_args"][0]
    assert passed is latent.X, "AnnData must be unwrapped to .X before predict_labels"
    assert fake_schpl["predict_labels_args"][1] is tree
    np.testing.assert_array_equal(predictions, expected)


def test_predict_schpl_accepts_raw_ndarray_unchanged(fake_schpl):
    arr = np.random.default_rng(1).normal(size=(5, 3))

    hierarchy.predict_schpl(arr, _Tree())

    assert fake_schpl["predict_labels_args"][0] is arr


# ---------------------------------------------------------------------------
# run_tree_arches_pipeline dispatch beyond mode='learn'.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_model():
    adata = make_adata(n_labels=5)
    model = setup_cytoanvi(adata)
    model.train(max_epochs=N_EPOCHS, accelerator="cpu")
    return model


def test_pipeline_rejects_invalid_mode(trained_model, monkeypatch):
    monkeypatch.setattr(hierarchy, "_require_schpl", lambda: None)
    with pytest.raises(ValueError, match="mode must be one of"):
        hierarchy.run_tree_arches_pipeline(
            reference_model=trained_model,
            batch_key=BATCH_KEY,
            batch_order=["b0"],
            cell_type_key=LABELS_KEY,
            mode="nonsense",
        )


@pytest.mark.parametrize("mode", ["update", "predict"])
def test_pipeline_requires_tree_for_update_and_predict(trained_model, monkeypatch, mode):
    monkeypatch.setattr(hierarchy, "_require_schpl", lambda: None)
    with pytest.raises(ValueError, match="tree is required"):
        hierarchy.run_tree_arches_pipeline(
            reference_model=trained_model,
            batch_key=BATCH_KEY,
            batch_order=["b0"],
            cell_type_key=LABELS_KEY,
            tree=None,
            mode=mode,
        )


def test_pipeline_predict_mode_dispatches_to_predict_schpl(trained_model, monkeypatch):
    monkeypatch.setattr(hierarchy, "_require_schpl", lambda: None)
    tree = _Tree()
    latent = _latent_adata()
    expected = np.array(["X", "Y"])
    seen = {}

    def fake_predict_schpl(query_latent, tree_arg, **kwargs):
        seen["query_latent"] = query_latent
        seen["tree"] = tree_arg
        return expected

    monkeypatch.setattr(hierarchy, "predict_schpl", fake_predict_schpl)

    result = hierarchy.run_tree_arches_pipeline(
        reference_model=trained_model,
        batch_key=BATCH_KEY,
        batch_order=["b0"],
        cell_type_key=LABELS_KEY,
        query_latent=latent,
        tree=tree,
        mode="predict",
    )

    assert seen["query_latent"] is latent
    assert seen["tree"] is tree
    np.testing.assert_array_equal(result["predictions"], expected)
    assert result["tree"] is tree


def test_pipeline_update_mode_prefers_combined_latent_over_concat(trained_model, monkeypatch):
    """Pins the documented "first match wins" precedence.

    Both ``combined_latent`` and a full ``reference_adata``/``query_adata``/``query_model`` set
    are supplied. The docstring says ``combined_latent`` wins; only the fallback path had any
    coverage before, so the precedence itself was unverified.
    """
    monkeypatch.setattr(hierarchy, "_require_schpl", lambda: None)
    tree = _Tree()
    sentinel = _latent_adata()
    new_tree = _Tree()
    seen = {}

    def fake_update_hierarchy(latent_adata, **kwargs):
        seen["latent_adata"] = latent_adata
        seen["kwargs"] = kwargs
        return new_tree, []

    monkeypatch.setattr(hierarchy, "update_hierarchy", fake_update_hierarchy)

    result = hierarchy.run_tree_arches_pipeline(
        reference_model=trained_model,
        batch_key=BATCH_KEY,
        batch_order=["b0", "b1"],
        cell_type_key=LABELS_KEY,
        reference_adata=make_adata(n_labels=5),
        query_adata=make_adata(n_labels=5),
        combined_latent=sentinel,
        tree=tree,
        mode="update",
        query_model=trained_model,
        batch_added=["b1"],
    )

    assert seen["latent_adata"] is sentinel, "combined_latent must win over the concat fallback"
    assert seen["kwargs"]["batch_added"] == ["b1"]
    assert seen["kwargs"]["tree"] is tree
    assert result["tree"] is new_tree
    assert result["combined_latent"] is sentinel
