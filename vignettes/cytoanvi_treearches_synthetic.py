"""Tracked executable treeArches examples used by docs, source tests, and wheel acceptance.

The fake scHPL backend exercises CytoANVI's orchestration contract without treating a synthetic
fixture as scientific validation. Real hierarchy learning still requires the pinned study protocol
and the optional ``cytoanvi-hierarchy`` dependency. See ``docs/usage_readiness.md`` for the
authoritative capability boundary.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import anndata
import numpy as np

import scvi
from cytoanvi import CytoANVI, hierarchy
from scvi.external import cytovi as cytovi_pp

LAYER = "scaled"
BATCH_KEY = "batch"
LABELS_KEY = "labels"
SAMPLE_KEY = "sample"
STUDY_KEY = "study"
TREE_LABEL_KEY = "cell_type"
UNLABELED = "label_0"


class _SyntheticTree:
    """Opaque tree token for the engineering-only fake scHPL backend."""

    def __init__(self, revision: int):
        self.revision = revision


@contextmanager
def _fake_schpl():
    """Install a deterministic in-process scHPL stand-in and restore module state afterward."""
    previous = {name: sys.modules.get(name) for name in ("scHPL", "scHPL.learn", "scHPL.predict")}

    def learn_tree(adata, *, tree=None, retrain=True, **kwargs):
        del kwargs
        revision = 1 if tree is None else tree.revision + 1
        if tree is not None and retrain is not False:
            raise AssertionError("treeArches update must forward retrain=False")
        return _SyntheticTree(revision=revision), []

    def predict_labels(values, tree, **kwargs):
        del kwargs
        values = np.asarray(values)
        labels = np.repeat(f"synthetic-tree-r{tree.revision}", values.shape[0])
        probabilities = np.ones(values.shape[0], dtype=np.float32)
        return labels, probabilities

    package = types.ModuleType("scHPL")
    learn_module = types.ModuleType("scHPL.learn")
    predict_module = types.ModuleType("scHPL.predict")
    learn_module.learn_tree = learn_tree
    predict_module.predict_labels = predict_labels
    package.learn = learn_module
    package.predict = predict_module
    sys.modules.update(
        {"scHPL": package, "scHPL.learn": learn_module, "scHPL.predict": predict_module}
    )
    try:
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


def make_synthetic_cytometry(seed: int, study: str, n_cells: int = 64):
    """Create a bounded same-panel engineering fixture."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        adata = scvi.data.synthetic_iid(
            batch_size=n_cells,
            n_genes=16,
            n_proteins=0,
            n_regions=0,
            n_batches=2,
            n_labels=4,
            rna_dist="normal",
        )
    finally:
        np.random.set_state(state)
    rng = np.random.default_rng(seed)
    adata.obs[SAMPLE_KEY] = rng.choice([f"{study}-s0", f"{study}-s1"], size=adata.n_obs)
    adata.obs[STUDY_KEY] = study
    adata.obs[TREE_LABEL_KEY] = adata.obs[LABELS_KEY].astype(str).to_numpy()
    adata.layers["raw"] = adata.X.copy()
    cytovi_pp.transform_arcsinh(adata)
    cytovi_pp.scale(adata)
    return adata


def train_reference(max_epochs: int = 1):
    """Train the small reference used by both advertised paths."""
    reference = make_synthetic_cytometry(seed=0, study="reference")
    CytoANVI.setup_anndata(
        reference,
        layer=LAYER,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(reference, n_latent=4)
    model.train(max_epochs=max_epochs, accelerator="cpu", enable_progress_bar=False)
    return reference, model


def run_direct_same_panel(max_epochs: int = 1) -> dict:
    """Run direct same-panel reference/query surgery without panel-padding preparation."""
    reference, reference_model = train_reference(max_epochs=max_epochs)
    query = make_synthetic_cytometry(seed=1, study="query")
    query.obs[LABELS_KEY] = UNLABELED
    query_model = CytoANVI.load_query_data(query, reference_model)
    query_model.train(max_epochs=max_epochs, accelerator="cpu", enable_progress_bar=False)
    predictions = np.asarray(query_model.predict())
    latent = query_model.get_latent_representation()
    if predictions.shape != (query.n_obs,) or latent.shape != (query.n_obs, 4):
        raise AssertionError("Direct same-panel surgery returned an unexpected schema")
    return {
        "reference": reference,
        "reference_model": reference_model,
        "query": query,
        "query_model": query_model,
        "predictions": predictions,
        "latent": latent,
    }


def run_one_shot_learn_update_predict(max_epochs: int = 1) -> dict:
    """Run the learn, update, and predict dispatcher branches on derived latent columns."""
    direct = run_direct_same_panel(max_epochs=max_epochs)
    reference = direct["reference"]
    reference_model = direct["reference_model"]
    query = direct["query"]
    query_model = direct["query_model"]
    obs_cols = [STUDY_KEY, TREE_LABEL_KEY]

    with _fake_schpl():
        learned = hierarchy.run_tree_arches_pipeline(
            reference_model=reference_model,
            reference_adata=reference,
            batch_key=STUDY_KEY,
            batch_order=["reference"],
            cell_type_key=TREE_LABEL_KEY,
            mode="learn",
            obs_cols=obs_cols,
        )
        query_latent = hierarchy.latent_to_anndata(query_model, query, obs_cols=obs_cols)
        combined_latent = anndata.concat(
            [learned["reference_latent"], query_latent], join="outer", index_unique="-"
        )
        updated = hierarchy.run_tree_arches_pipeline(
            reference_model=reference_model,
            combined_latent=combined_latent,
            batch_key=STUDY_KEY,
            batch_order=["reference", "query"],
            cell_type_key=TREE_LABEL_KEY,
            tree=learned["tree"],
            mode="update",
            batch_added=["query"],
        )
        predicted = hierarchy.run_tree_arches_pipeline(
            reference_model=reference_model,
            query_latent=query_latent,
            batch_key=STUDY_KEY,
            batch_order=["reference", "query"],
            cell_type_key=TREE_LABEL_KEY,
            tree=updated["tree"],
            mode="predict",
        )

    predictions = np.asarray(predicted["predictions"])
    if predictions.shape != (query.n_obs,):
        raise AssertionError("One-shot treeArches prediction returned an unexpected schema")
    return {
        **direct,
        "learned_tree": learned["tree"],
        "updated_tree": updated["tree"],
        "tree_predictions": predictions,
    }


if __name__ == "__main__":
    result = run_one_shot_learn_update_predict()
    print(
        f"treeArches synthetic acceptance: {result['tree_predictions'].shape[0]} query cells, "
        f"tree revision {result['updated_tree'].revision}"
    )
