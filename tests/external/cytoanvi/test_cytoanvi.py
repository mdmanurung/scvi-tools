import os

import numpy as np
import pytest

from scvi.data import synthetic_iid
from scvi.external import CYTOVI, CytoANVI
from scvi.external import cytovi as cytovi_pp

SCALED_LAYER_KEY = "scaled"
NAN_LAYER_KEY = "_nan_mask"
BATCH_KEY = "batch"
LABELS_KEY = "labels"
SAMPLE_KEY = "sample_key"
UNLABELED = "label_0"  # use an existing label value as the unlabeled category
N_EPOCHS = 2


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


@pytest.fixture
def adata():
    return _make_adata()


def test_cytoanvi_setup_indices(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    # labeled/unlabeled split matches the unlabeled category
    assert len(model._labeled_indices) == int((adata.obs[LABELS_KEY] != UNLABELED).sum())
    assert len(model._unlabeled_indices) == int((adata.obs[LABELS_KEY] == UNLABELED).sum())
    # n_labels excludes the unlabeled category
    assert model.n_labels == adata.obs[LABELS_KEY].nunique() - 1
    assert model.module.classifier.classifier[-1].out_features == model.n_labels


def test_cytoanvi_train_predict_latent(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    model.train(max_epochs=N_EPOCHS)
    assert model.is_trained

    latent = model.get_latent_representation()
    assert latent.shape == (adata.n_obs, model.module.n_latent)

    preds = model.predict()
    assert preds.shape[0] == adata.n_obs
    # predicted labels never include the unlabeled category and are valid observed labels
    observed = set(adata.obs[LABELS_KEY].unique()) - {UNLABELED}
    assert set(np.unique(preds)).issubset(observed)

    soft = model.predict(soft=True)
    assert soft.shape == (adata.n_obs, model.n_labels)


def test_cytoanvi_from_cytovi_reproduces_latent(adata):
    CYTOVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        sample_key=SAMPLE_KEY,
    )
    cytovi_model = CYTOVI(adata, n_latent=10)
    cytovi_model.train(max_epochs=N_EPOCHS)
    cytovi_model.module.eval()
    cytovi_latent = cytovi_model.get_latent_representation()

    cytoanvi_model = CytoANVI.from_cytovi_model(
        cytovi_model, unlabeled_category=UNLABELED, labels_key=LABELS_KEY
    )
    assert cytoanvi_model.was_pretrained
    # before any fine-tuning, the shared encoder reproduces the CytoVI latent space.
    # the model isn't fit yet, so flip the trained flag and put the module in eval mode
    # (get_latent_representation does not force eval; an untrained module is still in train
    # mode, which would otherwise activate dropout / batchnorm batch-stats).
    cytoanvi_model.is_trained = True
    cytoanvi_model.module.eval()
    cytoanvi_latent = cytoanvi_model.get_latent_representation()
    assert cytoanvi_latent.shape == cytovi_latent.shape
    np.testing.assert_allclose(cytoanvi_latent, cytovi_latent, atol=1e-4)

    cytoanvi_model.train(max_epochs=1)
    assert cytoanvi_model.is_trained


def test_cytoanvi_surgery(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    ref = CytoANVI(adata, n_latent=10)
    ref.train(max_epochs=N_EPOCHS)

    query = _make_adata()
    # query cells are all unlabeled for the label-transfer scenario
    query.obs[LABELS_KEY] = UNLABELED
    q = CytoANVI.load_query_data(query, ref)
    q.train(max_epochs=1, plan_kwargs={"weight_decay": 0.0})
    preds = q.predict()
    assert preds.shape[0] == query.n_obs


def test_cytoanvi_missing_markers():
    # build an overlapping-panel object with a nan_layer (multi-panel / missing markers)
    adata1 = _make_adata(n_genes=30, n_batches=1)
    adata2 = _make_adata(n_genes=20, n_batches=1)
    adata1.obs_names = "a1_" + adata1.obs_names
    adata2.obs_names = "a2_" + adata2.obs_names
    merged = cytovi_pp.merge_batches([adata1, adata2])
    assert NAN_LAYER_KEY in merged.layers

    # labels exist on the merged object; mark a subset unlabeled
    rng = np.random.default_rng(0)
    labs = merged.obs[LABELS_KEY].astype(str).to_numpy()
    labs[rng.random(merged.n_obs) < 0.3] = UNLABELED
    merged.obs[LABELS_KEY] = labs

    CytoANVI.setup_anndata(
        merged,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        nan_layer=NAN_LAYER_KEY,
    )
    model = CytoANVI(merged)
    # classifier operates on the shared latent z1, so encoding only backbone markers is valid
    assert model.module.encoder_marker_mask is not None
    model.train(max_epochs=N_EPOCHS)
    assert model.is_trained
    preds = model.predict()
    assert preds.shape[0] == merged.n_obs


def test_cytoanvi_save_load(adata, save_path):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    model.train(max_epochs=N_EPOCHS)
    latent = model.get_latent_representation()

    model_path = os.path.join(save_path, "test_cytoanvi")
    model.save(model_path, save_anndata=True, overwrite=True)
    model2 = CytoANVI.load(model_path)
    np.testing.assert_array_equal(model2.history_["elbo_train"], model.history_["elbo_train"])
    np.testing.assert_allclose(model2.get_latent_representation(), latent, atol=1e-5)
    assert model2.n_labels == model.n_labels


def test_cytoanvi_surgery_partial_labels(adata):
    # P0: query carrying *reference* labels (plus the unlabeled category) must keep the classifier
    # head fixed at the reference n_labels and train/predict without dimension mismatch.
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    ref = CytoANVI(adata, n_latent=10)
    ref.train(max_epochs=N_EPOCHS)

    query = _make_adata()
    # keep ~half of the (reference-valid) labels, blank the rest to the unlabeled category
    rng = np.random.default_rng(1)
    labs = query.obs[LABELS_KEY].astype(str).to_numpy()
    labs[rng.random(query.n_obs) < 0.5] = UNLABELED
    query.obs[LABELS_KEY] = labs

    q = CytoANVI.load_query_data(query, ref)
    assert q.n_labels == ref.n_labels
    assert q.module.classifier.classifier[-1].out_features == ref.n_labels
    q.train(max_epochs=1, plan_kwargs={"weight_decay": 0.0})
    preds = q.predict()
    assert preds.shape[0] == query.n_obs


def test_cytoanvi_y_prior_empirical(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10, y_prior="empirical")
    yp = model.module.y_prior.detach().cpu().numpy()
    assert yp.shape == (1, model.n_labels)
    np.testing.assert_allclose(yp.sum(), 1.0, atol=1e-5)
    # empirical prior is not uniform on imbalanced-ish synthetic labels
    model.train(max_epochs=1)
    assert model.is_trained

    with pytest.raises(ValueError):
        CytoANVI(adata, n_latent=10, y_prior="not-a-mode")


def test_cytoanvi_rejects_ln_latent(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    with pytest.raises(NotImplementedError):
        CytoANVI(adata, n_latent=10, latent_distribution="ln")
