import os
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
)

from cytoanvi import CytoANVI
from scvi.data._constants import _SETUP_ARGS_KEY
from scvi.external import CYTOVI
from scvi.external import cytovi as cytovi_pp

NAN_LAYER_KEY = "_nan_mask"


@pytest.fixture
def adata():
    return make_adata()


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


def test_cytoanvi_loss_scale_clamps_tiny_variance():
    from cytoanvi._module import _stable_normal_scale

    variance = torch.tensor([1.0, 0.0, 1e-12], dtype=torch.float32)

    scale = _stable_normal_scale(variance, "pz1_v")

    assert torch.all(torch.isfinite(scale))
    torch.testing.assert_close(scale, torch.tensor([1.0, 1e-3, 1e-3]))


@pytest.mark.parametrize(
    ("variance", "message"),
    [
        (torch.tensor([1.0, float("nan")]), "pz1_v.*non-finite.*1"),
        (torch.tensor([1.0, float("inf")]), "pz1_v.*non-finite.*1"),
        (torch.tensor([1.0, -0.25]), "pz1_v.*negative.*1"),
    ],
)
def test_cytoanvi_loss_scale_rejects_invalid_variance(variance, message):
    from cytoanvi._module import _stable_normal_scale

    with pytest.raises(ValueError, match=message):
        _stable_normal_scale(variance, "pz1_v")


def _decoder_pair(seed: int = 0):
    """A bounded and a stock Decoder sharing identical weights."""
    from scvi.nn import Decoder

    from cytoanvi._module import BoundedVarianceDecoder

    torch.manual_seed(seed)
    bounded = BoundedVarianceDecoder(4, 3, n_layers=1, n_hidden=8)
    torch.manual_seed(seed)
    stock = Decoder(4, 3, n_layers=1, n_hidden=8)
    stock.load_state_dict(bounded.state_dict())
    return bounded, stock


def test_bounded_variance_decoder_keeps_decoder_state_dict_keys():
    """The bound must not change checkpoint layout, or saved models stop loading."""
    bounded, stock = _decoder_pair()

    assert set(bounded.state_dict()) == set(stock.state_dict())


def test_bounded_variance_decoder_matches_decoder_below_the_bound():
    """Below the bound the subclass must be the stock decoder, not an approximation."""
    bounded, stock = _decoder_pair()
    x = torch.randn(6, 4)

    b_m, b_v = bounded(x)
    s_m, s_v = stock(x)

    torch.testing.assert_close(b_m, s_m)
    torch.testing.assert_close(b_v, s_v)
    assert torch.all(torch.isfinite(b_v))


def test_bounded_variance_decoder_gradient_is_finite_on_overflow():
    """The regression test for the defect a09a9e28 introduced.

    Clamping an already-overflowed variance saturates the forward value but makes the
    backward ``0 * inf = nan``. Bounding the pre-activation must leave the gradient finite.
    The stock decoder is asserted to overflow on the same input, so this pins the fix rather
    than restating an invariant that already held.
    """
    bounded, stock = _decoder_pair()
    with torch.no_grad():
        # Drive the variance pre-activation far past the float32 exp overflow line (88.7228).
        bounded.var_decoder.bias.fill_(100.0)
        stock.var_decoder.bias.fill_(100.0)
    x = torch.randn(6, 4)

    _, stock_v = stock(x)
    assert torch.isinf(stock_v).any(), "stock Decoder must overflow, or this test proves nothing"

    _, bounded_v = bounded(x)
    assert torch.all(torch.isfinite(bounded_v))

    bounded_v.sum().backward()
    for name, parameter in bounded.named_parameters():
        assert parameter.grad is None or torch.all(torch.isfinite(parameter.grad)), (
            f"non-finite gradient on {name} after a saturated variance pre-activation"
        )


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
    assert cytoanvi_model.n_labels == adata.obs[LABELS_KEY].nunique() - 1
    assert cytoanvi_model.module.classifier.classifier[-1].out_features == cytoanvi_model.n_labels

    cytovi_state = cytovi_model.module.state_dict()
    cytoanvi_state = cytoanvi_model.module.state_dict()
    for key in (
        "z_encoder.encoder.fc_layers.Layer 0.0.weight",
        "decoder.px_decoder.fc_layers.Layer 0.0.weight",
    ):
        assert key in cytovi_state
        assert key in cytoanvi_state
        torch.testing.assert_close(cytoanvi_state[key].cpu(), cytovi_state[key].cpu())
        assert cytoanvi_state[key].data_ptr() != cytovi_state[key].data_ptr()

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
    assert UNLABELED not in set(cytoanvi_model.predict())


def test_cytoanvi_from_cytovi_model_uses_cytovi_labels_key_by_default(adata):
    CYTOVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        sample_key=SAMPLE_KEY,
    )
    cytovi_model = CYTOVI(adata, n_latent=10)
    cytovi_model.train(max_epochs=1)

    cytoanvi_model = CytoANVI.from_cytovi_model(
        cytovi_model, unlabeled_category=UNLABELED, labels_key=None
    )

    assert cytoanvi_model.adata_manager.registry[_SETUP_ARGS_KEY]["labels_key"] == LABELS_KEY


def test_cytoanvi_from_cytovi_model_requires_labels_key_without_cytovi_labels(adata):
    CYTOVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        sample_key=SAMPLE_KEY,
    )
    cytovi_model = CYTOVI(adata, n_latent=10)
    cytovi_model.train(max_epochs=1)

    with pytest.raises(ValueError, match="labels_key"):
        CytoANVI.from_cytovi_model(cytovi_model, unlabeled_category=UNLABELED, labels_key=None)


def test_cytoanvi_from_cytovi_model_rejects_mismatched_labels_key(adata):
    adata.obs["other_labels"] = adata.obs[LABELS_KEY].astype(str).to_numpy()
    CYTOVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        sample_key=SAMPLE_KEY,
    )
    cytovi_model = CYTOVI(adata, n_latent=10)
    cytovi_model.train(max_epochs=1)

    with pytest.raises(ValueError, match="different labels_key"):
        CytoANVI.from_cytovi_model(
            cytovi_model, unlabeled_category=UNLABELED, labels_key="other_labels"
        )


def test_cytoanvi_from_cytovi_model_accepts_compatible_adata(adata):
    CYTOVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        sample_key=SAMPLE_KEY,
    )
    cytovi_model = CYTOVI(adata, n_latent=10)
    cytovi_model.train(max_epochs=1)
    compatible = adata.copy()

    cytoanvi_model = CytoANVI.from_cytovi_model(
        cytovi_model,
        unlabeled_category=UNLABELED,
        labels_key=LABELS_KEY,
        adata=compatible,
    )

    assert cytoanvi_model.adata is compatible
    assert cytoanvi_model.was_pretrained


def test_cytoanvi_from_cytovi_model_preserves_setup_args(adata):
    CYTOVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        sample_key=SAMPLE_KEY,
    )
    cytovi_model = CYTOVI(adata, n_latent=10)
    cytovi_model.train(max_epochs=1)
    original_non_kwargs = dict(cytovi_model.init_params_["non_kwargs"])
    original_kwargs = deepcopy(cytovi_model.init_params_["kwargs"])
    original_setup_args = deepcopy(cytovi_model.adata_manager.registry[_SETUP_ARGS_KEY])

    CytoANVI.from_cytovi_model(
        cytovi_model, unlabeled_category=UNLABELED, labels_key=LABELS_KEY
    )

    assert cytovi_model.init_params_["non_kwargs"] == original_non_kwargs
    assert cytovi_model.init_params_["kwargs"] == original_kwargs
    assert cytovi_model.adata_manager.registry[_SETUP_ARGS_KEY] == original_setup_args


def test_cytoanvi_from_cytovi_model_rejects_ln_latent_before_setup(adata):
    CYTOVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        sample_key=SAMPLE_KEY,
    )
    cytovi_model = CYTOVI(adata, n_latent=10, latent_distribution="ln")
    cytovi_model.train(max_epochs=1)
    original_setup_args = deepcopy(cytovi_model.adata_manager.registry[_SETUP_ARGS_KEY])

    with pytest.raises(NotImplementedError, match="latent_distribution='normal'"):
        CytoANVI.from_cytovi_model(
            cytovi_model, unlabeled_category=UNLABELED, labels_key=LABELS_KEY
        )

    assert cytovi_model.adata_manager.registry[_SETUP_ARGS_KEY] == original_setup_args


def test_cytoanvi_from_cytovi_model_query_save_roundtrip(adata, save_path):
    CYTOVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        sample_key=SAMPLE_KEY,
    )
    cytovi_model = CYTOVI(adata, n_latent=10)
    cytovi_model.train(max_epochs=1)
    ref = CytoANVI.from_cytovi_model(
        cytovi_model, unlabeled_category=UNLABELED, labels_key=LABELS_KEY
    )
    ref.train(max_epochs=1)

    path = os.path.join(save_path, "warmstart_ref")
    ref.save(path, overwrite=True, save_anndata=True)
    reloaded = CytoANVI.load(path)
    assert reloaded.n_labels == ref.n_labels

    query = make_adata()
    query.obs[LABELS_KEY] = UNLABELED
    q_from_model = CytoANVI.load_query_data(query.copy(), ref)
    q_from_model.train(max_epochs=1, plan_kwargs={"weight_decay": 0.0})
    assert q_from_model.predict().shape == (query.n_obs,)

    q_from_path = CytoANVI.load_query_data(query.copy(), path)
    q_from_path.train(max_epochs=1, plan_kwargs={"weight_decay": 0.0})
    assert q_from_path.predict().shape == (query.n_obs,)


def test_cytoanvi_from_cytovi_model_multipanel_prepare_query_preserves_nan_layer(save_path):
    a1 = make_adata(n_genes=30, n_batches=1)
    a2 = make_adata(n_genes=20, n_batches=1)
    a1.obs_names = "a1_" + a1.obs_names
    a2.obs_names = "a2_" + a2.obs_names
    merged = cytovi_pp.merge_batches([a1, a2])
    assert NAN_LAYER_KEY in merged.layers

    CYTOVI.setup_anndata(
        merged,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        nan_layer=NAN_LAYER_KEY,
    )
    cytovi_model = CYTOVI(merged, n_latent=10)
    cytovi_model.train(max_epochs=1)
    ref = CytoANVI.from_cytovi_model(
        cytovi_model, unlabeled_category=UNLABELED, labels_key=LABELS_KEY
    )
    ref.train(max_epochs=1)
    path = os.path.join(save_path, "warmstart_multipanel_ref")
    ref.save(path, overwrite=True, save_anndata=True)

    backbone = list(ref.adata.var_names[ref.encoder_marker_mask_])
    query = make_adata()[:, backbone].copy()
    query.obs[LABELS_KEY] = UNLABELED
    CytoANVI.prepare_query_anndata(query, path)

    assert list(query.var_names) == list(ref.adata.var_names)
    assert NAN_LAYER_KEY in query.layers


def test_cytoanvi_query_missing_labels_column_becomes_unlabeled(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    ref = CytoANVI(adata, n_latent=10)
    ref.train(max_epochs=1)
    query = make_adata()
    del query.obs[LABELS_KEY]

    q = CytoANVI.load_query_data(query, ref)
    q.train(max_epochs=1, plan_kwargs={"weight_decay": 0.0})
    assert q.predict().shape == (query.n_obs,)
    assert (q.adata.obs[LABELS_KEY] == UNLABELED).all()


def test_cytoanvi_query_known_partial_labels_trains(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    ref = CytoANVI(adata, n_latent=10)
    ref.train(max_epochs=1)
    query = make_adata()
    query.obs[LABELS_KEY] = query.obs[LABELS_KEY].astype(str)
    query.obs.loc[query.obs.index[: query.n_obs // 2], LABELS_KEY] = UNLABELED

    q = CytoANVI.load_query_data(query, ref)
    q.train(max_epochs=1, plan_kwargs={"weight_decay": 0.0})
    assert q.predict().shape == (query.n_obs,)


def test_cytoanvi_query_new_unseen_label_raises(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    ref = CytoANVI(adata, n_latent=10)
    ref.train(max_epochs=1)
    query = make_adata()
    query.obs[LABELS_KEY] = UNLABELED
    query.obs.iloc[0, query.obs.columns.get_loc(LABELS_KEY)] = "new_unseen_label"

    with pytest.raises(ValueError, match="new_unseen_label|Category"):
        CytoANVI.load_query_data(query, ref)


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

    query = make_adata()
    # query cells are all unlabeled for the label-transfer scenario
    query.obs[LABELS_KEY] = UNLABELED
    q = CytoANVI.load_query_data(query, ref)
    q.train(max_epochs=1, plan_kwargs={"weight_decay": 0.0})
    preds = q.predict()
    assert preds.shape[0] == query.n_obs


def test_cytoanvi_missing_markers():
    # build an overlapping-panel object with a nan_layer (multi-panel / missing markers)
    adata1 = make_adata(n_genes=30, n_batches=1)
    adata2 = make_adata(n_genes=20, n_batches=1)
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


def test_cytoanvi_save_load_cpu_inference(adata, save_path):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    model.train(max_epochs=1)

    model_path = os.path.join(save_path, "test_cytoanvi_cpu_load")
    model.save(model_path, save_anndata=True, overwrite=True)
    loaded = CytoANVI.load(model_path, accelerator="cpu", device="cpu")

    preds = loaded.predict()
    latent = loaded.get_latent_representation()
    assert preds.shape == (adata.n_obs,)
    assert latent.shape == (adata.n_obs, loaded.module.n_latent)


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

    query = make_adata()
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


def test_cytoanvi_class_weighting_default_none(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )

    model = CytoANVI(adata, n_latent=10)

    assert model.class_weighting_ == "none"
    assert model.class_weight_clip_ is None
    assert model.class_weights_ is None
    assert model.module.class_weights is None


def test_cytoanvi_sqrt_inverse_frequency_class_weights(adata):
    labels = np.asarray(adata.obs[LABELS_KEY].astype(str), dtype=object)
    labels[:] = UNLABELED
    labels[:30] = "label_1"
    labels[30:40] = "label_2"
    labels[40:45] = "label_3"
    labels[45:47] = "label_4"
    adata.obs[LABELS_KEY] = labels
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )

    model = CytoANVI(
        adata,
        n_latent=10,
        class_weighting="sqrt_inverse_frequency",
        class_weight_clip=10.0,
    )

    weights = model.module.class_weights.detach().cpu().numpy()
    assert model.class_weighting_ == "sqrt_inverse_frequency"
    assert model.class_weight_clip_ == 10.0
    assert weights.shape == (model.n_labels,)
    np.testing.assert_allclose(weights.mean(), 1.0, atol=1e-6)
    weights_by_label = dict(zip(model._observed_label_names(), weights, strict=True))
    assert (
        weights_by_label["label_1"]
        < weights_by_label["label_2"]
        < weights_by_label["label_3"]
        < weights_by_label["label_4"]
    )


def test_cytoanvi_class_weights_save_load(adata, tmp_path):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(
        adata,
        n_latent=10,
        class_weighting="sqrt_inverse_frequency",
        class_weight_clip=10.0,
    )
    weights = model.module.class_weights.detach().cpu().clone()
    model.train(max_epochs=1)
    model_path = os.path.join(tmp_path, "test_cytoanvi_class_weights")
    model.save(model_path, overwrite=True)

    loaded = CytoANVI.load(model_path, adata=adata)

    np.testing.assert_allclose(loaded.class_weights_, weights.numpy())
    torch.testing.assert_close(loaded.module.class_weights.detach().cpu(), weights)


def test_cytoanvi_class_weights_preserved_in_query_surgery(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    ref = CytoANVI(
        adata,
        n_latent=10,
        class_weighting="sqrt_inverse_frequency",
        class_weight_clip=10.0,
    )
    ref_weights = ref.module.class_weights.detach().cpu().clone()

    query = make_adata()
    query.obs[LABELS_KEY] = UNLABELED
    q = CytoANVI.load_query_data(query, ref)

    torch.testing.assert_close(q.module.class_weights.detach().cpu(), ref_weights)


def test_cytoanvi_y_prior_tensor_validation(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10, y_prior=torch.full((1, 4), 0.25))
    torch.testing.assert_close(model.module.y_prior.cpu(), torch.full((1, 4), 0.25))

    with pytest.raises(ValueError, match="shape"):
        CytoANVI(adata, n_latent=10, y_prior=torch.ones(adata.n_obs))
    with pytest.raises(ValueError, match="non-negative"):
        CytoANVI(adata, n_latent=10, y_prior=torch.tensor([[1.0, -0.1, 0.1, 0.0]]))
    with pytest.raises(ValueError, match="sum"):
        CytoANVI(adata, n_latent=10, y_prior=torch.zeros(1, 4))


def test_cytoanvi_y_prior_rejects_non_finite_tensor(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )

    with pytest.raises(ValueError, match="finite"):
        CytoANVI(adata, n_latent=10, y_prior=torch.tensor([[0.25, 0.25, float("nan"), 0.25]]))


@pytest.mark.parametrize("class_weight_clip", [0.0, -1.0, float("inf"), float("nan")])
def test_cytoanvi_class_weight_clip_validation(adata, class_weight_clip):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )

    with pytest.raises(ValueError, match="class_weight_clip"):
        CytoANVI(
            adata,
            n_latent=10,
            class_weighting="inverse_frequency",
            class_weight_clip=class_weight_clip,
        )


def test_cytoanvi_class_weight_tensor_validation(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )

    model = CytoANVI(adata, n_latent=10, class_weighting=torch.ones(4))
    torch.testing.assert_close(model.module.class_weights.cpu(), torch.ones(4))

    with pytest.raises(ValueError, match="shape"):
        CytoANVI(adata, n_latent=10, class_weighting=torch.ones(1, 4))
    with pytest.raises(ValueError, match="finite"):
        CytoANVI(
            adata,
            n_latent=10,
            class_weighting=torch.tensor([1.0, 1.0, float("inf"), 1.0]),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        CytoANVI(
            adata,
            n_latent=10,
            class_weighting=torch.tensor([1.0, 0.0, 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="class_weighting"):
        CytoANVI(adata, n_latent=10, class_weighting="balanced")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cytoanvi_y_prior_tensor_validation_cuda(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(
        adata,
        n_latent=10,
        y_prior=torch.full((1, 4), 0.25, device="cuda"),
    )
    assert model.module.y_prior.device.type == "cuda"
    torch.testing.assert_close(
        model.module.y_prior.cpu(),
        torch.full((1, 4), 0.25),
    )


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


def test_cytoanvi_get_uncertainty(adata):
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
    unc = model.get_uncertainty(tta_rep=3)
    assert unc.shape == (adata.n_obs,)
    assert np.all(np.isfinite(unc))


def test_cytoanvi_get_uncertainty_rejects_invalid_mode(adata):
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

    with pytest.raises(ValueError, match="mode must be"):
        model.get_uncertainty(tta_rep=3, mode="probability")


def test_cytoanvi_continual_update(adata):
    # reference
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

    replay = adata[:128].copy()  # buffer of reference cells
    control = make_adata()  # healthy controls sharing the panel
    query = make_adata()
    query.obs[LABELS_KEY] = UNLABELED

    q = CytoANVI.load_query_data_with_replay(
        query, ref, replay_adata=replay, control_adata=control
    )
    cont = q.module.continual
    assert cont is not None
    assert cont.old_params is not None
    assert cont.importances is not None
    assert cont.ctrl_importances is not None  # required (F_reference o F_query_ctrl)
    assert cont.combine_type == "product"  # paper default
    assert cont.replay_batches  # experience-replay buffer stored

    q.train(max_epochs=1, plan_kwargs={"ewc_importance": 1.0})
    assert q.is_trained
    preds = q.predict()
    assert preds.shape[0] == query.n_obs

    # control_adata is required (paper EWC term is F_reference o F_query_ctrl)
    with pytest.raises(ValueError):
        CytoANVI.load_query_data_with_replay(make_adata(), ref, replay_adata=replay)

    # additive combine is also available
    q2 = CytoANVI.load_query_data_with_replay(
        make_adata(),
        ref,
        replay_adata=replay,
        control_adata=make_adata(),
        combine_type="additive",
    )
    q2.train(max_epochs=1, plan_kwargs={"ewc_importance": 0.5})
    assert q2.is_trained


def test_cytoanvi_continual_new_batch(adata):
    # case-control scenario: the query comes from *new* batches, so surgery extends the batch
    # categories and resizes batch-dependent params. The EWC penalty must skip resized params
    # (size guard) rather than crash on a shape mismatch.
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

    query = make_adata()
    # relabel to brand-new batch categories not seen in the reference
    remap = {"batch_0": "batch_2", "batch_1": "batch_3"}
    query.obs[BATCH_KEY] = query.obs[BATCH_KEY].map(remap).astype(str)
    query.obs[LABELS_KEY] = UNLABELED

    q = CytoANVI.load_query_data_with_replay(
        query, ref, replay_adata=adata[:128].copy(), control_adata=make_adata()
    )
    assert q.summary_stats.n_batch > ref.summary_stats.n_batch
    q.train(max_epochs=1, plan_kwargs={"ewc_importance": 1.0})
    assert q.is_trained
    assert q.predict().shape[0] == query.n_obs


def test_cytoanvi_continual_query_batch_controls(adata):
    # controls drawn from the query cohort carry *query* batches; their importances must be
    # computed on the batch-extended query model (not the reference, which lacks those batches).
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

    query = make_adata()
    query.obs[BATCH_KEY] = query.obs[BATCH_KEY].map({"batch_0": "batch_2", "batch_1": "batch_3"})
    query.obs[BATCH_KEY] = query.obs[BATCH_KEY].astype(str)
    query.obs[LABELS_KEY] = UNLABELED
    control = query[:128].copy()  # healthy controls from the query cohort (query batches)

    q = CytoANVI.load_query_data_with_replay(
        query, ref, replay_adata=adata[:128].copy(), control_adata=control
    )
    assert q.module.continual.ctrl_importances is not None
    q.train(max_epochs=1, plan_kwargs={"ewc_importance": 1.0})
    assert q.is_trained


def _make_backbone_reference():
    """Reference with a genuine backbone (markers 0-24) + panel-specific markers (25-29).

    Markers 25-29 are masked in the first 10 cells, so they are not "present in all cells" and
    therefore fall outside the encoder backbone (which CytoVI derives from the nan mask).
    """
    adata = make_adata()
    mask = np.ones_like(adata.layers[SCALED_LAYER_KEY])
    mask[:10, 25:] = 0
    adata.layers[SCALED_LAYER_KEY][:10, 25:] = 0
    adata.layers[NAN_LAYER_KEY] = mask
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
        nan_layer=NAN_LAYER_KEY,
    )
    ref = CytoANVI(adata, n_latent=10)
    ref.train(max_epochs=N_EPOCHS)
    return ref


def test_cytoanvi_prepare_query_panel_aware():
    ref = _make_backbone_reference()
    ref_vars = ref.adata.var_names
    # backbone = first 25 markers; query was measured on the backbone panel only
    backbone = list(ref_vars[:25])
    panel_specific = list(ref_vars[25:])
    assert ref_vars[ref.module.encoder_marker_mask].tolist() == backbone

    query = make_adata()
    query = query[:, backbone].copy()
    query.obs[LABELS_KEY] = UNLABELED
    assert NAN_LAYER_KEY not in query.layers

    CytoANVI.prepare_query_anndata(query, ref)

    # padded + sorted to the reference panel, with a freshly built nan mask
    assert list(query.var_names) == list(ref_vars)
    assert NAN_LAYER_KEY in query.layers
    mask = np.asarray(query.layers[NAN_LAYER_KEY])
    assert (mask[:, query.var_names.get_indexer(panel_specific)] == 0).all()  # masked out
    assert (mask[:, query.var_names.get_indexer(backbone)] == 1).all()  # backbone kept

    # the query re-derives the reference backbone, so surgery + predict run end-to-end
    q = CytoANVI.load_query_data(query, ref)
    q.train(max_epochs=1, plan_kwargs={"weight_decay": 0.0})
    assert q.predict().shape[0] == query.n_obs


def test_cytoanvi_prepare_query_docs_snippet_uses_returned_anndata():
    ref = _make_backbone_reference()
    backbone = list(ref.adata.var_names[:25])
    query_adata = make_adata()[:, backbone].copy()
    query_adata.obs[LABELS_KEY] = UNLABELED

    query = CytoANVI.prepare_query_anndata(query_adata, reference_model=ref, inplace=False)

    assert query is not query_adata
    assert list(query.var_names) == list(ref.adata.var_names)
    assert NAN_LAYER_KEY in query.layers
    assert NAN_LAYER_KEY not in query_adata.layers


def test_cytoanvi_prepare_query_preserves_sparse_expression():
    from scipy import sparse

    ref = _make_backbone_reference()
    backbone = list(ref.adata.var_names[:25])
    query = make_adata()[:, backbone].copy()
    query.obs[LABELS_KEY] = UNLABELED
    query.X = sparse.csr_matrix(query.X)
    query.layers[SCALED_LAYER_KEY] = sparse.csr_matrix(query.layers[SCALED_LAYER_KEY])

    CytoANVI.prepare_query_anndata(query, ref)

    assert sparse.issparse(query.X)
    assert sparse.issparse(query.layers[SCALED_LAYER_KEY])
    assert NAN_LAYER_KEY in query.layers


def test_cytoanvi_prepare_query_rejects_missing_backbone():
    ref = _make_backbone_reference()
    # drop a backbone marker (index 0) — the encoder needs it, so prep must reject this
    query = make_adata()
    query = query[:, list(ref.adata.var_names[1:])].copy()
    with pytest.raises(ValueError, match="backbone"):
        CytoANVI.prepare_query_anndata(query, ref)


def test_cytoanvi_prepare_query_rejects_partial_backbone():
    # a query whose own nan_layer masks a *backbone* marker in some cells would re-derive a
    # smaller backbone than the reference -> reject up front (not a cryptic resize crash later)
    ref = _make_backbone_reference()
    backbone = list(ref.adata.var_names[:25])
    query = make_adata()
    query = query[:, backbone].copy()
    query.obs[LABELS_KEY] = UNLABELED
    qmask = np.ones_like(query.layers[SCALED_LAYER_KEY])
    qmask[:10, 0] = 0.0  # backbone marker (column 0) masked in some cells
    query.layers[NAN_LAYER_KEY] = qmask
    with pytest.raises(ValueError, match="backbone"):
        CytoANVI.prepare_query_anndata(query, ref)


def test_cytoanvi_prepare_query_rejects_path_reference(save_path):
    # saved path works for prep when encoder_marker_mask_ is persisted (scvi-tools >= 1.5)
    ref = _make_backbone_reference()
    path = os.path.join(save_path, "ref_for_prep")
    ref.save(path, overwrite=True, save_anndata=True)

    query = make_adata()[:, list(ref.adata.var_names[:25])].copy()
    names = CytoANVI.prepare_query_anndata(query, path, return_reference_var_names=True)
    assert list(names) == list(ref.adata.var_names)

    query2 = make_adata()[:, list(ref.adata.var_names[:25])].copy()
    CytoANVI.prepare_query_anndata(query2, path)
    assert NAN_LAYER_KEY in query2.layers


def test_cytoanvi_prepare_query_requires_nan_layer(adata):
    # a reference set up without a nan_layer cannot mask query-absent markers
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
    )
    ref = CytoANVI(adata, n_latent=10)
    ref.train(max_epochs=N_EPOCHS)

    query = make_adata()
    query = query[:, list(ref.adata.var_names[:-5])].copy()
    with pytest.raises(ValueError, match="nan_layer"):
        CytoANVI.prepare_query_anndata(query, ref)


def test_continual_update_penalty_math():
    # The deepened seam is directly testable: penalty = sum_k w_k (theta_k - theta_k^ref)^2,
    # with w_k = combine(reference Fisher, control Fisher). No training needed.
    import torch

    from cytoanvi._continual import ContinualUpdate

    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

        @property
        def device(self):
            return torch.device("cpu")

    m = _Stub()
    old = [("w", torch.tensor([0.0, 0.0]))]
    imp = [("w", torch.tensor([1.0, 1.0]))]
    ctrl = [("w", torch.tensor([2.0, 3.0]))]

    # product: w = imp*ctrl = [2, 3] -> 2*1^2 + 3*2^2 = 14
    assert float(ContinualUpdate(old, imp, ctrl, "product").penalty(m)) == 14.0
    # additive: w = imp+ctrl = [3, 4] -> 3*1 + 4*4 = 19
    assert float(ContinualUpdate(old, imp, ctrl, "additive").penalty(m)) == 19.0
    # no control Fisher: w = imp = [1, 1] -> 1*1 + 1*4 = 5
    assert float(ContinualUpdate(old, imp, None).penalty(m)) == 5.0
    # size-guard: an anchor param whose shape mismatches the live param is skipped
    mismatched = [("w", torch.tensor([0.0, 0.0, 0.0]))]
    assert float(ContinualUpdate(mismatched, imp, ctrl).penalty(m)) == 0.0


def test_cytoanvi_continual_save_load(adata, save_path):
    # the continual update (anchor + both Fishers + combine rule) survives save/load; the
    # session-scoped replay buffer does not.
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

    query = make_adata()
    query.obs[LABELS_KEY] = UNLABELED
    control = query[:128].copy()
    q = CytoANVI.load_query_data_with_replay(
        query, ref, replay_adata=adata[:128].copy(), control_adata=control
    )
    q.train(max_epochs=1, plan_kwargs={"ewc_importance": 1.0})
    assert q.module.continual is not None
    pen_before = float(q.module.continual.penalty(q.module))

    path = os.path.join(save_path, "test_cytoanvi_continual")
    q.save(path, overwrite=True, save_anndata=True)
    q2 = CytoANVI.load(path)

    # reattached on load, anchor + Fishers preserved (penalty reproduces), replay dropped
    assert q2.module.continual is not None
    assert q2.module.continual.combine_type == "product"
    assert q2.module.continual.replay_batches is None
    pen_after = float(q2.module.continual.penalty(q2.module))
    np.testing.assert_allclose(pen_after, pen_before, rtol=1e-5)
    assert q2.predict().shape[0] == query.n_obs


def test_cytoanvi_beta_likelihood(adata):
    # untested path: Beta protein likelihood. Beta needs expression strictly inside (0, 1), so
    # clip the min-max endpoints (exact 0/1 give -inf log-prob).
    adata.layers[SCALED_LAYER_KEY] = np.clip(adata.layers[SCALED_LAYER_KEY], 1e-3, 1 - 1e-3)
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10, protein_likelihood="beta")
    model.train(max_epochs=N_EPOCHS)
    assert model.is_trained
    assert np.all(np.isfinite(model.get_latent_representation()))
    assert model.predict().shape[0] == adata.n_obs


def test_cytoanvi_continual_on_multipanel():
    # untested path: continual case-control update on a multi-panel (nan_layer) reference, so the
    # EWC penalty and backbone masking coexist.
    a1 = make_adata(n_genes=30, n_batches=1)
    a2 = make_adata(n_genes=20, n_batches=1)
    a1.obs_names = "a1_" + a1.obs_names
    a2.obs_names = "a2_" + a2.obs_names
    merged = cytovi_pp.merge_batches([a1, a2])
    assert NAN_LAYER_KEY in merged.layers
    merged.obs[LABELS_KEY] = merged.obs[LABELS_KEY].astype(str)

    CytoANVI.setup_anndata(
        merged,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        nan_layer=NAN_LAYER_KEY,
    )
    ref = CytoANVI(merged, n_latent=10)
    ref.train(max_epochs=N_EPOCHS)

    half = merged.n_obs // 2
    query = merged[:half].copy()
    query.obs[LABELS_KEY] = UNLABELED
    control = query[:64].copy()
    replay = merged[half:][:128].copy()

    q = CytoANVI.load_query_data_with_replay(
        query, ref, replay_adata=replay, control_adata=control
    )
    assert q.module.continual is not None
    q.train(max_epochs=1, plan_kwargs={"ewc_importance": 1.0})
    assert q.is_trained
    assert q.predict().shape[0] == query.n_obs


def test_cytoanvi_uncertainty_flags_ood(adata):
    # untested path: get_uncertainty novelty discrimination. BI is >= 0 (Jensen on log-sum-exp);
    # far-out-of-distribution cells should score higher than in-distribution ones.
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    model.train(max_epochs=10)

    query = adata.copy()
    n = query.n_obs
    ood = np.zeros(n, dtype=bool)
    ood[: n // 2] = True
    # push the OOD half far outside the training [0, 1] range
    x = query.layers[SCALED_LAYER_KEY].copy()
    x[ood] = x[ood] * 8.0 + 5.0
    query.layers[SCALED_LAYER_KEY] = x

    unc = model.get_uncertainty(query, tta_rep=30)
    assert unc.shape == (n,)
    assert np.all(np.isfinite(unc))
    assert np.all(unc >= -1e-6)  # Bregman Information is non-negative
    assert unc[ood].mean() > unc[~ood].mean()  # OOD cells are flagged as more uncertain


def test_cytoanvi_all_unlabeled_raises(adata):
    adata.obs[LABELS_KEY] = UNLABELED
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
    )
    with pytest.raises(ValueError, match="at least one observed"):
        CytoANVI(adata, n_latent=10)


def test_cytoanvi_inherited_cytovi_smoke(adata):
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
    norm = model.get_normalized_expression()
    assert norm.shape[0] == adata.n_obs
    de = model.differential_expression(groupby=LABELS_KEY)
    assert len(de) > 0


def test_cytoanvi_select_replay_by_uncertainty(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
    )
    model = CytoANVI(adata, n_latent=10)
    model.train(max_epochs=N_EPOCHS)
    replay = CytoANVI.select_replay_by_uncertainty(model, adata, fraction=0.2)
    assert 0 < replay.n_obs < adata.n_obs


def test_cytoanvi_encoder_mask_saved_path_prep(adata, save_path):
    a1 = make_adata(n_genes=30, n_batches=1)
    a2 = make_adata(n_genes=20, n_batches=1)
    a1.obs_names = "a1_" + a1.obs_names
    a2.obs_names = "a2_" + a2.obs_names
    merged = cytovi_pp.merge_batches([a1, a2])
    CytoANVI.setup_anndata(
        merged,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        nan_layer=NAN_LAYER_KEY,
    )
    ref = CytoANVI(merged, n_latent=10)
    ref.train(max_epochs=N_EPOCHS)
    path = os.path.join(save_path, "ref_encoder_mask")
    ref.save(path, overwrite=True, save_anndata=True)

    backbone = list(ref.adata.var_names[ref.encoder_marker_mask_])
    query = make_adata()[:, backbone].copy()
    query.obs[LABELS_KEY] = UNLABELED
    CytoANVI.prepare_query_anndata(query, path)
    assert list(query.var_names) == list(ref.adata.var_names)


def test_cytoanvi_prepare_query_path_no_longer_rejects(save_path):
    ref = _make_backbone_reference()
    path = os.path.join(save_path, "ref_for_prep_v2")
    ref.save(path, overwrite=True, save_anndata=True)
    query = make_adata()[:, list(ref.adata.var_names[:25])].copy()
    CytoANVI.prepare_query_anndata(query, path)
    assert NAN_LAYER_KEY in query.layers


def test_cytoanvi_continual_resume_replay(adata, save_path):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
    )
    ref = CytoANVI(adata, n_latent=10)
    ref.train(max_epochs=N_EPOCHS)
    replay = adata[:128].copy()
    query = make_adata()
    query.obs[LABELS_KEY] = UNLABELED
    q = CytoANVI.load_query_data_with_replay(
        query, ref, replay_adata=replay, control_adata=query[:64].copy()
    )
    q.train(max_epochs=1, plan_kwargs={"ewc_importance": 1.0})
    path = os.path.join(save_path, "continual_resume")
    q.save(path, overwrite=True, save_anndata=True)
    q2 = CytoANVI.load(path)
    assert q2.module.continual.replay_batches is None
    with pytest.warns(UserWarning, match="replay buffer"):
        q2.train(max_epochs=1, plan_kwargs={"ewc_importance": 1.0})
    q3 = CytoANVI.load_query_data_with_replay(
        query, ref, replay_adata=replay, control_adata=query[:64].copy()
    )
    q3.train(max_epochs=1, plan_kwargs={"ewc_importance": 1.0})
    assert q3.module.continual.replay_batches is not None


def test_cytoanvi_uncertainty_multipanel():
    a1 = make_adata(n_genes=30, n_batches=1)
    a2 = make_adata(n_genes=20, n_batches=1)
    a1.obs_names = "a1_" + a1.obs_names
    a2.obs_names = "a2_" + a2.obs_names
    merged = cytovi_pp.merge_batches([a1, a2])
    CytoANVI.setup_anndata(
        merged,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        nan_layer=NAN_LAYER_KEY,
    )
    model = CytoANVI(merged, n_latent=10)
    model.train(max_epochs=N_EPOCHS)
    unc = model.get_uncertainty(tta_rep=3)
    assert unc.shape == (merged.n_obs,)
    assert np.all(np.isfinite(unc))


def test_example_reference_query_runs():
    example_path = (
        Path(__file__).parents[2]
        / "vignettes"
        / "cytoanvi_example_reference_query.py"
    )
    spec = spec_from_file_location("cytoanvi_example_reference_query", example_path)
    example_reference_query = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(example_reference_query)

    example_reference_query.main(max_epochs=2)


def test_vignette_warmstart_section_runs():
    from vignettes.cytoanvi_showcase import section_warmstart

    result = section_warmstart(max_epochs=1)
    assert result["dataset"] == "synthetic_warmstart"
    assert result["max_epochs"] == 1


def test_fisher_importances_raises_on_empty_adata(adata):
    """fisher_importances should raise ValueError, not silently return all-zero importances."""
    from cytoanvi._continual import fisher_importances

    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    model.train(max_epochs=1)

    # Slice to produce a valid but empty AnnData (zero cells).
    empty_adata = adata[:0].copy()
    assert empty_adata.n_obs == 0

    with pytest.raises(ValueError, match="non-empty"):
        fisher_importances(model, empty_adata)
