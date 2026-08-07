"""Defect-inverting tests for the MrTotalVI 0.2 usage-readiness contract."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from scipy import sparse

import scvi
from scvi.external import MrTotalVI
from scvi.external.mrtotalvi._contracts import (
    ordered_indices_sha256,
    validate_count_matrix,
    validate_sample_metadata,
)
from scvi.model._totalvi import TOTALVI


def _adata(*, labels: bool = False):
    adata = scvi.data.synthetic_iid()
    adata.obs["sample"] = np.asarray(
        [f"donor_{i % 4}" for i in range(adata.n_obs)],
        dtype=object,
    )
    if labels:
        adata.obs["cell_type"] = np.asarray(
            ["T" if i % 2 else "B" for i in range(adata.n_obs)],
            dtype=object,
        )
    return adata


def _setup(adata, *, labels: bool = False):
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        labels_key="cell_type" if labels else None,
    )


@pytest.mark.parametrize(
    ("prior", "legacy", "mixture"),
    [
        ("standard", None, False),
        ("mog", None, True),
        ("vamp", None, True),
        ("standard", False, False),
        ("mog", True, True),
        ("vamp", True, True),
    ],
)
def test_exact_prior_migration_table(prior, legacy, mixture):
    adata = _adata()
    _setup(adata)
    context = (
        pytest.warns(DeprecationWarning, match="u_prior_mixture")
        if legacy is not None
        else warnings.catch_warnings()
    )
    with context:
        model = MrTotalVI(
            adata,
            sample_key="sample",
            u_prior=prior,
            u_prior_mixture=legacy,
        )
    assert model.resolved_u_prior == prior
    assert model.module.u_prior_mixture is mixture
    assert model.init_params_["non_kwargs"]["u_prior"] == prior
    assert model.init_params_["non_kwargs"]["u_prior_mixture"] is None


@pytest.mark.parametrize(
    ("prior", "legacy"),
    [
        ("unknown", None),
        ("standard", True),
        ("mog", False),
        ("vamp", False),
    ],
)
def test_unknown_and_contradictory_prior_states_refuse_before_module(prior, legacy, monkeypatch):
    adata = _adata()
    _setup(adata)
    constructed = False

    def forbidden(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("module construction must not be reached")

    monkeypatch.setattr(TOTALVI, "__init__", forbidden)
    with pytest.raises((ValueError, TypeError)):
        MrTotalVI(
            adata,
            sample_key="sample",
            u_prior=prior,
            u_prior_mixture=legacy,
        )
    assert not constructed


def test_registered_labels_alone_leave_prior_unsupervised():
    adata = _adata(labels=True)
    _setup(adata, labels=True)
    model = MrTotalVI(adata, sample_key="sample", u_prior_mixture_k=7, n_latent_u=4)

    assert model.u_prior_supervision == "none"
    assert model.u_prior_label_weight == 0.0
    assert model.module.resolved_u_prior_mixture_k == 7
    assert model.module.u_prior_logits.shape == (7,)
    u = torch.zeros(2, 4)
    prior = model.module.build_u_prior(u, torch.tensor([[0], [1]]))
    assert prior.mixture_distribution.logits.ndim == 1
    assert "u_prior_supervision: none" in model._model_summary_string


def test_explicit_label_supervision_is_opt_in_and_persisted():
    adata = _adata(labels=True)
    _setup(adata, labels=True)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent_u=4,
        u_prior_supervision="labels",
        u_prior_label_weight=3.0,
    )

    assert model.module.resolved_u_prior_mixture_k == model.summary_stats.n_labels
    prior = model.module.build_u_prior(
        torch.zeros(2, 4),
        torch.tensor([[0], [1]]),
    )
    assert prior.mixture_distribution.logits.argmax(dim=-1).tolist() == [0, 1]
    init = model.init_params_["non_kwargs"]
    assert init["u_prior_supervision"] == "labels"
    assert init["u_prior_label_weight"] == 3.0
    assert "u_prior_supervision: labels" in model._model_summary_string


@pytest.mark.parametrize(
    ("labels", "mode", "weight", "match"),
    [
        (True, "none", 1.0, "requires u_prior_label_weight=0.0"),
        (True, "labels", 0.0, "finite positive"),
        (True, "labels", np.inf, "finite"),
        (False, "labels", 1.0, "requires labels_key"),
        (True, "invalid", 0.0, "exactly one"),
    ],
)
def test_invalid_supervision_states_fail_before_module(labels, mode, weight, match, monkeypatch):
    adata = _adata(labels=labels)
    _setup(adata, labels=labels)
    constructed = False

    def forbidden(*args, **kwargs):
        nonlocal constructed
        constructed = True
        raise AssertionError("module construction must not be reached")

    monkeypatch.setattr(TOTALVI, "__init__", forbidden)
    with pytest.raises(ValueError, match=match):
        MrTotalVI(
            adata,
            sample_key="sample",
            u_prior_supervision=mode,
            u_prior_label_weight=weight,
        )
    assert not constructed


def test_checkpoint_supervision_migration_resaves_explicit_metadata(tmp_path):
    adata = _adata(labels=True)
    _setup(adata, labels=True)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        u_prior_supervision="labels",
        u_prior_label_weight=10.0,
    )
    old_path = tmp_path / "old"
    model.save(old_path)

    payload = torch.load(old_path / "model.pt", map_location="cpu", weights_only=False)
    init = payload["attr_dict"]["init_params_"]["non_kwargs"]
    init.pop("u_prior_supervision")
    init["u_prior_mixture"] = True
    torch.save(payload, old_path / "model.pt")

    with pytest.warns(DeprecationWarning, match="Missing u_prior_supervision"):
        migrated = MrTotalVI.load(old_path, adata=adata)
    assert migrated.u_prior_supervision == "labels"
    assert migrated.u_prior_label_weight == 10.0

    new_path = tmp_path / "resaved"
    migrated.save(new_path)
    resaved = torch.load(new_path / "model.pt", map_location="cpu", weights_only=False)
    explicit = resaved["attr_dict"]["init_params_"]["non_kwargs"]
    assert explicit["u_prior_supervision"] == "labels"
    assert explicit["u_prior_mixture"] is None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reloaded = MrTotalVI.load(new_path, adata=adata)
    assert reloaded.u_prior_supervision == "labels"
    assert not any(issubclass(item.category, DeprecationWarning) for item in caught)


def test_label_free_checkpoint_migrates_historical_weight_to_unsupervised(tmp_path):
    adata = _adata(labels=False)
    _setup(adata, labels=False)
    model = MrTotalVI(adata, sample_key="sample")
    old_path = tmp_path / "old-label-free"
    model.save(old_path)

    payload = torch.load(old_path / "model.pt", map_location="cpu", weights_only=False)
    init = payload["attr_dict"]["init_params_"]["non_kwargs"]
    init.pop("u_prior_supervision")
    init["u_prior_label_weight"] = 10.0
    init["u_prior_mixture"] = True
    torch.save(payload, old_path / "model.pt")

    with pytest.warns(DeprecationWarning, match="historical effective objective"):
        migrated = MrTotalVI.load(old_path, adata=adata)
    assert migrated.u_prior_supervision == "none"
    assert migrated.u_prior_label_weight == 0.0

    new_path = tmp_path / "resaved-label-free"
    migrated.save(new_path)
    resaved = torch.load(new_path / "model.pt", map_location="cpu", weights_only=False)
    explicit = resaved["attr_dict"]["init_params_"]["non_kwargs"]
    assert explicit["u_prior_supervision"] == "none"
    assert explicit["u_prior_label_weight"] == 0.0
    assert explicit["u_prior_mixture"] is None


@pytest.mark.parametrize("prior", ["standard", "mog", "vamp"])
@pytest.mark.parametrize("labels", [False, True])
def test_legacy_checkpoint_prior_label_matrix_preserves_effective_objective(
    tmp_path,
    prior,
    labels,
):
    adata = _adata(labels=labels)
    _setup(adata, labels=labels)
    model_kwargs = {"u_prior": prior}
    if labels and prior in {"mog", "vamp"}:
        # Match the historical state shapes: labelled mixtures used one
        # component per registered label, while only MoG applied offsets.
        model_kwargs.update(
            u_prior_supervision="labels",
            u_prior_label_weight=10.0,
        )
    model = MrTotalVI(adata, sample_key="sample", **model_kwargs)
    old_path = tmp_path / f"old-{prior}-{labels}"
    model.save(old_path)

    payload = torch.load(old_path / "model.pt", map_location="cpu", weights_only=False)
    init = payload["attr_dict"]["init_params_"]["non_kwargs"]
    original_mixture_k = init["u_prior_mixture_k"]
    init.pop("u_prior_supervision")
    init["u_prior_label_weight"] = 10.0
    init["u_prior_mixture"] = prior != "standard"
    torch.save(payload, old_path / "model.pt")

    expected_supervision = "labels" if labels and prior == "mog" else "none"
    expected_weight = 10.0 if expected_supervision == "labels" else 0.0
    migration_match = (
        "explicit label supervision"
        if expected_supervision == "labels"
        else "historical effective objective"
    )
    with pytest.warns(DeprecationWarning, match=migration_match):
        migrated = MrTotalVI.load(old_path, adata=adata)
    assert migrated.u_prior_supervision == expected_supervision
    assert migrated.u_prior_label_weight == expected_weight

    if labels and prior in {"mog", "vamp"}:
        label_index = torch.tensor([[0], [1], [0]])
        distribution = migrated.module.build_u_prior(
            torch.zeros(3, migrated.module.n_latent_u),
            label_index,
        )
        expected_ndim = 2 if prior == "mog" else 1
        assert distribution.mixture_distribution.logits.ndim == expected_ndim
        assert migrated.module.resolved_u_prior_mixture_k == migrated.summary_stats.n_labels

    first_resave = tmp_path / f"resaved-once-{prior}-{labels}"
    migrated.save(first_resave)
    first_payload = torch.load(
        first_resave / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    first_init = first_payload["attr_dict"]["init_params_"]["non_kwargs"]
    assert first_init["u_prior_supervision"] == expected_supervision
    assert first_init["u_prior_label_weight"] == expected_weight
    assert first_init["u_prior_mixture"] is None
    expected_k = (
        migrated.summary_stats.n_labels
        if labels and prior == "vamp"
        else original_mixture_k
    )
    assert first_init["u_prior_mixture_k"] == expected_k

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reloaded = MrTotalVI.load(first_resave, adata=adata)
    assert not any(issubclass(item.category, DeprecationWarning) for item in caught)
    second_resave = tmp_path / f"resaved-twice-{prior}-{labels}"
    reloaded.save(second_resave)
    second_payload = torch.load(
        second_resave / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    second_init = second_payload["attr_dict"]["init_params_"]["non_kwargs"]
    assert second_init["u_prior_supervision"] == expected_supervision
    assert second_init["u_prior_label_weight"] == expected_weight
    assert second_init["u_prior_mixture"] is None
    assert second_init["u_prior_mixture_k"] == expected_k


def test_contradictory_historical_checkpoint_refuses(tmp_path):
    adata = _adata()
    _setup(adata)
    model = MrTotalVI(adata, sample_key="sample")
    path = tmp_path / "contradictory"
    model.save(path)
    payload = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    init = payload["attr_dict"]["init_params_"]["non_kwargs"]
    init["u_prior"] = "standard"
    init["u_prior_mixture"] = True
    torch.save(payload, path / "model.pt")

    with pytest.raises(ValueError, match="Contradictory u prior"):
        MrTotalVI.load(path, adata=adata)


@pytest.mark.parametrize(
    ("modality", "storage", "bad_value", "match"),
    [
        ("rna", "dense", -1.0, "negative"),
        ("rna", "sparse", 0.5, "non-integer-like"),
        ("rna", "sparse", np.inf, "non-finite"),
        ("protein", "dense", np.nan, "non-finite"),
        ("protein", "sparse", -1.0, "negative"),
        ("protein", "dataframe", 0.25, "non-integer-like"),
    ],
)
def test_setup_exhaustively_rejects_hidden_bad_counts_before_mutation(
    modality,
    storage,
    bad_value,
    match,
):
    adata = _adata()
    key = "protein_expression"
    source = np.asarray(
        adata.X if modality == "rna" else adata.obsm[key],
        dtype=np.float64,
    ).copy()
    source[-1, -1] = bad_value
    if storage == "sparse":
        source = sparse.csr_matrix(source)
    elif storage == "dataframe":
        source = pd.DataFrame(
            source,
            index=adata.obs_names,
            columns=[f"protein_{i}" for i in range(source.shape[1])],
        )
    if modality == "rna":
        adata.X = source
    else:
        adata.obsm[key] = source

    assert "_indices" not in adata.obs
    with pytest.raises(ValueError, match=match):
        _setup(adata)
    assert "_indices" not in adata.obs


@pytest.mark.parametrize("modality", ["rna", "protein"])
def test_setup_rejects_complex_count_dtypes_before_mutation(modality):
    adata = _adata()
    key = "protein_expression"
    source = np.asarray(
        adata.X if modality == "rna" else adata.obsm[key],
        dtype=np.complex128,
    ).copy()
    source[-1, -1] = 1.0 + 9.0j
    if modality == "rna":
        adata.X = source
    else:
        adata.obsm[key] = source

    assert "_indices" not in adata.obs
    with pytest.raises(ValueError, match="real-valued raw counts"):
        _setup(adata)
    assert "_indices" not in adata.obs


@pytest.mark.parametrize("modality", ["rna", "protein"])
def test_setup_rejects_object_count_dtypes_before_mutation(modality):
    adata = _adata()
    key = "protein_expression"
    source = np.asarray(
        adata.X if modality == "rna" else adata.obsm[key],
        dtype=object,
    ).copy()
    source[-1, -1] = "not-a-count"
    if modality == "rna":
        adata.X = source
    else:
        adata.obsm[key] = source

    assert "_indices" not in adata.obs
    with pytest.raises(ValueError, match="real numeric dtype"):
        _setup(adata)
    assert "_indices" not in adata.obs


def test_chunked_count_validator_finds_last_value(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "counts.h5"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("counts", shape=(2050, 3), dtype="f4", chunks=(64, 3))
        dataset[...] = 0.0
        dataset[-1, -1] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            validate_count_matrix(dataset, name="backed RNA", chunk_size=128)


def test_vamp_initialization_uses_only_frozen_training_indices(monkeypatch):
    first = _adata()
    second = first.copy()
    n_obs = first.n_obs
    train = np.arange(n_obs // 2, dtype=np.int64)
    val = np.arange(n_obs // 2, 3 * n_obs // 4, dtype=np.int64)
    test = np.arange(3 * n_obs // 4, n_obs, dtype=np.int64)
    second.X[val[0] :, :] = np.asarray(second.X[val[0] :, :]) + 17
    second.obsm["protein_expression"][val[0] :, :] += 11
    _setup(first)
    _setup(second)

    calls = []

    def fake_train(self, **kwargs):
        calls.append(kwargs)
        return "trainer-not-entered"

    monkeypatch.setattr(TOTALVI, "train", fake_train)
    kwargs = {
        "sample_key": "sample",
        "u_prior": "vamp",
        "u_prior_mixture_k": 3,
        "init_prior_from_data": True,
        "freeze_prior_after_init": True,
        "u_prior_init_seed": 19,
    }
    model_a = MrTotalVI(first, **kwargs)
    model_b = MrTotalVI(second, **kwargs)
    result_a = model_a.train(
        max_epochs=1,
        accelerator="cpu",
        external_indexing=[train, val, test],
    )
    result_b = model_b.train(
        max_epochs=1,
        accelerator="cpu",
        external_indexing=[train, val, test],
    )

    assert result_a == result_b == "trainer-not-entered"
    torch.testing.assert_close(model_a.module.u_vamp_pseudo, model_b.module.u_vamp_pseudo)
    expected_digest = ordered_indices_sha256(train)
    assert model_a.vamp_training_indices_sha256_ == expected_digest
    assert model_a.vamp_initialization_seed_ == 19
    assert not model_a.module.u_vamp_pseudo.requires_grad
    assert expected_digest in model_a._model_summary_string
    assert calls[0]["external_indexing"][0].tolist() == train.tolist()


def test_frozen_vamp_prior_remains_frozen_after_load_and_retrain(tmp_path):
    adata = _adata()
    _setup(adata)
    train = np.arange(adata.n_obs - 20, dtype=np.int64)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        u_prior="vamp",
        u_prior_mixture_k=3,
        init_prior_from_data=True,
        freeze_prior_after_init=True,
        u_prior_init_seed=19,
    )
    model._initialize_vamp_from_training_indices(train)
    expected_digest = model.vamp_training_indices_sha256_
    assert not model.module.u_vamp_pseudo.requires_grad

    path = tmp_path / "frozen-vamp"
    model.save(path)
    loaded = MrTotalVI.load(path, adata=adata, accelerator="cpu")

    assert loaded._freeze_prior_after_init is True
    assert loaded.vamp_training_indices_sha256_ == expected_digest
    assert not loaded.module.u_vamp_pseudo.requires_grad
    loaded._initialize_vamp_from_training_indices(train)
    assert not loaded.module.u_vamp_pseudo.requires_grad


def test_vamp_reinitialization_with_different_boundary_refuses(monkeypatch):
    adata = _adata()
    _setup(adata)
    monkeypatch.setattr(TOTALVI, "train", lambda self, **kwargs: None)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        u_prior="vamp",
        u_prior_mixture_k=3,
        init_prior_from_data=True,
    )
    n = adata.n_obs
    first = [np.arange(0, n - 20), np.arange(n - 20, n - 10), np.arange(n - 10, n)]
    second = [np.arange(10, n - 10), np.arange(0, 10), np.arange(n - 10, n)]
    model.train(max_epochs=1, accelerator="cpu", external_indexing=first)
    with pytest.raises(RuntimeError, match="different training boundary"):
        model.train(max_epochs=1, accelerator="cpu", external_indexing=second)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing_column", KeyError),
        ("null_sample", ValueError),
        ("null_covariate", ValueError),
        ("conflicting_covariate", ValueError),
        ("conflicting_donor", ValueError),
        ("unknown_subset", ValueError),
        ("duplicate_subset", ValueError),
        ("empty_subset", ValueError),
    ],
)
def test_sample_metadata_contract_refuses_ambiguous_inputs(mutation, error):
    obs = pd.DataFrame(
        {
            "sample": ["s0", "s0", "s1", "s1"],
            "condition": ["a", "a", "b", "b"],
            "donor": ["d0", "d0", "d1", "d1"],
        }
    )
    subset = ["s0"]
    covariates = ["condition"]
    if mutation == "missing_column":
        covariates = ["missing"]
    elif mutation == "null_sample":
        obs.loc[3, "sample"] = None
    elif mutation == "null_covariate":
        obs.loc[3, "condition"] = None
    elif mutation == "conflicting_covariate":
        obs.loc[1, "condition"] = "b"
    elif mutation == "conflicting_donor":
        obs.loc[1, "donor"] = "d1"
    elif mutation == "unknown_subset":
        subset = ["missing"]
    elif mutation == "duplicate_subset":
        subset = ["s0", "s0"]
    elif mutation == "empty_subset":
        subset = []

    with pytest.raises(error):
        validate_sample_metadata(
            obs,
            sample_key="sample",
            covariate_keys=covariates,
            donor_key="donor",
            sample_subset=subset,
            authoritative_order=["s0", "s1"],
        )


def test_sample_subset_preserves_declared_order():
    obs = pd.DataFrame(
        {
            "sample": ["s0", "s0", "s1", "s1", "s2", "s2"],
            "condition": ["a", "a", "b", "b", "a", "a"],
            "donor": ["d0", "d0", "d1", "d1", "d2", "d2"],
        }
    )
    selected, info = validate_sample_metadata(
        obs,
        sample_key="sample",
        covariate_keys=["condition"],
        donor_key="donor",
        sample_subset=["s2", "s0"],
        authoritative_order=["s0", "s1", "s2"],
    )
    assert selected == ["s2", "s0"]
    assert info.index.tolist() == ["s2", "s0"]


def test_da_is_descriptive_and_uses_validated_declared_order(monkeypatch):
    adata = _adata()
    adata.obs["condition"] = np.where(
        adata.obs["sample"].isin(["donor_0", "donor_1"]),
        "a",
        "b",
    )
    _setup(adata)
    model = MrTotalVI(adata, sample_key="sample")
    captured = {}

    def fake_da(*args, **kwargs):
        captured.update(kwargs)
        return xr.Dataset()

    monkeypatch.setattr("scvi.external.mrtotalvi._model._differential_abundance", fake_da)
    with pytest.warns(UserWarning, match="descriptive, non-inferential"):
        result = model.differential_abundance(
            sample_cov_keys=["condition"],
            sample_subset=["donor_2", "donor_0"],
        )
    assert captured["validated_sample_order"] == ["donor_2", "donor_0"]
    assert result.attrs["interpretation"] == "descriptive_non_inferential"
    assert result.attrs["biological_inference_supported"] is False


@pytest.mark.parametrize("hierarchy_mode", ["legacy", "centered_v2"])
def test_public_de_refuses_before_legacy_statistics(monkeypatch, hierarchy_mode):
    adata = _adata()
    _setup(adata)
    model = MrTotalVI(
        adata,
        sample_key="sample",
        hierarchy_mode=hierarchy_mode,
    )
    reached = False

    def forbidden(*args, **kwargs):
        nonlocal reached
        reached = True
        raise AssertionError("legacy statistics must not run")

    monkeypatch.setattr("scvi.external.mrtotalvi._model._differential_expression", forbidden)
    with pytest.raises(RuntimeError, match="donor-pseudobulk PyDESeq2, edgeR, or dreamlet"):
        model.differential_expression()
    assert not reached


def test_use_vmap_true_refuses_before_any_statistics(monkeypatch):
    adata = _adata()
    _setup(adata)
    model = MrTotalVI(adata, sample_key="sample")
    reached = False

    def forbidden(*args, **kwargs):
        nonlocal reached
        reached = True
        raise AssertionError("statistics must not run")

    monkeypatch.setattr("scvi.external.mrtotalvi._model._differential_expression", forbidden)
    with pytest.raises(NotImplementedError, match="use_vmap=True"):
        model.differential_expression(use_vmap=True)
    assert not reached


def test_private_legacy_de_false_loop_path_validates_inputs(monkeypatch):
    adata = _adata()
    adata.obs["condition"] = np.where(
        adata.obs["sample"].isin(["donor_0", "donor_1"]),
        "a",
        "b",
    )
    _setup(adata)
    model = MrTotalVI(adata, sample_key="sample")
    captured = {}

    def fake_de(*args, **kwargs):
        captured.update(kwargs)
        return xr.Dataset()

    monkeypatch.setattr("scvi.external.mrtotalvi._model._differential_expression", fake_de)
    result = model._legacy_differential_expression_for_reproducibility(
        sample_cov_keys=["condition"],
        sample_subset=["donor_3", "donor_1"],
        use_vmap=False,
    )
    assert captured["validated_sample_order"] == ["donor_3", "donor_1"]
    assert captured["use_vmap"] is False
    assert result.attrs["interpretation"] == "historical_private_non_inferential"

    adata.obs.loc[adata.obs["sample"] == "donor_1", "condition"] = None
    with pytest.raises(ValueError, match="null"):
        model._legacy_differential_expression_for_reproducibility(
            sample_cov_keys=["condition"],
        )


def test_latent_return_contract_documents_non_isomorphic_shapes():
    doc = inspect.getdoc(MrTotalVI.get_latent_representation)
    assert "(n_obs, n_latent)" in doc
    assert "(n_obs, n_latent_u)" in doc


def test_freeze_is_only_allowed_for_data_initialized_vamp():
    adata = _adata()
    _setup(adata)
    with pytest.raises(ValueError, match="data-initialized u_prior='vamp'"):
        MrTotalVI(
            adata,
            sample_key="sample",
            u_prior="mog",
            freeze_prior_after_init=True,
        )
