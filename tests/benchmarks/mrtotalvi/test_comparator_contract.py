"""Matched stock-comparator and diagnostic serialization contracts."""

from __future__ import annotations

import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks.mrtotalvi.comparator import (
    ComparatorRunConfig,
    best_checkpoint_identity,
    collect_mrtotalvi_diagnostics,
    export_named_representations,
    serialize_training_history,
    state_dict_digest,
    stock_comparator_spec,
    validate_external_split,
    validate_metric_comparability,
)
from benchmarks.mrtotalvi.versioning import (
    historical_redesign_contract_adapter,
    prospective_redesign_contract_adapter,
)


def test_stock_specs_freeze_matched_covariates_and_modality_boundaries():
    scvi = stock_comparator_spec("B0")
    totalvi = stock_comparator_spec("B1")

    assert scvi.model_family == "scvi"
    assert scvi.modalities == ("rna",)
    assert totalvi.model_family == "totalvi"
    assert totalvi.modalities == ("rna", "protein")
    assert scvi.biological_sample_key is None
    assert totalvi.biological_sample_key is None
    assert scvi.gene_likelihood == totalvi.gene_likelihood == "nb"
    with pytest.raises(ValueError, match="stock comparator"):
        stock_comparator_spec("B2")


def test_comparator_run_config_and_external_split_fail_closed():
    config = ComparatorRunConfig(
        candidate_id="B1",
        train_indices=(0, 1, 2),
        validation_indices=(3, 4),
        max_epochs=400,
        n_latent=3,
        n_hidden=16,
        n_layers=1,
        batch_size=4,
        learning_rate=1e-3,
        technical_batch_key="batch",
        training_seed=7,
        evaluation_seed=11,
        check_val_every_n_epoch=5,
        minimum_epochs=50,
        early_stopping_patience=30,
    )
    assert config.spec == stock_comparator_spec("B1")
    assert config.technical_batch_key == "batch"
    assert config.check_val_every_n_epoch == 5
    assert config.minimum_epochs == 50
    assert config.early_stopping_patience == 30
    with pytest.raises(ValueError, match="distinct streams"):
        ComparatorRunConfig(
            candidate_id="B1",
            train_indices=(0, 1, 2),
            validation_indices=(3, 4),
            max_epochs=5,
            n_latent=3,
            n_hidden=16,
            n_layers=1,
            batch_size=4,
            learning_rate=1e-3,
            technical_batch_key="batch",
            training_seed=7,
            evaluation_seed=7,
        )
    train, validation = validate_external_split(
        config.train_indices,
        config.validation_indices,
        n_obs=5,
    )
    np.testing.assert_array_equal(train, [0, 1, 2])
    np.testing.assert_array_equal(validation, [3, 4])

    with pytest.raises(ValueError, match="overlap"):
        validate_external_split([0, 1], [1, 2], n_obs=3)
    with pytest.raises(ValueError, match="complete partition"):
        validate_external_split([0], [2], n_obs=3)
    with pytest.raises(ValueError, match="duplicate"):
        validate_external_split([0, 0], [1, 2], n_obs=3)


class _FakeStock:
    def __init__(self, z):
        self.z = z
        self.calls = []

    def get_latent_representation(self, **kwargs):
        self.calls.append(kwargs)
        return self.z


class _FakeMr(_FakeStock):
    def __init__(self, u, z):
        super().__init__(z)
        self.u = u

    def get_latent_representation(self, **kwargs):
        self.calls.append(kwargs)
        return self.z if kwargs["give_z"] else self.u


class _RngConsumingMr(_FakeMr):
    def get_latent_representation(self, **kwargs):
        import torch

        random.random()
        np.random.random()
        torch.rand(3)
        return super().get_latent_representation(**kwargs)


def test_named_representation_exports_never_alias_u_and_factual_z():
    cells = np.asarray(["c0", "c1", "c2"])
    z = np.arange(6, dtype=float).reshape(3, 2)
    stock = _FakeStock(z)
    stock_export = export_named_representations(
        stock,
        candidate_id="B1",
        cell_ids=cells,
        indices=[0, 1, 2],
        batch_size=8,
    )
    assert set(stock_export) == {"factual_z"}
    np.testing.assert_array_equal(stock_export["factual_z"]["cell_ids"], cells)
    np.testing.assert_array_equal(stock_export["factual_z"]["values"], z)

    mr = _FakeMr(z + 10, z + 20)
    mr_export = export_named_representations(
        mr,
        candidate_id="D0",
        cell_ids=cells,
        indices=[0, 1, 2],
        batch_size=8,
    )
    assert set(mr_export) == {"u", "factual_z"}
    np.testing.assert_array_equal(mr_export["u"]["values"], z + 10)
    np.testing.assert_array_equal(mr_export["factual_z"]["values"], z + 20)
    assert [call["give_z"] for call in mr.calls] == [False, True]


def test_representation_export_restores_python_numpy_and_torch_rng_state():
    """A prospective pre-export cannot perturb unchanged stochastic metrics."""
    import torch

    random.seed(31)
    np.random.seed(37)
    torch.manual_seed(41)
    before_python = random.getstate()
    before_numpy = np.random.get_state()
    before_torch = torch.get_rng_state().clone()
    values = np.arange(12, dtype=np.float64).reshape(6, 2)

    export_named_representations(
        _RngConsumingMr(values + 1.0, values + 2.0),
        candidate_id="D0",
        cell_ids=np.asarray([f"cell-{index}" for index in range(6)]),
        indices=np.arange(6),
        batch_size=3,
    )

    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0]
    np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
    assert after_numpy[2:] == before_numpy[2:]
    assert torch.equal(torch.get_rng_state(), before_torch)


def test_rng_neutral_export_keeps_v1_v2_unchanged_metric_draws_identical():
    """The same checkpoint yields identical unchanged stochastic diagnostics."""
    import torch

    def downstream_signature():
        return (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )

    values = np.arange(12, dtype=np.float64).reshape(6, 2)
    seed = 20260731
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    historical = downstream_signature()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    export_named_representations(
        _RngConsumingMr(values + 1.0, values + 2.0),
        candidate_id="D0",
        cell_ids=np.asarray([f"cell-{index}" for index in range(6)]),
        indices=np.arange(6),
        batch_size=3,
        allow_integrity_failures=True,
    )
    prospective = downstream_signature()

    assert prospective[:2] == historical[:2]
    assert torch.equal(prospective[2], historical[2])


def test_same_checkpoint_v1_v2_collectors_preserve_unchanged_stochastic_metrics(
    tmp_path,
    monkeypatch,
):
    """Pre-export changes only v2 integrity fields, not scientific evidence."""
    import torch

    values = np.arange(8, dtype=np.float64).reshape(4, 2) + 1.0

    class FakeModule:
        def state_dict(self):
            return {}

        def parameters(self):
            return iter(())

    class FakeModel(_RngConsumingMr):
        def __init__(self):
            super().__init__(values + 10.0, values + 20.0)
            self.adata = SimpleNamespace(
                obs_names=np.asarray([f"cell-{index}" for index in range(4)])
            )
            self.module = FakeModule()

    history = {"elbo_validation": [{"epoch": 0, "value": 1.0}]}
    checkpoint = {
        "monitor": "elbo_validation",
        "mode": "min",
        "epoch": 0,
        "value": 1.0,
        "state_digest": "a" * 64,
        "artifact_name": "best",
    }

    def validation_arrays(*_args, **_kwargs):
        draw = random.random() + float(np.random.random()) + float(torch.rand(()))
        return {
            "rna_observed": np.full((4, 2), 2.0),
            "rna_log_prob": np.full((4, 2), -draw),
            "protein_observed": np.full((4, 1), 1.0),
            "protein_log_prob": np.full((4, 1), -(draw + 1.0)),
            "posterior_mean": np.full((4, 2), draw),
            "posterior_scale": np.full((4, 2), draw + 1.0),
            "kl_z": draw + 2.0,
            "kl_u": draw + 3.0,
            "registered_residual_magnitude": draw + 4.0,
            "centering_max_abs": 0.0,
        }

    def factual_scale(*_args, **_kwargs):
        draw = random.random() + float(np.random.random()) + float(torch.rand(()))
        return np.full((4, 2), draw + 1.0)

    def predictive_draws(*_args, **_kwargs):
        draw = random.random() + float(np.random.random()) + float(torch.rand(()))
        return np.full((2, 4, 2), draw + 2.0), np.full(
            (2, 4, 1),
            draw + 1.0,
        )

    def gradient_diagnostics(*_args, **_kwargs):
        draw = random.random() + float(np.random.random()) + float(torch.rand(()))
        return draw + 1.0, 1.0

    monkeypatch.setattr(
        "benchmarks.mrtotalvi.comparator.serialize_training_history",
        lambda _history: history,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.comparator.validate_checkpoint_identity",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.comparator._stock_validation_arrays",
        validation_arrays,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.comparator._mrtotalvi_factual_posterior_scale",
        factual_scale,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.comparator._posterior_predictive_draws",
        predictive_draws,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.comparator._mrtotalvi_residual_gradient_diagnostics",
        gradient_diagnostics,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.comparator.add_evaluation_annotation_metrics",
        lambda metrics, *_args, **_kwargs: metrics,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.comparator.validate_metric_payload",
        lambda payload, **_kwargs: payload,
    )

    def collect(adapter):
        random.seed(101)
        np.random.seed(103)
        torch.manual_seed(107)
        return collect_mrtotalvi_diagnostics(
            FakeModel(),
            candidate_id="D0",
            validation_indices=np.arange(4),
            gradient_indices=np.arange(4),
            batch_size=2,
            posterior_samples=32,
            posterior_predictive_draws=32,
            evaluation_seed=109,
            evaluation_annotations={
                "state_labels": np.asarray(["a", "a", "b", "b"]),
                "sample_labels": np.asarray(["s0", "s0", "s1", "s1"]),
                "technical_batch_labels": np.asarray(["b0", "b0", "b1", "b1"]),
            },
            training_history=history,
            checkpoint_identity=checkpoint,
            checkpoint_path=tmp_path / "best",
            wall_time_seconds=1.0,
            peak_memory_bytes=1,
            contract_adapter=adapter,
        )["metrics"]

    historical = collect(historical_redesign_contract_adapter())
    prospective = collect(prospective_redesign_contract_adapter())
    unchanged = (
        "rna_reconstruction_loss",
        "protein_reconstruction_loss",
        "kl_z",
        "kl_u",
        "registered_residual_magnitude",
        "registered_residual_gradient_norm",
        "registered_residual_gradient_coverage",
        "rna_heldout_negative_log_likelihood",
        "protein_heldout_negative_log_likelihood",
        "multimodal_heldout_predictive_loss",
    )
    assert {key: historical[key] for key in unchanged} == {
        key: prospective[key] for key in unchanged
    }


def test_training_history_and_best_checkpoint_identity_are_complete_and_stable():
    history = {
        "elbo_train": np.asarray([8.0, 6.0, 5.0]),
        "elbo_validation": np.asarray([9.0, 5.5, 6.0]),
    }
    serialized = serialize_training_history(history)
    assert serialized == {
        "elbo_train": [
            {"epoch": 0, "value": 8.0},
            {"epoch": 1, "value": 6.0},
            {"epoch": 2, "value": 5.0},
        ],
        "elbo_validation": [
            {"epoch": 0, "value": 9.0},
            {"epoch": 1, "value": 5.5},
            {"epoch": 2, "value": 6.0},
        ],
    }
    identity = best_checkpoint_identity(
        serialized,
        monitor="elbo_validation",
        mode="min",
        state_digest="a" * 64,
        artifact_name="best-0001",
    )
    assert identity == {
        "monitor": "elbo_validation",
        "mode": "min",
        "epoch": 1,
        "value": 5.5,
        "state_digest": "a" * 64,
        "artifact_name": "best-0001",
    }

    state = {
        "z.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "a.bias": np.asarray([1.0, 2.0]),
    }
    assert state_dict_digest(state) == state_dict_digest(deepcopy(state))
    changed = deepcopy(state)
    changed["a.bias"][0] += 1
    assert state_dict_digest(state) != state_dict_digest(changed)


def test_scvi_is_rejected_from_multimodal_and_protein_rankings():
    validate_metric_comparability("scvi", "rna_heldout_negative_log_likelihood")
    validate_metric_comparability("totalvi", "multimodal_heldout_predictive_loss")
    for metric_id in (
        "protein_heldout_negative_log_likelihood",
        "multimodal_heldout_predictive_loss",
        "multimodal_elbo",
    ):
        with pytest.raises(ValueError, match="RNA-only scVI"):
            validate_metric_comparability("scvi", metric_id)
