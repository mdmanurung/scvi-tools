"""Prospective v3 fit evidence from the real diagnosis entry point."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from benchmarks.mrtotalvi.convergence_runner import (
    PreparedDiagnosisFixture,
    run_diagnosis_fit,
)
from benchmarks.mrtotalvi.versioning import (
    prospective_redesign_contract_adapter,
    version_binding_fields,
)


def _fixture() -> PreparedDiagnosisFixture:
    return PreparedDiagnosisFixture(
        fixture_id="mixed",
        adata=SimpleNamespace(
            n_obs=20,
            n_vars=7,
            obsm={"protein_expression": np.zeros((20, 3))},
        ),
        train_indices=np.arange(16, dtype=np.int64),
        validation_indices=np.arange(16, 20, dtype=np.int64),
        state_labels=np.asarray(["a"] * 10 + ["b"] * 10),
        sample_labels=np.asarray(["s0"] * 10 + ["s1"] * 10),
        technical_batch_labels=np.asarray(["b0"] * 10 + ["b1"] * 10),
        count_layer="counts",
        protein_obsm_key="protein_expression",
        protein_names_uns_key="protein_names",
        technical_batch_key="technical_batch",
        sample_key="sample",
        n_latent=4,
        n_hidden=8,
        n_layers=1,
        n_prior_components=4,
        batch_size=8,
        learning_rate=1e-3,
        evaluation_seed=20_260_726,
        source_data_digest="a" * 64,
        state_annotation_digest="b" * 64,
        data_digest="c" * 64,
        split_digest="d" * 64,
        state_annotation_id="known_truth_cell_state",
    )


def _stock_result(*, finite: bool) -> dict:
    adapter = prospective_redesign_contract_adapter()
    metrics = dict.fromkeys(adapter.metric_ids)
    history = [
        {"epoch": 5 * (index + 1), "value": 100.0 - index}
        for index in range(31)
    ]
    checkpoint = {
        "monitor": "elbo_validation",
        "mode": "min",
        "epoch": 155,
        "value": 70.0,
        "state_digest": "e" * 64,
        "artifact_name": "best",
    }
    metrics.update(
        {
            "validation_objective_history": history,
            "best_checkpoint_identity": checkpoint,
            "factual_z_posterior_scale": 0.3 if finite else None,
            "factual_z_latent_variance": 0.2 if finite else None,
            "factual_z_effective_rank": 1.0 if finite else None,
            "factual_z_representation_all_finite": float(finite),
            "factual_z_exact_nonconstant_variation": 1.0,
            "factual_z_posterior_scales_all_valid": 1.0,
        }
    )
    values = np.arange(16, dtype=np.float64).reshape(4, 4)
    if not finite:
        values[0, 0] = np.nan
    return {
        "representations": {
            "factual_z": {
                "cell_ids": np.asarray([f"cell-{i}" for i in range(4)]),
                "values": values,
            }
        },
        "trainer_epochs": 155,
        "stopped_early": True,
        "training_history": {"elbo_validation": history},
        "best_checkpoint_identity": checkpoint,
        "optimization_identity": {"mock": True},
        "metrics": metrics,
    }


def test_prospective_fit_uses_v3_integrity_and_exact_bindings(
    tmp_path,
    monkeypatch,
):
    """Low rank is retained as an alert in a self-describing v3 fit."""
    adapter = prospective_redesign_contract_adapter()
    stock = _stock_result(finite=True)
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.convergence_runner.run_stock_comparator",
        lambda *_args, **_kwargs: stock,
    )

    result, representations = run_diagnosis_fit(
        _fixture(),
        candidate_id="B1",
        training_seed=0,
        checkpoint_dir=tmp_path / "checkpoint",
        contract_adapter=adapter,
    )

    assert result["schema_version"] == "mrtotalvi-convergence-fit-v3"
    assert "collapse" not in result
    assert result["latent_integrity"][
        "schema_version"
    ] == "mrtotalvi-latent-integrity-assessment-v3"
    assert result["latent_integrity"]["terminal_integrity_failed"] is False
    assert result["latent_integrity"][
        "effective_rank_screen_flags"
    ] == ["factual_z"]
    assert {
        name: result[name] for name in version_binding_fields(adapter)
    } == version_binding_fields(adapter)
    assert {
        name: result["latent_integrity"][name]
        for name in version_binding_fields(adapter)
    } == version_binding_fields(adapter)
    assert representations is stock["representations"]


def test_prospective_fit_serializes_declared_terminal_failure(
    tmp_path,
    monkeypatch,
):
    """A declared representation defect produces evidence instead of an abort."""
    adapter = prospective_redesign_contract_adapter()
    stock = _stock_result(finite=False)
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.convergence_runner.run_stock_comparator",
        lambda *_args, **_kwargs: stock,
    )

    result, _ = run_diagnosis_fit(
        _fixture(),
        candidate_id="B1",
        training_seed=0,
        checkpoint_dir=tmp_path / "checkpoint",
        contract_adapter=adapter,
    )

    factual = result["latent_integrity"]["representations"]["factual_z"]
    assert result["latent_integrity"]["terminal_integrity_failed"] is True
    assert factual["terminal_failure_reasons"] == [
        "factual_z_representation_all_finite"
    ]
    assert factual["eligible_for_geometry"] is False
