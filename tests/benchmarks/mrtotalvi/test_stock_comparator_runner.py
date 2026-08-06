"""CPU smoke test for exact-split stock scVI and TotalVI runners."""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.mrtotalvi import (
    ComparatorRunConfig,
    ScenarioConfig,
    generate_scenario,
    run_stock_comparator,
)
from benchmarks.mrtotalvi.comparator import _construct_stock_model


@pytest.mark.parametrize(
    ("candidate_id", "expected_representations", "has_protein"),
    [
        ("B0", {"factual_z"}, False),
        ("B1", {"factual_z"}, True),
    ],
)
def test_stock_comparator_runner_uses_exact_split_and_separate_modalities(
    tmp_path,
    candidate_id,
    expected_representations,
    has_protein,
):
    simulated = generate_scenario(
        ScenarioConfig(
            scenario="null",
            n_donors=2,
            cells_per_sample=8,
            n_states=2,
            n_genes=6,
            n_proteins=3,
            latent_truth_dim=2,
        ),
        seed=81,
    )
    n_obs = simulated.adata.n_obs
    validation = np.asarray(
        [
            index
            for index in range(n_obs)
            if index % 8 in {6, 7}
        ],
        dtype=np.int64,
    )
    train = np.setdiff1d(np.arange(n_obs), validation)
    config = ComparatorRunConfig(
        candidate_id=candidate_id,
        train_indices=tuple(train),
        validation_indices=tuple(validation),
        max_epochs=1,
        n_latent=2,
        n_hidden=8,
        n_layers=1,
        batch_size=16,
        learning_rate=1e-3,
        technical_batch_key="technical_batch",
        training_seed=91,
        evaluation_seed=97,
        posterior_predictive_draws=2,
    )
    result = run_stock_comparator(
        simulated.adata,
        config,
        checkpoint_dir=tmp_path / candidate_id,
        evaluation_annotations={
            "state_labels": simulated.truth.cell_states[validation],
            "sample_labels": simulated.truth.observed_sample_indices[validation],
            "technical_batch_labels": simulated.truth.sample_batches[
                simulated.truth.observed_sample_indices[validation]
            ],
        },
    )

    assert result["candidate_id"] == candidate_id
    assert result["biological_sample_key"] is None
    assert result["technical_batch_key"] == "technical_batch"
    assert result["training_seed"] == 91
    assert result["evaluation_seed"] == 97
    np.testing.assert_array_equal(result["train_indices"], train)
    np.testing.assert_array_equal(result["validation_indices"], validation)
    assert set(result["representations"]) == expected_representations
    assert len(result["training_history"]["elbo_validation"]) == 1
    assert len(result["best_checkpoint_identity"]["state_digest"]) == 64
    assert result["best_checkpoint_identity"]["artifact_name"].startswith("best-")
    metrics = result["metrics"]
    assert metrics["rna_heldout_negative_log_likelihood"] > 0.0
    assert metrics["factual_z_effective_rank"] > 0.0
    assert 0.0 <= metrics["factual_z_state_balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["factual_z_technical_batch_mixing"] <= 1.0
    assert metrics["trainable_parameter_count"] > 0
    assert metrics["latent_all_finite"] == 1.0
    if has_protein:
        assert metrics["protein_heldout_negative_log_likelihood"] > 0.0
        assert metrics["multimodal_heldout_predictive_loss"] > 0.0
    else:
        assert metrics["protein_heldout_negative_log_likelihood"] is None
        assert metrics["multimodal_heldout_predictive_loss"] is None


@pytest.mark.parametrize("candidate_id", ["B0", "B1"])
def test_stock_comparator_accepts_declared_canonical_batch_key_without_obs_adapter(
    candidate_id,
):
    import scvi

    simulated = generate_scenario(
        ScenarioConfig(
            scenario="null",
            n_donors=2,
            cells_per_sample=4,
            n_states=2,
            n_genes=4,
            n_proteins=2,
            latent_truth_dim=2,
        ),
        seed=101,
    )
    adata = simulated.adata.copy()
    adata.obs["batch"] = adata.obs["technical_batch"].copy()
    del adata.obs["technical_batch"]
    n_obs = adata.n_obs
    config = ComparatorRunConfig(
        candidate_id=candidate_id,
        train_indices=tuple(range(n_obs - 2)),
        validation_indices=(n_obs - 2, n_obs - 1),
        max_epochs=1,
        n_latent=2,
        n_hidden=8,
        n_layers=1,
        batch_size=8,
        learning_rate=1e-3,
        technical_batch_key="batch",
        training_seed=103,
        evaluation_seed=107,
        posterior_predictive_draws=2,
    )
    scvi.settings.seed = config.training_seed
    model = _construct_stock_model(adata, config)

    assert model.adata is adata
    assert "batch" in adata.obs
    assert "technical_batch" not in adata.obs


def test_same_seed_fit_and_evaluation_ignore_ambient_rng_state(tmp_path):
    import torch

    simulated = generate_scenario(
        ScenarioConfig(
            scenario="mixed",
            n_donors=2,
            cells_per_sample=8,
            n_states=2,
            n_genes=6,
            n_proteins=3,
            latent_truth_dim=2,
        ),
        seed=109,
    )
    n_obs = simulated.adata.n_obs
    validation = np.asarray(
        [index for index in range(n_obs) if index % 8 in {6, 7}],
        dtype=np.int64,
    )
    train = np.setdiff1d(np.arange(n_obs), validation)
    config = ComparatorRunConfig(
        candidate_id="B0",
        train_indices=tuple(train),
        validation_indices=tuple(validation),
        max_epochs=1,
        n_latent=2,
        n_hidden=8,
        n_layers=1,
        batch_size=16,
        learning_rate=1e-3,
        technical_batch_key="technical_batch",
        training_seed=113,
        evaluation_seed=127,
        posterior_predictive_draws=2,
    )
    annotations = {
        "state_labels": simulated.truth.cell_states[validation],
        "sample_labels": simulated.truth.observed_sample_indices[validation],
        "technical_batch_labels": simulated.truth.sample_batches[
            simulated.truth.observed_sample_indices[validation]
        ],
    }

    torch.manual_seed(131)
    np.random.seed(137)
    torch.randn(37)
    np.random.random(37)
    first = run_stock_comparator(
        simulated.adata.copy(),
        config,
        checkpoint_dir=tmp_path / "first",
        evaluation_annotations=annotations,
    )

    torch.manual_seed(139)
    np.random.seed(149)
    torch.randn(101)
    np.random.random(101)
    second = run_stock_comparator(
        simulated.adata.copy(),
        config,
        checkpoint_dir=tmp_path / "second",
        evaluation_annotations=annotations,
    )

    assert (
        first["best_checkpoint_identity"]["state_digest"]
        == second["best_checkpoint_identity"]["state_digest"]
    )
    assert first["training_history"] == second["training_history"]
    np.testing.assert_array_equal(
        first["representations"]["factual_z"]["values"],
        second["representations"]["factual_z"]["values"],
    )
    for metric_id in (
        "rna_posterior_predictive_calibration",
        "factual_z_state_balanced_accuracy",
        "factual_z_technical_batch_mixing",
    ):
        assert first["metrics"][metric_id] == second["metrics"][metric_id]
