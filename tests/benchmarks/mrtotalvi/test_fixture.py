"""CPU end-to-end MrTotalVI benchmark fixture."""

from __future__ import annotations

import math

from benchmarks.mrtotalvi import (
    FixtureRunConfig,
    ScenarioConfig,
    run_candidate_fixture,
)


def test_centered_sample_blind_fixture_trains_decodes_and_scores_end_to_end():
    """One public call crosses simulation, training, decoding, and scoring."""
    result = run_candidate_fixture(
        FixtureRunConfig(
            candidate="C4",
            scenario=ScenarioConfig(
                scenario="mixed",
                n_donors=2,
                cells_per_sample=24,
                n_genes=10,
                n_proteins=4,
                latent_truth_dim=3,
            ),
            seed=3,
            max_epochs=1,
            batch_size=32,
            n_latent=3,
            n_prior_components=4,
        )
    )

    assert result["candidate"] == "C4"
    assert result["scenario"] == "mixed"
    assert result["seed"] == 3
    assert result["n_train_cells"] > result["n_validation_cells"] > 0
    assert result["mode"] == {
        "hierarchy_mode": "centered_v2",
        "u_encoder_mode": "sample_blind",
        "scale_observations": False,
    }
    for name in (
        "heldout_elbo",
        "state_balanced_accuracy",
        "sample_balanced_accuracy_within_state",
        "knn_state_accuracy",
        "median_target_distance_spearman",
        "normalized_composition_rmse",
    ):
        assert math.isfinite(result["metrics"][name]), name
    assert result["metrics"]["centering_max_abs"] <= 1e-6
    assert result["metrics"]["n_finite_cells"] == result["n_validation_cells"]
    assert result["wall_seconds"] > 0.0
