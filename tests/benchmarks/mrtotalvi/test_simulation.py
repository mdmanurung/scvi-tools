"""Public exogenous-truth simulation contracts."""

from __future__ import annotations

import numpy as np
from benchmarks.mrtotalvi import SCENARIO_NAMES, ScenarioConfig, generate_scenario


def test_null_scenario_is_deterministic_and_keeps_truth_out_of_model_inputs():
    """One seed regenerates one fixture without exposing counterfactual truth."""
    config = ScenarioConfig(
        scenario="null",
        n_donors=2,
        cells_per_sample=18,
        n_genes=10,
        n_proteins=4,
    )
    first = generate_scenario(config, seed=17)
    second = generate_scenario(config, seed=17)
    changed = generate_scenario(config, seed=18)

    np.testing.assert_array_equal(first.adata.X, second.adata.X)
    np.testing.assert_array_equal(
        first.adata.obsm["protein_expression"],
        second.adata.obsm["protein_expression"],
    )
    np.testing.assert_array_equal(
        first.truth.rna_mean_by_target,
        second.truth.rna_mean_by_target,
    )
    assert not np.array_equal(first.adata.X, changed.adata.X)

    assert np.issubdtype(first.adata.X.dtype, np.integer)
    assert np.issubdtype(
        first.adata.obsm["protein_expression"].dtype,
        np.integer,
    )
    assert np.all(first.adata.X >= 0)
    assert np.all(first.adata.obsm["protein_expression"] >= 0)
    assert not any(name.startswith("truth_") for name in first.adata.obs)
    assert not any(name.startswith("truth_") for name in first.adata.uns)
    assert first.truth.truth_seed != first.truth.training_seed
    assert first.truth.truth_seed != first.truth.evaluation_seed


def test_scenario_family_encodes_distinct_exogenous_truth():
    """Every frozen scenario is generated with its intended isolated perturbation."""
    assert SCENARIO_NAMES == (
        "null",
        "da_only",
        "de_only",
        "mixed",
        "rare_state",
        "unequal_cells",
        "continuous",
        "batch_confounded",
    )
    generated = {
        name: generate_scenario(
            ScenarioConfig(
                scenario=name,
                n_donors=3,
                cells_per_sample=60,
                n_genes=12,
                n_proteins=5,
            ),
            seed=29,
        )
        for name in SCENARIO_NAMES
    }

    null = generated["null"].truth
    assert not null.da_state_mask.any()
    assert not null.de_gene_mask.any()
    assert not null.de_protein_mask.any()
    assert np.unique(null.sample_cell_counts).size == 1

    da_only = generated["da_only"].truth
    assert da_only.da_state_mask.any()
    assert not da_only.de_gene_mask.any()
    assert not da_only.de_protein_mask.any()
    w00 = da_only.sample_state_counts[da_only.sample_conditions == 0].sum(axis=0)
    w22 = da_only.sample_state_counts[da_only.sample_conditions == 1].sum(axis=0)
    assert np.max(np.abs(w22 / w22.sum() - w00 / w00.sum())) >= 0.15

    de_only = generated["de_only"].truth
    assert not de_only.da_state_mask.any()
    assert de_only.de_gene_mask.any()
    assert de_only.de_protein_mask.any()
    for donor in range(3):
        np.testing.assert_array_equal(
            de_only.sample_state_counts[2 * donor],
            de_only.sample_state_counts[2 * donor + 1],
        )

    mixed = generated["mixed"].truth
    assert mixed.da_state_mask.any()
    assert mixed.de_gene_mask.any()
    assert mixed.de_protein_mask.any()

    rare = generated["rare_state"].truth
    state_frequency = rare.sample_state_counts.sum(axis=0)
    state_frequency = state_frequency / state_frequency.sum()
    assert state_frequency.min() <= 0.05

    unequal = generated["unequal_cells"].truth.sample_cell_counts
    assert unequal.max() >= 2 * unequal.min()

    continuous = generated["continuous"].truth
    observed_condition = continuous.sample_conditions[
        continuous.observed_sample_indices
    ]
    mean_difference = (
        continuous.continuous_state[observed_condition == 1].mean()
        - continuous.continuous_state[observed_condition == 0].mean()
    )
    assert mean_difference >= 0.25

    confounded = generated["batch_confounded"].truth
    assert confounded.design_matrix_rank == 3
    assert confounded.sample_batches[confounded.sample_conditions == 1].mean() > (
        confounded.sample_batches[confounded.sample_conditions == 0].mean()
    )
