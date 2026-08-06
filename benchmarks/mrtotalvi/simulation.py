"""Exogenous-truth CITE-seq fixtures for MrTotalVI benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

SCENARIO_NAMES = (
    "null",
    "da_only",
    "de_only",
    "mixed",
    "rare_state",
    "unequal_cells",
    "continuous",
    "batch_confounded",
)
ScenarioName = Literal[
    "null",
    "da_only",
    "de_only",
    "mixed",
    "rare_state",
    "unequal_cells",
    "continuous",
    "batch_confounded",
]


@dataclass(frozen=True)
class ScenarioConfig:
    """Dimensions of one deterministic simulation scenario."""

    scenario: ScenarioName = "null"
    n_donors: int = 3
    cells_per_sample: int = 60
    n_states: int = 3
    n_genes: int = 24
    n_proteins: int = 8
    latent_truth_dim: int = 4
    effect_size: float = 0.8
    rare_state_fraction: float = 0.03
    imbalance_ratio: float = 3.0

    def __post_init__(self) -> None:
        """Validate scenario dimensions and effect controls."""
        if self.scenario not in SCENARIO_NAMES:
            raise ValueError(
                f"scenario must be one of {SCENARIO_NAMES}, got {self.scenario!r}."
            )
        for name in (
            "n_donors",
            "cells_per_sample",
            "n_states",
            "n_genes",
            "n_proteins",
            "latent_truth_dim",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if not 0.0 < self.rare_state_fraction < 0.5:
            raise ValueError("rare_state_fraction must be inside (0, 0.5).")
        if self.imbalance_ratio < 1.0:
            raise ValueError("imbalance_ratio must be at least one.")
        if self.effect_size <= 0.0:
            raise ValueError("effect_size must be positive.")


@dataclass(frozen=True)
class SimulationTruth:
    """Evaluation-only truth that is never registered as model input."""

    scenario: ScenarioName
    truth_seed: int
    training_seed: int
    evaluation_seed: int
    sample_names: tuple[str, ...]
    sample_conditions: np.ndarray
    sample_batches: np.ndarray
    sample_cell_counts: np.ndarray
    sample_state_counts: np.ndarray
    observed_sample_indices: np.ndarray
    cell_states: np.ndarray
    continuous_state: np.ndarray
    da_state_mask: np.ndarray
    de_gene_mask: np.ndarray
    de_protein_mask: np.ndarray
    design_matrix_rank: int
    latent_by_target: np.ndarray
    rna_mean_by_target: np.ndarray
    rna_mean_by_target_observed_context: np.ndarray
    protein_mean_by_target: np.ndarray


@dataclass(frozen=True)
class SimulatedCITESeq:
    """Observed AnnData and its separate exogenous evaluation truth."""

    adata: ad.AnnData
    truth: SimulationTruth


def _stream_seeds(seed: int) -> tuple[int, int, int]:
    streams = np.random.SeedSequence(seed).spawn(3)
    return tuple(
        int(stream.generate_state(1, dtype=np.uint32)[0])
        for stream in streams
    )


def _balanced_states(
    *,
    n_cells: int,
    n_states: int,
    rng: np.random.Generator,
) -> np.ndarray:
    states = np.arange(n_cells, dtype=np.int64) % n_states
    rng.shuffle(states)
    return states


def _allocated_states(
    *,
    n_cells: int,
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    expected = probabilities * n_cells
    counts = np.floor(expected).astype(np.int64)
    remainder = n_cells - int(counts.sum())
    if remainder:
        order = np.argsort(-(expected - counts), kind="stable")
        counts[order[:remainder]] += 1
    states = np.repeat(np.arange(len(probabilities), dtype=np.int64), counts)
    rng.shuffle(states)
    return states


def _state_probabilities(
    config: ScenarioConfig,
    *,
    condition: int,
) -> np.ndarray:
    probabilities = np.full(config.n_states, 1.0 / config.n_states)
    if config.scenario in {"da_only", "mixed"} and config.n_states > 1:
        delta = min(0.2, probabilities[1] * 0.8)
        if condition == 1:
            probabilities[0] += delta
            probabilities[1] -= delta
    if config.scenario == "rare_state" and config.n_states > 1:
        probabilities.fill(
            (1.0 - config.rare_state_fraction) / (config.n_states - 1)
        )
        probabilities[-1] = config.rare_state_fraction
    return probabilities


def generate_scenario(config: ScenarioConfig, *, seed: int) -> SimulatedCITESeq:
    """Generate one deterministic exogenous-truth CITE-seq fixture."""
    truth_seed, training_seed, evaluation_seed = _stream_seeds(seed)
    rng = np.random.default_rng(truth_seed)

    conditions = ("W00", "W22")
    sample_names = tuple(
        f"donor_{donor}_{condition}"
        for donor in range(config.n_donors)
        for condition in conditions
    )
    sample_conditions = np.tile(
        np.arange(len(conditions), dtype=np.int64),
        config.n_donors,
    )
    if config.scenario == "batch_confounded":
        sample_batches = np.asarray(
            [
                1 if condition == 1 or donor == config.n_donors - 1 else 0
                for donor in range(config.n_donors)
                for condition in range(len(conditions))
            ],
            dtype=np.int64,
        )
    else:
        sample_batches = np.asarray(
            [
                (donor + condition) % 2
                for donor in range(config.n_donors)
                for condition in range(len(conditions))
            ],
            dtype=np.int64,
        )

    state_embedding = rng.normal(
        0.0,
        0.7,
        size=(config.n_states, config.latent_truth_dim),
    )
    sample_embedding = rng.normal(
        0.0,
        0.25,
        size=(len(sample_names), config.latent_truth_dim),
    )
    sample_embedding -= sample_embedding.mean(axis=0, keepdims=True)
    gene_loadings = rng.normal(
        0.0,
        0.35,
        size=(config.latent_truth_dim, config.n_genes),
    )
    protein_loadings = rng.normal(
        0.0,
        0.3,
        size=(config.latent_truth_dim, config.n_proteins),
    )
    gene_intercept = rng.normal(1.0, 0.2, size=config.n_genes)
    protein_intercept = rng.normal(0.5, 0.2, size=config.n_proteins)
    batch_scale = 0.45 if config.scenario == "batch_confounded" else 0.12
    gene_batch_effect = rng.normal(0.0, batch_scale, size=(2, config.n_genes))
    protein_batch_effect = rng.normal(
        0.0,
        batch_scale * 0.8,
        size=(2, config.n_proteins),
    )
    continuous_direction = rng.normal(
        0.0,
        0.4,
        size=config.latent_truth_dim,
    )

    observed_samples: list[int] = []
    observed_states: list[int] = []
    continuous_states: list[float] = []
    cell_latents: list[np.ndarray] = []
    sample_state_counts = np.zeros(
        (len(sample_names), config.n_states),
        dtype=np.int64,
    )
    sample_cell_counts = np.empty(len(sample_names), dtype=np.int64)
    for sample_index in range(len(sample_names)):
        condition = int(sample_conditions[sample_index])
        if config.scenario == "unequal_cells":
            n_sample_cells = (
                max(2 * config.n_states, int(config.cells_per_sample / config.imbalance_ratio))
                if condition == 0
                else config.cells_per_sample
            )
        else:
            n_sample_cells = config.cells_per_sample
        sample_cell_counts[sample_index] = n_sample_cells

        if config.scenario == "continuous":
            sample_continuous = rng.normal(
                loc=0.5 * condition,
                scale=0.6,
                size=n_sample_cells,
            )
            state_edges = np.linspace(-1.0, 1.0, config.n_states + 1)[1:-1]
            states = np.digitize(sample_continuous, state_edges).astype(np.int64)
        else:
            probabilities = _state_probabilities(config, condition=condition)
            states = _allocated_states(
                n_cells=n_sample_cells,
                probabilities=probabilities,
                rng=rng,
            )
            sample_continuous = rng.normal(0.0, 0.2, size=n_sample_cells)
        sample_state_counts[sample_index] = np.bincount(
            states,
            minlength=config.n_states,
        )

        for state, continuous in zip(states, sample_continuous, strict=True):
            observed_samples.append(sample_index)
            observed_states.append(int(state))
            continuous_states.append(float(continuous))
            cell_latents.append(
                state_embedding[state]
                + continuous * continuous_direction
                + rng.normal(0.0, 0.18, size=config.latent_truth_dim)
            )

    observed_samples_array = np.asarray(observed_samples, dtype=np.int64)
    observed_states_array = np.asarray(observed_states, dtype=np.int64)
    continuous_state_array = np.asarray(continuous_states, dtype=np.float32)
    cell_latent_array = np.asarray(cell_latents)
    n_cells = len(observed_samples)
    n_targets = len(sample_names)
    latent_by_target = (
        cell_latent_array[:, None, :] + sample_embedding[None, :, :]
    ).astype(np.float32)

    rna_mean_by_target = np.empty(
        (n_cells, n_targets, config.n_genes),
        dtype=np.float32,
    )
    rna_mean_by_target_observed_context = np.empty_like(rna_mean_by_target)
    protein_mean_by_target = np.empty(
        (n_cells, n_targets, config.n_proteins),
        dtype=np.float32,
    )
    library_factors = rng.lognormal(0.0, 0.2, size=n_cells)
    da_state_mask = np.zeros(config.n_states, dtype=bool)
    if config.scenario in {"da_only", "mixed"}:
        da_state_mask[0] = True
    de_gene_mask = np.zeros(config.n_genes, dtype=bool)
    de_protein_mask = np.zeros(config.n_proteins, dtype=bool)
    if config.scenario in {"de_only", "mixed"}:
        de_gene_mask[: max(1, config.n_genes // 4)] = True
        de_protein_mask[: max(1, config.n_proteins // 3)] = True
    de_cells = observed_states_array == 0
    observed_batches = sample_batches[observed_samples_array]
    for target in range(n_targets):
        target_latent = cell_latent_array + sample_embedding[target]
        rna_log_mean = (
            gene_intercept
            + target_latent @ gene_loadings
            + gene_batch_effect[sample_batches[target]]
        )
        protein_log_mean = (
            protein_intercept
            + target_latent @ protein_loadings
            + protein_batch_effect[sample_batches[target]]
        )
        rna_log_mean_observed_context = (
            gene_intercept
            + target_latent @ gene_loadings
            + gene_batch_effect[observed_batches]
        )
        if sample_conditions[target] == 1:
            rna_log_mean[np.ix_(de_cells, de_gene_mask)] += config.effect_size
            rna_log_mean_observed_context[
                np.ix_(de_cells, de_gene_mask)
            ] += config.effect_size
            protein_log_mean[
                np.ix_(de_cells, de_protein_mask)
            ] += config.effect_size
        rna_mean_by_target[:, target] = (
            np.exp(np.clip(rna_log_mean, -4.0, 4.0))
            * library_factors[:, None]
        ).astype(np.float32)
        rna_mean_by_target_observed_context[:, target] = (
            np.exp(np.clip(rna_log_mean_observed_context, -4.0, 4.0))
            * library_factors[:, None]
        ).astype(np.float32)
        protein_mean_by_target[:, target] = np.exp(
            np.clip(protein_log_mean, -4.0, 4.0)
        ).astype(np.float32)

    cell_indices = np.arange(n_cells)
    factual_rna = rna_mean_by_target[cell_indices, observed_samples_array]
    factual_protein = protein_mean_by_target[cell_indices, observed_samples_array]
    x = rng.poisson(factual_rna).astype(np.int64)
    protein = rng.poisson(factual_protein).astype(np.int64)

    obs = pd.DataFrame(
        {
            "sample": np.asarray(sample_names, dtype=object)[observed_samples_array],
            "donor": [
                f"donor_{sample_index // 2}"
                for sample_index in observed_samples_array
            ],
            "condition": np.asarray(conditions, dtype=object)[
                sample_conditions[observed_samples_array]
            ],
            "technical_batch": [
                f"batch_{sample_batches[sample_index]}"
                for sample_index in observed_samples_array
            ],
            "state": [
                f"state_{state}"
                for state in observed_states_array
            ],
        },
        index=[
            f"sim_{seed}_cell_{cell_index:05d}"
            for cell_index in range(n_cells)
        ],
    )
    adata = ad.AnnData(
        X=x,
        obs=obs,
        var=pd.DataFrame(
            index=[f"gene_{index:03d}" for index in range(config.n_genes)]
        ),
    )
    adata.obsm["protein_expression"] = protein
    adata.uns["protein_names"] = np.asarray(
        [f"protein_{index:03d}" for index in range(config.n_proteins)],
        dtype=object,
    )
    adata.layers["counts"] = x.copy()

    design_matrix = np.column_stack(
        [
            np.ones(len(sample_names)),
            sample_conditions,
            sample_batches,
        ]
    )
    truth = SimulationTruth(
        scenario=config.scenario,
        truth_seed=truth_seed,
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        sample_names=sample_names,
        sample_conditions=sample_conditions,
        sample_batches=sample_batches,
        sample_cell_counts=sample_cell_counts,
        sample_state_counts=sample_state_counts,
        observed_sample_indices=observed_samples_array,
        cell_states=observed_states_array,
        continuous_state=continuous_state_array,
        da_state_mask=da_state_mask,
        de_gene_mask=de_gene_mask,
        de_protein_mask=de_protein_mask,
        design_matrix_rank=int(np.linalg.matrix_rank(design_matrix)),
        latent_by_target=latent_by_target,
        rna_mean_by_target=rna_mean_by_target,
        rna_mean_by_target_observed_context=rna_mean_by_target_observed_context,
        protein_mean_by_target=protein_mean_by_target,
    )
    return SimulatedCITESeq(adata=adata, truth=truth)
