"""CPU-scale end-to-end runner for controlled MrTotalVI fixtures."""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import candidate_configs
from .metrics import counterfactual_recovery_metrics, representation_metrics
from .simulation import generate_scenario

if TYPE_CHECKING:
    from typing import Literal

    from .simulation import ScenarioConfig


@dataclass(frozen=True)
class FixtureRunConfig:
    """One bounded candidate/scenario/seed training request."""

    candidate: Literal["C0", "C1", "C2", "C3", "C4"]
    scenario: ScenarioConfig
    seed: int
    max_epochs: int = 3
    batch_size: int = 64
    n_latent: int = 4
    n_hidden: int = 32
    n_prior_components: int = 8
    train_size: float = 0.8
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        """Validate dimensions and the frozen candidate name."""
        if self.candidate not in candidate_configs():
            raise ValueError(f"Unknown candidate {self.candidate!r}.")
        for name in (
            "max_epochs",
            "batch_size",
            "n_latent",
            "n_hidden",
            "n_prior_components",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if not 0.5 <= self.train_size < 1.0:
            raise ValueError("train_size must be inside [0.5, 1.0).")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.n_latent != self.scenario.latent_truth_dim:
            raise ValueError(
                "The mechanism fixture requires n_latent to equal latent_truth_dim."
            )


def _max_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _target_reorder(model, sample_names: tuple[str, ...]) -> tuple[list[str], np.ndarray]:
    target_names = [str(value) for value in model.sample_order]
    if set(target_names) != set(sample_names):
        raise ValueError("Registered model samples do not match simulation truth.")
    truth_indices = np.asarray(
        [sample_names.index(name) for name in target_names],
        dtype=np.int64,
    )
    return target_names, truth_indices


def run_candidate_fixture(config: FixtureRunConfig) -> dict:
    """Train, decode, and score one controlled MrTotalVI fixture."""
    import scvi
    from scvi.external import MrTotalVI

    started = time.perf_counter()
    rss_before = _max_rss_bytes()
    simulated = generate_scenario(config.scenario, seed=config.seed)
    adata = simulated.adata
    truth = simulated.truth
    scvi.settings.seed = truth.training_seed

    MrTotalVI.setup_anndata(
        adata,
        layer="counts",
        protein_expression_obsm_key="protein_expression",
        protein_names_uns_key="protein_names",
        sample_key="sample",
        batch_key="technical_batch",
    )
    candidate = candidate_configs()[config.candidate]
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=config.n_latent,
        n_latent_u=config.n_latent,
        n_latent_sample=config.n_latent,
        z_u_prior=True,
        u_prior_mixture=True,
        u_prior_mixture_k=config.n_prior_components,
        use_map=True,
        hierarchy_mode=candidate.hierarchy_mode,
        u_encoder_mode=candidate.u_encoder_mode,
        scale_observations=candidate.scale_observations,
        u_prior=candidate.u_prior,
        init_prior_from_data=candidate.init_prior_from_data,
        freeze_prior_after_init=candidate.freeze_prior_after_init,
        use_batch_norm="none",
        use_layer_norm="both",
        encode_covariates=True,
        n_hidden=config.n_hidden,
        n_layers_encoder=1,
        n_layers_decoder=1,
        qu_kwargs={"n_hidden": config.n_hidden, "n_layers": 1},
        qz_kwargs={"n_hidden": config.n_hidden, "n_layers": 1},
    )
    model.train(
        max_epochs=config.max_epochs,
        accelerator="cpu",
        devices=1,
        train_size=config.train_size,
        validation_size=1.0 - config.train_size,
        shuffle_set_split=True,
        batch_size=config.batch_size,
        early_stopping=False,
        plan_kwargs={"lr": config.learning_rate},
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    validation_indices = np.asarray(model.validation_indices, dtype=np.int64)
    train_indices = np.asarray(model.train_indices, dtype=np.int64)
    if len(validation_indices) == 0:
        raise RuntimeError("The fixture produced no validation cells.")
    heldout_elbo = float(
        model.get_elbo(
            indices=validation_indices,
            batch_size=config.batch_size,
        )
    )
    sample_elbos = []
    for sample_index in range(len(truth.sample_names)):
        sample_validation = validation_indices[
            truth.observed_sample_indices[validation_indices] == sample_index
        ]
        if len(sample_validation):
            sample_elbos.append(
                float(
                    model.get_elbo(
                        indices=sample_validation,
                        batch_size=config.batch_size,
                    )
                )
            )
    representation_indices = np.arange(adata.n_obs, dtype=np.int64)
    u = model.get_latent_representation(
        indices=representation_indices,
        give_z=False,
        give_mean=True,
        batch_size=config.batch_size,
    )
    representation = representation_metrics(
        u,
        state_labels=truth.cell_states,
        sample_labels=truth.observed_sample_indices,
        k=min(15, len(representation_indices) - 1),
        random_state=truth.evaluation_seed,
    )

    target_names, truth_target_indices = _target_reorder(model, truth.sample_names)
    factual_truth_targets = truth.observed_sample_indices[validation_indices]
    inverse_target_order = np.empty(len(truth_target_indices), dtype=np.int64)
    inverse_target_order[truth_target_indices] = np.arange(len(truth_target_indices))
    observed_targets = inverse_target_order[factual_truth_targets]
    truth_latent = truth.latent_by_target[
        validation_indices
    ][:, truth_target_indices]

    if candidate.hierarchy_mode == "centered_v2":
        latent = model.get_counterfactual_latent(
            indices=validation_indices,
            target_samples=target_names,
            inference_mode="latent_mean",
            n_draws=1,
            batch_size=config.batch_size,
            random_state=truth.evaluation_seed,
        )
        predicted_latent = (
            latent["z"]
            .isel(draw=0)
            .transpose("cell_name", "target_sample", "latent_dim")
            .to_numpy()
        )
        centering_max_abs = float(
            np.max(
                np.abs(
                    latent["eps_centered"]
                    .isel(draw=0)
                    .mean("target_sample")
                    .to_numpy()
                )
            )
        )
        expression = model.get_counterfactual_expression(
            indices=validation_indices,
            target_samples=target_names,
            gene_list=list(adata.var_names),
            protein_list=None,
            inference_mode="latent_mean",
            n_draws=1,
            batch_policy="observed",
            panel_policy="observed",
            library_policy="observed",
            batch_size=config.batch_size,
            random_state=truth.evaluation_seed,
        )
        predicted_rna = (
            expression["rna_scale"]
            .isel(draw=0)
            .transpose("cell_name", "target_sample", "gene")
            .to_numpy()
        )
        truth_rna = truth.rna_mean_by_target_observed_context[
            validation_indices
        ][:, truth_target_indices]
        expression_recovery = counterfactual_recovery_metrics(
            predicted_rna,
            truth=truth_rna,
            observed_target_indices=observed_targets,
        )
        normalized_composition_rmse = expression_recovery[
            "normalized_composition_rmse"
        ]
    else:
        predicted_latent = (
            model.get_local_sample_representation(
                indices=validation_indices,
                batch_size=config.batch_size,
                use_mean=True,
            )
            .sel(sample=target_names)
            .transpose("cell_name", "sample", "latent_dim")
            .to_numpy()
        )
        centering_max_abs = float("nan")
        normalized_composition_rmse = float("nan")

    latent_recovery = counterfactual_recovery_metrics(
        predicted_latent,
        truth=truth_latent,
        observed_target_indices=observed_targets,
    )
    metrics = {
        "heldout_elbo": heldout_elbo,
        "heldout_sample_elbo_sd": float(
            np.std(sample_elbos, ddof=1)
            if len(sample_elbos) > 1
            else 0.0
        ),
        "heldout_sample_elbo_range": float(
            np.ptp(sample_elbos) if sample_elbos else float("nan")
        ),
        **representation,
        "median_target_distance_spearman": latent_recovery[
            "median_target_distance_spearman"
        ],
        "mean_target_distance_spearman": latent_recovery[
            "mean_target_distance_spearman"
        ],
        "n_finite_cells": latent_recovery["n_finite_cells"],
        "normalized_composition_rmse": normalized_composition_rmse,
        "centering_max_abs": centering_max_abs,
    }
    return {
        "schema_version": "mrtotalvi-fixture-result-v1",
        "candidate": config.candidate,
        "scenario": config.scenario.scenario,
        "seed": config.seed,
        "truth_seed": truth.truth_seed,
        "training_seed": truth.training_seed,
        "evaluation_seed": truth.evaluation_seed,
        "n_train_cells": len(train_indices),
        "n_validation_cells": len(validation_indices),
        "n_representation_cells": len(representation_indices),
        "mode": {
            "hierarchy_mode": candidate.hierarchy_mode,
            "u_encoder_mode": candidate.u_encoder_mode,
            "scale_observations": candidate.scale_observations,
        },
        "metrics": metrics,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_increase_bytes": max(0, _max_rss_bytes() - rss_before),
        "scientific_scope": (
            "synthetic mechanism pilot; not biological validation; "
            "not candidate-selection evidence"
        ),
    }
