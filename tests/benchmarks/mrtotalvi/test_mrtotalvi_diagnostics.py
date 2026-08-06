"""CPU smoke test for per-fit MrTotalVI redesign diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.mrtotalvi import (
    ScenarioConfig,
    best_checkpoint_identity,
    collect_mrtotalvi_diagnostics,
    generate_scenario,
    serialize_training_history,
    state_dict_digest,
    validate_checkpoint_identity,
)


def test_mrtotalvi_diagnostics_record_distinct_latents_kl_residuals_and_gradients(
    tmp_path,
):
    import scvi
    from scvi.external import MrTotalVI

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
        seed=17,
    )
    adata = simulated.adata
    validation = np.asarray(
        [
            index
            for index in range(adata.n_obs)
            if index % 8 in {6, 7}
        ],
        dtype=np.int64,
    )
    train = np.setdiff1d(np.arange(adata.n_obs), validation)
    scvi.settings.seed = 23
    MrTotalVI.setup_anndata(
        adata,
        layer="counts",
        protein_expression_obsm_key="protein_expression",
        protein_names_uns_key="protein_names",
        sample_key="sample",
        batch_key="technical_batch",
    )
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=2,
        n_latent_u=2,
        n_latent_sample=2,
        z_u_prior=True,
        u_prior_mixture=True,
        u_prior_mixture_k=4,
        use_map=True,
        hierarchy_mode="centered_v2",
        u_encoder_mode="sample_blind",
        scale_observations=False,
        u_prior="vamp",
        init_prior_from_data=True,
        freeze_prior_after_init=True,
        use_batch_norm="none",
        use_layer_norm="both",
        encode_covariates=True,
        n_hidden=8,
        n_layers_encoder=1,
        n_layers_decoder=1,
        qu_kwargs={"n_hidden": 8, "n_layers": 1},
        qz_kwargs={"n_hidden": 8, "n_layers": 1},
    )
    model.train(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        train_size=None,
        validation_size=None,
        shuffle_set_split=False,
        batch_size=16,
        early_stopping=False,
        external_indexing=[
            train,
            validation,
            np.asarray([], dtype=np.int64),
        ],
        plan_kwargs={"lr": 1e-3},
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    history = serialize_training_history(model.history)
    checkpoint_path = tmp_path / "best-0000"
    model.save(checkpoint_path, save_anndata=False)
    identity = best_checkpoint_identity(
        history,
        monitor="elbo_validation",
        mode="min",
        state_digest=state_dict_digest(model.module.state_dict()),
        artifact_name=checkpoint_path.name,
    )
    assert validate_checkpoint_identity(
        history,
        identity,
        current_state_dict=model.module.state_dict(),
        checkpoint_path=checkpoint_path,
    ) == identity

    wrong = dict(identity)
    wrong["state_digest"] = "a" * 64
    with pytest.raises(ValueError, match="current model state"):
        validate_checkpoint_identity(
            history,
            wrong,
            current_state_dict=model.module.state_dict(),
            checkpoint_path=checkpoint_path,
        )
    wrong = dict(identity)
    wrong["epoch"] = int(identity["epoch"]) + 1
    with pytest.raises(ValueError, match="best history record"):
        validate_checkpoint_identity(
            history,
            wrong,
            current_state_dict=model.module.state_dict(),
            checkpoint_path=checkpoint_path,
        )
    wrong = dict(identity)
    wrong["value"] = float(identity["value"]) + 1.0
    with pytest.raises(ValueError, match="best history record"):
        validate_checkpoint_identity(
            history,
            wrong,
            current_state_dict=model.module.state_dict(),
            checkpoint_path=checkpoint_path,
        )
    with pytest.raises(ValueError, match="requires exactly"):
        validate_checkpoint_identity(
            history,
            {"nonsense": 1},
            current_state_dict=model.module.state_dict(),
            checkpoint_path=checkpoint_path,
        )
    import torch

    original_state = {
        name: value.detach().clone()
        for name, value in model.module.state_dict().items()
    }
    with torch.no_grad():
        next(model.module.parameters()).add_(1.0)
    mutated_identity = best_checkpoint_identity(
        history,
        monitor="elbo_validation",
        mode="min",
        state_digest=state_dict_digest(model.module.state_dict()),
        artifact_name=checkpoint_path.name,
    )
    with pytest.raises(ValueError, match="saved best checkpoint artifact"):
        validate_checkpoint_identity(
            history,
            mutated_identity,
            current_state_dict=model.module.state_dict(),
            checkpoint_path=checkpoint_path,
        )
    model.module.load_state_dict(original_state)

    result = collect_mrtotalvi_diagnostics(
        model,
        candidate_id="D0",
        validation_indices=validation,
        gradient_indices=train,
        batch_size=16,
        posterior_samples=2,
        posterior_predictive_draws=2,
        evaluation_seed=29,
        evaluation_annotations={
            "state_labels": simulated.truth.cell_states[validation],
            "sample_labels": simulated.truth.observed_sample_indices[validation],
            "technical_batch_labels": simulated.truth.sample_batches[
                simulated.truth.observed_sample_indices[validation]
            ],
        },
        training_history=model.history,
        checkpoint_identity=identity,
        checkpoint_path=checkpoint_path,
        wall_time_seconds=1.0,
        peak_memory_bytes=1,
    )

    assert set(result["representations"]) == {"u", "factual_z"}
    assert not np.shares_memory(
        result["representations"]["u"]["values"],
        result["representations"]["factual_z"]["values"],
    )
    metrics = result["metrics"]
    for name in (
        "rna_reconstruction_loss",
        "protein_reconstruction_loss",
        "kl_z",
        "kl_u",
        "u_posterior_scale",
        "factual_z_posterior_scale",
        "registered_residual_magnitude",
        "registered_residual_gradient_norm",
        "u_effective_rank",
        "factual_z_effective_rank",
        "u_state_balanced_accuracy",
        "factual_z_state_balanced_accuracy",
    ):
        assert np.isfinite(metrics[name]), name
    assert metrics["registered_residual_gradient_norm"] > 0.0
    assert metrics["registered_residual_gradient_coverage"] == 1.0
    assert metrics["centering_max_abs"] <= 1e-6
