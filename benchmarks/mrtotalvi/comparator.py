"""Matched stock scVI and TotalVI comparator contracts and runners."""

from __future__ import annotations

import hashlib
import json
import random
import re
import resource
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .diagnostics import (
    heldout_prediction_metrics,
    latent_diagnostics,
    latent_diagnostics_v2,
    representation_diagnostics,
)
from .metric_schema import metric_payload_template, validate_metric_payload
from .redesign_contract import redesign_candidate_configs
from .versioning import (
    RedesignContractAdapter,
    historical_redesign_contract_adapter,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Literal

    from anndata import AnnData


@dataclass(frozen=True)
class StockComparatorSpec:
    """Frozen training-input boundary for one stock comparator."""

    candidate_id: Literal["B0", "B1"]
    model_family: Literal["scvi", "totalvi"]
    modalities: tuple[str, ...]
    count_layer: str
    protein_obsm_key: str | None
    protein_names_uns_key: str | None
    biological_sample_key: None
    gene_likelihood: Literal["nb"]


_STOCK_COMPARATOR_SPECS = {
    "B0": StockComparatorSpec(
        candidate_id="B0",
        model_family="scvi",
        modalities=("rna",),
        count_layer="counts",
        protein_obsm_key=None,
        protein_names_uns_key=None,
        biological_sample_key=None,
        gene_likelihood="nb",
    ),
    "B1": StockComparatorSpec(
        candidate_id="B1",
        model_family="totalvi",
        modalities=("rna", "protein"),
        count_layer="counts",
        protein_obsm_key="protein_expression",
        protein_names_uns_key="protein_names",
        biological_sample_key=None,
        gene_likelihood="nb",
    ),
}
FROZEN_OPTIMIZER_NAME = "Adam"
FROZEN_OPTIMIZER_CLASS = "torch.optim.adam.Adam"
FROZEN_WEIGHT_DECAY = 1e-6
FROZEN_OPTIMIZER_EPS = 0.01
FROZEN_OPTIMIZER_BETAS = (0.9, 0.999)
MRTOTALVI_FACTUAL_Z_POSTERIOR_DRAWS = 32
MRTOTALVI_FACTUAL_Z_POSTERIOR_DDOF = 1


def stock_comparator_spec(candidate_id: str) -> StockComparatorSpec:
    """Return an exact B0/B1 spec and reject every non-stock row."""
    try:
        return _STOCK_COMPARATOR_SPECS[candidate_id]
    except KeyError as error:
        raise ValueError(
            f"{candidate_id!r} is not a stock comparator; expected B0 or B1."
        ) from error


@dataclass(frozen=True)
class ComparatorRunConfig:
    """One exact-split stock comparator training request."""

    candidate_id: Literal["B0", "B1"]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    max_epochs: int
    n_latent: int
    n_hidden: int
    n_layers: int
    batch_size: int
    learning_rate: float
    technical_batch_key: str
    training_seed: int
    evaluation_seed: int
    check_val_every_n_epoch: int = 1
    minimum_epochs: int = 1
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    posterior_predictive_draws: int = 32
    explicit_totalvi_optimization: bool = False

    def __post_init__(self) -> None:
        """Reject unknown rows and invalid shared hyperparameters."""
        stock_comparator_spec(self.candidate_id)
        for name in (
            "max_epochs",
            "n_latent",
            "n_hidden",
            "n_layers",
            "batch_size",
            "check_val_every_n_epoch",
            "minimum_epochs",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.minimum_epochs > self.max_epochs:
            raise ValueError("minimum_epochs cannot exceed max_epochs.")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be positive or None.")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be nonnegative.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if (
            not isinstance(self.technical_batch_key, str)
            or not self.technical_batch_key
        ):
            raise ValueError("technical_batch_key must be a non-empty string.")
        for name in ("training_seed", "evaluation_seed"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.training_seed == self.evaluation_seed:
            raise ValueError(
                "training_seed and evaluation_seed must identify distinct streams."
            )
        if self.posterior_predictive_draws < 2:
            raise ValueError("posterior_predictive_draws must be at least two.")
        if (
            self.explicit_totalvi_optimization
            and self.candidate_id != "B1"
        ):
            raise ValueError(
                "Explicit TotalVI optimization applies only to B1."
            )

    @property
    def spec(self) -> StockComparatorSpec:
        """Return this request's frozen stock model spec."""
        return stock_comparator_spec(self.candidate_id)


def frozen_totalvi_optimization_kwargs(n_obs: int) -> dict[str, object]:
    """Return explicit optimizer/warmup controls shared by RDX-03 rows."""
    if isinstance(n_obs, bool) or not isinstance(n_obs, int) or n_obs < 2:
        raise ValueError("n_obs must be an integer of at least two.")
    return {
        "n_steps_kl_warmup": int(0.75 * n_obs),
        "n_epochs_kl_warmup": None,
        "adversarial_classifier": False,
        "plan_kwargs": {
            "optimizer": FROZEN_OPTIMIZER_NAME,
            "weight_decay": FROZEN_WEIGHT_DECAY,
            "eps": FROZEN_OPTIMIZER_EPS,
            "gradient_clip_norm": None,
        },
    }


def _realized_totalvi_optimization_identity(
    model,
    *,
    n_obs: int,
    learning_rate: float,
) -> dict[str, object]:
    """Verify and serialize the optimizer realized by a completed fit."""
    declared_kwargs = frozen_totalvi_optimization_kwargs(n_obs)
    optimizers = list(model.trainer.optimizers)
    if len(optimizers) != 1:
        raise ValueError("RDX-03 requires exactly one realized optimizer.")
    optimizer = optimizers[0]
    optimizer_class = (
        f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    )
    parameter_groups = []
    for group in optimizer.param_groups:
        parameter_groups.append(
            {
                "lr": float(group["lr"]),
                "betas": [float(value) for value in group["betas"]],
                "eps": float(group["eps"]),
                "weight_decay": float(group["weight_decay"]),
                "amsgrad": bool(group["amsgrad"]),
            }
        )
    expected_group = {
        "lr": float(learning_rate),
        "betas": list(FROZEN_OPTIMIZER_BETAS),
        "eps": FROZEN_OPTIMIZER_EPS,
        "weight_decay": FROZEN_WEIGHT_DECAY,
        "amsgrad": False,
    }
    if (
        optimizer_class != FROZEN_OPTIMIZER_CLASS
        or parameter_groups != [expected_group]
    ):
        raise ValueError("Realized optimizer drifted from the frozen contract.")

    plan = model.trainer.lightning_module
    scheduler_configs = list(model.trainer.lr_scheduler_configs)
    if scheduler_configs:
        raise ValueError("RDX-03 forbids a realized learning-rate scheduler.")
    realized_plan = {
        "class": f"{type(plan).__module__}.{type(plan).__qualname__}",
        "n_steps_kl_warmup": plan.n_steps_kl_warmup,
        "n_epochs_kl_warmup": plan.n_epochs_kl_warmup,
        "reduce_lr_on_plateau": bool(plan.reduce_lr_on_plateau),
        "adversarial_classifier": (
            False if plan.adversarial_classifier is False else "enabled"
        ),
        "gradient_clip_norm": plan.gradient_clip_norm,
    }
    expected_plan = {
        "class": "scvi.train._trainingplans.AdversarialTrainingPlan",
        "n_steps_kl_warmup": declared_kwargs["n_steps_kl_warmup"],
        "n_epochs_kl_warmup": None,
        "reduce_lr_on_plateau": False,
        "adversarial_classifier": False,
        "gradient_clip_norm": None,
    }
    if realized_plan != expected_plan:
        raise ValueError(
            "Realized training plan drifted from the frozen contract."
        )
    identity = {
        "schema_version": "mrtotalvi-optimization-identity-v1",
        "declared": {
            "optimizer": FROZEN_OPTIMIZER_NAME,
            "optimizer_class": FROZEN_OPTIMIZER_CLASS,
            "learning_rate": float(learning_rate),
            "weight_decay": FROZEN_WEIGHT_DECAY,
            "eps": FROZEN_OPTIMIZER_EPS,
            "betas": list(FROZEN_OPTIMIZER_BETAS),
            "n_steps_kl_warmup": declared_kwargs[
                "n_steps_kl_warmup"
            ],
            "n_epochs_kl_warmup": None,
            "adversarial_classifier": False,
            "reduce_lr_on_plateau": False,
            "gradient_clip_norm": None,
            "scheduler_enabled": False,
        },
        "realized": {
            "optimizer_count": 1,
            "optimizer_class": optimizer_class,
            "parameter_groups": parameter_groups,
            "training_plan": realized_plan,
            "scheduler_states": [],
        },
    }
    return validate_frozen_optimization_identity(
        identity,
        n_obs=n_obs,
        learning_rate=learning_rate,
    )


def validate_frozen_optimization_identity(
    identity: Mapping[str, object],
    *,
    n_obs: int,
    learning_rate: float,
) -> dict[str, object]:
    """Validate a persisted optimizer identity without trusting live defaults."""
    expected = frozen_optimization_identity(
        n_obs=n_obs,
        learning_rate=learning_rate,
    )
    if identity != expected:
        raise ValueError(
            "Persisted optimizer/training-plan identity drifted."
        )
    return dict(identity)


def frozen_optimization_identity(
    *,
    n_obs: int,
    learning_rate: float,
) -> dict[str, object]:
    """Return the exact persisted RDX-03 optimizer identity."""
    declared_kwargs = frozen_totalvi_optimization_kwargs(n_obs)
    expected_group = {
        "lr": float(learning_rate),
        "betas": list(FROZEN_OPTIMIZER_BETAS),
        "eps": FROZEN_OPTIMIZER_EPS,
        "weight_decay": FROZEN_WEIGHT_DECAY,
        "amsgrad": False,
    }
    return {
        "schema_version": "mrtotalvi-optimization-identity-v1",
        "declared": {
            "optimizer": FROZEN_OPTIMIZER_NAME,
            "optimizer_class": FROZEN_OPTIMIZER_CLASS,
            "learning_rate": float(learning_rate),
            "weight_decay": FROZEN_WEIGHT_DECAY,
            "eps": FROZEN_OPTIMIZER_EPS,
            "betas": list(FROZEN_OPTIMIZER_BETAS),
            "n_steps_kl_warmup": declared_kwargs[
                "n_steps_kl_warmup"
            ],
            "n_epochs_kl_warmup": None,
            "adversarial_classifier": False,
            "reduce_lr_on_plateau": False,
            "gradient_clip_norm": None,
            "scheduler_enabled": False,
        },
        "realized": {
            "optimizer_count": 1,
            "optimizer_class": FROZEN_OPTIMIZER_CLASS,
            "parameter_groups": [expected_group],
            "training_plan": {
                "class": (
                    "scvi.train._trainingplans."
                    "AdversarialTrainingPlan"
                ),
                "n_steps_kl_warmup": declared_kwargs[
                    "n_steps_kl_warmup"
                ],
                "n_epochs_kl_warmup": None,
                "reduce_lr_on_plateau": False,
                "adversarial_classifier": False,
                "gradient_clip_norm": None,
            },
            "scheduler_states": [],
        },
    }


def _integer_indices(values, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
    if not np.issubdtype(raw.dtype, np.integer):
        try:
            numeric = raw.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must contain integers.") from error
        if not np.all(np.equal(numeric, np.floor(numeric))):
            raise ValueError(f"{name} must contain integers.")
    result = raw.astype(np.int64)
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices.")
    return result


def validate_external_split(
    train_indices,
    validation_indices,
    *,
    n_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Require disjoint, duplicate-free, complete train/validation coverage."""
    if n_obs < 2:
        raise ValueError("n_obs must be at least two.")
    train = _integer_indices(train_indices, name="train_indices")
    validation = _integer_indices(validation_indices, name="validation_indices")
    for name, values in (("train_indices", train), ("validation_indices", validation)):
        if np.any(values < 0) or np.any(values >= n_obs):
            raise ValueError(f"{name} contain out-of-bounds indices.")
    if np.intersect1d(train, validation).size:
        raise ValueError("Training and validation indices overlap.")
    combined = np.concatenate([train, validation])
    if len(combined) != n_obs or not np.array_equal(
        np.sort(combined),
        np.arange(n_obs, dtype=np.int64),
    ):
        raise ValueError(
            "Training and validation indices must form a complete partition."
        )
    return train, validation


def _representation_payload(
    cell_ids,
    values,
    *,
    allow_integrity_failures: bool = False,
) -> dict[str, np.ndarray]:
    cells = np.asarray(cell_ids, dtype=str)
    embedding = np.asarray(values, dtype=np.float64)
    if (
        cells.ndim != 1
        or embedding.ndim != 2
        or len(cells) != len(embedding)
        or len(np.unique(cells)) != len(cells)
    ):
        raise ValueError("Representation values require unique, cell-aligned IDs.")
    if not allow_integrity_failures and not np.all(np.isfinite(embedding)):
        raise ValueError("Representation values must be finite.")
    return {"cell_ids": cells.copy(), "values": embedding.copy()}


@contextmanager
def _preserved_export_rng(model):
    """Restore Python, NumPy, torch CPU, and active CUDA RNG streams."""
    import torch

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = []
    module = getattr(model, "module", None)
    module_device = getattr(module, "device", None)
    if getattr(module_device, "type", None) == "cuda":
        devices.append(module_device)
    try:
        with torch.random.fork_rng(devices=devices, enabled=True):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def export_named_representations(
    model,
    *,
    candidate_id: str,
    cell_ids,
    indices,
    batch_size: int,
    allow_integrity_failures: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """Export factual ``z`` and, only where defined, ``u`` under distinct names."""
    configs = redesign_candidate_configs()
    if candidate_id not in configs:
        raise ValueError(f"Unknown redesign row {candidate_id!r}.")
    candidate = configs[candidate_id]
    index_values = _integer_indices(indices, name="indices")
    cells = np.asarray(cell_ids, dtype=str)
    if len(cells) != len(index_values):
        raise ValueError("cell_ids must align with requested indices.")

    common = {
        "indices": index_values,
        "give_mean": True,
        "batch_size": batch_size,
    }
    with _preserved_export_rng(model):
        if candidate.model_family in {"scvi", "totalvi"}:
            factual_z = model.get_latent_representation(**common)
            return {
                "factual_z": _representation_payload(
                    cells,
                    factual_z,
                    allow_integrity_failures=allow_integrity_failures,
                )
            }

        u = model.get_latent_representation(give_z=False, **common)
        factual_z = model.get_latent_representation(give_z=True, **common)
        return {
            "u": _representation_payload(
                cells,
                u,
                allow_integrity_failures=allow_integrity_failures,
            ),
            "factual_z": _representation_payload(
                cells,
                factual_z,
                allow_integrity_failures=allow_integrity_failures,
            ),
        }


def _history_values(value, *, metric_name: str) -> tuple[list, list]:
    if hasattr(value, "columns") and hasattr(value, "index"):
        columns = list(value.columns)
        if metric_name in columns:
            series = value[metric_name]
        elif len(columns) == 1:
            series = value[columns[0]]
        else:
            raise ValueError(
                f"History metric {metric_name!r} has ambiguous columns {columns!r}."
            )
        return list(series.index), list(series.to_numpy())
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"History metric {metric_name!r} must be one-dimensional.")
    return list(range(len(array))), list(array)


def serialize_training_history(history: Mapping) -> dict[str, list[dict[str, float | int]]]:
    """Serialize every logged history value without dropping intermediate epochs."""
    if not isinstance(history, Mapping) or not history:
        raise ValueError("Training history must be a non-empty mapping.")
    serialized: dict[str, list[dict[str, float | int]]] = {}
    for metric_name in sorted(history):
        indices, values = _history_values(
            history[metric_name],
            metric_name=str(metric_name),
        )
        records = []
        for index, value in zip(indices, values, strict=True):
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(
                    f"Training history metric {metric_name!r} contains nonfinite values."
                )
            if isinstance(index, (int, np.integer)):
                epoch: int | float = int(index)
            else:
                epoch_value = float(index)
                epoch = (
                    int(epoch_value)
                    if epoch_value.is_integer()
                    else epoch_value
                )
            records.append({"epoch": epoch, "value": numeric})
        if not records:
            raise ValueError(f"Training history metric {metric_name!r} is empty.")
        serialized[str(metric_name)] = records
    return serialized


def _validate_sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal.") from error


def best_checkpoint_identity(
    serialized_history: Mapping[str, Sequence[Mapping]],
    *,
    monitor: str,
    mode: Literal["min", "max"],
    state_digest: str,
    artifact_name: str,
) -> dict[str, str | float | int]:
    """Identify the best logged epoch and bind it to an exact state digest."""
    _validate_sha256(state_digest, name="state_digest")
    if mode not in {"min", "max"}:
        raise ValueError("Checkpoint mode must be 'min' or 'max'.")
    if monitor not in serialized_history:
        raise ValueError(f"Checkpoint monitor {monitor!r} is absent from history.")
    records = list(serialized_history[monitor])
    if not records:
        raise ValueError("Checkpoint monitor history is empty.")
    if (
        not isinstance(artifact_name, str)
        or not artifact_name
        or Path(artifact_name).name != artifact_name
    ):
        raise ValueError("artifact_name must be one non-empty basename.")
    selector = min if mode == "min" else max
    best = selector(records, key=lambda record: float(record["value"]))
    return {
        "monitor": monitor,
        "mode": mode,
        "epoch": best["epoch"],
        "value": float(best["value"]),
        "state_digest": state_digest,
        "artifact_name": artifact_name,
    }


def state_dict_digest(state_dict: Mapping[str, object]) -> str:
    """Hash tensor names, dtypes, shapes, and canonical CPU bytes."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("state_dict must be a non-empty mapping.")
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous().numpy()
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"State value {name!r} has unsupported object dtype.")
        contiguous = np.ascontiguousarray(array)
        metadata = json.dumps(
            {
                "name": str(name),
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(contiguous.tobytes()).to_bytes(8, "big"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _checkpoint_artifact_state_digest(checkpoint_path: Path) -> str:
    if not checkpoint_path.is_dir():
        raise ValueError(
            f"Checkpoint artifact is not a directory: {checkpoint_path}"
        )
    model_path = checkpoint_path / "model.pt"
    if not model_path.is_file():
        raise ValueError(
            f"Checkpoint artifact has no model.pt: {checkpoint_path}"
        )
    from scvi.model.base._save_load import _load_saved_files

    _, _, saved_state, _ = _load_saved_files(
        str(checkpoint_path),
        load_adata=False,
        map_location="cpu",
    )
    if not isinstance(saved_state, Mapping):
        raise ValueError("Checkpoint model state is not a mapping.")
    saved_state = dict(saved_state)
    saved_state.pop("pyro_param_store", None)
    return state_dict_digest(saved_state)


def validate_checkpoint_identity(
    serialized_history: Mapping[str, Sequence[Mapping]],
    checkpoint_identity: Mapping,
    *,
    current_state_dict: Mapping[str, object],
    checkpoint_path: str | Path,
    monitor: str = "elbo_validation",
    mode: Literal["min", "max"] = "min",
) -> dict[str, str | float | int]:
    """Bind one best-history record to both live and saved model state."""
    expected_fields = {
        "monitor",
        "mode",
        "epoch",
        "value",
        "state_digest",
        "artifact_name",
    }
    if (
        not isinstance(checkpoint_identity, Mapping)
        or set(checkpoint_identity) != expected_fields
    ):
        raise ValueError(
            "Checkpoint identity requires exactly monitor, mode, epoch, value, "
            "state_digest, and artifact_name."
        )
    if checkpoint_identity["monitor"] != monitor:
        raise ValueError("Checkpoint monitor does not match the required monitor.")
    if checkpoint_identity["mode"] != mode:
        raise ValueError("Checkpoint mode does not match the required mode.")
    live_digest = state_dict_digest(current_state_dict)
    expected = best_checkpoint_identity(
        serialized_history,
        monitor=monitor,
        mode=mode,
        state_digest=live_digest,
        artifact_name=str(checkpoint_identity["artifact_name"]),
    )
    if dict(checkpoint_identity) != expected:
        raise ValueError(
            "Checkpoint identity does not match the exact best history record "
            "and current model state."
        )
    artifact_path = Path(checkpoint_path)
    if artifact_path.name != checkpoint_identity["artifact_name"]:
        raise ValueError(
            "Checkpoint artifact basename does not match checkpoint identity."
        )
    artifact_digest = _checkpoint_artifact_state_digest(artifact_path)
    if artifact_digest != live_digest:
        raise ValueError(
            "Current model state does not match the saved best checkpoint artifact."
        )
    return expected


_SCVI_FORBIDDEN_METRICS = {
    "protein_reconstruction_loss",
    "protein_heldout_negative_log_likelihood",
    "protein_posterior_predictive_calibration",
    "multimodal_heldout_predictive_loss",
    "multimodal_elbo",
}


def validate_metric_comparability(model_family: str, metric_id: str) -> None:
    """Reject RNA-only scVI from every protein or multimodal ranking."""
    if model_family not in {"scvi", "totalvi", "mrtotalvi"}:
        raise ValueError(f"Unknown model family {model_family!r}.")
    if not isinstance(metric_id, str) or not metric_id:
        raise ValueError("metric_id must be a non-empty string.")
    if model_family == "scvi" and metric_id in _SCVI_FORBIDDEN_METRICS:
        raise ValueError(
            f"RNA-only scVI cannot be ranked by {metric_id!r}; "
            "use an RNA-only estimand."
        )


def _max_rss_bytes() -> int:
    multiplier = 1 if __import__("sys").platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier)


def _dense_counts(values) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError("Observed counts must be a finite nonnegative matrix.")
    return result


def _stock_validation_arrays(model, *, indices: np.ndarray, batch_size: int):
    """Collect exact per-feature held-out log probabilities and posterior moments."""
    import torch

    from scvi import REGISTRY_KEYS
    from scvi.distributions import NegativeBinomial, NegativeBinomialMixture
    from scvi.module._constants import MODULE_KEYS

    rna_observed = []
    rna_log_prob = []
    protein_observed = []
    protein_log_prob = []
    posterior_mean = []
    posterior_scale = []
    kl_z = []
    kl_u = []
    residual_absolute_sum = 0.0
    residual_value_count = 0
    centering_max_abs = 0.0
    loader = model._make_data_loader(
        adata=model.adata,
        indices=indices,
        batch_size=batch_size,
    )
    was_training = model.module.training
    model.module.eval()
    try:
        with torch.inference_mode():
            for tensors in loader:
                inference, generative, loss = model.module(
                    tensors,
                    loss_kwargs={"kl_weight": 1.0},
                )
                qz = inference[MODULE_KEYS.QZ_KEY]
                posterior_mean.append(qz.loc.detach().cpu().numpy())
                posterior_scale.append(qz.scale.detach().cpu().numpy())
                x = tensors[REGISTRY_KEYS.X_KEY]
                rna_observed.append(x.detach().cpu().numpy())
                if MODULE_KEYS.PX_KEY in generative:
                    rna_dist = generative[MODULE_KEYS.PX_KEY]
                else:
                    px = generative["px_"]
                    rna_dist = NegativeBinomial(mu=px["rate"], theta=px["r"])
                rna_log_prob.append(
                    rna_dist.log_prob(x).detach().cpu().numpy()
                )
                if REGISTRY_KEYS.PROTEIN_EXP_KEY in tensors:
                    y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
                    py = generative["py_"]
                    efficiency = generative["per_batch_efficiency"]
                    rate_back = py["rate_back"]
                    rate_fore = py["rate_fore"]
                    if efficiency is not None:
                        rate_back = efficiency * rate_back
                        rate_fore = efficiency * rate_fore
                    protein_dist = NegativeBinomialMixture(
                        mu1=rate_back,
                        mu2=rate_fore,
                        theta1=py["r"],
                        mixture_logits=py["mixing"],
                    )
                    protein_observed.append(y.detach().cpu().numpy())
                    protein_log_prob.append(
                        protein_dist.log_prob(y).detach().cpu().numpy()
                    )
                kl_mapping = loss.kl_local
                z_key = (
                    "kl_div_z"
                    if "kl_div_z" in kl_mapping
                    else MODULE_KEYS.KL_Z_KEY
                )
                combined_kl = kl_mapping[z_key]
                if hasattr(model.module, "kl_u") and "u" in inference:
                    batch_kl_u = model.module.kl_u(
                        inference.get("qu", qz),
                        inference["u"],
                        tensors[REGISTRY_KEYS.LABELS_KEY],
                    )
                    u_weight = float(model.module.kl_u_weight)
                    z_weight = float(model.module.kl_z_weight)
                    if z_weight <= 0.0:
                        raise RuntimeError(
                            "MrTotalVI kl_z_weight must be positive for diagnostics."
                        )
                    batch_kl_z = (
                        combined_kl - u_weight * batch_kl_u
                    ) / z_weight
                    kl_u.append(batch_kl_u.detach().cpu().numpy())
                    kl_z.append(batch_kl_z.detach().cpu().numpy())

                    _, residual_raw, residual_centered, _ = (
                        model.module._all_sample_residuals(qz.loc)
                    )
                    residual = (
                        residual_centered
                        if model.module.hierarchy_mode == "centered_v2"
                        else residual_raw
                    )
                    residual_absolute_sum += float(residual.abs().sum().item())
                    residual_value_count += int(residual.numel())
                    if model.module.hierarchy_mode == "centered_v2":
                        sample_dim = 1 if residual_centered.ndim == 3 else 2
                        centering_max_abs = max(
                            centering_max_abs,
                            float(
                                residual_centered.mean(dim=sample_dim)
                                .abs()
                                .max()
                                .item()
                            ),
                        )
                else:
                    kl_z.append(combined_kl.detach().cpu().numpy())
    finally:
        model.module.train(was_training)
    return {
        "rna_observed": np.concatenate(rna_observed),
        "rna_log_prob": np.concatenate(rna_log_prob),
        "protein_observed": (
            np.concatenate(protein_observed) if protein_observed else None
        ),
        "protein_log_prob": (
            np.concatenate(protein_log_prob) if protein_log_prob else None
        ),
        "posterior_mean": np.concatenate(posterior_mean),
        "posterior_scale": np.concatenate(posterior_scale),
        "kl_z": float(np.mean(np.concatenate(kl_z))),
        "kl_u": (
            float(np.mean(np.concatenate(kl_u))) if kl_u else None
        ),
        "registered_residual_magnitude": (
            residual_absolute_sum / residual_value_count
            if residual_value_count
            else None
        ),
        "centering_max_abs": (
            centering_max_abs
            if residual_value_count
            and getattr(model.module, "hierarchy_mode", None) == "centered_v2"
            else None
        ),
    }


def _mrtotalvi_factual_posterior_scale(
    model,
    *,
    indices: np.ndarray,
    batch_size: int,
    n_samples: int,
    random_state: int,
) -> np.ndarray:
    import torch

    from scvi.module._constants import MODULE_KEYS

    if n_samples < 2:
        raise ValueError("Factual-z posterior scale requires at least two samples.")
    scales = []
    loader = model._make_data_loader(
        adata=model.adata,
        indices=indices,
        batch_size=batch_size,
    )
    was_training = model.module.training
    model.module.eval()
    devices = (
        [model.module.device]
        if model.module.device.type == "cuda"
        else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(random_state)
            with torch.inference_mode():
                for tensors in loader:
                    inputs = model.module._get_inference_input(tensors)
                    inference = model.module.inference(
                        **inputs,
                        n_samples=n_samples,
                    )
                    factual_z = inference[MODULE_KEYS.Z_KEY]
                    if factual_z.ndim != 3:
                        raise RuntimeError(
                            "MrTotalVI posterior sampling did not return "
                            "(draws, cells, latent) factual z."
                        )
                    scales.append(
                        factual_z.std(
                            dim=0,
                            correction=MRTOTALVI_FACTUAL_Z_POSTERIOR_DDOF,
                        )
                        .cpu()
                        .numpy()
                    )
    finally:
        model.module.train(was_training)
    return np.concatenate(scales)


def _mrtotalvi_residual_gradient_diagnostics(
    model,
    *,
    indices: np.ndarray,
    batch_size: int,
    allow_integrity_failures: bool = False,
) -> tuple[float | None, float | None]:
    import torch

    if not hasattr(model.module, "qz") or not hasattr(
        model.module.qz,
        "embedding",
    ):
        if allow_integrity_failures:
            return None, None
        raise ValueError("Model has no registered residual embedding table.")
    accumulated = torch.zeros(
        model.module.qz.embedding.num_embeddings,
        dtype=torch.float64,
    )
    loader = model._make_data_loader(
        adata=model.adata,
        indices=indices,
        batch_size=batch_size,
        shuffle=False,
    )
    was_training = model.module.training
    model.module.eval()
    invalid_gradient = False
    try:
        for tensors in loader:
            model.module.zero_grad(set_to_none=True)
            _, _, loss = model.module(
                tensors,
                loss_kwargs={"kl_weight": 1.0},
            )
            loss.loss.backward()
            gradient = model.module.qz.embedding.weight.grad
            if gradient is None or not torch.isfinite(gradient).all():
                if allow_integrity_failures:
                    invalid_gradient = True
                    break
                raise RuntimeError(
                    "Registered residual embedding gradient is missing or nonfinite."
                )
            accumulated += gradient.detach().double().square().sum(dim=1).cpu()
    finally:
        model.module.zero_grad(set_to_none=True)
        model.module.train(was_training)
    if invalid_gradient:
        return None, None
    norm = float(torch.sqrt(accumulated.sum()).item())
    coverage = float((accumulated > 0.0).double().mean().item())
    return norm, coverage


def _posterior_predictive_draws(
    model,
    *,
    model_family: str,
    indices: np.ndarray,
    batch_size: int,
    n_draws: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    device = model.module.device
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(random_state)
        sampled = model.posterior_predictive_sample(
            indices=indices,
            n_samples=n_draws,
            batch_size=batch_size,
        )
    if model_family == "scvi":
        if hasattr(sampled, "todense"):
            sampled = sampled.todense()
        rna = np.asarray(sampled)
        if rna.shape[-1] != n_draws:
            raise RuntimeError("Unexpected scVI posterior-predictive sample shape.")
        return np.moveaxis(rna, -1, 0), None
    rna = np.asarray(sampled["rna"])
    protein = np.asarray(sampled["protein"])
    if rna.shape[-1] != n_draws or protein.shape[-1] != n_draws:
        raise RuntimeError("Unexpected TotalVI posterior-predictive sample shape.")
    return np.moveaxis(rna, -1, 0), np.moveaxis(protein, -1, 0)


def _construct_stock_model(adata: AnnData, config: ComparatorRunConfig):
    spec = config.spec
    if config.technical_batch_key not in adata.obs:
        raise ValueError(
            "AnnData is missing technical batch column "
            f"{config.technical_batch_key!r}."
        )
    if spec.model_family == "scvi":
        from scvi.model import SCVI

        SCVI.setup_anndata(
            adata,
            layer=spec.count_layer,
            batch_key=config.technical_batch_key,
        )
        return SCVI(
            adata,
            n_latent=config.n_latent,
            n_hidden=config.n_hidden,
            n_layers=config.n_layers,
            gene_likelihood=spec.gene_likelihood,
        )

    from scvi.model import TOTALVI

    if spec.protein_obsm_key not in adata.obsm:
        raise ValueError(
            f"AnnData is missing protein matrix {spec.protein_obsm_key!r}."
        )
    TOTALVI.setup_anndata(
        adata,
        layer=spec.count_layer,
        protein_expression_obsm_key=spec.protein_obsm_key,
        protein_names_uns_key=spec.protein_names_uns_key,
        batch_key=config.technical_batch_key,
    )
    return TOTALVI(
        adata,
        n_latent=config.n_latent,
        gene_likelihood=spec.gene_likelihood,
        n_hidden=config.n_hidden,
        n_layers_encoder=config.n_layers,
        n_layers_decoder=config.n_layers,
        empirical_protein_background_prior=False,
    )


def add_evaluation_annotation_metrics(
    metrics: dict,
    representations: dict[str, dict[str, np.ndarray]],
    *,
    state_labels,
    sample_labels,
    technical_batch_labels,
    random_state: int,
    n_permutations: int = 100,
    k: int = 15,
) -> dict:
    """Add evaluation-only state, sample, and batch endpoints by representation."""
    updated = dict(metrics)
    for representation_name, payload in representations.items():
        values = payload["values"]
        diagnostics = representation_diagnostics(
            values,
            state_labels=state_labels,
            sample_labels=sample_labels,
            technical_batch_labels=technical_batch_labels,
            evaluation_indices=np.arange(len(values), dtype=np.int64),
            cell_ids=payload["cell_ids"],
            k=min(k, len(values) - 1),
            random_state=random_state,
            n_permutations=n_permutations,
        )
        prefix = "u" if representation_name == "u" else "factual_z"
        updated[f"{prefix}_state_balanced_accuracy"] = diagnostics[
            "state_balanced_accuracy"
        ]
        updated[f"{prefix}_knn_state_accuracy_k15"] = diagnostics[
            "knn_state_accuracy"
        ]
        updated[f"{prefix}_technical_batch_mixing"] = diagnostics[
            "technical_batch_mixing"
        ]
        if representation_name == "u":
            updated["u_within_state_sample_predictability"] = diagnostics[
                "within_state_sample_predictability"
            ]
            updated[
                "u_within_state_sample_predictability_permutation_p95"
            ] = diagnostics[
                "within_state_sample_predictability_permutation_p95"
            ]
    return updated


def _selected_contract_adapter(
    contract_adapter: RedesignContractAdapter | None,
) -> RedesignContractAdapter:
    """Resolve an explicit adapter or the frozen historical compatibility path."""
    if contract_adapter is None:
        return historical_redesign_contract_adapter()
    if not isinstance(contract_adapter, RedesignContractAdapter):
        raise TypeError("contract_adapter must be a RedesignContractAdapter.")
    return contract_adapter


_PREDICTION_OUTPUT_IDS = (
    "rna_negative_log_likelihood",
    "protein_negative_log_likelihood",
    "multimodal_predictive_loss",
    "rna_calibration_error",
    "protein_calibration_error",
)
_RNA_PREDICTION_METRIC_IDS = (
    "rna_reconstruction_loss",
    "rna_heldout_negative_log_likelihood",
    "rna_posterior_predictive_calibration",
)
_MULTIMODAL_PREDICTION_METRIC_IDS = (
    "protein_reconstruction_loss",
    "protein_heldout_negative_log_likelihood",
    "multimodal_heldout_predictive_loss",
    "protein_posterior_predictive_calibration",
)


def _nonfinite_representation_reasons(
    representations: Mapping[str, Mapping[str, object]],
) -> list[str]:
    """Return stable reason codes for explicitly nonfinite exports."""
    return [
        f"{representation_name}_representation_all_finite"
        for representation_name, payload in representations.items()
        if not np.all(np.isfinite(np.asarray(payload["values"])))
    ]


def _prediction_no_call_payload(
    *,
    candidate_id: str,
    reasons: list[str],
) -> tuple[dict, dict[str, list[str]]]:
    """Return empty prediction outputs and applicable metric no-calls."""
    candidate = redesign_candidate_configs().get(candidate_id)
    if candidate is None:
        raise ValueError(f"Unknown redesign row {candidate_id!r}.")
    metric_ids = list(_RNA_PREDICTION_METRIC_IDS)
    if candidate.model_family in {"totalvi", "mrtotalvi"}:
        metric_ids.extend(_MULTIMODAL_PREDICTION_METRIC_IDS)
    return (
        dict.fromkeys(_PREDICTION_OUTPUT_IDS),
        {metric_id: list(reasons) for metric_id in metric_ids},
    )


def _diagnostic_with_integrity_continuation(
    diagnostic: Callable[[], object],
    *,
    representations: Mapping[str, Mapping[str, object]],
) -> tuple[object | None, list[str]]:
    """Continue only explicit numeric validation failures after a bad export."""
    try:
        return diagnostic(), []
    except FloatingPointError:
        reasons = _nonfinite_representation_reasons(representations)
        if not reasons:
            raise
        return None, reasons
    except ValueError as error:
        message = str(error).lower()
        explicitly_nonfinite = (
            "nonfinite" in message
            or "non-finite" in message
            or "must be finite" in message
            or "only finite" in message
            or re.search(r"\b(?:nan|inf|infinity)\b", message) is not None
        )
        if not explicitly_nonfinite:
            raise
        reasons = _nonfinite_representation_reasons(representations)
        if not reasons:
            raise
        return None, reasons


def _prediction_metrics_with_integrity_continuation(
    diagnostic: Callable[[], dict],
    *,
    candidate_id: str,
    representations: Mapping[str, Mapping[str, object]],
) -> tuple[dict, dict[str, list[str]]]:
    """No-call failed predictive diagnostics only after a nonfinite export."""
    result, reasons = _diagnostic_with_integrity_continuation(
        diagnostic,
        representations=representations,
    )
    if not reasons:
        return result, {}
    return _prediction_no_call_payload(
        candidate_id=candidate_id,
        reasons=reasons,
    )


def _validation_arrays_with_integrity_continuation(
    diagnostic: Callable[[], dict],
    *,
    candidate_id: str,
    representations: Mapping[str, Mapping[str, object]],
) -> tuple[dict | None, dict[str, list[str]]]:
    """No-call validation-dependent metrics after a nonfinite export."""
    candidate = redesign_candidate_configs().get(candidate_id)
    if candidate is None:
        raise ValueError(f"Unknown redesign row {candidate_id!r}.")
    result, reasons = _diagnostic_with_integrity_continuation(
        diagnostic,
        representations=representations,
    )
    if not reasons:
        return result, {}
    _, no_call_reasons = _prediction_no_call_payload(
        candidate_id=candidate_id,
        reasons=reasons,
    )
    no_call_reasons["kl_z"] = list(reasons)
    if candidate.model_family == "mrtotalvi":
        for metric_id in ("kl_u", "registered_residual_magnitude"):
            no_call_reasons[metric_id] = list(reasons)
        if candidate.hierarchy_mode == "centered_v2":
            no_call_reasons["centering_max_abs"] = list(reasons)
    return None, no_call_reasons


def _representation_terminal_reasons(
    diagnostics: Mapping[str, object],
    *,
    representation_name: str,
    gradient_coverage: float | None = 1.0,
) -> list[str]:
    """Return exact prospective terminal-integrity reason codes."""
    reasons = []
    for indicator in (
        "representation_all_finite",
        "exact_nonconstant_variation",
        "posterior_scales_all_valid",
    ):
        if diagnostics.get(indicator) != 1.0:
            reasons.append(f"{representation_name}_{indicator}")
    if representation_name == "factual_z" and gradient_coverage != 1.0:
        reasons.append("registered_residual_gradient_coverage")
    return reasons


def _add_prospective_representation_metrics(
    metrics: dict,
    representations: dict[str, dict[str, np.ndarray]],
    *,
    latent_by_representation: Mapping[str, Mapping[str, object]],
    gradient_coverage: float | None,
    state_labels,
    sample_labels,
    technical_batch_labels,
    random_state: int,
) -> tuple[dict, dict[str, list[str]]]:
    """Evaluate valid representations and explicitly no-call terminal ones."""
    updated = dict(metrics)
    no_call_reasons: dict[str, list[str]] = {}
    annotation_metric_ids = {
        "factual_z": (
            "factual_z_state_balanced_accuracy",
            "factual_z_knn_state_accuracy_k15",
            "factual_z_technical_batch_mixing",
        ),
        "u": (
            "u_state_balanced_accuracy",
            "u_knn_state_accuracy_k15",
            "u_within_state_sample_predictability",
            "u_within_state_sample_predictability_permutation_p95",
            "u_technical_batch_mixing",
        ),
    }
    descriptive_metric_ids = {
        "factual_z": (
            "factual_z_posterior_scale",
            "factual_z_latent_variance",
            "factual_z_effective_rank",
        ),
        "u": (
            "u_posterior_scale",
            "u_latent_variance",
            "u_effective_rank",
        ),
    }
    for representation_name, representation in representations.items():
        diagnostics = latent_by_representation[representation_name]
        reasons = _representation_terminal_reasons(
            diagnostics,
            representation_name=representation_name,
            gradient_coverage=gradient_coverage,
        )
        embedding_usable = (
            diagnostics.get("representation_all_finite") == 1.0
            and diagnostics.get("exact_nonconstant_variation") == 1.0
        )
        if not embedding_usable:
            for metric_id in annotation_metric_ids[representation_name]:
                no_call_reasons[metric_id] = list(reasons)
        else:
            updated = add_evaluation_annotation_metrics(
                updated,
                {representation_name: representation},
                state_labels=state_labels,
                sample_labels=sample_labels,
                technical_batch_labels=technical_batch_labels,
                random_state=random_state,
            )
        for metric_id in descriptive_metric_ids[representation_name]:
            if updated.get(metric_id) is None:
                diagnostic_id = metric_id.removeprefix(
                    f"{representation_name}_"
                )
                diagnostic_reasons = diagnostics.get(
                    "diagnostic_no_call_reasons",
                    {},
                ).get(diagnostic_id, [])
                no_call_reasons[metric_id] = list(
                    reasons
                    or (
                        f"{representation_name}_{reason}"
                        for reason in diagnostic_reasons
                    )
                )
    if gradient_coverage is None:
        reasons = ["registered_residual_gradient_coverage"]
        no_call_reasons["registered_residual_gradient_norm"] = reasons
        no_call_reasons["registered_residual_gradient_coverage"] = reasons
    return updated, no_call_reasons


def _validate_evaluation_annotations(value: object) -> dict:
    expected = {
        "state_labels",
        "sample_labels",
        "technical_batch_labels",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(
            "evaluation_annotations must contain exactly state_labels, "
            "sample_labels, and technical_batch_labels."
        )
    return value


def collect_mrtotalvi_diagnostics(
    model,
    *,
    candidate_id: str,
    validation_indices,
    gradient_indices,
    batch_size: int,
    posterior_samples: int,
    posterior_predictive_draws: int,
    evaluation_seed: int,
    evaluation_annotations: dict,
    training_history,
    checkpoint_identity: dict,
    checkpoint_path: str | Path,
    wall_time_seconds: float,
    peak_memory_bytes: int,
    contract_adapter: RedesignContractAdapter | None = None,
) -> dict:
    """Collect all per-fit MrTotalVI optimization and representation diagnostics."""
    adapter = _selected_contract_adapter(contract_adapter)
    prospective = adapter.integrity_version == "v2"
    candidate = redesign_candidate_configs().get(candidate_id)
    if candidate is None or candidate.model_family != "mrtotalvi":
        raise ValueError("collect_mrtotalvi_diagnostics requires an MrTotalVI row.")
    validation = _integer_indices(
        validation_indices,
        name="validation_indices",
    )
    gradient = _integer_indices(gradient_indices, name="gradient_indices")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if evaluation_seed < 0:
        raise ValueError("evaluation_seed must be nonnegative.")
    if wall_time_seconds < 0.0 or peak_memory_bytes < 0:
        raise ValueError("Runtime and memory diagnostics must be nonnegative.")
    annotations = _validate_evaluation_annotations(evaluation_annotations)
    history = serialize_training_history(training_history)
    checkpoint = validate_checkpoint_identity(
        history,
        checkpoint_identity,
        current_state_dict=model.module.state_dict(),
        checkpoint_path=checkpoint_path,
    )

    cell_ids = np.asarray(model.adata.obs_names, dtype=str)[validation]
    representations = None
    if prospective:
        representations = export_named_representations(
            model,
            candidate_id=candidate_id,
            cell_ids=cell_ids,
            indices=validation,
            batch_size=batch_size,
            allow_integrity_failures=True,
        )

    def validation_diagnostic() -> dict:
        return _stock_validation_arrays(
            model,
            indices=validation,
            batch_size=batch_size,
        )

    if prospective:
        validation_arrays, validation_no_call_reasons = (
            _validation_arrays_with_integrity_continuation(
                validation_diagnostic,
                candidate_id=candidate_id,
                representations=representations,
            )
        )
    else:
        validation_arrays = validation_diagnostic()
        validation_no_call_reasons = {}

    def factual_scale_diagnostic() -> np.ndarray:
        return _mrtotalvi_factual_posterior_scale(
            model,
            indices=validation,
            batch_size=batch_size,
            n_samples=posterior_samples,
            random_state=evaluation_seed,
        )

    if prospective:
        factual_scale, _ = _diagnostic_with_integrity_continuation(
            factual_scale_diagnostic,
            representations=representations,
        )
    else:
        factual_scale = factual_scale_diagnostic()

    def prediction_diagnostic() -> dict:
        rna_draws, protein_draws = _posterior_predictive_draws(
            model,
            model_family="mrtotalvi",
            indices=validation,
            batch_size=batch_size,
            n_draws=posterior_predictive_draws,
            random_state=evaluation_seed,
        )
        return heldout_prediction_metrics(
            rna_log_prob=validation_arrays["rna_log_prob"],
            rna_observed=validation_arrays["rna_observed"],
            rna_predictive_draws=rna_draws,
            protein_log_prob=validation_arrays["protein_log_prob"],
            protein_observed=validation_arrays["protein_observed"],
            protein_predictive_draws=protein_draws,
        )

    if prospective and validation_arrays is None:
        prediction = dict.fromkeys(_PREDICTION_OUTPUT_IDS)
        prediction_no_call_reasons = dict(validation_no_call_reasons)
    elif prospective:
        prediction, prediction_no_call_reasons = (
            _prediction_metrics_with_integrity_continuation(
                prediction_diagnostic,
                candidate_id=candidate_id,
                representations=representations,
            )
        )
        prediction_no_call_reasons = {
            **validation_no_call_reasons,
            **prediction_no_call_reasons,
        }
    else:
        prediction = prediction_diagnostic()
        prediction_no_call_reasons = {}
        representations = export_named_representations(
            model,
            candidate_id=candidate_id,
            cell_ids=cell_ids,
            indices=validation,
            batch_size=batch_size,
        )
    u_values = representations["u"]["values"]
    factual_values = representations["factual_z"]["values"]
    diagnostic = latent_diagnostics_v2 if prospective else latent_diagnostics
    u_latent = diagnostic(
        u_values,
        posterior_scale=(
            None
            if validation_arrays is None
            else validation_arrays["posterior_scale"]
        ),
    )
    factual_latent = diagnostic(
        factual_values,
        posterior_scale=factual_scale,
    )
    gradient_norm, gradient_coverage = (
        _mrtotalvi_residual_gradient_diagnostics(
            model,
            indices=gradient,
            batch_size=batch_size,
            allow_integrity_failures=prospective,
        )
    )
    n_rna = (
        None
        if validation_arrays is None
        else validation_arrays["rna_observed"].shape[1]
    )
    n_protein = (
        None
        if validation_arrays is None
        else validation_arrays["protein_observed"].shape[1]
    )
    metrics = metric_payload_template(contract_adapter=adapter)
    metrics.update(
        {
            "validation_objective_history": history["elbo_validation"],
            "best_checkpoint_identity": checkpoint,
            "rna_reconstruction_loss": (
                None
                if n_rna is None
                or prediction["rna_negative_log_likelihood"] is None
                else prediction["rna_negative_log_likelihood"] * n_rna
            ),
            "protein_reconstruction_loss": (
                None
                if n_protein is None
                or prediction["protein_negative_log_likelihood"] is None
                else prediction["protein_negative_log_likelihood"] * n_protein
            ),
            "kl_z": (
                None if validation_arrays is None else validation_arrays["kl_z"]
            ),
            "kl_u": (
                None if validation_arrays is None else validation_arrays["kl_u"]
            ),
            "u_posterior_scale": u_latent["posterior_scale"],
            "factual_z_posterior_scale": factual_latent["posterior_scale"],
            "u_latent_variance": u_latent["latent_variance"],
            "factual_z_latent_variance": factual_latent["latent_variance"],
            "u_effective_rank": u_latent["effective_rank"],
            "factual_z_effective_rank": factual_latent["effective_rank"],
            "registered_residual_magnitude": (
                None
                if validation_arrays is None
                else validation_arrays["registered_residual_magnitude"]
            ),
            "registered_residual_gradient_norm": gradient_norm,
            "registered_residual_gradient_coverage": gradient_coverage,
            "trainable_parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in model.module.parameters()
                    if parameter.requires_grad
                )
            ),
            "wall_time_seconds": float(wall_time_seconds),
            "peak_memory_bytes": int(peak_memory_bytes),
            "rna_heldout_negative_log_likelihood": prediction[
                "rna_negative_log_likelihood"
            ],
            "protein_heldout_negative_log_likelihood": prediction[
                "protein_negative_log_likelihood"
            ],
            "multimodal_heldout_predictive_loss": prediction[
                "multimodal_predictive_loss"
            ],
            "rna_posterior_predictive_calibration": prediction[
                "rna_calibration_error"
            ],
            "protein_posterior_predictive_calibration": prediction[
                "protein_calibration_error"
            ],
            "centering_max_abs": (
                None
                if validation_arrays is None
                else validation_arrays["centering_max_abs"]
            ),
            "latent_all_finite": min(
                u_latent["all_finite"],
                factual_latent["all_finite"],
            ),
        }
    )
    if prospective:
        metrics.update(
            {
                f"u_{metric_id}": u_latent[metric_id]
                for metric_id in (
                    "representation_all_finite",
                    "exact_nonconstant_variation",
                    "posterior_scales_all_valid",
                )
            }
        )
        metrics.update(
            {
                f"factual_z_{metric_id}": factual_latent[metric_id]
                for metric_id in (
                    "representation_all_finite",
                    "exact_nonconstant_variation",
                    "posterior_scales_all_valid",
                )
            }
        )
        metrics, representation_no_call_reasons = (
            _add_prospective_representation_metrics(
                metrics,
                representations,
                latent_by_representation={
                    "u": u_latent,
                    "factual_z": factual_latent,
                },
                gradient_coverage=gradient_coverage,
                state_labels=annotations["state_labels"],
                sample_labels=annotations["sample_labels"],
                technical_batch_labels=annotations[
                    "technical_batch_labels"
                ],
                random_state=evaluation_seed,
            )
        )
        no_call_reasons = {
            **prediction_no_call_reasons,
            **representation_no_call_reasons,
        }
    else:
        metrics = add_evaluation_annotation_metrics(
            metrics,
            representations,
            state_labels=annotations["state_labels"],
            sample_labels=annotations["sample_labels"],
            technical_batch_labels=annotations["technical_batch_labels"],
            random_state=evaluation_seed,
        )
        no_call_reasons = {}
    validate_metric_payload(
        metrics,
        candidate_id=candidate_id,
        lifecycle="per_fit",
        contract_adapter=adapter,
        required_no_call_reasons=no_call_reasons,
    )
    return {
        "representations": representations,
        "training_history": history,
        "best_checkpoint_identity": checkpoint,
        "evaluation_seed": evaluation_seed,
        "metrics": metrics,
        "metric_no_call_reasons": no_call_reasons,
    }


def run_stock_comparator(
    adata: AnnData,
    config: ComparatorRunConfig,
    *,
    checkpoint_dir: str | Path,
    evaluation_annotations: dict,
    contract_adapter: RedesignContractAdapter | None = None,
) -> dict:
    """Train B0 or B1 on an exact split and return estimand-separated evidence."""
    adapter = _selected_contract_adapter(contract_adapter)
    prospective = adapter.integrity_version == "v2"
    import scvi
    from scvi.train import SaveCheckpoint

    train, validation = validate_external_split(
        config.train_indices,
        config.validation_indices,
        n_obs=adata.n_obs,
    )
    annotations = _validate_evaluation_annotations(evaluation_annotations)
    if config.technical_batch_key not in adata.obs:
        raise ValueError(
            "AnnData is missing technical batch column "
            f"{config.technical_batch_key!r}."
        )
    scvi.settings.seed = config.training_seed
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists():
        raise FileExistsError(
            f"Checkpoint directory already exists: {checkpoint_path}"
        )
    checkpoint_path.mkdir(parents=True, exist_ok=False)
    callback = SaveCheckpoint(
        dirpath=str(checkpoint_path),
        filename="best-{epoch:04d}-{elbo_validation:.8f}",
        monitor="elbo_validation",
        mode="min",
        save_top_k=1,
        load_best_on_end=True,
    )
    model = _construct_stock_model(adata, config)
    started = time.perf_counter()
    rss_before = _max_rss_bytes()
    common_train = {
        "max_epochs": config.max_epochs,
        "min_epochs": config.minimum_epochs,
        "accelerator": "cpu",
        "devices": 1,
        "train_size": None,
        "validation_size": None,
        "shuffle_set_split": False,
        "batch_size": config.batch_size,
        "early_stopping": config.early_stopping_patience is not None,
        "check_val_every_n_epoch": config.check_val_every_n_epoch,
        "callbacks": [callback],
        "enable_checkpointing": True,
        "enable_progress_bar": False,
    }
    if config.early_stopping_patience is not None:
        common_train.update(
            {
                "early_stopping_monitor": "elbo_validation",
                "early_stopping_mode": "min",
                "early_stopping_patience": config.early_stopping_patience,
                "early_stopping_min_delta": config.early_stopping_min_delta,
            }
        )
    external = [train, validation, np.asarray([], dtype=np.int64)]
    if config.spec.model_family == "scvi":
        model.train(
            **common_train,
            datasplitter_kwargs={"external_indexing": external},
            plan_kwargs={"lr": config.learning_rate},
        )
    else:
        explicit_optimization = (
            frozen_totalvi_optimization_kwargs(adata.n_obs)
            if config.explicit_totalvi_optimization
            else {}
        )
        model.train(
            **common_train,
            external_indexing=external,
            lr=config.learning_rate,
            reduce_lr_on_plateau=False,
            **explicit_optimization,
        )
    wall_seconds = float(time.perf_counter() - started)
    trainer_epochs = int(model.trainer.current_epoch)
    stopped_early = bool(
        model.trainer.should_stop and trainer_epochs < config.max_epochs
    )
    history = serialize_training_history(model.history)
    best_artifact_path = Path(callback.best_model_path)
    artifact_digest = _checkpoint_artifact_state_digest(best_artifact_path)
    checkpoint = best_checkpoint_identity(
        history,
        monitor="elbo_validation",
        mode="min",
        state_digest=artifact_digest,
        artifact_name=best_artifact_path.name,
    )
    checkpoint = validate_checkpoint_identity(
        history,
        checkpoint,
        current_state_dict=model.module.state_dict(),
        checkpoint_path=best_artifact_path,
    )
    optimization_identity = (
        _realized_totalvi_optimization_identity(
            model,
            n_obs=adata.n_obs,
            learning_rate=config.learning_rate,
        )
        if config.explicit_totalvi_optimization
        else None
    )

    cell_ids = np.asarray(adata.obs_names, dtype=str)[validation]
    representations = None
    if prospective:
        representations = export_named_representations(
            model,
            candidate_id=config.candidate_id,
            cell_ids=cell_ids,
            indices=validation,
            batch_size=config.batch_size,
            allow_integrity_failures=True,
        )

    def validation_diagnostic() -> dict:
        return _stock_validation_arrays(
            model,
            indices=validation,
            batch_size=config.batch_size,
        )

    if prospective:
        validation_arrays, validation_no_call_reasons = (
            _validation_arrays_with_integrity_continuation(
                validation_diagnostic,
                candidate_id=config.candidate_id,
                representations=representations,
            )
        )
    else:
        validation_arrays = validation_diagnostic()
        validation_no_call_reasons = {}

    def prediction_diagnostic() -> dict:
        rna_draws, protein_draws = _posterior_predictive_draws(
            model,
            model_family=config.spec.model_family,
            indices=validation,
            batch_size=config.batch_size,
            n_draws=config.posterior_predictive_draws,
            random_state=config.evaluation_seed,
        )
        return heldout_prediction_metrics(
            rna_log_prob=validation_arrays["rna_log_prob"],
            rna_observed=validation_arrays["rna_observed"],
            rna_predictive_draws=rna_draws,
            protein_log_prob=validation_arrays["protein_log_prob"],
            protein_observed=validation_arrays["protein_observed"],
            protein_predictive_draws=protein_draws,
        )

    if prospective and validation_arrays is None:
        prediction = dict.fromkeys(_PREDICTION_OUTPUT_IDS)
        prediction_no_call_reasons = dict(validation_no_call_reasons)
    elif prospective:
        prediction, prediction_no_call_reasons = (
            _prediction_metrics_with_integrity_continuation(
                prediction_diagnostic,
                candidate_id=config.candidate_id,
                representations=representations,
            )
        )
        prediction_no_call_reasons = {
            **validation_no_call_reasons,
            **prediction_no_call_reasons,
        }
    else:
        prediction = prediction_diagnostic()
        prediction_no_call_reasons = {}
        representations = export_named_representations(
            model,
            candidate_id=config.candidate_id,
            cell_ids=cell_ids,
            indices=validation,
            batch_size=config.batch_size,
        )
    diagnostic = latent_diagnostics_v2 if prospective else latent_diagnostics
    latent = diagnostic(
        representations["factual_z"]["values"],
        posterior_scale=(
            None
            if validation_arrays is None
            else validation_arrays["posterior_scale"]
        ),
    )
    n_rna = (
        None
        if validation_arrays is None
        else validation_arrays["rna_observed"].shape[1]
    )
    protein_observed = (
        None
        if validation_arrays is None
        else validation_arrays["protein_observed"]
    )
    n_protein = None if protein_observed is None else protein_observed.shape[1]
    metrics = metric_payload_template(contract_adapter=adapter)
    metrics.update({
        "validation_objective_history": history["elbo_validation"],
        "best_checkpoint_identity": checkpoint,
        "rna_reconstruction_loss": (
            None
            if n_rna is None
            or prediction["rna_negative_log_likelihood"] is None
            else prediction["rna_negative_log_likelihood"] * n_rna
        ),
        "protein_reconstruction_loss": (
            None
            if n_protein is None
            or prediction["protein_negative_log_likelihood"] is None
            else prediction["protein_negative_log_likelihood"] * n_protein
        ),
        "kl_z": (
            None if validation_arrays is None else validation_arrays["kl_z"]
        ),
        "kl_u": None,
        "factual_z_posterior_scale": latent["posterior_scale"],
        "factual_z_latent_variance": latent["latent_variance"],
        "factual_z_effective_rank": latent["effective_rank"],
        "trainable_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.module.parameters()
                if parameter.requires_grad
            )
        ),
        "wall_time_seconds": wall_seconds,
        "peak_memory_bytes": max(0, _max_rss_bytes() - rss_before),
        "latent_all_finite": latent["all_finite"],
        "rna_heldout_negative_log_likelihood": prediction[
            "rna_negative_log_likelihood"
        ],
        "protein_heldout_negative_log_likelihood": prediction[
            "protein_negative_log_likelihood"
        ],
        "multimodal_heldout_predictive_loss": prediction[
            "multimodal_predictive_loss"
        ],
        "rna_posterior_predictive_calibration": prediction[
            "rna_calibration_error"
        ],
        "protein_posterior_predictive_calibration": prediction[
            "protein_calibration_error"
        ],
    })
    if prospective:
        metrics.update(
            {
                f"factual_z_{metric_id}": latent[metric_id]
                for metric_id in (
                    "representation_all_finite",
                    "exact_nonconstant_variation",
                    "posterior_scales_all_valid",
                )
            }
        )
        metrics, representation_no_call_reasons = (
            _add_prospective_representation_metrics(
                metrics,
                representations,
                latent_by_representation={"factual_z": latent},
                gradient_coverage=1.0,
                state_labels=annotations["state_labels"],
                sample_labels=annotations["sample_labels"],
                technical_batch_labels=annotations[
                    "technical_batch_labels"
                ],
                random_state=config.evaluation_seed,
            )
        )
        no_call_reasons = {
            **prediction_no_call_reasons,
            **representation_no_call_reasons,
        }
    else:
        metrics = add_evaluation_annotation_metrics(
            metrics,
            representations,
            state_labels=annotations["state_labels"],
            sample_labels=annotations["sample_labels"],
            technical_batch_labels=annotations["technical_batch_labels"],
            random_state=config.evaluation_seed,
        )
        no_call_reasons = {}
    validate_metric_payload(
        metrics,
        candidate_id=config.candidate_id,
        lifecycle="per_fit",
        contract_adapter=adapter,
        required_no_call_reasons=no_call_reasons,
    )
    return {
        "schema_version": "mrtotalvi-stock-comparator-result-v2",
        "candidate_id": config.candidate_id,
        "model_family": config.spec.model_family,
        "modalities": list(config.spec.modalities),
        "technical_batch_key": config.technical_batch_key,
        "biological_sample_key": None,
        "training_seed": config.training_seed,
        "evaluation_seed": config.evaluation_seed,
        "training_control": {
            "check_val_every_n_epoch": config.check_val_every_n_epoch,
            "minimum_epochs": config.minimum_epochs,
            "maximum_epochs": config.max_epochs,
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
        },
        "trainer_epochs": trainer_epochs,
        "stopped_early": stopped_early,
        "train_indices": train,
        "validation_indices": validation,
        "training_history": history,
        "best_checkpoint_identity": checkpoint,
        "optimization_identity": optimization_identity,
        "representations": representations,
        "metrics": metrics,
        "metric_no_call_reasons": no_call_reasons,
    }
