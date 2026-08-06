"""Counterfactual datasets for the opt-in centered MrTotalVI hierarchy."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.distributions import Normal

from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS

SCHEMA_VERSION = "mrtotalvi-counterfactual-v1"
MAX_IN_MEMORY_BYTES = 512 * 1024 * 1024
INTERPRETATION = "registered-sample model transformation; non-causal"


def _as_indices(adata, indices, *, name: str) -> np.ndarray:
    if indices is None:
        result = np.arange(adata.n_obs, dtype=np.int64)
    else:
        result = np.asarray(indices)
        if result.dtype == np.bool_:
            if result.ndim != 1 or result.size != adata.n_obs:
                raise ValueError(f"{name} boolean mask must have length adata.n_obs.")
            result = np.flatnonzero(result)
        result = result.astype(np.int64, copy=False)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if np.any(result < 0) or np.any(result >= adata.n_obs):
        raise IndexError(f"{name} contains an out-of-range observation index.")
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} contains duplicate observation indices.")
    return result


def _validate_quantiles(quantiles) -> np.ndarray:
    values = np.asarray(quantiles, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("quantiles must be a non-empty one-dimensional sequence.")
    if np.unique(values).size != values.size:
        raise ValueError("quantiles must be unique.")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("quantiles must lie strictly inside (0, 1).")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("quantiles must be strictly increasing.")
    return values


def _validate_inference(inference_mode: str, n_draws: int, quantiles) -> np.ndarray:
    if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws < 1:
        raise ValueError("n_draws must be a positive integer.")
    if inference_mode == "latent_mean":
        if n_draws != 1:
            raise ValueError("latent_mean requires n_draws=1.")
    elif inference_mode == "posterior_mc":
        if n_draws < 2:
            raise ValueError("posterior_mc requires at least two draws.")
    else:
        raise ValueError("inference_mode must be one of {'latent_mean', 'posterior_mc'}.")
    return _validate_quantiles(quantiles)


def _validate_chunk_size(value, *, name: str) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer or None.")


def _validate_targets(model, target_samples) -> tuple[np.ndarray, np.ndarray]:
    registered = np.asarray(model.sample_order)
    registered_strings = np.asarray([str(value) for value in registered])
    if np.unique(registered_strings).size != registered_strings.size:
        raise ValueError("Registered sample labels are not unique after string conversion.")
    if target_samples is None:
        return np.arange(registered.size, dtype=np.int64), registered_strings
    requested = np.asarray([str(value) for value in target_samples])
    if requested.size == 0:
        raise ValueError("target_samples must contain at least one target.")
    if np.unique(requested).size != requested.size:
        raise ValueError("target_samples contains duplicate targets.")
    lookup = {value: index for index, value in enumerate(registered_strings)}
    unknown = [value for value in requested if value not in lookup]
    if unknown:
        raise ValueError(f"Unknown target sample(s): {unknown}.")
    return np.asarray([lookup[value] for value in requested], dtype=np.int64), requested


def _validate_features(registered, requested, *, feature_type: str):
    registered = np.asarray([str(value) for value in registered])
    if np.unique(registered).size != registered.size:
        raise ValueError(f"Registered {feature_type} names are not unique.")
    if requested is None:
        return np.arange(registered.size, dtype=np.int64), registered
    requested = np.asarray([str(value) for value in requested])
    if requested.size == 0:
        raise ValueError(f"{feature_type}_list must contain at least one feature.")
    if np.unique(requested).size != requested.size:
        raise ValueError(f"{feature_type}_list contains duplicate features.")
    lookup = {value: index for index, value in enumerate(registered)}
    unknown = [value for value in requested if value not in lookup]
    if unknown:
        raise ValueError(f"Unknown {feature_type}(s): {unknown}.")
    return np.asarray([lookup[value] for value in requested]), requested


def _counter_normal_noise(
    *,
    seed: int,
    cell_names: np.ndarray,
    n_draws: int,
    n_latent: int,
) -> np.ndarray:
    """Generate Philox noise keyed by seed, cell ID, draw, and coordinate."""
    noise = np.empty((n_draws, cell_names.size, n_latent), dtype=np.float32)
    for cell_index, cell_name in enumerate(cell_names):
        cell_digest = hashlib.sha256(str(cell_name).encode("utf-8")).digest()
        cell_key = int.from_bytes(cell_digest[:8], byteorder="little", signed=False)
        for draw in range(n_draws):
            seed_sequence = np.random.SeedSequence(
                [
                    int(seed) & 0xFFFFFFFF,
                    cell_key & 0xFFFFFFFF,
                    (cell_key >> 32) & 0xFFFFFFFF,
                    int(draw),
                ]
            )
            generator = np.random.Generator(np.random.Philox(seed_sequence))
            noise[draw, cell_index] = generator.standard_normal(
                n_latent,
                dtype=np.float32,
            )
    return noise


def _encode_u_params(model, adata, indices: np.ndarray, batch_size: int):
    loader = model._make_data_loader(
        adata=adata,
        indices=indices,
        batch_size=batch_size,
    )
    locs = []
    scales = []
    observed_samples = []
    was_training = model.module.training
    model.module.eval()
    try:
        with torch.inference_mode():
            for tensors in loader:
                inputs = model.module._get_inference_input(tensors)
                qu = model.module.qu(
                    inputs["x"],
                    inputs["y"],
                    inputs["sample_index"],
                    batch_index=(
                        inputs.get("batch_index")
                        if model.module.encode_covariates
                        else None
                    ),
                    cont_covs=(
                        inputs.get("cont_covs")
                        if model.module.encode_covariates
                        else None
                    ),
                    cat_covs=(
                        inputs.get("cat_covs")
                        if model.module.encode_covariates
                        else None
                    ),
                )
                locs.append(qu.loc.detach().cpu())
                scales.append(qu.scale.detach().cpu())
                observed_samples.append(
                    tensors[REGISTRY_KEYS.SAMPLE_KEY].long().flatten().cpu()
                )
    finally:
        model.module.train(was_training)
    return (
        torch.cat(locs).numpy().astype(np.float32, copy=False),
        torch.cat(scales).numpy().astype(np.float32, copy=False),
        torch.cat(observed_samples).numpy().astype(np.int64, copy=False),
    )


def _collect_observed_context(model, adata, indices: np.ndarray, batch_size: int):
    loader = model._make_data_loader(
        adata=adata,
        indices=indices,
        batch_size=batch_size,
    )
    batches = []
    panels = []
    samples = []
    libraries = []
    cat_covariates = []
    cont_covariates = []
    has_cat = False
    has_cont = False
    has_size_factor = (
        REGISTRY_KEYS.SIZE_FACTOR_KEY in model.adata_manager.data_registry
    )
    uses_latent_library = not model.module.use_observed_lib_size
    if has_size_factor:
        library_estimand = "registered_size_factor"
    elif uses_latent_library:
        library_estimand = "posterior_lognormal_mean"
    else:
        library_estimand = "observed_total_count"

    was_training = model.module.training
    if uses_latent_library:
        model.module.eval()
    try:
        for tensors in loader:
            inputs = model.module._get_inference_input(tensors)
            batches.append(inputs["batch_index"].long().flatten().cpu())
            panels.append(inputs["panel_index"].long().flatten().cpu())
            samples.append(
                tensors[REGISTRY_KEYS.SAMPLE_KEY].long().flatten().cpu()
            )
            if has_size_factor:
                library = tensors[REGISTRY_KEYS.SIZE_FACTOR_KEY].float()
            elif uses_latent_library:
                with torch.inference_mode():
                    inference_outputs = model.module.inference(**inputs)
                ql = inference_outputs[MODULE_KEYS.QL_KEY]
                library = torch.exp(ql.loc + 0.5 * ql.scale.square())
            else:
                library = inputs["x"].sum(dim=-1, keepdim=True).float()
            libraries.append(library.reshape(-1).cpu())
            if inputs.get("cat_covs") is not None:
                has_cat = True
                cat_covariates.append(inputs["cat_covs"].cpu())
            if inputs.get("cont_covs") is not None:
                has_cont = True
                cont_covariates.append(inputs["cont_covs"].cpu())
    finally:
        if uses_latent_library:
            model.module.train(was_training)
    return {
        "batch": torch.cat(batches).numpy().astype(np.int64),
        "panel": torch.cat(panels).numpy().astype(np.int64),
        "sample": torch.cat(samples).numpy().astype(np.int64),
        "library": torch.cat(libraries).numpy().astype(np.float32),
        "cat_covs": (
            torch.cat(cat_covariates).numpy().astype(np.int64)
            if has_cat
            else None
        ),
        "cont_covs": (
            torch.cat(cont_covariates).numpy().astype(np.float32)
            if has_cont
            else None
        ),
        "library_estimand": library_estimand,
    }


def _mapping(model, registry_key: str) -> np.ndarray:
    return np.asarray(
        [
            str(value)
            for value in model.adata_manager.get_state_registry(
                registry_key
            ).categorical_mapping
        ]
    )


def _specified_category(value, mapping: np.ndarray, *, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required when its policy is 'specified'.")
    value = str(value)
    positions = np.flatnonzero(mapping == value)
    if positions.size != 1:
        raise ValueError(f"Unsupported {name} category: {value!r}.")
    return int(positions[0])


def _specified_library(value, n_cells: int) -> np.ndarray:
    if value is None:
        raise ValueError(
            "specified_library_size is required when library_policy='specified'."
        )
    values = np.asarray(value, dtype=np.float32)
    if values.ndim == 0:
        values = np.full(n_cells, float(values), dtype=np.float32)
    elif values.shape != (n_cells,):
        raise ValueError(
            "specified_library_size must be one scalar or a cell-aligned vector."
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("specified_library_size values must be finite and positive.")
    return values


def _validate_context_request(
    *,
    model,
    n_cells: int,
    batch_policy: str,
    specified_batch,
    panel_policy: str,
    specified_panel,
    library_policy: str,
    specified_library_size,
    has_marginal_reference: bool,
) -> None:
    """Validate every context policy before any model inference."""
    allowed = {"observed", "specified", "sample_balanced_marginal"}
    for name, value in (
        ("batch_policy", batch_policy),
        ("panel_policy", panel_policy),
        ("library_policy", library_policy),
    ):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}.")

    batch_marginal = batch_policy == "sample_balanced_marginal"
    panel_marginal = panel_policy == "sample_balanced_marginal"
    library_marginal = library_policy == "sample_balanced_marginal"
    if batch_marginal != panel_marginal:
        raise ValueError(
            "Batch and panel must be marginalized jointly; requesting "
            "sample_balanced_marginal for only one is unsupported."
        )
    if (batch_marginal or library_marginal) and not has_marginal_reference:
        raise ValueError(
            "A non-empty marginal reference is required for "
            "sample_balanced_marginal policies."
        )

    if batch_policy == "specified":
        _specified_category(
            specified_batch,
            _mapping(model, REGISTRY_KEYS.BATCH_KEY),
            name="specified_batch",
        )
    if panel_policy == "specified":
        panel_mapping = (
            _mapping(model, "panel")
            if model.module.panel_key == "panel"
            else _mapping(model, REGISTRY_KEYS.BATCH_KEY)
        )
        _specified_category(
            specified_panel,
            panel_mapping,
            name="specified_panel",
        )
    if library_policy == "specified":
        _specified_library(specified_library_size, n_cells)


def _build_context_scenarios(
    *,
    model,
    observed_context,
    batch_policy: str,
    specified_batch,
    panel_policy: str,
    specified_panel,
    library_policy: str,
    specified_library_size,
    reference_context=None,
) -> tuple[list[dict[str, np.ndarray]], str]:
    allowed = {"observed", "specified", "sample_balanced_marginal"}
    for name, value in (
        ("batch_policy", batch_policy),
        ("panel_policy", panel_policy),
        ("library_policy", library_policy),
    ):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {sorted(allowed)}.")
    batch_marginal = batch_policy == "sample_balanced_marginal"
    panel_marginal = panel_policy == "sample_balanced_marginal"
    library_marginal = library_policy == "sample_balanced_marginal"
    if batch_marginal != panel_marginal:
        raise ValueError(
            "Batch and panel must be marginalized jointly; requesting "
            "sample_balanced_marginal for only one is unsupported."
        )
    if (batch_marginal or library_marginal) and reference_context is None:
        raise ValueError(
            "A non-empty marginal reference is required for "
            "sample_balanced_marginal policies."
        )

    n_cells = observed_context["batch"].size
    batch_mapping = _mapping(model, REGISTRY_KEYS.BATCH_KEY)
    panel_mapping = (
        _mapping(model, "panel")
        if model.module.panel_key == "panel"
        else batch_mapping
    )
    batch = observed_context["batch"].copy()
    panel = observed_context["panel"].copy()
    library = observed_context["library"].copy()
    if batch_policy == "specified":
        batch.fill(
            _specified_category(
                specified_batch,
                batch_mapping,
                name="specified_batch",
            )
        )
    if panel_policy == "specified":
        panel.fill(
            _specified_category(
                specified_panel,
                panel_mapping,
                name="specified_panel",
            )
        )
    if library_policy == "specified":
        library = _specified_library(specified_library_size, n_cells)

    if not (batch_marginal or library_marginal):
        scenarios = [
            {
                "batch": batch,
                "panel": panel,
                "library": library,
                "weight": np.ones(n_cells, dtype=np.float32),
            }
        ]
    else:
        reference_samples = reference_context["sample"]
        represented_samples = np.unique(reference_samples)
        if represented_samples.size == 0:
            raise ValueError(
                "marginal_reference_indices must retain at least one reference cell."
            )

        # Each represented biological sample receives equal total mass. Within
        # a sample, empirical joint technical contexts retain their frequency.
        context_weights: dict[tuple[int, int, float], float] = {}
        for sample_index in represented_samples:
            sample_positions = np.flatnonzero(reference_samples == sample_index)
            sample_weight = 1.0 / represented_samples.size
            row_weight = sample_weight / sample_positions.size
            for position in sample_positions:
                key = (
                    (
                        int(reference_context["batch"][position])
                        if batch_marginal
                        else -1
                    ),
                    (
                        int(reference_context["panel"][position])
                        if panel_marginal
                        else -1
                    ),
                    (
                        float(reference_context["library"][position])
                        if library_marginal
                        else 0.0
                    ),
                )
                context_weights[key] = context_weights.get(key, 0.0) + row_weight

        scenarios = []
        for (context_batch, context_panel, context_library), weight in sorted(
            context_weights.items()
        ):
            scenario_batch = batch.copy()
            scenario_panel = panel.copy()
            scenario_library = library.copy()
            if batch_marginal:
                scenario_batch.fill(context_batch)
                scenario_panel.fill(context_panel)
            if library_marginal:
                scenario_library.fill(context_library)
            scenarios.append(
                {
                    "batch": scenario_batch,
                    "panel": scenario_panel,
                    "library": scenario_library,
                    "weight": np.full(n_cells, weight, dtype=np.float32),
                }
            )

    context_payload = json.dumps(
        [
            {
                "batch": scenario["batch"].tolist(),
                "library": [
                    float(value) for value in scenario["library"]
                ],
                "panel": scenario["panel"].tolist(),
                "weight": [
                    float(value) for value in scenario["weight"]
                ],
            }
            for scenario in scenarios
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return scenarios, hashlib.sha256(context_payload.encode()).hexdigest()


def _max_component_log_prob(
    point: np.ndarray,
    component_loc: np.ndarray,
    component_scale: np.ndarray,
    keep: np.ndarray,
) -> float:
    if not np.any(keep):
        return math.nan
    point_tensor = torch.as_tensor(point, dtype=torch.float32)
    loc_tensor = torch.as_tensor(component_loc[keep], dtype=torch.float32)
    scale_tensor = torch.as_tensor(component_scale[keep], dtype=torch.float32)
    values = Normal(loc_tensor, scale_tensor).log_prob(point_tensor).sum(dim=-1)
    return float(values.max().item())


def _support_and_admissibility(
    *,
    model,
    adata,
    query_indices: np.ndarray,
    query_loc: np.ndarray,
    target_indices: np.ndarray,
    reference_indices: np.ndarray,
    support_quantile: float,
    admissibility_threshold: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < support_quantile < 1.0:
        raise ValueError("support_quantile must lie strictly inside (0, 1).")
    if not np.isfinite(admissibility_threshold):
        raise ValueError("admissibility_threshold must be finite.")

    reference_loc, reference_scale, reference_sample = _encode_u_params(
        model,
        adata,
        reference_indices,
        batch_size,
    )
    query_names = np.asarray(adata.obs_names[query_indices], dtype=str)
    reference_names = np.asarray(adata.obs_names[reference_indices], dtype=str)
    support = np.zeros((query_indices.size, target_indices.size), dtype=bool)
    admissible = np.zeros_like(support)

    for target_position, target_index in enumerate(target_indices):
        target_mask = reference_sample == target_index
        loc = reference_loc[target_mask]
        scale = reference_scale[target_mask]
        names = reference_names[target_mask]
        keep_by_query = [names != query_name for query_name in query_names]
        for query_position, keep in enumerate(keep_by_query):
            support[query_position, target_position] = bool(np.any(keep))
        if loc.shape[0] < 2:
            continue

        reference_scores = np.empty(loc.shape[0], dtype=np.float64)
        for component_index in range(loc.shape[0]):
            keep = np.ones(loc.shape[0], dtype=bool)
            keep[component_index] = False
            reference_scores[component_index] = _max_component_log_prob(
                loc[component_index],
                loc,
                scale,
                keep,
            )
        threshold = float(np.quantile(reference_scores, support_quantile))

        for query_position, point in enumerate(query_loc):
            keep = keep_by_query[query_position]
            score = _max_component_log_prob(point, loc, scale, keep)
            admissible[query_position, target_position] = bool(
                np.isfinite(score)
                and score - threshold > admissibility_threshold
            )
    return support, admissible


def _add_posterior_summaries(
    dataset: xr.Dataset,
    *,
    quantiles: np.ndarray,
) -> xr.Dataset:
    additions = {}
    for variable_name, variable in dataset.data_vars.items():
        if "draw" not in variable.dims or not np.issubdtype(variable.dtype, np.floating):
            continue
        draw_axis = variable.dims.index("draw")
        values = variable.to_numpy()
        mean_values = values.mean(axis=draw_axis, dtype=np.float64).astype(np.float32)
        remaining_dims = tuple(dim for dim in variable.dims if dim != "draw")
        additions[f"{variable_name}_posterior_mean"] = (
            remaining_dims,
            mean_values,
        )
        quantile_values = np.quantile(
            values,
            quantiles,
            axis=draw_axis,
        ).astype(np.float32)
        additions[f"{variable_name}_posterior_quantile"] = (
            ("quantile", *remaining_dims),
            quantile_values,
        )
    return dataset.assign(additions).assign_coords(quantile=quantiles)


def _estimate_latent_bytes(
    *,
    n_cells: int,
    n_targets: int,
    n_draws: int,
    n_latent_u: int,
    n_latent: int,
    n_quantiles: int,
    posterior_mc: bool,
) -> int:
    float_elements = n_draws * n_cells * (
        n_latent_u + n_latent + 3 * n_targets * n_latent
    )
    if posterior_mc:
        float_elements += (1 + n_quantiles) * n_cells * (
            n_latent_u + n_latent + 3 * n_targets * n_latent
        )
    raw_bytes = 4 * float_elements + 2 * n_cells * n_targets
    return math.ceil(raw_bytes * 1.2)


def _estimate_expression_bytes(
    *,
    n_cells: int,
    n_targets: int,
    n_draws: int,
    n_genes: int,
    n_proteins: int,
    n_quantiles: int,
    posterior_mc: bool,
) -> int:
    per_draw = n_cells * n_targets * (2 * n_genes + 7 * n_proteins)
    float_elements = n_draws * per_draw
    if posterior_mc:
        float_elements += (1 + n_quantiles) * per_draw
    raw_bytes = 4 * float_elements + n_cells * n_targets * n_proteins
    return math.ceil(raw_bytes * 1.2)


def _write_atomic_zarr(
    dataset: xr.Dataset,
    *,
    zarr_path,
    zarr_chunks,
) -> xr.Dataset:
    try:
        import dask  # noqa: F401
        import zarr
    except ImportError as error:  # pragma: no cover - optional dependency environment
        raise ImportError(
            "Zarr output requires dask and zarr. Install the existing 'parallel' extra."
        ) from error

    destination = Path(zarr_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing Zarr store: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        effective_chunks = _resolve_zarr_chunks(dataset, zarr_chunks)
        group = zarr.open_group(temporary, mode="w", zarr_format=2)
        group.attrs.update(dict(dataset.attrs))
        group.attrs["storage_mode"] = "atomic_zarr_regions"
        group.attrs["chunks"] = json.dumps(
            effective_chunks,
            sort_keys=True,
            separators=(",", ":"),
        )

        for coordinate_name, coordinate in dataset.coords.items():
            values = coordinate.to_numpy()
            if values.dtype.kind in {"O", "U", "S"}:
                values = np.asarray(values, dtype=str)
            dimensions = list(coordinate.dims)
            chunks = tuple(
                min(effective_chunks[dimension], size)
                for dimension, size in zip(
                    coordinate.dims,
                    coordinate.shape,
                    strict=True,
                )
            )
            group.create_array(
                coordinate_name,
                data=values,
                chunks=chunks or None,
                fill_value=None,
                attributes={"_ARRAY_DIMENSIONS": dimensions},
            )

        for variable_name, variable in dataset.data_vars.items():
            dimensions = list(variable.dims)
            chunks = tuple(
                min(effective_chunks[dimension], size)
                for dimension, size in zip(
                    variable.dims,
                    variable.shape,
                    strict=True,
                )
            )
            target = group.create_array(
                variable_name,
                shape=variable.shape,
                dtype=variable.dtype,
                chunks=chunks or None,
                fill_value=None,
                attributes={"_ARRAY_DIMENSIONS": dimensions},
            )
            if not variable.dims:
                target[...] = variable.to_numpy()
                continue
            starts = [
                range(0, size, chunk)
                for size, chunk in zip(
                    variable.shape,
                    chunks,
                    strict=True,
                )
            ]
            for chunk_starts in itertools.product(*starts):
                region = tuple(
                    slice(start, min(start + chunk, size))
                    for start, chunk, size in zip(
                        chunk_starts,
                        chunks,
                        variable.shape,
                        strict=True,
                    )
                )
                target[region] = variable.to_numpy()[region]
        zarr.consolidate_metadata(temporary, zarr_format=2)
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return xr.open_zarr(destination, chunks="auto", consolidated=True)


def _resolve_zarr_chunks(dataset: xr.Dataset, zarr_chunks) -> dict[str, int]:
    defaults = {
        "draw": 1,
        "cell_name": 256,
        "target_sample": 8,
        "gene": 256,
        "protein": 256,
        "quantile": max(1, dataset.sizes.get("quantile", 1)),
    }
    if zarr_chunks is None:
        requested = {}
    elif isinstance(zarr_chunks, dict):
        requested = dict(zarr_chunks)
    else:
        raise TypeError("zarr_chunks must be a mapping from dimension to size.")
    unknown = set(requested) - set(dataset.dims)
    if unknown:
        raise ValueError(f"zarr_chunks contains unknown dimensions: {sorted(unknown)}.")
    result = {}
    for dimension, size in dataset.sizes.items():
        chunk = requested.get(dimension, defaults.get(dimension, size))
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
            raise ValueError(
                f"Zarr chunk size for {dimension!r} must be a positive integer."
            )
        result[dimension] = min(chunk, max(1, size))
    return result


def _preflight_zarr_path(zarr_path) -> None:
    if zarr_path is None:
        return
    destination = Path(zarr_path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing Zarr store: {destination}")


class _AtomicZarrRegionWriter:
    """Write cell-aligned xarray pieces into one atomic sibling store."""

    def __init__(
        self,
        *,
        destination,
        template: xr.Dataset,
        cell_names: np.ndarray,
        zarr_chunks,
        attrs: dict,
    ):
        try:
            import dask  # noqa: F401
            import zarr
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "Zarr output requires dask and zarr. "
                "Install the existing 'parallel' extra."
            ) from error

        self._zarr = zarr
        self.destination = Path(destination)
        _preflight_zarr_path(self.destination)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.destination.with_name(
            f".{self.destination.name}.tmp-{uuid.uuid4().hex}"
        )
        self.full_sizes = dict(template.sizes)
        self.full_sizes["cell_name"] = cell_names.size
        sizing = template.drop_dims("cell_name")
        sizing = sizing.assign_coords(cell_name=cell_names)
        self.chunks = _resolve_zarr_chunks(sizing, zarr_chunks)

        try:
            self.group = zarr.open_group(
                self.temporary,
                mode="w",
                zarr_format=2,
            )
            self.group.attrs.update(dict(attrs))
            self.group.attrs["chunks"] = json.dumps(
                self.chunks,
                sort_keys=True,
                separators=(",", ":"),
            )
            for coordinate_name, coordinate in template.coords.items():
                values = (
                    cell_names
                    if coordinate_name == "cell_name"
                    else coordinate.to_numpy()
                )
                if values.dtype.kind in {"O", "U", "S"}:
                    values = np.asarray(values, dtype=str)
                shape = tuple(
                    self.full_sizes[dimension] for dimension in coordinate.dims
                )
                coordinate_chunks = tuple(
                    min(self.chunks[dimension], size)
                    for dimension, size in zip(
                        coordinate.dims,
                        shape,
                        strict=True,
                    )
                )
                self.group.create_array(
                    coordinate_name,
                    data=values,
                    chunks=coordinate_chunks or None,
                    fill_value=None,
                    attributes={"_ARRAY_DIMENSIONS": list(coordinate.dims)},
                )
            for variable_name, variable in template.data_vars.items():
                shape = tuple(
                    self.full_sizes[dimension] for dimension in variable.dims
                )
                variable_chunks = tuple(
                    min(self.chunks[dimension], size)
                    for dimension, size in zip(
                        variable.dims,
                        shape,
                        strict=True,
                    )
                )
                self.group.create_array(
                    variable_name,
                    shape=shape,
                    dtype=variable.dtype,
                    chunks=variable_chunks or None,
                    fill_value=None,
                    attributes={"_ARRAY_DIMENSIONS": list(variable.dims)},
                )
        except Exception:
            self.abort()
            raise

    def write_cell_region(
        self,
        dataset: xr.Dataset,
        *,
        start: int,
        stop: int,
    ) -> None:
        for variable_name, variable in dataset.data_vars.items():
            region = tuple(
                slice(start, stop)
                if dimension == "cell_name"
                else slice(None)
                for dimension in variable.dims
            )
            self.group[variable_name][region] = variable.to_numpy()

    def finalize(self) -> xr.Dataset:
        try:
            self._zarr.consolidate_metadata(
                self.temporary,
                zarr_format=2,
            )
            os.replace(self.temporary, self.destination)
        except Exception:
            self.abort()
            raise
        return xr.open_zarr(
            self.destination,
            chunks="auto",
            consolidated=True,
        )

    def abort(self) -> None:
        if getattr(self, "temporary", None) is not None and self.temporary.exists():
            shutil.rmtree(self.temporary)


def _stream_cell_regions_to_zarr(
    *,
    build_piece,
    query_indices: np.ndarray,
    cell_names: np.ndarray,
    cell_chunk_size: int,
    zarr_path,
    zarr_chunks,
    global_estimated_bytes: int,
    attrs_override: dict | None = None,
) -> xr.Dataset:
    writer = None
    try:
        for start in range(0, query_indices.size, cell_chunk_size):
            stop = min(start + cell_chunk_size, query_indices.size)
            piece = build_piece(query_indices[start:stop], start, stop)
            if writer is None:
                attrs = dict(piece.attrs)
                attrs["estimated_bytes_with_overhead"] = int(
                    global_estimated_bytes
                )
                attrs["storage_mode"] = "atomic_zarr_cell_regions"
                if attrs_override:
                    attrs.update(attrs_override)
                writer = _AtomicZarrRegionWriter(
                    destination=zarr_path,
                    template=piece,
                    cell_names=cell_names,
                    zarr_chunks=zarr_chunks,
                    attrs=attrs,
                )
            writer.write_cell_region(piece, start=start, stop=stop)
        if writer is None:
            raise ValueError("indices must retain at least one query cell.")
        return writer.finalize()
    except Exception:
        if writer is not None:
            writer.abort()
        raise


def get_counterfactual_latent(
    model,
    adata=None,
    indices=None,
    *,
    target_samples=None,
    inference_mode="latent_mean",
    n_draws=1,
    quantiles=(0.025, 0.5, 0.975),
    reference_indices=None,
    support_quantile=0.05,
    admissibility_threshold=0.0,
    batch_size=256,
    target_chunk_size=None,
    random_state=0,
    zarr_path=None,
    zarr_chunks=None,
) -> xr.Dataset:
    """Build the registered-sample centered latent dataset."""
    _preflight_zarr_path(zarr_path)
    if model.hierarchy_mode != "centered_v2":
        raise RuntimeError(
            "get_counterfactual_latent() requires hierarchy_mode='centered_v2'. "
            "Legacy models retain get_local_sample_representation()."
        )
    model._check_if_trained(warn=False)
    adata = model._validate_anndata(adata)
    query_indices = _as_indices(adata, indices, name="indices")
    reference_indices = _as_indices(
        adata,
        reference_indices,
        name="reference_indices",
    )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    _validate_chunk_size(target_chunk_size, name="target_chunk_size")
    quantile_values = _validate_inference(inference_mode, n_draws, quantiles)
    target_indices, target_labels = _validate_targets(model, target_samples)

    cell_names = np.asarray(adata.obs_names[query_indices], dtype=str)
    if np.unique(cell_names).size != cell_names.size:
        raise ValueError("Query cell IDs must be unique for deterministic posterior draws.")
    estimated_bytes = _estimate_latent_bytes(
        n_cells=query_indices.size,
        n_targets=target_indices.size,
        n_draws=n_draws,
        n_latent_u=model.module.n_latent_u,
        n_latent=model.module.n_latent,
        n_quantiles=quantile_values.size,
        posterior_mc=inference_mode == "posterior_mc",
    )
    if zarr_path is None and estimated_bytes > MAX_IN_MEMORY_BYTES:
        raise MemoryError(
            f"Estimated in-memory materialization is {estimated_bytes} bytes, "
            f"exceeding the hard 512 MiB ({MAX_IN_MEMORY_BYTES}-byte) limit. "
            "Subset the request or provide zarr_path."
        )
    if zarr_path is not None and estimated_bytes > MAX_IN_MEMORY_BYTES:
        per_cell_bytes = _estimate_latent_bytes(
            n_cells=1,
            n_targets=target_indices.size,
            n_draws=n_draws,
            n_latent_u=model.module.n_latent_u,
            n_latent=model.module.n_latent,
            n_quantiles=quantile_values.size,
            posterior_mc=inference_mode == "posterior_mc",
        )
        if per_cell_bytes > MAX_IN_MEMORY_BYTES:
            raise MemoryError(
                "One query cell exceeds the 512 MiB working-set limit even "
                "with Zarr output; reduce draws or target samples."
            )
        cell_chunk_size = max(
            1,
            min(
                batch_size,
                query_indices.size,
                MAX_IN_MEMORY_BYTES // max(1, per_cell_bytes),
            ),
        )

        def build_piece(piece_indices, _start, _stop):
            return get_counterfactual_latent(
                model,
                adata=adata,
                indices=piece_indices,
                target_samples=target_labels,
                inference_mode=inference_mode,
                n_draws=n_draws,
                quantiles=quantile_values,
                reference_indices=reference_indices,
                support_quantile=support_quantile,
                admissibility_threshold=admissibility_threshold,
                batch_size=batch_size,
                target_chunk_size=target_chunk_size,
                random_state=random_state,
            )

        return _stream_cell_regions_to_zarr(
            build_piece=build_piece,
            query_indices=query_indices,
            cell_names=cell_names,
            cell_chunk_size=cell_chunk_size,
            zarr_path=zarr_path,
            zarr_chunks=zarr_chunks,
            global_estimated_bytes=estimated_bytes,
        )

    loc, scale, observed_indices = _encode_u_params(
        model,
        adata,
        query_indices,
        batch_size,
    )
    if inference_mode == "latent_mean":
        u_values = loc[None, ...]
    else:
        noise = _counter_normal_noise(
            seed=int(random_state),
            cell_names=cell_names,
            n_draws=n_draws,
            n_latent=model.module.n_latent_u,
        )
        u_values = loc[None, ...] + scale[None, ...] * noise
    u_values = u_values.astype(np.float32, copy=False)

    z_base_parts = []
    eps_raw_parts = []
    eps_centered_parts = []
    z_parts = []
    was_training = model.module.training
    model.module.eval()
    try:
        with torch.inference_mode():
            for start in range(0, query_indices.size, batch_size):
                stop = min(start + batch_size, query_indices.size)
                u_batch = torch.as_tensor(u_values[:, start:stop])
                z_base, eps_raw, eps_centered, z_all = (
                    model.module._all_sample_residuals(
                        u_batch,
                        target_chunk_size=target_chunk_size,
                    )
                )
                z_base_parts.append(z_base.detach().cpu().numpy())
                eps_raw_parts.append(
                    eps_raw[:, :, target_indices].detach().cpu().numpy()
                )
                eps_centered_parts.append(
                    eps_centered[:, :, target_indices].detach().cpu().numpy()
                )
                z_parts.append(z_all[:, :, target_indices].detach().cpu().numpy())
    finally:
        model.module.train(was_training)

    z_base_values = np.concatenate(z_base_parts, axis=1).astype(
        np.float32, copy=False
    )
    eps_raw_values = np.concatenate(eps_raw_parts, axis=1).astype(
        np.float32, copy=False
    )
    eps_centered_values = np.concatenate(eps_centered_parts, axis=1).astype(
        np.float32, copy=False
    )
    z_values = np.concatenate(z_parts, axis=1).astype(np.float32, copy=False)

    support, admissible = _support_and_admissibility(
        model=model,
        adata=adata,
        query_indices=query_indices,
        query_loc=loc,
        target_indices=target_indices,
        reference_indices=reference_indices,
        support_quantile=float(support_quantile),
        admissibility_threshold=float(admissibility_threshold),
        batch_size=batch_size,
    )
    registered_labels = np.asarray([str(value) for value in model.sample_order])
    observed_labels = registered_labels[observed_indices]
    dataset = xr.Dataset(
        {
            "u": (
                ("draw", "cell_name", "latent_u_dim"),
                u_values,
            ),
            "z_base": (
                ("draw", "cell_name", "latent_dim"),
                z_base_values,
            ),
            "eps_raw": (
                ("draw", "cell_name", "target_sample", "latent_dim"),
                eps_raw_values,
            ),
            "eps_centered": (
                ("draw", "cell_name", "target_sample", "latent_dim"),
                eps_centered_values,
            ),
            "z": (
                ("draw", "cell_name", "target_sample", "latent_dim"),
                z_values,
            ),
            "admissible": (
                ("cell_name", "target_sample"),
                admissible,
            ),
            "target_support": (
                ("cell_name", "target_sample"),
                support,
            ),
            "observed_sample": (("cell_name",), observed_labels),
        },
        coords={
            "draw": np.arange(n_draws),
            "cell_name": cell_names,
            "latent_u_dim": np.arange(model.module.n_latent_u),
            "latent_dim": np.arange(model.module.n_latent),
            "target_sample": target_labels,
        },
        attrs={
            "schema_version": SCHEMA_VERSION,
            "hierarchy_mode": model.hierarchy_mode,
            "u_encoder_mode": model.u_encoder_mode,
            "inference_mode": inference_mode,
            "rng": "numpy.Philox keyed by seed, cell ID, draw, and latent coordinate",
            "random_state": int(random_state),
            "centering": "full registered sample universe in registry order",
            "raw_residual_penalty": "mean_sample(sum_latent(-log Normal(eps_raw)))",
            "dtype": "float32",
            "chunks": json.dumps(
                {
                    "cell_name": int(batch_size),
                    "target_sample": int(
                        target_chunk_size or model.module._n_sample
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "storage_mode": "in_memory",
            "estimated_bytes_with_overhead": int(estimated_bytes),
            "registered_target_limitation": "registered samples only; no new-sample extrapolation",
            "interpretation": INTERPRETATION,
        },
    )
    if inference_mode == "posterior_mc":
        dataset = _add_posterior_summaries(
            dataset,
            quantiles=quantile_values,
        )
    if zarr_path is not None:
        return _write_atomic_zarr(
            dataset,
            zarr_path=zarr_path,
            zarr_chunks=zarr_chunks,
        )
    return dataset


def get_counterfactual_expression(
    model,
    adata=None,
    indices=None,
    *,
    target_samples=None,
    gene_list=None,
    protein_list=None,
    inference_mode="latent_mean",
    n_draws=1,
    quantiles=(0.025, 0.5, 0.975),
    batch_policy="observed",
    specified_batch=None,
    panel_policy="observed",
    specified_panel=None,
    library_policy="observed",
    specified_library_size=None,
    marginal_reference_indices=None,
    batch_size=256,
    target_chunk_size=None,
    feature_chunk_size=None,
    random_state=0,
    zarr_path=None,
    zarr_chunks=None,
) -> xr.Dataset:
    """Decode deterministic RNA/protein expectations for registered samples."""
    _preflight_zarr_path(zarr_path)
    if model.hierarchy_mode != "centered_v2":
        raise RuntimeError(
            "get_counterfactual_expression() requires hierarchy_mode='centered_v2'."
        )
    model._check_if_trained(warn=False)
    adata = model._validate_anndata(adata)
    query_indices = _as_indices(adata, indices, name="indices")
    uses_marginal_context = "sample_balanced_marginal" in {
        batch_policy,
        panel_policy,
        library_policy,
    }
    reference_indices = None
    if uses_marginal_context or marginal_reference_indices is not None:
        reference_indices = _as_indices(
            adata,
            marginal_reference_indices,
            name="marginal_reference_indices",
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    _validate_chunk_size(target_chunk_size, name="target_chunk_size")
    _validate_chunk_size(feature_chunk_size, name="feature_chunk_size")

    quantile_values = _validate_inference(inference_mode, n_draws, quantiles)
    target_indices, target_labels = _validate_targets(model, target_samples)
    gene_indices, gene_labels = _validate_features(
        adata.var_names,
        gene_list,
        feature_type="gene",
    )
    protein_registry = model.adata_manager.get_state_registry(
        REGISTRY_KEYS.PROTEIN_EXP_KEY
    )
    protein_indices, protein_labels = _validate_features(
        protein_registry.column_names,
        protein_list,
        feature_type="protein",
    )
    _validate_context_request(
        model=model,
        n_cells=query_indices.size,
        batch_policy=batch_policy,
        specified_batch=specified_batch,
        panel_policy=panel_policy,
        specified_panel=specified_panel,
        library_policy=library_policy,
        specified_library_size=specified_library_size,
        has_marginal_reference=(
            reference_indices is not None and reference_indices.size > 0
        ),
    )
    estimated_bytes = _estimate_expression_bytes(
        n_cells=query_indices.size,
        n_targets=target_indices.size,
        n_draws=n_draws,
        n_genes=gene_indices.size,
        n_proteins=protein_indices.size,
        n_quantiles=quantile_values.size,
        posterior_mc=inference_mode == "posterior_mc",
    )
    if zarr_path is None and estimated_bytes > MAX_IN_MEMORY_BYTES:
        raise MemoryError(
            f"Estimated in-memory materialization is {estimated_bytes} bytes, "
            f"exceeding the hard 512 MiB ({MAX_IN_MEMORY_BYTES}-byte) limit. "
            "Subset the request or provide zarr_path."
        )

    observed_context = _collect_observed_context(
        model,
        adata,
        query_indices,
        batch_size,
    )
    reference_context = (
        _collect_observed_context(
            model,
            adata,
            reference_indices,
            batch_size,
        )
        if uses_marginal_context
        else None
    )
    scenarios, context_hash = _build_context_scenarios(
        model=model,
        observed_context=observed_context,
        batch_policy=batch_policy,
        specified_batch=specified_batch,
        panel_policy=panel_policy,
        specified_panel=specified_panel,
        library_policy=library_policy,
        specified_library_size=specified_library_size,
        reference_context=reference_context,
    )

    cell_names = np.asarray(adata.obs_names[query_indices], dtype=str)
    if np.unique(cell_names).size != cell_names.size:
        raise ValueError("Query cell IDs must be unique for deterministic posterior draws.")
    if zarr_path is not None and estimated_bytes > MAX_IN_MEMORY_BYTES:
        per_cell_bytes = _estimate_expression_bytes(
            n_cells=1,
            n_targets=target_indices.size,
            n_draws=n_draws,
            n_genes=gene_indices.size,
            n_proteins=protein_indices.size,
            n_quantiles=quantile_values.size,
            posterior_mc=inference_mode == "posterior_mc",
        )
        if per_cell_bytes > MAX_IN_MEMORY_BYTES:
            raise MemoryError(
                "One query cell exceeds the 512 MiB working-set limit even "
                "with Zarr output; reduce draws, targets, or features."
            )
        cell_chunk_size = max(
            1,
            min(
                batch_size,
                query_indices.size,
                MAX_IN_MEMORY_BYTES // max(1, per_cell_bytes),
            ),
        )
        specified_library_array = (
            None
            if specified_library_size is None
            else np.asarray(specified_library_size)
        )

        def build_piece(piece_indices, start, stop):
            piece_library = specified_library_size
            if (
                specified_library_array is not None
                and specified_library_array.ndim == 1
            ):
                piece_library = specified_library_array[start:stop]
            return get_counterfactual_expression(
                model,
                adata=adata,
                indices=piece_indices,
                target_samples=target_labels,
                gene_list=gene_labels,
                protein_list=protein_labels,
                inference_mode=inference_mode,
                n_draws=n_draws,
                quantiles=quantile_values,
                batch_policy=batch_policy,
                specified_batch=specified_batch,
                panel_policy=panel_policy,
                specified_panel=specified_panel,
                library_policy=library_policy,
                specified_library_size=piece_library,
                marginal_reference_indices=reference_indices,
                batch_size=batch_size,
                target_chunk_size=target_chunk_size,
                feature_chunk_size=feature_chunk_size,
                random_state=random_state,
            )

        return _stream_cell_regions_to_zarr(
            build_piece=build_piece,
            query_indices=query_indices,
            cell_names=cell_names,
            cell_chunk_size=cell_chunk_size,
            zarr_path=zarr_path,
            zarr_chunks=zarr_chunks,
            global_estimated_bytes=estimated_bytes,
            attrs_override={"context_table_sha256": context_hash},
        )

    loc, scale, _ = _encode_u_params(model, adata, query_indices, batch_size)
    if inference_mode == "latent_mean":
        u_values = loc[None, ...]
    else:
        noise = _counter_normal_noise(
            seed=int(random_state),
            cell_names=cell_names,
            n_draws=n_draws,
            n_latent=model.module.n_latent_u,
        )
        u_values = loc[None, ...] + scale[None, ...] * noise
    u_values = u_values.astype(np.float32, copy=False)

    z_parts = []
    was_training = model.module.training
    model.module.eval()
    try:
        with torch.inference_mode():
            for start in range(0, query_indices.size, batch_size):
                stop = min(start + batch_size, query_indices.size)
                _, _, _, z_all = model.module._all_sample_residuals(
                    torch.as_tensor(u_values[:, start:stop]),
                    target_chunk_size=target_chunk_size,
                )
                z_parts.append(
                    z_all[:, :, target_indices].detach().cpu().numpy()
                )
    finally:
        model.module.train(was_training)
    z_values = np.concatenate(z_parts, axis=1).astype(np.float32, copy=False)

    n_cells = query_indices.size
    n_targets = target_indices.size
    draw_cell_target = (n_draws, n_cells, n_targets)
    rna_shape = (*draw_cell_target, gene_indices.size)
    protein_shape = (*draw_cell_target, protein_indices.size)
    rna_scale = np.zeros(rna_shape, dtype=np.float32)
    rna_rate = np.zeros(rna_shape, dtype=np.float32)
    protein_values = {
        name: np.zeros(protein_shape, dtype=np.float32)
        for name in (
            "protein_background_component_mean",
            "protein_foreground_component_mean",
            "protein_foreground_probability",
            "protein_background_contribution",
            "protein_foreground_contribution",
            "protein_total_mean",
            "protein_batch_efficiency",
        )
    }
    protein_available = np.ones(
        (n_cells, protein_indices.size),
        dtype=bool,
    )

    flat_z = z_values.reshape(-1, model.module.n_latent)
    device = model.module.device
    was_training = model.module.training
    model.module.eval()
    try:
        for scenario in scenarios:
            batch_values = np.broadcast_to(
                scenario["batch"][None, :, None],
                draw_cell_target,
            ).reshape(-1, 1).copy()
            library_values = np.broadcast_to(
                scenario["library"][None, :, None],
                draw_cell_target,
            ).reshape(-1, 1).copy()
            weights = scenario["weight"][None, :, None, None]
            cat_covs = observed_context["cat_covs"]
            if cat_covs is not None:
                cat_covs = np.broadcast_to(
                    cat_covs[None, :, None, :],
                    (*draw_cell_target, cat_covs.shape[1]),
                ).reshape(-1, cat_covs.shape[1]).copy()
            cont_covs = observed_context["cont_covs"]
            if cont_covs is not None:
                cont_covs = np.broadcast_to(
                    cont_covs[None, :, None, :],
                    (*draw_cell_target, cont_covs.shape[1]),
                ).reshape(-1, cont_covs.shape[1]).copy()

            decoded = model.module._deterministic_decoder_parameters(
                torch.as_tensor(flat_z, dtype=torch.float32, device=device),
                torch.as_tensor(
                    library_values,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(batch_values, dtype=torch.long, device=device),
                cont_covs=(
                    torch.as_tensor(
                        cont_covs,
                        dtype=torch.float32,
                        device=device,
                    )
                    if cont_covs is not None
                    else None
                ),
                cat_covs=(
                    torch.as_tensor(cat_covs, dtype=torch.long, device=device)
                    if cat_covs is not None
                    else None
                ),
            )
            gene_step = feature_chunk_size or gene_indices.size
            for feature_start in range(0, gene_indices.size, gene_step):
                feature_stop = min(
                    feature_start + gene_step,
                    gene_indices.size,
                )
                selected = gene_indices[feature_start:feature_stop]
                selected_shape = (
                    *draw_cell_target,
                    feature_stop - feature_start,
                )
                decoded_rna_scale = (
                    decoded["rna_scale"][:, selected]
                    .reshape(selected_shape)
                    .detach()
                    .cpu()
                    .numpy()
                )
                decoded_rna_rate = (
                    decoded["rna_rate"][:, selected]
                    .reshape(selected_shape)
                    .detach()
                    .cpu()
                    .numpy()
                )
                rna_scale[..., feature_start:feature_stop] += (
                    weights * decoded_rna_scale
                )
                rna_rate[..., feature_start:feature_stop] += (
                    weights * decoded_rna_rate
                )

            protein_step = feature_chunk_size or protein_indices.size
            for feature_start in range(0, protein_indices.size, protein_step):
                feature_stop = min(
                    feature_start + protein_step,
                    protein_indices.size,
                )
                selected = protein_indices[feature_start:feature_stop]
                selected_shape = (
                    *draw_cell_target,
                    feature_stop - feature_start,
                )
                for variable_name in protein_values:
                    decoded_values = (
                        decoded[variable_name][:, selected]
                        .reshape(selected_shape)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    protein_values[variable_name][
                        ..., feature_start:feature_stop
                    ] += weights * decoded_values

            if model.module.protein_batch_mask is not None:
                for cell_position, panel_index in enumerate(scenario["panel"]):
                    if scenario["weight"][cell_position] <= 0:
                        continue
                    panel_mask = model.module.protein_batch_mask[
                        str(int(panel_index))
                    ].astype(bool)
                    protein_available[cell_position] &= panel_mask[protein_indices]
    finally:
        model.module.train(was_training)

    availability = np.broadcast_to(
        protein_available[:, None, :],
        (n_cells, n_targets, protein_indices.size),
    ).copy()
    availability_draw = availability[None, ...]
    for variable_name in protein_values:
        protein_values[variable_name] = np.where(
            availability_draw,
            protein_values[variable_name],
            np.nan,
        ).astype(np.float32, copy=False)

    variables = {
        "rna_scale": (
            ("draw", "cell_name", "target_sample", "gene"),
            rna_scale,
        ),
        "rna_rate": (
            ("draw", "cell_name", "target_sample", "gene"),
            rna_rate,
        ),
        **{
            variable_name: (
                ("draw", "cell_name", "target_sample", "protein"),
                values,
            )
            for variable_name, values in protein_values.items()
        },
        "protein_available": (
            ("cell_name", "target_sample", "protein"),
            availability,
        ),
    }
    dataset = xr.Dataset(
        variables,
        coords={
            "draw": np.arange(n_draws),
            "cell_name": cell_names,
            "target_sample": target_labels,
            "gene": gene_labels,
            "protein": protein_labels,
        },
        attrs={
            "schema_version": SCHEMA_VERSION,
            "hierarchy_mode": model.hierarchy_mode,
            "u_encoder_mode": model.u_encoder_mode,
            "inference_mode": inference_mode,
            "rng": "numpy.Philox keyed by seed, cell ID, draw, and latent coordinate",
            "random_state": int(random_state),
            "batch_policy": batch_policy,
            "panel_policy": panel_policy,
            "library_policy": library_policy,
            "observed_library_estimand": observed_context[
                "library_estimand"
            ],
            "specified_batch": (
                "" if specified_batch is None else str(specified_batch)
            ),
            "specified_panel": (
                "" if specified_panel is None else str(specified_panel)
            ),
            "protein_mean_formula": (
                "efficiency * exp(back_alpha + 0.5 * back_beta**2); "
                "foreground = background * fore_scale; "
                "p_foreground = 1 - sigmoid(mixing)"
            ),
            "context_table_sha256": context_hash,
            "dtype": "float32",
            "chunks": json.dumps(
                {
                    "cell_name": int(batch_size),
                    "feature": int(
                        feature_chunk_size
                        or max(gene_indices.size, protein_indices.size)
                    ),
                    "target_sample": int(
                        target_chunk_size or model.module._n_sample
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            "storage_mode": "in_memory",
            "estimated_bytes_with_overhead": int(estimated_bytes),
            "registered_target_limitation": "registered samples only; no new-sample extrapolation",
            "interpretation": INTERPRETATION,
        },
    )
    if inference_mode == "posterior_mc":
        dataset = _add_posterior_summaries(
            dataset,
            quantiles=quantile_values,
        )
    if zarr_path is not None:
        return _write_atomic_zarr(
            dataset,
            zarr_path=zarr_path,
            zarr_chunks=zarr_chunks,
        )
    return dataset


def _hash_subsample_positions(
    names: np.ndarray,
    *,
    sample_label: str,
    limit: int | None,
    random_state: int,
) -> np.ndarray:
    if limit is None or names.size <= limit:
        return np.arange(names.size)
    scores = [
        hashlib.sha256(
            f"{int(random_state)}\0{sample_label}\0{name}".encode()
        ).digest()
        for name in names
    ]
    return np.asarray(
        sorted(range(names.size), key=lambda position: (scores[position], position))[
            :limit
        ],
        dtype=np.int64,
    )


def _sample_covariate_values(
    *,
    adata,
    sample_key: str,
    target_labels: np.ndarray,
    covariate_key: str,
) -> np.ndarray:
    if covariate_key not in adata.obs:
        raise KeyError(f"{covariate_key!r} is not present in reference_adata.obs.")
    sample_values = adata.obs[sample_key].astype(str).to_numpy()
    covariate_values = adata.obs[covariate_key].astype(str).to_numpy()
    result = []
    for target_label in target_labels:
        values = np.unique(covariate_values[sample_values == target_label])
        if values.size != 1:
            raise ValueError(
                f"{covariate_key!r} must be constant within sample "
                f"{target_label!r}; observed {values.tolist()}."
            )
        result.append(values[0])
    return np.asarray(result)


def _mixture_log_density(
    query: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate an equal-component Gaussian mixture without materializing it."""
    query_tensor = torch.as_tensor(query, dtype=torch.float32)
    log_total = None
    for start in range(0, loc.shape[0], chunk_size):
        stop = min(start + chunk_size, loc.shape[0])
        component_log_prob = Normal(
            torch.as_tensor(loc[start:stop], dtype=torch.float32),
            torch.as_tensor(scale[start:stop], dtype=torch.float32),
        ).log_prob(query_tensor[:, None, :]).sum(dim=-1)
        chunk_total = torch.logsumexp(component_log_prob, dim=-1)
        log_total = (
            chunk_total
            if log_total is None
            else torch.logaddexp(log_total, chunk_total)
        )
    return (
        log_total - math.log(loc.shape[0])
    ).numpy().astype(np.float32, copy=False)


def local_sample_enrichment(
    model,
    adata=None,
    indices=None,
    *,
    target_samples=None,
    reference_adata=None,
    reference_indices=None,
    inference_mode="latent_mean",
    n_draws=1,
    quantiles=(0.025, 0.5, 0.975),
    group_key=None,
    contrast=None,
    donor_key=None,
    max_reference_cells_per_sample=None,
    batch_size=256,
    reference_chunk_size=None,
    random_state=0,
) -> xr.Dataset:
    """Compute descriptive local densities under registered-sample posteriors."""
    if model.hierarchy_mode != "centered_v2":
        raise RuntimeError(
            "local_sample_enrichment() requires hierarchy_mode='centered_v2'."
        )
    model._check_if_trained(warn=False)
    adata = model._validate_anndata(adata)
    reference_adata = model._validate_anndata(
        adata if reference_adata is None else reference_adata
    )
    query_indices = _as_indices(adata, indices, name="indices")
    reference_indices = _as_indices(
        reference_adata,
        reference_indices,
        name="reference_indices",
    )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if reference_chunk_size is None:
        reference_chunk_size = batch_size
    if (
        isinstance(reference_chunk_size, bool)
        or not isinstance(reference_chunk_size, int)
        or reference_chunk_size < 1
    ):
        raise ValueError("reference_chunk_size must be a positive integer or None.")
    if max_reference_cells_per_sample is not None and (
        isinstance(max_reference_cells_per_sample, bool)
        or not isinstance(max_reference_cells_per_sample, int)
        or max_reference_cells_per_sample < 1
    ):
        raise ValueError(
            "max_reference_cells_per_sample must be a positive integer or None."
        )

    quantile_values = _validate_inference(inference_mode, n_draws, quantiles)
    target_indices, target_labels = _validate_targets(model, target_samples)
    cell_names = np.asarray(adata.obs_names[query_indices], dtype=str)
    if np.unique(cell_names).size != cell_names.size:
        raise ValueError("Query cell IDs must be unique for deterministic posterior draws.")

    query_loc, query_scale, observed_samples = _encode_u_params(
        model,
        adata,
        query_indices,
        batch_size,
    )
    if inference_mode == "latent_mean":
        query_draws = query_loc[None, ...]
    else:
        query_draws = query_loc[None, ...] + query_scale[None, ...] * (
            _counter_normal_noise(
                seed=int(random_state),
                cell_names=cell_names,
                n_draws=n_draws,
                n_latent=model.module.n_latent_u,
            )
        )
    query_draws = query_draws.astype(np.float32, copy=False)

    reference_loc, reference_scale, reference_samples = _encode_u_params(
        model,
        reference_adata,
        reference_indices,
        batch_size,
    )
    reference_names = np.asarray(
        reference_adata.obs_names[reference_indices],
        dtype=str,
    )

    selected_reference: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for target_index, target_label in zip(
        target_indices,
        target_labels,
        strict=True,
    ):
        positions = np.flatnonzero(reference_samples == target_index)
        keep = _hash_subsample_positions(
            reference_names[positions],
            sample_label=str(target_label),
            limit=max_reference_cells_per_sample,
            random_state=int(random_state),
        )
        positions = positions[keep]
        selected_reference[int(target_index)] = (
            reference_loc[positions],
            reference_scale[positions],
            reference_names[positions],
        )

    density = np.full(
        (n_draws, query_indices.size, target_indices.size),
        np.nan,
        dtype=np.float32,
    )
    n_reference_cells = np.zeros(
        (query_indices.size, target_indices.size),
        dtype=np.int64,
    )
    self_excluded = np.zeros_like(n_reference_cells, dtype=bool)
    finite_support = np.zeros_like(n_reference_cells, dtype=bool)
    for query_position, (query_name, factual_sample) in enumerate(
        zip(cell_names, observed_samples, strict=True)
    ):
        for target_position, target_index in enumerate(target_indices):
            loc, scale, names = selected_reference[int(target_index)]
            keep = np.ones(names.size, dtype=bool)
            if target_index == factual_sample:
                self_matches = names == query_name
                self_excluded[query_position, target_position] = bool(
                    np.any(self_matches)
                )
                keep &= ~self_matches
            loc = loc[keep]
            scale = scale[keep]
            n_reference_cells[query_position, target_position] = loc.shape[0]
            if loc.shape[0] == 0:
                continue
            finite_support[query_position, target_position] = True
            density[:, query_position, target_position] = _mixture_log_density(
                query_draws[:, query_position],
                loc,
                scale,
                chunk_size=reference_chunk_size,
            )

    data_vars = {
        "log_density": (
            ("draw", "cell_name", "target_sample"),
            density,
        ),
        "n_reference_cells": (
            ("cell_name", "target_sample"),
            n_reference_cells,
        ),
        "self_excluded": (
            ("cell_name", "target_sample"),
            self_excluded,
        ),
        "finite_support": (
            ("cell_name", "target_sample"),
            finite_support,
        ),
    }
    coords: dict[str, np.ndarray] = {
        "draw": np.arange(n_draws),
        "cell_name": cell_names,
        "target_sample": target_labels,
    }

    group_values = None
    group_order = None
    if group_key is not None:
        group_values = _sample_covariate_values(
            adata=reference_adata,
            sample_key=model.sample_key,
            target_labels=target_labels,
            covariate_key=group_key,
        )
        group_order = np.asarray(list(dict.fromkeys(group_values.tolist())))
        group_density = np.empty(
            (n_draws, query_indices.size, group_order.size),
            dtype=np.float32,
        )
        for group_position, group in enumerate(group_order):
            positions = np.flatnonzero(group_values == group)
            group_density[..., group_position] = (
                np.logaddexp.reduce(density[..., positions], axis=-1)
                - math.log(positions.size)
            )
        data_vars["group_log_density"] = (
            ("draw", "cell_name", "group"),
            group_density,
        )
        coords["group"] = group_order
    elif contrast is not None or donor_key is not None:
        raise ValueError("group_key is required when contrast or donor_key is set.")

    numerator = denominator = None
    if contrast is not None:
        contrast_values = np.asarray([str(value) for value in contrast])
        if (
            contrast_values.shape != (2,)
            or contrast_values[0] == contrast_values[1]
        ):
            raise ValueError(
                "contrast must contain two distinct values: "
                "(numerator, denominator)."
            )
        unknown = [
            value for value in contrast_values if value not in set(group_order)
        ]
        if unknown:
            raise ValueError(f"Unknown contrast group(s): {unknown}.")
        numerator, denominator = contrast_values
        numerator_position = int(np.flatnonzero(group_order == numerator)[0])
        denominator_position = int(np.flatnonzero(group_order == denominator)[0])
        data_vars["log_ratio"] = (
            ("draw", "cell_name"),
            group_density[..., numerator_position]
            - group_density[..., denominator_position],
        )

    if donor_key is not None:
        if contrast is None:
            raise ValueError("contrast is required for paired donor summaries.")
        donor_values = _sample_covariate_values(
            adata=reference_adata,
            sample_key=model.sample_key,
            target_labels=target_labels,
            covariate_key=donor_key,
        )
        numerator_positions = np.flatnonzero(group_values == numerator)
        denominator_positions = np.flatnonzero(group_values == denominator)
        numerator_donors = donor_values[numerator_positions]
        denominator_donors = donor_values[denominator_positions]
        if (
            np.unique(numerator_donors).size != numerator_donors.size
            or np.unique(denominator_donors).size != denominator_donors.size
            or set(numerator_donors) != set(denominator_donors)
        ):
            raise ValueError(
                "Paired contrasts require exactly one numerator and one "
                "denominator sample for every donor."
            )
        donor_order = np.asarray(list(dict.fromkeys(numerator_donors.tolist())))
        donor_log_ratio = np.empty(
            (n_draws, query_indices.size, donor_order.size),
            dtype=np.float32,
        )
        for donor_position, donor in enumerate(donor_order):
            numerator_sample = numerator_positions[
                np.flatnonzero(numerator_donors == donor)[0]
            ]
            denominator_sample = denominator_positions[
                np.flatnonzero(denominator_donors == donor)[0]
            ]
            donor_log_ratio[..., donor_position] = (
                density[..., numerator_sample] - density[..., denominator_sample]
            )
        data_vars["donor_log_ratio"] = (
            ("draw", "cell_name", "donor"),
            donor_log_ratio,
        )
        data_vars["donor_log_ratio_mean"] = (
            ("draw", "cell_name"),
            np.mean(donor_log_ratio, axis=-1, dtype=np.float64).astype(np.float32),
        )
        data_vars["donor_log_ratio_median"] = (
            ("draw", "cell_name"),
            np.median(donor_log_ratio, axis=-1).astype(np.float32),
        )
        coords["donor"] = donor_order

    dataset = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "schema_version": SCHEMA_VERSION,
            "hierarchy_mode": model.hierarchy_mode,
            "u_encoder_mode": model.u_encoder_mode,
            "inference_mode": inference_mode,
            "rng": "numpy.Philox keyed by seed, cell ID, draw, and latent coordinate",
            "random_state": int(random_state),
            "density": "equal-component aggregated posterior Gaussian mixture",
            "group_density": "equal-sample logmeanexp; no cell-count weighting",
            "self_exclusion": "query cell excluded only from its factual sample mixture",
            "interpretation": INTERPRETATION,
        },
    )
    if inference_mode == "posterior_mc":
        dataset = _add_posterior_summaries(
            dataset,
            quantiles=quantile_values,
        )
    return dataset
