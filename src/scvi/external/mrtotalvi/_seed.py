"""Training-seed aggregation for MrTotalVI counterfactual datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Mapping


def combine_mrtotalvi_seed_results(
    results: Mapping[int, xr.Dataset],
) -> xr.Dataset:
    """Combine seed-specific datasets without pooling uncertainty sources."""
    if not results:
        raise ValueError("results must be a non-empty mapping.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in results):
        raise TypeError("Every results key must be an integer training seed.")
    if any(not isinstance(dataset, xr.Dataset) for dataset in results.values()):
        raise TypeError("Every results value must be an xarray.Dataset.")

    seeds = sorted(results)
    datasets = [results[seed] for seed in seeds]
    template = datasets[0]
    if "draw" not in template.dims:
        raise ValueError("Every seed result must retain a draw dimension.")
    for dataset in datasets[1:]:
        if set(dataset.data_vars) != set(template.data_vars):
            raise ValueError("Seed results must contain identical data variables.")
        if {
            name: variable.dims for name, variable in dataset.data_vars.items()
        } != {
            name: variable.dims for name, variable in template.data_vars.items()
        }:
            raise ValueError("Seed result variable dimensions must match exactly.")
        if dataset.attrs.get("schema_version") != template.attrs.get(
            "schema_version"
        ):
            raise ValueError("Seed result schema_version attributes must match.")
        try:
            xr.align(template, dataset, join="exact", copy=False)
        except ValueError as error:
            raise ValueError("Seed result coordinates must match exactly.") from error

    combined = xr.concat(
        datasets,
        dim=xr.IndexVariable("training_seed", seeds),
        data_vars="all",
        coords="minimal",
        compat="equals",
        join="exact",
        combine_attrs="drop",
    )
    additions = {}
    for variable_name, variable in combined.data_vars.items():
        if "draw" not in variable.dims or not np.issubdtype(
            variable.dtype,
            np.floating,
        ):
            continue
        within_mean_name = f"{variable_name}_within_seed_posterior_mean"
        within_sd_name = f"{variable_name}_within_seed_posterior_sd"
        between_mean_name = f"{variable_name}_between_seed_mean"
        between_sd_name = f"{variable_name}_between_seed_sd"
        conflicts = {
            within_mean_name,
            within_sd_name,
            between_mean_name,
            between_sd_name,
        } & set(combined.data_vars)
        if conflicts:
            raise ValueError(
                "Seed summary variable name conflict(s): "
                f"{sorted(conflicts)}."
            )
        within_mean = variable.mean("draw")
        additions[within_mean_name] = within_mean
        additions[within_sd_name] = variable.std("draw", ddof=1)
        additions[between_mean_name] = within_mean.mean("training_seed")
        additions[between_sd_name] = within_mean.std("training_seed", ddof=1)

    combined = combined.assign(additions)
    combined.attrs = dict(template.attrs)
    combined.attrs["uncertainty_separation"] = (
        "within-seed posterior draws and between-seed training variation "
        "are reported separately and never pooled"
    )
    combined.attrs["training_seed_order"] = ",".join(map(str, seeds))
    return combined
