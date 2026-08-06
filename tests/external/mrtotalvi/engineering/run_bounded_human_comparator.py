"""Run the bounded, non-biological MrTotalVI v2 engineering smoke."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

import scvi
from scvi.external import MrTotalVI

if TYPE_CHECKING:
    import xarray as xr

EXPECTED_INPUT_SHA256 = (
    "520ff544daae6192efd7f3501669e05b0122e6fbaf8e9de0246122cecd1de2da"
)
EXPECTED_PANEL_SHA256 = (
    "90a467f72abe9b347784ceae0eeb1e4fb14476bbe65a3e736f77785f95cd4570"
)
EXPECTED_PROTEIN_ORDER_SHA256 = (
    "09bbe2e870a7005d411b9557711266e7e8d4ac3d3a5e2804ddd658467873418b"
)
EXPECTED_GENE_ORDER_SHA256 = (
    "2e2f27973e64360e839e896a7d955fc17ae4268485c519d721ea2baac96f681c"
)
SEED = 0
N_GROUPS = 20
CELLS_PER_GROUP = 25
N_GENES = 1000
N_PROTEINS = 130
QUERY_CELLS = 64
QUERY_GENES = 100
QUERY_PROTEINS = 20
TARGET_CHUNKS = (1, 5, 20)
FORWARD_BACKWARD_LIMIT_SECONDS = 120.0
FULL_SMOKE_LIMIT_SECONDS = 600.0
RSS_INCREASE_LIMIT_BYTES = 2 * 1024**3
ENGINEERING_CHUNK_RTOL = 1e-5
ENGINEERING_CHUNK_ATOL = 1e-5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        default=Path(
            "/exports/para-lipg-hpc/mdmanurung/schisto_citeseq/"
            "analysis/harmonized_integration/outputs/human_immune_joint.h5ad"
        ),
    )
    parser.add_argument(
        "--panel-map",
        type=Path,
        default=Path(
            "/exports/para-lipg-hpc/mdmanurung/schisto_citeseq/"
            "analysis/adt-denoising/outputs/adt_panel_map.tsv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".scratch/mrtotalvi-v2/engineering-runs"),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lines(values: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_lines(path: Path, values: list[str]) -> None:
    path.write_text(
        "".join(f"{value}\n" for value in values),
        encoding="utf-8",
    )


def _max_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _array_digest(values: np.ndarray) -> bytes:
    values = np.asarray(values)
    if values.dtype.kind in {"O", "U", "S"}:
        return json.dumps(
            values.astype(str).tolist(),
            separators=(",", ":"),
        ).encode()
    return np.ascontiguousarray(values).tobytes()


def _dataset_sha256(dataset: xr.Dataset) -> str:
    digest = hashlib.sha256()
    for name in sorted(dataset.coords):
        coordinate = dataset.coords[name]
        digest.update(f"coord:{name}:{coordinate.dims}:{coordinate.dtype}".encode())
        digest.update(_array_digest(coordinate.to_numpy()))
    for name in sorted(dataset.data_vars):
        variable = dataset[name]
        digest.update(f"var:{name}:{variable.dims}:{variable.dtype}".encode())
        digest.update(_array_digest(variable.to_numpy()))
    return digest.hexdigest()


def _assert_dataset_values_equal(
    actual: xr.Dataset,
    expected: xr.Dataset,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    if set(actual.coords) != set(expected.coords):
        raise AssertionError("Dataset coordinate names differ.")
    if set(actual.data_vars) != set(expected.data_vars):
        raise AssertionError("Dataset variable names differ.")
    for name in actual.coords:
        np.testing.assert_array_equal(
            actual.coords[name].to_numpy(),
            expected.coords[name].to_numpy(),
        )
    for name in actual.data_vars:
        left = actual[name]
        right = expected[name]
        if left.dims != right.dims:
            raise AssertionError(f"Variable dimensions differ for {name!r}.")
        if np.issubdtype(left.dtype, np.floating):
            np.testing.assert_allclose(
                left.to_numpy(),
                right.to_numpy(),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        else:
            np.testing.assert_array_equal(left.to_numpy(), right.to_numpy())


def _decode_h5_strings(dataset: h5py.Dataset) -> np.ndarray:
    values = dataset[:]
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        ],
        dtype=str,
    )


def _read_h5_categorical(group: h5py.Group) -> np.ndarray:
    categories = _decode_h5_strings(group["categories"])
    codes = np.asarray(group["codes"][:], dtype=np.int64)
    if np.any(codes < 0) or np.any(codes >= categories.size):
        raise ValueError(f"Categorical field {group.name!r} contains missing codes.")
    return categories[codes]


def _read_h5_csr_subset(
    group: h5py.Group,
    *,
    rows: np.ndarray,
    columns: np.ndarray,
) -> sp.csr_matrix:
    if group.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError(f"{group.name!r} must use H5AD CSR encoding.")
    shape = tuple(int(value) for value in group.attrs["shape"])
    indptr = np.asarray(group["indptr"][:], dtype=np.int64)
    column_lookup = np.full(shape[1], -1, dtype=np.int32)
    column_lookup[columns] = np.arange(columns.size, dtype=np.int32)
    output_indptr = np.zeros(rows.size + 1, dtype=np.int64)
    output_indices = []
    output_data = []
    for output_row, source_row in enumerate(rows):
        start = int(indptr[source_row])
        stop = int(indptr[source_row + 1])
        source_columns = np.asarray(group["indices"][start:stop], dtype=np.int64)
        mapped_columns = column_lookup[source_columns]
        keep = mapped_columns >= 0
        output_indices.append(mapped_columns[keep])
        output_data.append(np.asarray(group["data"][start:stop])[keep])
        output_indptr[output_row + 1] = (
            output_indptr[output_row] + int(np.count_nonzero(keep))
        )
    indices = (
        np.concatenate(output_indices)
        if output_indices
        else np.empty(0, dtype=np.int32)
    )
    data = (
        np.concatenate(output_data)
        if output_data
        else np.empty(0, dtype=group["data"].dtype)
    )
    return sp.csr_matrix(
        (data, indices, output_indptr),
        shape=(rows.size, columns.size),
    )


def _select_fixture(
    input_h5ad: Path,
    panel_map_path: Path,
) -> tuple[ad.AnnData, dict]:
    panel = pd.read_csv(panel_map_path, sep="\t")
    required_panel_columns = {"feature", "is_control"}
    if not required_panel_columns.issubset(panel.columns):
        raise ValueError(
            f"Panel map must contain {sorted(required_panel_columns)}."
        )
    is_control = panel["is_control"].map(
        lambda value: str(value).strip().lower() in {"1", "true"}
    )
    biological_proteins = panel.loc[~is_control, "feature"].astype(str).tolist()
    if len(biological_proteins) != N_PROTEINS:
        raise ValueError(
            f"Expected {N_PROTEINS} biological proteins, found "
            f"{len(biological_proteins)}."
        )
    protein_hash = _sha256_lines(biological_proteins)
    if protein_hash != EXPECTED_PROTEIN_ORDER_SHA256:
        raise ValueError(
            "The ordered biological-protein selection does not match its "
            "frozen SHA-256."
        )

    with h5py.File(input_h5ad, "r") as source:
        required_paths = {
            "obs/_index",
            "obs/timepoint",
            "obs/donor_timepoint",
            "obs/batch",
            "var/_index",
            "var/hvg_pearson_residuals_rank",
            "var/hvg_pearson_residuals_2000",
            "layers/counts",
            "obsm/protein",
            "uns/protein_names",
        }
        missing_paths = sorted(path for path in required_paths if path not in source)
        if missing_paths:
            raise ValueError(f"Input H5AD is missing paths {missing_paths}.")

        source_names = _decode_h5_strings(source["obs/_index"])
        source_timepoints = _read_h5_categorical(source["obs/timepoint"])
        source_groups = _read_h5_categorical(source["obs/donor_timepoint"])
        source_batches = _read_h5_categorical(source["obs/batch"])
        selected_timepoint = np.isin(source_timepoints, ["W00", "W22"])
        selected_groups = sorted(
            np.unique(source_groups[selected_timepoint]).astype(str).tolist()
        )
        if len(selected_groups) != N_GROUPS:
            raise ValueError(
                f"Expected {N_GROUPS} W00/W22 donor-timepoint groups, found "
                f"{len(selected_groups)}."
            )

        selected_positions = []
        selection_rows = []
        for group in selected_groups:
            positions = np.flatnonzero(source_groups == group)
            if positions.size < CELLS_PER_GROUP:
                raise ValueError(
                    f"{group!r} has only {positions.size} cells; "
                    f"{CELLS_PER_GROUP} are required."
                )
            scored = sorted(
                (
                    hashlib.sha256(
                        f"{SEED}\0{source_names[position]}".encode()
                    ).hexdigest(),
                    int(position),
                )
                for position in positions
            )
            for cell_hash, position in scored[:CELLS_PER_GROUP]:
                selected_positions.append(position)
                selection_rows.append(
                    {
                        "donor_timepoint": group,
                        "cell_name": source_names[position],
                        "selection_sha256": cell_hash,
                    }
                )
        selected_positions = np.asarray(sorted(selected_positions), dtype=np.int64)

        source_genes = _decode_h5_strings(source["var/_index"])
        hvg_rank = np.asarray(
            source["var/hvg_pearson_residuals_rank"][:],
            dtype=np.float64,
        )
        hvg_2000 = np.asarray(
            source["var/hvg_pearson_residuals_2000"][:],
            dtype=bool,
        )
        hvg_positions = np.flatnonzero(hvg_2000)
        if hvg_positions.size != 2000:
            raise ValueError(
                f"Expected 2,000 stored Pearson-residual HVGs, found "
                f"{hvg_positions.size}."
            )
        rank_order = np.lexsort((hvg_positions, hvg_rank[hvg_positions]))
        gene_positions = hvg_positions[rank_order[:N_GENES]]
        selected_genes = source_genes[gene_positions].tolist()
        gene_hash = _sha256_lines(selected_genes)
        if gene_hash != EXPECTED_GENE_ORDER_SHA256:
            raise ValueError(
                "The ordered 1,000-gene selection does not match its frozen "
                "SHA-256."
            )

        protein_names = _decode_h5_strings(source["uns/protein_names"]).tolist()
        if len(protein_names) != len(set(protein_names)):
            raise ValueError("Input protein names are not unique.")
        protein_lookup = {name: index for index, name in enumerate(protein_names)}
        missing_proteins = [
            name for name in biological_proteins if name not in protein_lookup
        ]
        if missing_proteins:
            raise ValueError(
                f"Panel-map proteins are absent from the input: {missing_proteins}."
            )
        protein_positions = [
            protein_lookup[name] for name in biological_proteins
        ]

        counts = _read_h5_csr_subset(
            source["layers/counts"],
            rows=selected_positions,
            columns=gene_positions,
        )
        proteins = np.asarray(
            source["obsm/protein"][selected_positions, :],
            dtype=np.float32,
        )[:, protein_positions]
        obs = pd.DataFrame(
            {
                "donor_timepoint": source_groups[selected_positions],
                "batch": source_batches[selected_positions],
            },
            index=pd.Index(source_names[selected_positions]),
        )
        var = pd.DataFrame(
            {
                "hvg_pearson_residuals_rank": hvg_rank[gene_positions],
                "hvg_pearson_residuals_2000": hvg_2000[gene_positions],
            },
            index=pd.Index(selected_genes),
        )

    if not np.isfinite(proteins).all() or np.any(proteins < 0):
        raise ValueError("Selected protein counts must be finite and nonnegative.")
    count_values = counts.data if sp.issparse(counts) else np.asarray(counts)
    if not np.isfinite(count_values).all() or np.any(count_values < 0):
        raise ValueError("Selected RNA counts must be finite and nonnegative.")
    if not np.allclose(count_values, np.rint(count_values), rtol=0.0, atol=0.0):
        raise ValueError("Selected RNA counts must be integer-valued.")

    obs["donor_timepoint"] = pd.Categorical(
        obs["donor_timepoint"].astype(str),
        categories=selected_groups,
        ordered=True,
    )
    obs["batch"] = pd.Categorical(obs["batch"].astype(str))
    fixture = ad.AnnData(
        X=counts.copy(),
        obs=obs,
        var=var,
    )
    fixture.layers["counts"] = counts
    fixture.obsm["protein_expression"] = proteins
    fixture.uns["protein_names_engineering"] = np.asarray(
        biological_proteins,
        dtype=object,
    )

    counts_by_group = (
        fixture.obs["donor_timepoint"].astype(str).value_counts().sort_index()
    )
    if (
        fixture.n_obs != N_GROUPS * CELLS_PER_GROUP
        or counts_by_group.shape[0] != N_GROUPS
        or not np.all(counts_by_group.to_numpy() == CELLS_PER_GROUP)
    ):
        raise AssertionError("The bounded fixture is not exactly balanced.")

    metadata = {
        "cell_selection": selection_rows,
        "selected_cell_order": fixture.obs_names.astype(str).tolist(),
        "selected_groups": selected_groups,
        "group_counts": {
            str(name): int(value) for name, value in counts_by_group.items()
        },
        "selected_genes": selected_genes,
        "selected_proteins": biological_proteins,
        "selected_cell_order_sha256": _sha256_lines(
            fixture.obs_names.astype(str).tolist()
        ),
        "selected_gene_order_sha256": gene_hash,
        "selected_protein_order_sha256": protein_hash,
    }
    return fixture, metadata


def _sample_blind_checks(model: MrTotalVI, tensors: dict) -> dict:
    module = model.module.eval()
    inputs = module._get_inference_input(tensors)
    x = inputs["x"]
    y = inputs["y"]
    batch_index = inputs.get("batch_index")
    cont_covs = inputs.get("cont_covs")
    cat_covs = inputs.get("cat_covs")
    sample_zero = torch.zeros_like(inputs["sample_index"])
    sample_one = torch.ones_like(inputs["sample_index"])
    with torch.inference_mode():
        qu_zero = module.qu(
            x,
            y,
            sample_zero,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        qu_one = module.qu(
            x,
            y,
            sample_one,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        torch.testing.assert_close(qu_zero.loc, qu_one.loc, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            qu_zero.scale,
            qu_one.scale,
            rtol=0.0,
            atol=0.0,
        )
        if batch_index is None or model.summary_stats.n_batch < 2:
            raise AssertionError(
                "The engineering fixture needs at least two registered batches."
            )
        alternate_batch = (batch_index + 1) % int(model.summary_stats.n_batch)
        qu_other_batch = module.qu(
            x,
            y,
            sample_zero,
            batch_index=alternate_batch,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
    technical_difference = float(
        torch.max(torch.abs(qu_zero.loc - qu_other_batch.loc)).cpu()
    )
    if technical_difference <= 0.0:
        raise AssertionError("Declared batch covariates do not affect sample-blind u.")
    return {
        "sample_loc_max_abs_difference": float(
            torch.max(torch.abs(qu_zero.loc - qu_one.loc)).cpu()
        ),
        "sample_scale_max_abs_difference": float(
            torch.max(torch.abs(qu_zero.scale - qu_one.scale)).cpu()
        ),
        "technical_batch_loc_max_abs_difference": technical_difference,
    }


def _forward_backward_checks(model: MrTotalVI, tensors: dict) -> dict:
    module = model.module.train()
    module.zero_grad(set_to_none=True)
    torch.manual_seed(SEED)
    started = time.perf_counter()
    inference_outputs, _, loss_output = module(
        tensors,
        inference_kwargs={"target_chunk_size": 5},
        loss_kwargs={"kl_weight": 1.0, "pro_recons_weight": 1.0},
    )
    loss_output.loss.backward()
    elapsed = time.perf_counter() - started
    if elapsed > FORWARD_BACKWARD_LIMIT_SECONDS:
        raise AssertionError(
            f"Forward/backward took {elapsed:.3f}s, exceeding "
            f"{FORWARD_BACKWARD_LIMIT_SECONDS:.0f}s."
        )
    if not torch.isfinite(loss_output.loss):
        raise AssertionError("Forward/backward produced a non-finite loss.")

    embedding_gradient = module.qz.embedding.weight.grad
    if embedding_gradient is None:
        raise AssertionError("The registered residual embedding has no gradient.")
    row_norms = embedding_gradient.norm(dim=1)
    if (
        row_norms.shape[0] != N_GROUPS
        or not torch.isfinite(row_norms).all()
        or torch.any(row_norms <= 0)
    ):
        raise AssertionError(
            "Every registered residual embedding row must have a finite, "
            "nonzero gradient."
        )
    conditioning_parameters = (
        module.qu.cond_norm1.gamma_embedding.weight,
        module.qu.cond_norm1.beta_embedding.weight,
        module.qu.cond_norm2.gamma_embedding.weight,
        module.qu.cond_norm2.beta_embedding.weight,
        module.qu.sample_embed.weight,
    )
    conditioning_nonzero = []
    for parameter in conditioning_parameters:
        gradient = parameter.grad
        conditioning_nonzero.append(
            0 if gradient is None else int(torch.count_nonzero(gradient).cpu())
        )
    if any(conditioning_nonzero):
        raise AssertionError(
            "Sample-conditioning parameters received gradients in sample-blind mode."
        )
    core_gradient = module.qu.fc1.weight.grad
    if (
        core_gradient is None
        or not torch.isfinite(core_gradient).all()
        or torch.count_nonzero(core_gradient) == 0
    ):
        raise AssertionError("The sample-blind encoder core lacks a valid gradient.")
    return {
        "seconds": elapsed,
        "loss": float(loss_output.loss.detach().cpu()),
        "residual_embedding_gradient_row_norms": row_norms.detach()
        .cpu()
        .tolist(),
        "sample_conditioning_nonzero_gradient_counts": conditioning_nonzero,
        "core_nonzero_gradient_count": int(
            torch.count_nonzero(core_gradient).cpu()
        ),
        "inference_eps_raw_shape": list(
            inference_outputs["eps_raw_all"].shape
        ),
    }


def _target_chunk_checks(model: MrTotalVI, tensors: dict) -> dict:
    module = model.module.eval()
    inference_inputs = module._get_inference_input(tensors)
    with torch.inference_mode():
        qu = module.qu(
            inference_inputs["x"],
            inference_inputs["y"],
            inference_inputs["sample_index"],
            batch_index=inference_inputs.get("batch_index"),
            cont_covs=inference_inputs.get("cont_covs"),
            cat_covs=inference_inputs.get("cat_covs"),
        )
    base_u = qu.loc[:16].detach()
    snapshots = {}
    for chunk_size in TARGET_CHUNKS:
        module.zero_grad(set_to_none=True)
        u = base_u.clone().requires_grad_(True)
        outputs = module._all_sample_residuals(
            u,
            target_chunk_size=chunk_size,
        )
        sum(output.square().sum() for output in outputs).backward()
        gradient = module.qz.embedding.weight.grad
        if gradient is None:
            raise AssertionError("Target chunking produced no embedding gradient.")
        snapshots[chunk_size] = {
            "outputs": [output.detach().cpu() for output in outputs],
            "u_gradient": u.grad.detach().cpu(),
            "embedding_gradient": gradient.detach().cpu().clone(),
        }

    expected = snapshots[N_GROUPS]
    maximum_value_difference = 0.0
    maximum_gradient_difference = 0.0
    for chunk_size in TARGET_CHUNKS[:-1]:
        actual = snapshots[chunk_size]
        for left, right in zip(
            actual["outputs"],
            expected["outputs"],
            strict=True,
        ):
            difference = float(torch.max(torch.abs(left - right)))
            maximum_value_difference = max(maximum_value_difference, difference)
            torch.testing.assert_close(
                left,
                right,
                rtol=ENGINEERING_CHUNK_RTOL,
                atol=ENGINEERING_CHUNK_ATOL,
            )
        for name in ("u_gradient", "embedding_gradient"):
            difference = float(
                torch.max(torch.abs(actual[name] - expected[name]))
            )
            maximum_gradient_difference = max(
                maximum_gradient_difference,
                difference,
            )
            torch.testing.assert_close(
                actual[name],
                expected[name],
                rtol=ENGINEERING_CHUNK_RTOL,
                atol=ENGINEERING_CHUNK_ATOL,
            )

    _, eps_raw, eps_centered, z_all = expected["outputs"]
    centering_error = float(
        torch.max(torch.abs(eps_centered.mean(dim=1)))
    )
    pairwise_error = float(
        torch.max(
            torch.abs(
                (z_all[:, 0] - z_all[:, -1])
                - (eps_raw[:, 0] - eps_raw[:, -1])
            )
        )
    )
    if centering_error > 1e-6 or pairwise_error > 1e-6:
        raise AssertionError(
            "Centered or pairwise hierarchy identities exceed 1e-6."
        )
    return {
        "target_chunk_sizes": list(TARGET_CHUNKS),
        "comparison_rtol": ENGINEERING_CHUNK_RTOL,
        "comparison_atol": ENGINEERING_CHUNK_ATOL,
        "maximum_value_difference": maximum_value_difference,
        "maximum_gradient_difference": maximum_gradient_difference,
        "centering_max_abs_error": centering_error,
        "pairwise_max_abs_error": pairwise_error,
    }


def _package_sources() -> list[Path]:
    return [
        Path("src/scvi/external/__init__.py"),
        Path("src/scvi/external/mrtotalvi/__init__.py"),
        Path("src/scvi/external/mrtotalvi/_components.py"),
        Path("src/scvi/external/mrtotalvi/_counterfactual.py"),
        Path("src/scvi/external/mrtotalvi/_model.py"),
        Path("src/scvi/external/mrtotalvi/_module.py"),
        Path("src/scvi/external/mrtotalvi/_seed.py"),
        Path("src/scvi/external/mrtotalvi/_stats.py"),
        Path(__file__).resolve().relative_to(Path.cwd().resolve()),
    ]


def _artifact_checksums(run_directory: Path) -> list[str]:
    lines = []
    for path in sorted(
        candidate
        for candidate in run_directory.rglob("*")
        if candidate.is_file() and candidate.name != "checksums.sha256"
    ):
        relative = path.relative_to(run_directory)
        lines.append(f"{_sha256_file(path)}  {relative.as_posix()}")
    return lines


def _run(args: argparse.Namespace, temporary: Path) -> tuple[str, dict]:
    smoke_started = time.perf_counter()
    initial_rss = _max_rss_bytes()
    input_hash = _sha256_file(args.input_h5ad)
    panel_hash = _sha256_file(args.panel_map)
    if input_hash != EXPECTED_INPUT_SHA256:
        raise ValueError(
            f"Input H5AD SHA-256 is {input_hash}, expected "
            f"{EXPECTED_INPUT_SHA256}."
        )
    if panel_hash != EXPECTED_PANEL_SHA256:
        raise ValueError(
            f"Panel-map SHA-256 is {panel_hash}, expected "
            f"{EXPECTED_PANEL_SHA256}."
        )

    fixture, selection = _select_fixture(args.input_h5ad, args.panel_map)
    _write_lines(
        temporary / "selected_cells.txt",
        selection["selected_cell_order"],
    )
    _write_lines(
        temporary / "selected_genes.txt",
        selection["selected_genes"],
    )
    _write_lines(
        temporary / "selected_proteins.txt",
        selection["selected_proteins"],
    )
    _write_json(
        temporary / "cell_selection.json",
        selection["cell_selection"],
    )

    scvi.settings.seed = SEED
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    MrTotalVI.setup_anndata(
        fixture,
        protein_expression_obsm_key="protein_expression",
        protein_names_uns_key="protein_names_engineering",
        sample_key="donor_timepoint",
        batch_key="batch",
        layer="counts",
    )
    model = MrTotalVI(
        fixture,
        sample_key="donor_timepoint",
        n_latent=20,
        hierarchy_mode="centered_v2",
        u_encoder_mode="sample_blind",
        use_map=True,
        z_u_prior=True,
        encode_covariates=True,
    )

    training_started = time.perf_counter()
    model.train(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        train_size=1.0,
        validation_size=None,
        batch_size=64,
        early_stopping=False,
        check_val_every_n_epoch=None,
        reduce_lr_on_plateau=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    training_seconds = time.perf_counter() - training_started

    batch_indices = np.arange(64)
    tensors = next(
        iter(
            model._make_data_loader(
                adata=fixture,
                indices=batch_indices,
                batch_size=64,
                shuffle=False,
            )
        )
    )
    sample_blind = _sample_blind_checks(model, tensors)
    forward_backward = _forward_backward_checks(model, tensors)
    target_chunks = _target_chunk_checks(model, tensors)

    target_samples = [str(value) for value in model.sample_order]
    query_indices = np.arange(QUERY_CELLS)
    reference_indices = np.arange(fixture.n_obs)
    latent_outputs = {}
    for chunk_size in TARGET_CHUNKS:
        latent_outputs[chunk_size] = model.get_counterfactual_latent(
            indices=query_indices,
            target_samples=target_samples,
            reference_indices=reference_indices,
            batch_size=64,
            target_chunk_size=chunk_size,
            random_state=SEED,
        )
    for chunk_size in TARGET_CHUNKS[:-1]:
        _assert_dataset_values_equal(
            latent_outputs[chunk_size],
            latent_outputs[N_GROUPS],
        )
    latent = latent_outputs[N_GROUPS]
    np.savez_compressed(
        temporary / "latent_output.npz",
        **{
            f"coord__{name}": coordinate.to_numpy()
            for name, coordinate in latent.coords.items()
        },
        **{
            f"variable__{name}": variable.to_numpy()
            for name, variable in latent.data_vars.items()
        },
    )

    checkpoint = temporary / "checkpoint"
    model.save(checkpoint, save_anndata=True)
    loaded = MrTotalVI.load(checkpoint, accelerator="cpu")
    checkpoint_reference_latent = latent_outputs[5]
    loaded_latent = loaded.get_counterfactual_latent(
        indices=query_indices,
        target_samples=target_samples,
        reference_indices=reference_indices,
        batch_size=64,
        target_chunk_size=5,
        random_state=SEED,
    )
    _assert_dataset_values_equal(
        loaded_latent,
        checkpoint_reference_latent,
        rtol=0.0,
        atol=0.0,
    )

    query_genes = selection["selected_genes"][:QUERY_GENES]
    query_proteins = selection["selected_proteins"][:QUERY_PROTEINS]
    expression_kwargs = {
        "indices": query_indices,
        "target_samples": target_samples,
        "gene_list": query_genes,
        "protein_list": query_proteins,
        "batch_policy": "observed",
        "panel_policy": "observed",
        "library_policy": "observed",
        "batch_size": 64,
        "target_chunk_size": 5,
        "feature_chunk_size": 25,
        "random_state": SEED,
    }
    expression = loaded.get_counterfactual_expression(**expression_kwargs)
    expression_zarr_path = temporary / "counterfactual_expression.zarr"
    expression_zarr = loaded.get_counterfactual_expression(
        **expression_kwargs,
        zarr_path=expression_zarr_path,
        zarr_chunks={
            "draw": 1,
            "cell_name": 16,
            "target_sample": 5,
            "gene": 25,
            "protein": 10,
        },
    )
    expression_from_zarr = expression_zarr.load()
    _assert_dataset_values_equal(expression_from_zarr, expression)
    expression_zarr.close()

    smoke_seconds = time.perf_counter() - smoke_started
    peak_rss = _max_rss_bytes()
    rss_increase = max(0, peak_rss - initial_rss)
    if smoke_seconds > FULL_SMOKE_LIMIT_SECONDS:
        raise AssertionError(
            f"Full smoke took {smoke_seconds:.3f}s, exceeding "
            f"{FULL_SMOKE_LIMIT_SECONDS:.0f}s."
        )
    if rss_increase >= RSS_INCREASE_LIMIT_BYTES:
        raise AssertionError(
            f"Peak RSS increased by {rss_increase} bytes, exceeding the "
            f"{RSS_INCREASE_LIMIT_BYTES}-byte limit."
        )

    source_hashes = {
        path.as_posix(): _sha256_file(path) for path in _package_sources()
    }
    configuration = {
        "scope": "historical human comparator engineering fixture only",
        "scientific_status": "not canonical, not QC-pass, not biological validation",
        "seed": SEED,
        "input_h5ad": str(args.input_h5ad.resolve()),
        "input_h5ad_sha256": input_hash,
        "panel_map": str(args.panel_map.resolve()),
        "panel_map_sha256": panel_hash,
        "sample_key": "donor_timepoint",
        "timepoints": ["W00", "W22"],
        "n_groups": N_GROUPS,
        "cells_per_group": CELLS_PER_GROUP,
        "n_cells": fixture.n_obs,
        "n_genes": fixture.n_vars,
        "n_proteins": N_PROTEINS,
        "n_latent": 20,
        "batch_size": 64,
        "max_epochs": 1,
        "hierarchy_mode": "centered_v2",
        "u_encoder_mode": "sample_blind",
        "use_map": True,
        "z_u_prior": True,
        "target_chunks": list(TARGET_CHUNKS),
        "query_cells": QUERY_CELLS,
        "query_genes": QUERY_GENES,
        "query_proteins": QUERY_PROTEINS,
        "qc_field_accessed": False,
        "latest_pointer_created": False,
    }
    _write_json(temporary / "configuration.json", configuration)
    _write_json(
        temporary / "environment.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "git_head": _git_output("rev-parse", "HEAD"),
            "git_status_source_scope": _git_output(
                "status",
                "--short",
                "--",
                "src/scvi/external",
                "tests/external/mrtotalvi",
            ).splitlines(),
            "source_sha256": source_hashes,
            "versions": {
                distribution: importlib.metadata.version(distribution)
                for distribution in (
                    "anndata",
                    "dask",
                    "numpy",
                    "pandas",
                    "scipy",
                    "torch",
                    "xarray",
                    "zarr",
                )
            },
            "torch_num_threads": torch.get_num_threads(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
        },
    )
    metrics = {
        "status": "pass",
        "interpretation": (
            "Bounded package engineering evidence only; no biological, QC, "
            "causal, comparative, or promotion claim."
        ),
        "training_seconds": training_seconds,
        "forward_backward": forward_backward,
        "sample_blind": sample_blind,
        "target_chunks": target_chunks,
        "latent_dataset_sha256": _dataset_sha256(
            checkpoint_reference_latent
        ),
        "loaded_latent_dataset_sha256": _dataset_sha256(loaded_latent),
        "expression_dataset_sha256": _dataset_sha256(expression),
        "zarr_expression_dataset_sha256": _dataset_sha256(expression_from_zarr),
        "save_load_output_identity": True,
        "zarr_output_identity": True,
        "registered_residual_embedding_rows_exercised": N_GROUPS,
        "full_smoke_seconds": smoke_seconds,
        "initial_max_rss_bytes": initial_rss,
        "peak_max_rss_bytes": peak_rss,
        "peak_rss_increase_bytes": rss_increase,
        "limits": {
            "forward_backward_seconds": FORWARD_BACKWARD_LIMIT_SECONDS,
            "full_smoke_seconds": FULL_SMOKE_LIMIT_SECONDS,
            "peak_rss_increase_bytes": RSS_INCREASE_LIMIT_BYTES,
        },
    }
    if metrics["latent_dataset_sha256"] != metrics[
        "loaded_latent_dataset_sha256"
    ]:
        raise AssertionError("Save/load changed the latent dataset digest.")
    if metrics["expression_dataset_sha256"] != metrics[
        "zarr_expression_dataset_sha256"
    ]:
        raise AssertionError("Zarr storage changed the expression dataset digest.")
    _write_json(temporary / "validation_metrics.json", metrics)
    return input_hash[:8], metrics


def main() -> None:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".mrtotalvi-v2-engineering-tmp-",
            dir=args.output_root,
        )
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        input_tag, metrics = _run(args, temporary)
        run_name = f"{timestamp}-seed0-{input_tag}"
        destination = args.output_root / run_name
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite engineering run {destination}."
            )
        checksum_lines = _artifact_checksums(temporary)
        _write_lines(temporary / "checksums.sha256", checksum_lines)
        os.replace(temporary, destination)
        print(
            json.dumps(
                {
                    "run_directory": str(destination.resolve()),
                    "status": metrics["status"],
                    "full_smoke_seconds": metrics["full_smoke_seconds"],
                    "peak_rss_increase_bytes": metrics[
                        "peak_rss_increase_bytes"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except Exception as error:
        _write_json(
            temporary / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        failed = args.output_root / f"{timestamp}-failed"
        if failed.exists():
            failed = args.output_root / (
                f"{timestamp}-failed-{hashlib.sha256(str(error).encode()).hexdigest()[:8]}"
            )
        os.replace(temporary, failed)
        raise


if __name__ == "__main__":
    main()
