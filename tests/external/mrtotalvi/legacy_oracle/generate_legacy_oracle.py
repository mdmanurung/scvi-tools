"""Generate immutable pre-v2 MrTotalVI and MrMultiVI regression oracles.

This script must be executed with ``PYTHONPATH`` pointing at a clean archive of
commit ``d8c8e997``. It intentionally refuses to overwrite an existing output
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

import scvi
from scvi import REGISTRY_KEYS
from scvi.external import MrMultiVI, MrTotalVI

SOURCE_COMMIT = "d8c8e997a67997a53f55923eb3ab14e6cf06f94c"
MODEL_SEED = 7301
FORWARD_SEED = 9817
BATCH_SIZE = 7
KL_WEIGHT = 0.73
PROTEIN_RECONSTRUCTION_WEIGHT = 0.61


def _reset_seed() -> None:
    scvi.settings.seed = MODEL_SEED
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)


def _array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _capture_model(name: str, model, output_dir: Path) -> None:
    model_dir = output_dir / name
    model_dir.mkdir()
    checkpoint_dir = model_dir / "checkpoint"
    model.save(checkpoint_dir, save_anndata=True)
    model = type(model).load(checkpoint_dir, accelerator="cpu")

    module = model.module
    module.eval()
    loader = model._make_data_loader(
        adata=model.adata,
        indices=np.arange(BATCH_SIZE),
        batch_size=BATCH_SIZE,
    )
    tensors = next(iter(loader))

    module.zero_grad(set_to_none=True)
    torch.manual_seed(FORWARD_SEED)
    loss_kwargs = {"kl_weight": KL_WEIGHT}
    if isinstance(model, MrTotalVI):
        loss_kwargs["pro_recons_weight"] = PROTEIN_RECONSTRUCTION_WEIGHT
    inference_outputs, _, loss_output = module(tensors, loss_kwargs=loss_kwargs)

    arrays: dict[str, np.ndarray] = {
        "qu.loc": _array(inference_outputs["qu"].loc),
        "qu.scale": _array(inference_outputs["qu"].scale),
        "u": _array(inference_outputs["u"]),
        "z_base": _array(inference_outputs["z_base"]),
        "eps_raw_legacy": _array(inference_outputs["eps"]),
        "z": _array(inference_outputs["z"]),
        "loss": np.asarray(loss_output.loss.detach().cpu(), dtype=np.float32),
    }
    for component, value in loss_output.reconstruction_loss.items():
        arrays[f"reconstruction_loss.{component}"] = _array(value)
    for component, value in loss_output.kl_local.items():
        arrays[f"kl_local.{component}"] = _array(value)
    if REGISTRY_KEYS.INDICES_KEY in tensors:
        arrays["cell_indices"] = _array(tensors[REGISTRY_KEYS.INDICES_KEY]).astype(np.int64)

    loss_output.loss.backward()
    gradient_parts: list[np.ndarray] = []
    gradient_manifest: dict[str, dict[str, object]] = {}
    for parameter_name, parameter in module.named_parameters():
        gradient = (
            torch.zeros_like(parameter)
            if parameter.grad is None
            else parameter.grad.detach()
        )
        gradient_array = _array(gradient)
        arrays[f"gradient.{parameter_name}"] = gradient_array
        gradient_parts.append(gradient_array.reshape(-1))
        gradient_manifest[parameter_name] = {
            "present": parameter.grad is not None,
            "shape": list(parameter.shape),
            "selected": parameter_name
            in {
                "qu.fc1.weight",
                "qu.output_nn.loc.weight",
                "qz.embedding.weight",
                "qz.attention_block.query_proj.weight",
            },
        }
    arrays["gradient.full"] = np.concatenate(gradient_parts)
    np.savez_compressed(model_dir / "oracle_arrays.npz", **arrays)

    state_manifest = {
        key: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
        for key, value in module.state_dict().items()
    }
    _write_json(model_dir / "state_manifest.json", state_manifest)
    _write_json(model_dir / "gradient_manifest.json", gradient_manifest)
    _write_json(
        model_dir / "run_manifest.json",
        {
            "batch_size": BATCH_SIZE,
            "forward_seed": FORWARD_SEED,
            "kl_weight": KL_WEIGHT,
            "model_seed": MODEL_SEED,
            "model_type": type(model).__name__,
            "oracle_stage": "post_checkpoint_roundtrip",
            "protein_reconstruction_weight": (
                PROTEIN_RECONSTRUCTION_WEIGHT if isinstance(model, MrTotalVI) else None
            ),
            "sample_order": list(map(str, model.sample_order)),
            "source_commit": SOURCE_COMMIT,
            "tolerances": {"atol": 1e-7, "rtol": 1e-6},
        },
    )


def _make_mrtotalvi(output_dir: Path) -> None:
    _reset_seed()
    adata = scvi.data.synthetic_iid(
        batch_size=8,
        n_genes=6,
        n_proteins=4,
        n_regions=5,
        n_batches=2,
        n_labels=3,
    )
    adata.obs["sample"] = np.asarray(
        [f"sample_{index % 3}" for index in range(adata.n_obs)]
    )
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(
        adata,
        sample_key="sample",
        n_latent=3,
        n_latent_sample=4,
        n_hidden=16,
        n_layers_encoder=1,
        n_layers_decoder=1,
        u_prior_mixture_k=3,
        qu_kwargs={"n_hidden": 16, "n_layers": 1},
        qz_kwargs={"n_channels": 2, "n_heads": 1, "n_hidden": 8, "n_layers": 1},
    )
    _capture_model("mrtotalvi", model, output_dir)


def _make_mrmultivi(output_dir: Path) -> None:
    _reset_seed()
    mdata = scvi.data.synthetic_iid(
        batch_size=8,
        n_genes=6,
        n_proteins=4,
        n_regions=5,
        n_batches=2,
        n_labels=3,
        return_mudata=True,
    )
    mdata.obs["donor"] = np.asarray(
        [f"sample_{index % 3}" for index in range(mdata.n_obs)]
    )
    MrMultiVI.setup_mudata(
        mdata,
        sample_key="donor",
        batch_key="batch",
        modalities={
            "rna_layer": "rna",
            "atac_layer": "accessibility",
            "protein_layer": "protein_expression",
        },
    )
    model = MrMultiVI(
        mdata,
        sample_key="donor",
        n_latent=3,
        n_latent_sample=4,
        n_hidden=16,
        n_layers_encoder=1,
        n_layers_decoder=1,
        u_prior_mixture_k=3,
        qu_kwargs={"n_hidden": 16, "n_layers": 1},
        qz_kwargs={"n_channels": 2, "n_heads": 1, "n_hidden": 8, "n_layers": 1},
    )
    _capture_model("mrmultivi", model, output_dir)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    args = parser.parse_args()

    if args.source_commit != SOURCE_COMMIT:
        raise ValueError(f"Expected source commit {SOURCE_COMMIT}, got {args.source_commit}.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable oracle directory: {args.output}")

    args.output.mkdir(parents=True)
    _make_mrtotalvi(args.output)
    _make_mrmultivi(args.output)
    _write_json(
        args.output / "environment_manifest.json",
        {
            "anndata": importlib.metadata.version("anndata"),
            "executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": sys.version,
            "scvi_tools": scvi.__version__,
            "source_commit": SOURCE_COMMIT,
            "source_root": os.environ.get("MRTOTALVI_ORACLE_SOURCE_ROOT"),
            "torch": torch.__version__,
        },
    )

    checksum_lines = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_lines.append(f"{_sha256(path)}  {path.relative_to(args.output)}")
    (args.output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
