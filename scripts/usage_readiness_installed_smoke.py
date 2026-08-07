#!/usr/bin/env python3
"""Installed-wheel smoke for the exact CytoANVI and MrTotalVI core workflows."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_digest(hex_digest: str) -> str:
    """Encode a hexadecimal SHA-256 digest in wheel RECORD form."""
    encoded = base64.urlsafe_b64encode(bytes.fromhex(hex_digest)).decode().rstrip("=")
    return f"sha256={encoded}"


def validate_console_wrapper(path: Path, target: str) -> None:
    """Require the exact standard pip launcher semantics for a sealed entry point."""
    module, separator, function = target.partition(":")
    if separator != ":" or not module or not function or ":" in function:
        raise RuntimeError(f"Unsupported console-script target: {target!r}")
    lines = path.read_text(encoding="utf-8").splitlines()
    expected_shebang = f"#!{sys.executable}"
    if not lines or lines[0] != expected_shebang:
        raise RuntimeError(
            f"Console-script shebang differs from the fresh interpreter: {path}"
        )
    observed = ast.dump(ast.parse("\n".join(lines[1:])), include_attributes=False)
    expected_source = f"""\
import re
import sys
from {module} import {function}
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])
    sys.exit({function}())
"""
    expected = ast.dump(ast.parse(expected_source), include_attributes=False)
    if observed != expected:
        raise RuntimeError(f"Console-script wrapper differs from the sealed target: {path}")


def installed_distribution_inventory(
    distribution: importlib.metadata.Distribution, console_scripts: dict[str, str]
) -> dict:
    """Hash the final owned file set and verify that installed RECORD accounts for it."""
    site_root = Path(distribution.locate_file("")).resolve()
    dist_info = Path(distribution._path).resolve()
    if site_root not in dist_info.parents:
        raise RuntimeError(f"Distribution metadata is outside site-packages: {dist_info}")
    dist_info_relative = dist_info.relative_to(site_root).as_posix()
    record_relative = f"{dist_info_relative}/RECORD"
    record_path = site_root / record_relative
    record_lines = record_path.read_text(encoding="utf-8").splitlines()
    rows = list(csv.reader(record_lines))
    if any(len(row) != 3 for row in rows):
        raise RuntimeError("Installed RECORD does not contain exactly three columns per row")
    record_paths = [row[0] for row in rows]
    if len(record_paths) != len(set(record_paths)):
        raise RuntimeError("Installed RECORD contains duplicate paths")

    scripts_dir = Path(sys.executable).parent.resolve()
    if scripts_dir.name not in {"bin", "Scripts"}:
        raise RuntimeError(f"Unexpected fresh-environment script directory: {scripts_dir}")
    if any(not name or Path(name).name != name for name in console_scripts):
        raise RuntimeError("Expected console-script names are not unique basenames")
    expected_console_paths = {}
    for name in sorted(console_scripts):
        script_path = scripts_dir / name
        if not script_path.is_file() or script_path.is_symlink():
            raise RuntimeError(
                f"Expected console script is not a regular non-symlink file: {script_path}"
            )
        validate_console_wrapper(script_path, console_scripts[name])
        relative = Path(os.path.relpath(script_path, start=site_root)).as_posix()
        expected_console_paths[relative] = (name, script_path.resolve())
    missing_scripts = sorted(set(expected_console_paths) - set(record_paths))
    if missing_scripts:
        raise RuntimeError(f"Installed RECORD omits expected console scripts: {missing_scripts}")

    actual_paths: set[str] = set()
    for relative in record_paths:
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise RuntimeError(f"Installed RECORD path escapes site-packages: {relative}")
        if ".." in relative_path.parts:
            if relative not in expected_console_paths:
                raise RuntimeError(f"Installed RECORD path escapes site-packages: {relative}")
            located = expected_console_paths[relative][1]
        else:
            located = (site_root / relative_path).resolve()
            if site_root != located and site_root not in located.parents:
                raise RuntimeError(f"Installed RECORD path escapes site-packages: {relative}")
        if not located.is_file() or located.is_symlink():
            raise RuntimeError(f"Installed RECORD path is not a regular file: {relative}")
        actual_paths.add(relative_path.as_posix())

    for owned_root in (site_root / "cytoanvi", site_root / "scvi", dist_info):
        if not owned_root.is_dir():
            raise RuntimeError(f"Expected installed path is absent: {owned_root}")
        for path in owned_root.rglob("*"):
            if path.is_file():
                if path.is_symlink():
                    raise RuntimeError(f"Installed distribution contains a symlink: {path}")
                actual_paths.add(path.relative_to(site_root).as_posix())
    if set(record_paths) != actual_paths:
        missing = sorted(actual_paths - set(record_paths))
        extra = sorted(set(record_paths) - actual_paths)
        raise RuntimeError(
            "Installed RECORD does not account for the final file set: "
            f"missing={missing}, extra={extra}"
        )

    inventory = []
    by_path = {}
    for relative in sorted(actual_paths):
        path = site_root / relative
        entry = {
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        inventory.append(entry)
        by_path[relative] = entry
    for relative, encoded_hash, encoded_size in rows:
        entry = by_path[relative]
        if relative == record_relative:
            if encoded_hash or encoded_size:
                raise RuntimeError("Installed RECORD self-entry must omit hash and size")
            continue
        if not encoded_hash and not encoded_size:
            if "/__pycache__/" not in relative or not relative.endswith(".pyc"):
                raise RuntimeError(
                    f"Installed RECORD omits hash/size for payload file: {relative}"
                )
            continue
        if not encoded_hash or not encoded_size:
            raise RuntimeError(f"Installed RECORD partially omits hash/size for {relative}")
        if encoded_hash != record_digest(entry["sha256"]):
            raise RuntimeError(f"Installed RECORD SHA-256 mismatch for {relative}")
        if encoded_size != str(entry["size_bytes"]):
            raise RuntimeError(f"Installed RECORD size mismatch for {relative}")

    generated = {}
    for name in ("INSTALLER", "REQUESTED", "direct_url.json"):
        relative = f"{dist_info_relative}/{name}"
        if relative in by_path:
            path = site_root / relative
            if name == "direct_url.json":
                generated[name] = json.loads(path.read_text(encoding="utf-8"))
            else:
                generated[name] = path.read_text(encoding="utf-8")
    generated["console_scripts"] = [
        {
            "name": name,
            "target": console_scripts[name],
            "path": relative,
            "sha256": by_path[relative]["sha256"],
            "size_bytes": by_path[relative]["size_bytes"],
        }
        for relative, (name, _path) in sorted(expected_console_paths.items())
    ]
    return {
        "site_root": str(site_root),
        "dist_info": dist_info_relative,
        "files": inventory,
        "record": {"path": record_relative, "lines": record_lines},
        "pip_generated": generated,
    }


def distribution_identity(console_scripts: dict[str, str]) -> dict:
    """Prove namespace and file ownership before importing model code."""
    distributions = importlib.metadata.packages_distributions()
    cyto_dist = importlib.metadata.distribution("cytoanvi")
    try:
        importlib.metadata.distribution("scvi-tools")
    except importlib.metadata.PackageNotFoundError:
        scvi_tools_absent = True
    else:
        scvi_tools_absent = False
    if not scvi_tools_absent:
        raise RuntimeError("The isolated environment contains forbidden distribution scvi-tools")
    for namespace in ("cytoanvi", "scvi"):
        owners = sorted(distributions.get(namespace, []))
        if owners != ["cytoanvi"]:
            raise RuntimeError(
                f"Namespace {namespace!r} owners are {owners}, expected ['cytoanvi']"
            )
    installed_inventory = installed_distribution_inventory(cyto_dist, console_scripts)
    metadata_files = sorted(str(path) for path in (cyto_dist.files or []))
    if metadata_files != sorted(entry["path"] for entry in installed_inventory["files"]):
        raise RuntimeError("importlib metadata file set differs from installed RECORD inventory")
    return {
        "distribution_version": cyto_dist.version,
        "distribution_path": str(cyto_dist._path),
        "scvi_tools_absent": scvi_tools_absent,
        "namespace_owners": {
            name: sorted(distributions.get(name, [])) for name in ("cytoanvi", "scvi")
        },
        "installed_inventory": installed_inventory,
    }


def cytoanvi_smoke(workdir: Path) -> dict:
    """Train/predict/latent/save-load and same-panel query using installed code."""
    import numpy as np

    import cytoanvi
    import scvi
    from cytoanvi import CytoANVI
    from scvi.external import cytovi as cytovi_pp

    scvi.settings.seed = 0
    adata = scvi.data.synthetic_iid(
        batch_size=64,
        n_genes=16,
        n_proteins=0,
        n_regions=0,
        n_batches=2,
        n_labels=4,
        rna_dist="normal",
    )
    adata.obs["sample"] = np.resize(np.array(["s0", "s1"]), adata.n_obs)
    adata.layers["raw"] = adata.X.copy()
    cytovi_pp.transform_arcsinh(adata)
    cytovi_pp.scale(adata)
    CytoANVI.setup_anndata(
        adata,
        layer="scaled",
        batch_key="batch",
        labels_key="labels",
        unlabeled_category="label_0",
        sample_key="sample",
    )
    model = CytoANVI(adata, n_latent=4)
    model.train(max_epochs=1, accelerator="cpu", enable_progress_bar=False)
    first = model.predict(soft=True).to_numpy()
    second = model.predict(soft=True).to_numpy()
    np.testing.assert_array_equal(first, second)
    latent = model.get_latent_representation()
    if latent.shape != (adata.n_obs, 4) or not np.isfinite(latent).all():
        raise AssertionError("CytoANVI latent schema is invalid")

    save_path = workdir / "cytoanvi-model"
    model.save(save_path, overwrite=False, save_anndata=True)
    loaded = CytoANVI.load(save_path, accelerator="cpu", device="cpu")
    np.testing.assert_array_equal(first, loaded.predict(soft=True).to_numpy())

    query = adata[:16].copy()
    query.obs["labels"] = "label_0"
    query_model = CytoANVI.load_query_data(query, loaded)
    query_model.train(max_epochs=1, accelerator="cpu", enable_progress_bar=False)
    query_predictions = np.asarray(query_model.predict())
    if query_predictions.shape != (query.n_obs,):
        raise AssertionError("CytoANVI same-panel query prediction schema is invalid")

    return {
        "cytoanvi_version": cytoanvi.__version__,
        "cytoanvi_file": str(Path(cytoanvi.__file__).resolve()),
        "scvi_file": str(Path(scvi.__file__).resolve()),
        "n_cells": adata.n_obs,
        "latent_shape": list(latent.shape),
        "query_cells": query.n_obs,
        "deterministic_predict": True,
        "save_load": True,
    }


def mrtotalvi_smoke(workdir: Path) -> dict:
    """Train/export u,z/summary/save-load using finite integer raw counts."""
    import numpy as np

    import scvi
    from scvi.external import MrTotalVI

    scvi.settings.seed = 1
    adata = scvi.data.synthetic_iid(batch_size=64, n_genes=24, n_proteins=8, n_batches=2)
    adata.obs["sample"] = np.resize(np.array(["d0", "d1", "d2", "d3"]), adata.n_obs)
    MrTotalVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    model = MrTotalVI(adata, sample_key="sample", n_latent=4, n_latent_u=3)
    model.train(max_epochs=1, accelerator="cpu", enable_progress_bar=False)
    z = model.get_latent_representation(give_z=True, give_mean=True)
    u = model.get_latent_representation(give_z=False, give_mean=True)
    if z.shape != (adata.n_obs, 4) or u.shape != (adata.n_obs, 3):
        raise AssertionError(f"MrTotalVI latent schema mismatch: z={z.shape}, u={u.shape}")
    if not np.isfinite(z).all() or not np.isfinite(u).all():
        raise AssertionError("MrTotalVI latent contains non-finite values")
    summary = str(model)
    for token in ("hierarchy_mode", "u_encoder_mode", "u_prior", "u_prior_supervision"):
        if token not in summary:
            raise AssertionError(f"MrTotalVI summary omits {token}")

    save_path = workdir / "mrtotalvi-model"
    model.save(save_path, overwrite=False)
    loaded = MrTotalVI.load(save_path, adata=adata)
    if loaded.get_latent_representation(give_z=False, give_mean=True).shape != u.shape:
        raise AssertionError("MrTotalVI save/load changed u schema")
    return {
        "n_cells": adata.n_obs,
        "z_shape": list(z.shape),
        "u_shape": list(u.shape),
        "summary_metadata": True,
        "save_load": True,
    }


def run_tutorial_smoke(tutorial_path: Path) -> dict:
    """Execute both reusable treeArches paths from the copied tutorial module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("cytoanvi_treearches_synthetic", tutorial_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load tutorial module {tutorial_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    direct = module.run_direct_same_panel()
    one_shot = module.run_one_shot_learn_update_predict()
    return {
        "direct_query_cells": int(direct["query"].n_obs),
        "one_shot_query_cells": int(one_shot["query"].n_obs),
        "updated_tree_revision": int(one_shot["updated_tree"].revision),
    }


def main() -> int:
    """Run the installed-distribution smoke checks and write their identity receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tutorial", required=True, type=Path)
    parser.add_argument("--console-script", action="append", default=[])
    args = parser.parse_args()
    console_scripts = {}
    for spec in args.console_script:
        name, separator, target = spec.partition("=")
        if separator != "=" or not name or not target or name in console_scripts:
            raise ValueError(f"Invalid or duplicate --console-script specification: {spec!r}")
        console_scripts[name] = target
    with tempfile.TemporaryDirectory(prefix="cytoanvi-installed-smoke-") as temporary:
        workdir = Path(temporary)
        result = {
            "identity": distribution_identity(console_scripts),
            "cytoanvi": cytoanvi_smoke(workdir),
            "mrtotalvi": mrtotalvi_smoke(workdir),
            "tree_arches": run_tutorial_smoke(args.tutorial),
        }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
