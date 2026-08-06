"""Run and atomically seal the preregistered RDX-03 convergence diagnosis.

The default request is the exact 48-fit grid. Any explicit subset is sealed as
non-authoritative probe evidence and cannot complete RDX-03.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .comparator import (
    MRTOTALVI_FACTUAL_Z_POSTERIOR_DDOF,
    MRTOTALVI_FACTUAL_Z_POSTERIOR_DRAWS,
    _checkpoint_artifact_state_digest,
    validate_frozen_optimization_identity,
)
from .convergence import (
    LATENT_INTEGRITY_ASSESSMENT_SCHEMA,
    aggregate_diagnosis_grid,
    aggregate_diagnosis_grid_v2,
    assess_convergence,
    assess_latent_collapse,
    assess_latent_integrity_v2,
)
from .convergence_runner import (
    CANONICAL_HUMAN_CONTRACT_SHA256,
    CANONICAL_HUMAN_RUN_ID,
    CANONICAL_HUMAN_SHA256,
    CANONICAL_HUMAN_SPLIT_SHA256,
    prepare_diagnosis_fixture,
)
from .governance import _rename_no_replace
from .latent_integrity import latent_integrity_policy_v2
from .manifest import (
    ArtifactRecord,
    RunManifest,
    make_run_id,
    sha256_file,
    verify_run_manifest,
)
from .metric_schema import (
    load_metric_dictionary,
    validate_metric_dictionary,
    validate_metric_payload,
)
from .redesign_contract import (
    redesign_run_contract,
    redesign_run_contract_v2,
)
from .run_pilot import (
    _environment_manifest,
    _git,
    _parse_csv,
    _parse_seeds,
    _payload_digest,
    _strict_jsonable,
    _write_json,
)
from .versioning import (
    prospective_redesign_contract_adapter,
    resolve_redesign_contract_adapter,
    validate_version_binding,
    version_binding_fields,
)

if TYPE_CHECKING:
    from argparse import Namespace

    from .convergence_runner import PreparedDiagnosisFixture
    from .versioning import RedesignContractAdapter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = Path(
    ".scratch/mrtotalvi-v2-redesign/convergence-runs"
)
SCIENTIFIC_SCOPE = (
    "RDX-03 convergence diagnosis under the frozen 400-epoch ceiling; "
    "no candidate selection, Milo gate, or factual human DA"
)
LEGACY_EXECUTION_SCHEMA = "mrtotalvi-convergence-execution-v2"
LEGACY_FIT_SCHEMA = "mrtotalvi-convergence-fit-v2"
LEGACY_PARTIAL_AGGREGATE_SCHEMA = "mrtotalvi-convergence-partial-v2"
LEGACY_FIT_WORKER_SCHEMA = "mrtotalvi-convergence-fit-worker-v2"
LEGACY_CODE_MANIFEST_SCHEMA = "mrtotalvi-convergence-code-manifest-v3"
EXECUTION_SCHEMA = "mrtotalvi-convergence-execution-v3"
FIT_SCHEMA = "mrtotalvi-convergence-fit-v3"
PARTIAL_AGGREGATE_SCHEMA = "mrtotalvi-convergence-partial-v3"
CODE_MANIFEST_SCHEMA = "mrtotalvi-convergence-code-manifest-v4"
SOURCE_SNAPSHOT_ROOT = "code-snapshot"
FIT_WORKER_MODULE = "benchmarks.mrtotalvi.run_convergence_fit"
FIT_WORKER_SCHEMA = "mrtotalvi-convergence-fit-worker-v3"
FIT_MEMORY_SCOPE = "fresh_process_per_fit"


def _diagnostic_estimator_configuration() -> dict[str, object]:
    """Return the exact prospective factual-z scale estimators."""
    return {
        "schema_version": "mrtotalvi-rdx03-diagnostic-estimators-v1",
        "factual_z_posterior_scale": {
            "stock_scvi_totalvi": {
                "estimator": "analytic_qz.scale",
                "aggregation": "mean_over_cells_and_coordinates",
            },
            "mrtotalvi": {
                "estimator": "seeded_monte_carlo_sample_standard_deviation",
                "draws": MRTOTALVI_FACTUAL_Z_POSTERIOR_DRAWS,
                "ddof": MRTOTALVI_FACTUAL_Z_POSTERIOR_DDOF,
                "rng_stream": "evaluation_seed",
                "aggregation": "mean_over_cells_and_coordinates",
            },
        },
    }


def _validate_diagnostic_estimator_configuration(
    payload: object,
) -> dict[str, object]:
    """Reject any prospective posterior-scale estimator drift."""
    expected = _diagnostic_estimator_configuration()
    if payload != expected:
        raise ValueError("Prospective diagnostic estimator configuration drifted.")
    return expected


def _resolve_execution_contract_adapter(
    configuration: Mapping[str, object],
    *,
    run_contract_payload: Mapping[str, object] | None = None,
    latent_integrity_policy_payload: Mapping[str, object] | None = None,
) -> RedesignContractAdapter:
    """Resolve a sealed execution from stored version metadata only."""
    return resolve_redesign_contract_adapter(
        configuration,
        run_contract_payload=run_contract_payload,
        latent_integrity_policy_payload=latent_integrity_policy_payload,
    )


def _prospective_version_context() -> tuple[
    RedesignContractAdapter,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Build and cross-check every complete prospective identity payload."""
    adapter = prospective_redesign_contract_adapter()
    contract = redesign_run_contract_v2().to_dict()
    policy = latent_integrity_policy_v2()
    dictionary = load_metric_dictionary(contract_adapter=adapter)
    validate_metric_dictionary(
        dictionary,
        contract_adapter=adapter,
    )
    resolved = _resolve_execution_contract_adapter(
        {
            "schema_version": EXECUTION_SCHEMA,
            **version_binding_fields(adapter),
        },
        run_contract_payload=contract,
        latent_integrity_policy_payload=policy,
    )
    if resolved != adapter:
        raise ValueError("Prospective version context failed exact replay.")
    return adapter, contract, policy, dictionary


def _convergence_control_payload(
    contract_adapter: RedesignContractAdapter,
) -> dict[str, object]:
    """Return selected convergence control without consulting live defaults."""
    frozen = contract_adapter.run_contract_section("convergence")
    return {
        "check_every_epochs": frozen["check_every_epochs"],
        "minimum_epochs": frozen["minimum_epochs"],
        "maximum_epochs": frozen["maximum_epochs"],
        "patience_checks": frozen["patience_checks"],
        "restore_best_checkpoint": frozen["restore_best_checkpoint"],
        "candidate_specific_retuning": frozen[
            "candidate_specific_retuning"
        ],
        "monitor": "elbo_validation",
        "mode": "min",
        "min_delta": 0.0,
    }


@dataclass(frozen=True)
class DiagnosisExecutionRequest:
    """One ordered exact or explicitly non-authoritative diagnosis grid."""

    fixtures: tuple[str, ...]
    rows: tuple[str, ...]
    seeds: tuple[int, ...]
    authoritative_full_grid: bool

    @property
    def n_fits(self) -> int:
        """Return the exact number of requested fits."""
        return len(self.fixtures) * len(self.rows) * len(self.seeds)


def _validate_ordered_subset(
    values: tuple,
    *,
    declared: tuple,
    name: str,
) -> tuple:
    if (
        not isinstance(values, tuple)
        or not values
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{name} must be a non-empty unique tuple.")
    unknown = [value for value in values if value not in declared]
    if unknown:
        raise ValueError(f"Unknown {name}: {unknown}.")
    positions = [declared.index(value) for value in values]
    if positions != sorted(positions):
        raise ValueError(f"{name} must preserve the frozen declared order.")
    return values


def validate_diagnosis_request(
    *,
    fixtures: tuple[str, ...],
    rows: tuple[str, ...],
    seeds: tuple[int, ...],
    contract_adapter: RedesignContractAdapter | None = None,
) -> DiagnosisExecutionRequest:
    """Validate an exact grid or label a strict ordered subset as a probe."""
    if contract_adapter is None:
        diagnosis = redesign_run_contract().diagnosis
        declared_fixtures = diagnosis.fixtures
        declared_rows = diagnosis.rows
        declared_seeds = diagnosis.training_seeds
    else:
        diagnosis = contract_adapter.run_contract_section("diagnosis")
        declared_fixtures = tuple(diagnosis["fixtures"])
        declared_rows = tuple(diagnosis["rows"])
        declared_seeds = tuple(diagnosis["training_seeds"])
    checked_fixtures = _validate_ordered_subset(
        fixtures,
        declared=declared_fixtures,
        name="fixtures",
    )
    checked_rows = _validate_ordered_subset(
        rows,
        declared=declared_rows,
        name="rows",
    )
    checked_seeds = _validate_ordered_subset(
        seeds,
        declared=declared_seeds,
        name="seeds",
    )
    authoritative = (
        checked_fixtures == declared_fixtures
        and checked_rows == declared_rows
        and checked_seeds == declared_seeds
    )
    return DiagnosisExecutionRequest(
        fixtures=checked_fixtures,
        rows=checked_rows,
        seeds=checked_seeds,
        authoritative_full_grid=authoritative,
    )


def _parse_args() -> argparse.Namespace:
    contract = redesign_run_contract_v2().diagnosis
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--fixtures",
        type=_parse_csv,
        default=contract.fixtures,
    )
    parser.add_argument(
        "--rows",
        type=_parse_csv,
        default=contract.rows,
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=contract.training_seeds,
    )
    return parser.parse_args()


def _fixture_manifest(
    fixture: PreparedDiagnosisFixture,
) -> dict[str, object]:
    payload = {
        "fixture_id": fixture.fixture_id,
        "data_digest": fixture.data_digest,
        "source_data_digest": fixture.source_data_digest,
        "state_annotation_digest": fixture.state_annotation_digest,
        "split_digest": fixture.split_digest,
        "n_cells": fixture.adata.n_obs,
        "n_genes": fixture.adata.n_vars,
        "n_proteins": fixture.adata.obsm[
            fixture.protein_obsm_key
        ].shape[1],
        "n_train_cells": len(fixture.train_indices),
        "n_validation_cells": len(fixture.validation_indices),
        "state_annotation_id": fixture.state_annotation_id,
        "technical_batch_key": fixture.technical_batch_key,
        "sample_key": fixture.sample_key,
        "evaluation_seed": fixture.evaluation_seed,
        "latent_dimension": fixture.n_latent,
        "batch_size": fixture.batch_size,
        "learning_rate": fixture.learning_rate,
    }
    if fixture.fixture_id == "canonical_human_if_available":
        payload["human_lineage"] = {
            "run_id": CANONICAL_HUMAN_RUN_ID,
            "h5ad_sha256": CANONICAL_HUMAN_SHA256,
            "contract_sha256": CANONICAL_HUMAN_CONTRACT_SHA256,
            "split_assignment_sha256": CANONICAL_HUMAN_SPLIT_SHA256,
        }
    return payload


def _artifact_records(run_dir: Path) -> tuple[ArtifactRecord, ...]:
    return tuple(
        ArtifactRecord(
            path=path.relative_to(run_dir).as_posix(),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
        )
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "run-manifest.json"
    )


def _convergence_code_manifest(
    *,
    contract_adapter: RedesignContractAdapter,
) -> dict[str, object]:
    """Describe an exact reconstructible snapshot of scientific Python code."""
    paths = sorted(
        {
            REPO_ROOT / "benchmarks" / "__init__.py",
            *(
                path
                for path in REPO_ROOT.glob("benchmarks/mrtotalvi/*")
                if path.is_file() and path.suffix in {".json", ".py"}
            ),
            *REPO_ROOT.glob("src/scvi/**/*.py"),
        }
    )
    files = {
        path.relative_to(REPO_ROOT).as_posix(): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "snapshot_path": (
                Path(SOURCE_SNAPSHOT_ROOT)
                / path.relative_to(REPO_ROOT)
            ).as_posix(),
        }
        for path in paths
    }
    payload = {
        "schema_version": CODE_MANIFEST_SCHEMA,
        **version_binding_fields(contract_adapter),
        "snapshot_strategy": "exact_runtime_source_snapshot",
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status_short": _git("status", "--short").splitlines(),
        "runtime_imports": _runtime_import_identity(),
        "files": files,
    }
    payload["code_digest"] = _payload_digest(payload)
    return payload


def _runtime_namespace_roots() -> dict[str, Path]:
    return {
        "benchmarks": (REPO_ROOT / "benchmarks").resolve(),
        "scvi": (REPO_ROOT / "src" / "scvi").resolve(),
    }


def _runtime_module_paths(module: object) -> tuple[Path | None, tuple[Path, ...]]:
    file_value = getattr(module, "__file__", None)
    file_path = Path(file_value).resolve() if file_value is not None else None
    package_value = getattr(module, "__path__", ())
    try:
        package_paths = tuple(Path(value).resolve() for value in package_value)
    except TypeError as error:
        raise ValueError("Runtime module __path__ is not iterable.") from error
    return file_path, package_paths


def _runtime_relative_path(
    path: Path,
    *,
    module_name: str,
    namespace_root: Path,
) -> str:
    try:
        path.relative_to(namespace_root)
        relative = path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Runtime import {module_name} is outside the declared source tree."
        ) from error
    return relative.as_posix()


def _loaded_runtime_modules() -> dict[str, dict[str, object]]:
    roots = _runtime_namespace_roots()
    observed: dict[str, dict[str, object]] = {}
    for module_name, module in sorted(sys.modules.items()):
        namespace = next(
            (
                name
                for name in roots
                if module_name == name or module_name.startswith(f"{name}.")
            ),
            None,
        )
        if namespace is None or module is None:
            continue
        file_path, package_paths = _runtime_module_paths(module)
        if file_path is None and not package_paths:
            raise ValueError(
                f"Runtime import {module_name} has no declared source path."
            )
        root = roots[namespace]
        observed[module_name] = {
            "file": (
                _runtime_relative_path(
                    file_path,
                    module_name=module_name,
                    namespace_root=root,
                )
                if file_path is not None
                else None
            ),
            "package_paths": [
                _runtime_relative_path(
                    path,
                    module_name=module_name,
                    namespace_root=root,
                )
                for path in package_paths
            ],
        }
    return observed


def _runtime_import_identity() -> dict[str, object]:
    """Bind every benchmark/scvi import and the exact interpreter search path."""
    import benchmarks
    import scvi

    expected = {
        "benchmarks": REPO_ROOT / "benchmarks" / "__init__.py",
        "scvi": REPO_ROOT / "src" / "scvi" / "__init__.py",
    }
    observed = {"benchmarks": Path(benchmarks.__file__), "scvi": Path(scvi.__file__)}
    for name, expected_path in expected.items():
        if observed[name].resolve() != expected_path.resolve():
            raise ValueError(
                f"Runtime import {name} is outside the declared source tree."
            )
    return {
        "schema_version": "mrtotalvi-runtime-import-identity-v2",
        "entrypoints": {
            name: path.relative_to(REPO_ROOT).as_posix()
            for name, path in expected.items()
        },
        "sys_path": [
            str(Path(value).resolve()) for value in sys.path if value
        ],
        "loaded_modules": _loaded_runtime_modules(),
    }


def _validate_runtime_import_identity(payload: object) -> None:
    """Reject worker path drift and every parent/worker namespace escape."""
    current = _runtime_import_identity()
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "entrypoints",
        "sys_path",
        "loaded_modules",
    }:
        raise ValueError("Runtime import identity drifted from code manifest.")
    if (
        payload["schema_version"] != current["schema_version"]
        or payload["entrypoints"] != current["entrypoints"]
        or payload["sys_path"] != current["sys_path"]
    ):
        raise ValueError("Runtime import identity drifted from code manifest.")
    loaded = payload["loaded_modules"]
    if not isinstance(loaded, dict) or not {"benchmarks", "scvi"}.issubset(loaded):
        raise ValueError("Runtime import identity drifted from code manifest.")
    roots = _runtime_namespace_roots()
    repository = REPO_ROOT.resolve()
    for module_name, record in loaded.items():
        namespace = next(
            (
                name
                for name in roots
                if module_name == name or module_name.startswith(f"{name}.")
            ),
            None,
        )
        if namespace is None or not isinstance(record, dict) or set(record) != {
            "file",
            "package_paths",
        }:
            raise ValueError("Runtime import identity drifted from code manifest.")
        values = [record["file"], *record["package_paths"]]
        relative_paths = [value for value in values if value is not None]
        if not relative_paths or any(not isinstance(value, str) for value in relative_paths):
            raise ValueError("Runtime import identity drifted from code manifest.")
        for value in relative_paths:
            resolved = (repository / value).resolve()
            _runtime_relative_path(
                resolved,
                module_name=module_name,
                namespace_root=roots[namespace],
            )


def _write_code_snapshot(
    run_dir: Path,
    code_manifest: Mapping[str, object],
) -> None:
    """Copy every declared source byte into the immutable run payload."""
    if code_manifest.get("schema_version") not in {
        LEGACY_CODE_MANIFEST_SCHEMA,
        CODE_MANIFEST_SCHEMA,
    }:
        raise ValueError("Unknown convergence code-manifest schema.")
    for relative, record in code_manifest["files"].items():
        source = REPO_ROOT / relative
        destination = run_dir / record["snapshot_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if (
            destination.stat().st_size != record["bytes"]
            or sha256_file(destination) != record["sha256"]
        ):
            raise ValueError(
                f"Source snapshot drifted while copying {relative}."
            )


def _verify_code_snapshot(
    run_dir: Path,
    code_manifest: Mapping[str, object],
) -> None:
    """Verify the exact declared source snapshot without consulting live code."""
    if code_manifest.get("schema_version") not in {
        LEGACY_CODE_MANIFEST_SCHEMA,
        CODE_MANIFEST_SCHEMA,
    }:
        raise ValueError("Unknown convergence code-manifest schema.")
    expected = set()
    for relative, record in code_manifest["files"].items():
        snapshot_path = Path(record["snapshot_path"])
        expected.add(snapshot_path.as_posix())
        path = run_dir / snapshot_path
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"Sealed source snapshot drifted for {relative}.")
    observed = {
        path.relative_to(run_dir).as_posix()
        for path in (run_dir / SOURCE_SNAPSHOT_ROOT).rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise ValueError("Sealed source snapshot file set drifted.")


def _verify_live_sources_against_manifest(
    code_manifest: Mapping[str, object],
) -> None:
    """Fail if a worker would execute source bytes outside the snapshot."""
    if code_manifest.get("schema_version") not in {
        LEGACY_CODE_MANIFEST_SCHEMA,
        CODE_MANIFEST_SCHEMA,
    }:
        raise ValueError("Unknown convergence code-manifest schema.")
    for relative, record in code_manifest["files"].items():
        path = REPO_ROOT / relative
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(
                f"Live worker source differs from snapshot for {relative}."
            )


def _ordered_metric_payload(
    payload: Mapping,
    *,
    contract_adapter: RedesignContractAdapter | None = None,
) -> dict:
    """Restore the frozen metric order after sorted-key JSON persistence."""
    adapter = contract_adapter
    expected = (
        redesign_run_contract().metric_ids
        if adapter is None
        else adapter.metric_ids
    )
    if (
        not isinstance(payload, Mapping)
        or len(payload) != len(expected)
        or set(payload) != set(expected)
    ):
        raise ValueError("Persisted metric IDs differ from the frozen contract.")
    return {metric_id: payload[metric_id] for metric_id in expected}


def _write_metric_dictionary(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    """Persist the contract-defined metric order without key sorting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _strict_jsonable(payload),
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _publish_run_no_replace(
    source: str | Path,
    destination: str | Path,
) -> None:
    """Publish a nested run without replacing an existing destination.

    ``renameat2(RENAME_NOREPLACE)`` is preferred. On filesystems that do not
    support it, an atomic destination-directory reservation is followed by
    moving regular files, with ``run-manifest.json`` moved last as the
    completion record.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    try:
        _rename_no_replace(source_path, destination_path)
        return
    except RuntimeError as error:
        if "supports only non-empty regular-file runs" not in str(error):
            raise

    entries = tuple(source_path.rglob("*"))
    if (
        not entries
        or any(
            entry.is_symlink()
            or (not entry.is_file() and not entry.is_dir())
            for entry in entries
        )
        or not (source_path / "run-manifest.json").is_file()
    ):
        raise RuntimeError(
            "Nested no-replace publication requires regular files, "
            "directories, and one run-manifest.json."
        )
    destination_path.mkdir()
    directories = sorted(
        (entry for entry in entries if entry.is_dir()),
        key=lambda path: len(path.relative_to(source_path).parts),
    )
    for directory in directories:
        (destination_path / directory.relative_to(source_path)).mkdir()
    files = sorted(
        (entry for entry in entries if entry.is_file()),
        key=lambda path: (
            path.name == "run-manifest.json",
            path.relative_to(source_path).as_posix(),
        ),
    )
    for file_path in files:
        target = destination_path / file_path.relative_to(source_path)
        file_path.rename(target)
    for directory in reversed(directories):
        directory.rmdir()
    source_path.rmdir()


def _seal_manifest(
    run_dir: Path,
    *,
    run_id: str,
    created_at: datetime,
    code_digest: str,
    config_digest: str,
    data_digest: str,
    status: str,
) -> None:
    manifest = RunManifest(
        schema_version="mrtotalvi-benchmark-run-v1",
        run_id=run_id,
        created_at=created_at.isoformat(),
        code_digest=code_digest,
        config_digest=config_digest,
        data_digest=data_digest,
        evidence_tier="pilot_cache",
        scientific_scope=SCIENTIFIC_SCOPE,
        status=status,
        artifacts=_artifact_records(run_dir),
    )
    _write_json(run_dir / "run-manifest.json", manifest.to_dict())
    verify_run_manifest(run_dir / "run-manifest.json")


def _write_representations(
    path: Path,
    representations: Mapping[str, Mapping[str, object]],
) -> None:
    expected = {"factual_z"} | ({"u"} if "u" in representations else set())
    if set(representations) != expected:
        raise ValueError(
            f"Unexpected representation names: {sorted(representations)}."
        )
    arrays = {}
    for name in sorted(representations):
        payload = representations[name]
        if set(payload) != {"cell_ids", "values"}:
            raise ValueError(
                f"Representation {name!r} has an invalid payload."
            )
        arrays[f"{name}__cell_ids"] = np.asarray(
            payload["cell_ids"],
            dtype=str,
        )
        arrays[f"{name}__values"] = np.asarray(
            payload["values"],
            dtype=np.float64,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _read_representations(
    path: Path,
    *,
    candidate_id: str,
) -> dict[str, dict[str, np.ndarray]]:
    expected_names = (
        {"factual_z"} if candidate_id == "B1" else {"u", "factual_z"}
    )
    with np.load(path, allow_pickle=False) as loaded:
        expected_keys = {
            f"{name}__{suffix}"
            for name in expected_names
            for suffix in ("cell_ids", "values")
        }
        if set(loaded.files) != expected_keys:
            raise ValueError(
                f"Representation archive keys drifted for {candidate_id}."
            )
        return {
            name: {
                "cell_ids": np.asarray(
                    loaded[f"{name}__cell_ids"],
                    dtype=str,
                ),
                "values": np.asarray(
                    loaded[f"{name}__values"],
                    dtype=np.float64,
                ),
            }
            for name in expected_names
        }


def _partial_aggregate_v1(
    results: list[dict],
    *,
    request: DiagnosisExecutionRequest,
) -> dict[str, object]:
    """Reproduce the exact historical non-authoritative aggregate."""
    failures = [
        {
            "fixture_id": result["fixture_id"],
            "candidate_id": result["candidate_id"],
            "training_seed": result["training_seed"],
            "convergence_status": result["convergence"]["status"],
            "collapse_failed": result["collapse"]["failed"],
        }
        for result in results
        if result["convergence"]["status"] != "converged"
        or result["collapse"]["failed"]
    ]
    return {
        "schema_version": LEGACY_PARTIAL_AGGREGATE_SCHEMA,
        "authoritative_full_grid": False,
        "n_fits": len(results),
        "requested_fits": request.n_fits,
        "fixtures": list(request.fixtures),
        "rows": list(request.rows),
        "training_seeds": list(request.seeds),
        "fit_failures": failures,
        "d0_decision": "not_available_for_non_authoritative_grid",
        "rdx03_completion": "prohibited",
        "scientific_scope": SCIENTIFIC_SCOPE,
    }


def _partial_aggregate(
    results: list[dict],
    *,
    request: DiagnosisExecutionRequest,
    contract_adapter: RedesignContractAdapter,
) -> dict[str, object]:
    if request.authoritative_full_grid:
        raise ValueError("Partial aggregate is only valid for a probe.")
    failures = []
    alerts = []
    terminal_failed = False
    for result in results:
        integrity = result["latent_integrity"]
        flagged = integrity["effective_rank_screen_flags"]
        if flagged:
            alerts.append(
                {
                    "fixture_id": result["fixture_id"],
                    "candidate_id": result["candidate_id"],
                    "training_seed": result["training_seed"],
                    "representations": list(flagged),
                }
            )
        failed = integrity["terminal_integrity_failed"]
        terminal_failed = terminal_failed or failed
        if result["convergence"]["status"] != "converged" or failed:
            failures.append(
                {
                    "fixture_id": result["fixture_id"],
                    "candidate_id": result["candidate_id"],
                    "training_seed": result["training_seed"],
                    "convergence_status": result["convergence"]["status"],
                    "terminal_integrity_failed": failed,
                }
            )
    return {
        "schema_version": PARTIAL_AGGREGATE_SCHEMA,
        **version_binding_fields(contract_adapter),
        "authoritative_full_grid": False,
        "run_purpose": "probe",
        "n_fits": len(results),
        "requested_fits": request.n_fits,
        "fixtures": list(request.fixtures),
        "rows": list(request.rows),
        "training_seeds": list(request.seeds),
        "fit_failures": failures,
        "terminal_integrity_failed": terminal_failed,
        "effective_rank_screen_flags": alerts,
        "all_fits_converged_with_terminal_integrity": not failures,
        "d0_decision": "not_available_for_non_authoritative_grid",
        "rdx03_completion": "prohibited",
        "scientific_scope": SCIENTIFIC_SCOPE,
    }


def _validate_fit_result(
    result: dict,
    *,
    fixture_manifest: Mapping[str, object],
    candidate_id: str,
    training_seed: int,
    checkpoint_root: Path,
    contract_adapter: RedesignContractAdapter,
) -> None:
    validate_version_binding(
        result,
        contract_adapter,
        expected_schema=FIT_SCHEMA,
    )
    validate_version_binding(
        result.get("latent_integrity"),
        contract_adapter,
        expected_schema=LATENT_INTEGRITY_ASSESSMENT_SCHEMA,
    )
    expected = {
        "fixture_id": fixture_manifest["fixture_id"],
        "candidate_id": candidate_id,
        "training_seed": training_seed,
        "evaluation_seed": fixture_manifest["evaluation_seed"],
        "data_digest": fixture_manifest["data_digest"],
        "source_data_digest": fixture_manifest["source_data_digest"],
        "state_annotation_digest": fixture_manifest[
            "state_annotation_digest"
        ],
        "split_digest": fixture_manifest["split_digest"],
        "learning_rate": fixture_manifest["learning_rate"],
        "control": _convergence_control_payload(contract_adapter),
    }
    for name, value in expected.items():
        if result.get(name) != value:
            raise ValueError(
                f"Diagnosis fit {name} drifted for "
                f"{fixture_manifest['fixture_id']}/"
                f"{candidate_id}/seed{training_seed}."
            )
    validate_metric_payload(
        _ordered_metric_payload(
            result["metrics"],
            contract_adapter=contract_adapter,
        ),
        candidate_id=candidate_id,
        lifecycle="per_fit",
        contract_adapter=contract_adapter,
        required_no_call_reasons=result.get(
            "metric_no_call_reasons",
            {},
        ),
    )
    validate_frozen_optimization_identity(
        result["optimization_identity"],
        n_obs=fixture_manifest["n_cells"],
        learning_rate=fixture_manifest["learning_rate"],
    )
    checkpoint = result["best_checkpoint_identity"]
    artifact = checkpoint_root / checkpoint["artifact_name"]
    if not artifact.is_dir():
        raise ValueError(f"Best checkpoint artifact is missing: {artifact}.")
    if (
        _checkpoint_artifact_state_digest(artifact)
        != checkpoint["state_digest"]
    ):
        raise ValueError("Best checkpoint artifact digest drifted after fit.")


def _execution_environment(
    *,
    contract_adapter: RedesignContractAdapter,
) -> dict:
    import scvi

    environment = _environment_manifest()
    environment["scientific_scope"] = SCIENTIFIC_SCOPE
    environment["accelerator_contract"] = contract_adapter.run_contract_section(
        "runtime"
    )["accelerator_policy"]
    environment["cuda_visible_devices"] = os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    try:
        version = importlib.metadata.version("scvi-tools")
    except importlib.metadata.PackageNotFoundError:
        version = None
    environment["scvi_tools"] = {
        "version": version,
        "import_path": str(Path(scvi.__file__).resolve()),
    }
    return environment


def _result_stem(
    fixture_id: str,
    candidate_id: str,
    seed: int,
) -> str:
    return f"{fixture_id}--{candidate_id}--seed{seed}"


def _fit_worker_command(
    *,
    fixture_id: str,
    candidate_id: str,
    seed: int,
    output_dir: Path,
    code_manifest_path: Path,
    configuration_path: Path,
    run_contract_path: Path,
    latent_integrity_policy_path: Path,
    metric_dictionary_path: Path,
) -> tuple[str, ...]:
    """Construct one fresh-interpreter worker command for one exact fit."""
    return (
        str(Path(sys.executable).resolve()),
        "-P",
        "-m",
        FIT_WORKER_MODULE,
        "--fixture",
        fixture_id,
        "--candidate",
        candidate_id,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir.resolve()),
        "--code-manifest",
        str(code_manifest_path.resolve()),
        "--configuration",
        str(configuration_path.resolve()),
        "--redesign-run-contract",
        str(run_contract_path.resolve()),
        "--latent-integrity-policy",
        str(latent_integrity_policy_path.resolve()),
        "--metric-dictionary",
        str(metric_dictionary_path.resolve()),
    )


def _fit_worker_environment() -> dict[str, str]:
    """Run child imports from this source tree without writing bytecode there."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str((REPO_ROOT / "src").resolve()),
            str(REPO_ROOT.resolve()),
        )
    )
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _validate_fit_worker_manifest(
    payload: Mapping[str, object],
    *,
    fixture_id: str,
    candidate_id: str,
    seed: int,
    code_digest: str,
    contract_adapter: RedesignContractAdapter,
) -> None:
    """Require evidence that every fit had an isolated process memory scope."""
    prospective = contract_adapter.integrity_version == "v2"
    if prospective:
        validate_version_binding(
            payload,
            contract_adapter,
            expected_schema=FIT_WORKER_SCHEMA,
        )
    expected = {
        "schema_version": (
            FIT_WORKER_SCHEMA
            if prospective
            else LEGACY_FIT_WORKER_SCHEMA
        ),
        "fixture_id": fixture_id,
        "candidate_id": candidate_id,
        "training_seed": seed,
        "python_executable": str(Path(sys.executable).resolve()),
        "memory_scope": FIT_MEMORY_SCOPE,
        "code_digest": code_digest,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"Fit-worker manifest {name} drifted.")
    process_id = payload.get("process_id")
    parent_process_id = payload.get("parent_process_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id < 1
        or isinstance(parent_process_id, bool)
        or not isinstance(parent_process_id, int)
        or parent_process_id < 1
        or process_id == parent_process_id
    ):
        raise ValueError("Fit-worker process identity is not isolated.")


def _run_isolated_diagnosis_fit(
    *,
    fixture_id: str,
    candidate_id: str,
    seed: int,
    run_dir: Path,
    contract_adapter: RedesignContractAdapter,
) -> tuple[dict, dict[str, dict[str, np.ndarray]], Path, Path]:
    """Run one fit in a new interpreter and return its staged artifacts."""
    run_dir = Path(run_dir).resolve()
    stem = _result_stem(fixture_id, candidate_id, seed)
    worker_parent = run_dir / "worker-output"
    worker_parent.mkdir(exist_ok=True)
    worker_dir = worker_parent / stem
    workspace_parent = run_dir / "worker-workspaces"
    workspace_parent.mkdir(exist_ok=True)
    worker_workspace = workspace_parent / stem
    worker_workspace.mkdir()
    log_dir = run_dir / "worker-logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{stem}.log"
    command = _fit_worker_command(
        fixture_id=fixture_id,
        candidate_id=candidate_id,
        seed=seed,
        output_dir=worker_dir,
        code_manifest_path=run_dir / "code-manifest.json",
        configuration_path=run_dir / "configuration.json",
        run_contract_path=run_dir / "redesign-run-contract.json",
        latent_integrity_policy_path=(
            run_dir / "latent-integrity-policy.json"
        ),
        metric_dictionary_path=run_dir / "metric-dictionary.json",
    )
    code_manifest = json.loads(
        (run_dir / "code-manifest.json").read_text(encoding="utf-8")
    )
    _verify_code_snapshot(run_dir, code_manifest)
    with log_path.open("xb") as log_handle:
        try:
            completed = subprocess.run(
                command,
                cwd=worker_workspace,
                env=_fit_worker_environment(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        finally:
            _verify_code_snapshot(run_dir, code_manifest)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Isolated fit worker failed with exit {completed.returncode}; "
            f"see {log_path.relative_to(run_dir)}."
        )
    worker_manifest_path = worker_dir / "worker-manifest.json"
    worker_manifest = json.loads(
        worker_manifest_path.read_text(encoding="utf-8")
    )
    _validate_fit_worker_manifest(
        worker_manifest,
        fixture_id=fixture_id,
        candidate_id=candidate_id,
        seed=seed,
        code_digest=code_manifest["code_digest"],
        contract_adapter=contract_adapter,
    )
    result = json.loads(
        (worker_dir / "result.json").read_text(encoding="utf-8")
    )
    representations = _read_representations(
        worker_dir / "representation.npz",
        candidate_id=candidate_id,
    )
    return result, representations, worker_dir, worker_manifest_path


def run_convergence_diagnosis(
    args: Namespace,
) -> Path:
    """Execute and seal one exact diagnosis grid or bounded probe."""
    (
        contract_adapter,
        run_contract_payload,
        latent_integrity_policy_payload,
        metric_dictionary_payload,
    ) = _prospective_version_context()
    request = validate_diagnosis_request(
        fixtures=tuple(args.fixtures),
        rows=tuple(args.rows),
        seeds=tuple(args.seeds),
        contract_adapter=contract_adapter,
    )
    base_fixtures = {
        fixture_id: prepare_diagnosis_fixture(
            fixture_id,
            repo_root=REPO_ROOT,
        )
        for fixture_id in request.fixtures
    }
    fixture_manifests = [
        _fixture_manifest(base_fixtures[fixture_id])
        for fixture_id in request.fixtures
    ]
    fixture_manifest_by_id = {
        item["fixture_id"]: item for item in fixture_manifests
    }
    del base_fixtures
    gc.collect()
    execution_config = {
        "schema_version": EXECUTION_SCHEMA,
        **version_binding_fields(contract_adapter),
        "control": _convergence_control_payload(contract_adapter),
        "fixtures": list(request.fixtures),
        "rows": list(request.rows),
        "training_seeds": list(request.seeds),
        "n_fits": request.n_fits,
        "authoritative_full_grid": request.authoritative_full_grid,
        "run_purpose": (
            "authoritative_full_grid"
            if request.authoritative_full_grid
            else "probe"
        ),
        "rdx03_completion": (
            "eligible_after_all_gates"
            if request.authoritative_full_grid
            else "prohibited"
        ),
        "fixture_manifests": fixture_manifests,
        "accelerator": "cpu",
        "candidate_specific_retuning": False,
        "factual_human_da": "locked_not_computed_or_inspected",
        "scientific_scope": SCIENTIFIC_SCOPE,
        "diagnostic_estimators": _diagnostic_estimator_configuration(),
    }
    validate_version_binding(
        execution_config,
        contract_adapter,
        expected_schema=EXECUTION_SCHEMA,
    )
    code_manifest = _convergence_code_manifest(
        contract_adapter=contract_adapter,
    )
    validate_version_binding(
        code_manifest,
        contract_adapter,
        expected_schema=CODE_MANIFEST_SCHEMA,
    )
    code_digest = code_manifest["code_digest"]
    config_digest = _payload_digest(execution_config)
    data_digest = _payload_digest(fixture_manifests)
    created_at = datetime.now(UTC).replace(microsecond=0)
    run_id = make_run_id(
        timestamp=created_at,
        code_digest=code_digest,
        config_digest=config_digest,
        data_digest=data_digest,
    )
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    failed_destination = output_root / f"{run_id}-failed"
    for path in (destination, failed_destination):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"Refusing existing convergence destination {path}."
            )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output_root)
    )

    try:
        _write_json(temporary / "configuration.json", execution_config)
        _write_json(
            temporary / "redesign-run-contract.json",
            run_contract_payload,
        )
        _write_json(
            temporary / "latent-integrity-policy.json",
            latent_integrity_policy_payload,
        )
        _write_metric_dictionary(
            temporary / "metric-dictionary.json",
            metric_dictionary_payload,
        )
        _write_json(temporary / "code-manifest.json", code_manifest)
        _write_code_snapshot(temporary, code_manifest)
        _write_json(
            temporary / "environment.json",
            _execution_environment(
                contract_adapter=contract_adapter,
            ),
        )
        results = []
        representations = {}
        for fixture_id in request.fixtures:
            fixture_manifest = fixture_manifest_by_id[fixture_id]
            for candidate_id in request.rows:
                for seed in request.seeds:
                    stem = _result_stem(
                        fixture_id,
                        candidate_id,
                        seed,
                    )
                    checkpoint_root = temporary / "checkpoints" / stem
                    print(
                        json.dumps(
                            {
                                "event": "fit_start",
                                "fixture_id": fixture_id,
                                "candidate_id": candidate_id,
                                "training_seed": seed,
                                "authoritative_full_grid": (
                                    request.authoritative_full_grid
                                ),
                            }
                        ),
                        flush=True,
                    )
                    (
                        result,
                        representation,
                        worker_dir,
                        worker_manifest_source,
                    ) = _run_isolated_diagnosis_fit(
                        fixture_id=fixture_id,
                        candidate_id=candidate_id,
                        seed=seed,
                        run_dir=temporary,
                        contract_adapter=contract_adapter,
                    )
                    worker_checkpoint = worker_dir / "checkpoint"
                    _validate_fit_result(
                        result,
                        fixture_manifest=fixture_manifest,
                        candidate_id=candidate_id,
                        training_seed=seed,
                        checkpoint_root=worker_checkpoint,
                        contract_adapter=contract_adapter,
                    )
                    checkpoint_root.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    worker_checkpoint.rename(checkpoint_root)
                    representation_path = (
                        temporary / "representations" / f"{stem}.npz"
                    )
                    representation_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    (worker_dir / "representation.npz").rename(
                        representation_path
                    )
                    worker_manifest_path = (
                        temporary / "workers" / f"{stem}.json"
                    )
                    worker_manifest_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    worker_manifest_source.rename(worker_manifest_path)
                    (worker_dir / "result.json").unlink()
                    worker_dir.rmdir()
                    result["checkpoint_relative_path"] = (
                        checkpoint_root.relative_to(temporary).as_posix()
                    )
                    result["representation_relative_path"] = (
                        representation_path.relative_to(temporary).as_posix()
                    )
                    result["worker_manifest_relative_path"] = (
                        worker_manifest_path.relative_to(
                            temporary
                        ).as_posix()
                    )
                    _write_json(
                        temporary / "results" / f"{stem}.json",
                        result,
                    )
                    results.append(result)
                    representations[
                        (candidate_id, fixture_id, seed)
                    ] = representation
                    print(
                        json.dumps(
                            {
                                "event": "fit_complete",
                                "fixture_id": fixture_id,
                                "candidate_id": candidate_id,
                                "training_seed": seed,
                                "trainer_epochs": result["convergence"][
                                    "trainer_epochs"
                                ],
                                "convergence_status": result[
                                    "convergence"
                                ]["status"],
                                "terminal_integrity_failed": result[
                                    "latent_integrity"
                                ]["terminal_integrity_failed"],
                                "wall_time_seconds": result["metrics"][
                                    "wall_time_seconds"
                                ],
                            }
                        ),
                        flush=True,
                    )
                    del representation
                    gc.collect()
            gc.collect()
        (temporary / "worker-output").rmdir()

        if request.authoritative_full_grid:
            aggregate = aggregate_diagnosis_grid_v2(
                results,
                representations=representations,
                contract_adapter=contract_adapter,
            )
        else:
            aggregate = _partial_aggregate(
                results,
                request=request,
                contract_adapter=contract_adapter,
            )
        _verify_live_sources_against_manifest(code_manifest)
        _write_json(temporary / "aggregate.json", aggregate)
        _seal_manifest(
            temporary,
            run_id=run_id,
            created_at=created_at,
            code_digest=code_digest,
            config_digest=config_digest,
            data_digest=data_digest,
            status=(
                "complete"
                if request.authoritative_full_grid
                else "inconclusive"
            ),
        )
        verify_convergence_run(temporary)
        _publish_run_no_replace(temporary, destination)
        verify_run_manifest(destination / "run-manifest.json")
        print(
            json.dumps({"event": "sealed", "path": str(destination)}),
            flush=True,
        )
        return destination
    except BaseException as error:
        _write_json(
            temporary / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        try:
            _seal_manifest(
                temporary,
                run_id=run_id,
                created_at=created_at,
                code_digest=code_digest,
                config_digest=config_digest,
                data_digest=data_digest,
                status="failed",
            )
        except (OSError, TypeError, ValueError) as seal_error:
            _write_json(
                temporary / "manifest-seal-failure.json",
                {
                    "error_type": type(seal_error).__name__,
                    "error": str(seal_error),
                    "traceback": traceback.format_exc(),
                },
            )
        _publish_run_no_replace(temporary, failed_destination)
        raise


def _validate_manifest_identity_bindings(
    manifest: RunManifest,
    *,
    code_manifest: Mapping[str, object],
    configuration: Mapping[str, object],
) -> None:
    """Bind the generic artifact inventory to scientific identity payloads."""
    code_payload = dict(code_manifest)
    recorded_code_digest = code_payload.pop("code_digest", None)
    recomputed_code_digest = _payload_digest(code_payload)
    if (
        recorded_code_digest != recomputed_code_digest
        or manifest.code_digest != recomputed_code_digest
    ):
        raise ValueError("Sealed code identity is internally inconsistent.")
    recomputed_config_digest = _payload_digest(configuration)
    if manifest.config_digest != recomputed_config_digest:
        raise ValueError(
            "Sealed configuration identity is internally inconsistent."
        )
    fixture_manifests = configuration.get("fixture_manifests")
    if not isinstance(fixture_manifests, list):
        raise ValueError("Sealed fixture manifests are missing.")
    recomputed_data_digest = _payload_digest(fixture_manifests)
    if manifest.data_digest != recomputed_data_digest:
        raise ValueError("Sealed data identity is internally inconsistent.")


def _validate_recomputed_fit_identity(
    result: Mapping[str, object],
    *,
    artifact_state_digest: str,
    contract_adapter: RedesignContractAdapter | None = None,
) -> None:
    """Recompute checkpoint, convergence, and collapse identities on re-read."""
    schema = result.get("schema_version")
    if schema not in {LEGACY_FIT_SCHEMA, FIT_SCHEMA}:
        raise ValueError("Unexpected sealed diagnosis fit schema.")
    prospective = schema == FIT_SCHEMA
    if prospective:
        if (
            contract_adapter is None
            or contract_adapter.integrity_version != "v2"
        ):
            raise ValueError(
                "Prospective fit verification requires its v2 adapter."
            )
        validate_version_binding(
            result,
            contract_adapter,
            expected_schema=FIT_SCHEMA,
        )
        validate_version_binding(
            result.get("latent_integrity"),
            contract_adapter,
            expected_schema=LATENT_INTEGRITY_ASSESSMENT_SCHEMA,
        )
    metrics = result["metrics"]
    history = result["training_history"]
    if (
        history.get("elbo_validation")
        != metrics.get("validation_objective_history")
    ):
        raise ValueError(
            "Sealed validation history is not bound to the metric payload."
        )
    if (
        result.get("best_checkpoint_identity")
        != metrics.get("best_checkpoint_identity")
    ):
        raise ValueError(
            "Sealed checkpoint identity is not bound to the metric payload."
        )
    checkpoint = result["best_checkpoint_identity"]
    if artifact_state_digest != checkpoint.get("state_digest"):
        raise ValueError("Sealed checkpoint state identity drifted.")
    convergence = result["convergence"]
    selected_control = result.get("control")
    if selected_control is None and contract_adapter is not None:
        selected_control = _convergence_control_payload(contract_adapter)
    recomputed_convergence = assess_convergence(
        history["elbo_validation"],
        trainer_epochs=convergence["trainer_epochs"],
        stopped_early=convergence["stopped_early"],
        control=selected_control,
    )
    if recomputed_convergence != convergence:
        raise ValueError("Sealed convergence classification drifted.")
    if prospective:
        recomputed_integrity = {
            **assess_latent_integrity_v2(
                metrics,
                candidate_id=result["candidate_id"],
                latent_dimension=result["latent_dimension"],
            ),
            **version_binding_fields(contract_adapter),
        }
        if recomputed_integrity != result["latent_integrity"]:
            raise ValueError(
                "Sealed latent-integrity classification drifted."
            )
        validate_metric_payload(
            _ordered_metric_payload(
                metrics,
                contract_adapter=contract_adapter,
            ),
            candidate_id=result["candidate_id"],
            lifecycle="per_fit",
            contract_adapter=contract_adapter,
            required_no_call_reasons=result.get(
                "metric_no_call_reasons",
                {},
            ),
        )
    else:
        recomputed_collapse = assess_latent_collapse(
            metrics,
            candidate_id=result["candidate_id"],
            latent_dimension=result["latent_dimension"],
        )
        if recomputed_collapse != result["collapse"]:
            raise ValueError("Sealed collapse classification drifted.")
    validate_frozen_optimization_identity(
        result["optimization_identity"],
        n_obs=result["n_cells"],
        learning_rate=result["learning_rate"],
    )


def verify_convergence_run(run_dir: str | Path) -> dict[str, object]:
    """Re-read a sealed convergence run and recompute its aggregate."""
    root = Path(run_dir).resolve()
    manifest = verify_run_manifest(root / "run-manifest.json")
    code_manifest = json.loads(
        (root / "code-manifest.json").read_text(encoding="utf-8")
    )
    configuration = json.loads(
        (root / "configuration.json").read_text(encoding="utf-8")
    )
    prospective = configuration.get("schema_version") == EXECUTION_SCHEMA
    if prospective:
        run_contract_payload = json.loads(
            (root / "redesign-run-contract.json").read_text(
                encoding="utf-8"
            )
        )
        latent_integrity_policy_payload = json.loads(
            (root / "latent-integrity-policy.json").read_text(
                encoding="utf-8"
            )
        )
        metric_dictionary_payload = json.loads(
            (root / "metric-dictionary.json").read_text(
                encoding="utf-8"
            )
        )
        contract_adapter = _resolve_execution_contract_adapter(
            configuration,
            run_contract_payload=run_contract_payload,
            latent_integrity_policy_payload=(
                latent_integrity_policy_payload
            ),
        )
        validate_version_binding(
            configuration,
            contract_adapter,
            expected_schema=EXECUTION_SCHEMA,
        )
        validate_version_binding(
            code_manifest,
            contract_adapter,
            expected_schema=CODE_MANIFEST_SCHEMA,
        )
        validate_metric_dictionary(
            metric_dictionary_payload,
            contract_adapter=contract_adapter,
        )
        _validate_diagnostic_estimator_configuration(
            configuration.get("diagnostic_estimators")
        )
    else:
        contract_adapter = _resolve_execution_contract_adapter(
            configuration
        )
        if contract_adapter.integrity_version != "v1":
            raise ValueError("Unsupported convergence execution version.")
    _validate_manifest_identity_bindings(
        manifest,
        code_manifest=code_manifest,
        configuration=configuration,
    )
    _verify_code_snapshot(root, code_manifest)
    request = validate_diagnosis_request(
        fixtures=tuple(configuration["fixtures"]),
        rows=tuple(configuration["rows"]),
        seeds=tuple(configuration["training_seeds"]),
        contract_adapter=contract_adapter,
    )
    expected_run_purpose = (
        "authoritative_full_grid"
        if request.authoritative_full_grid
        else "probe"
    )
    if (
        configuration["authoritative_full_grid"]
        != request.authoritative_full_grid
        or configuration["n_fits"] != request.n_fits
        or configuration["control"]
        != _convergence_control_payload(contract_adapter)
        or (
            prospective
            and configuration.get("run_purpose")
            != expected_run_purpose
        )
        or (
            prospective
            and configuration.get("rdx03_completion")
            != (
                "eligible_after_all_gates"
                if request.authoritative_full_grid
                else "prohibited"
            )
        )
    ):
        raise ValueError("Convergence execution configuration drifted.")

    expected_stems = {
        _result_stem(fixture_id, candidate_id, seed)
        for fixture_id in request.fixtures
        for candidate_id in request.rows
        for seed in request.seeds
    }
    for directory, suffix in (
        ("results", ".json"),
        ("representations", ".npz"),
        ("workers", ".json"),
        ("worker-logs", ".log"),
    ):
        observed = {
            path.name.removesuffix(suffix)
            for path in (root / directory).glob(f"*{suffix}")
            if path.is_file()
        }
        if observed != expected_stems:
            raise ValueError(
                f"Sealed {directory} file set differs from the exact grid."
            )

    results = []
    representations = {}
    expected_fit_schema = FIT_SCHEMA if prospective else LEGACY_FIT_SCHEMA
    for fixture_id in request.fixtures:
        fixture_manifest = next(
            item
            for item in configuration["fixture_manifests"]
            if item["fixture_id"] == fixture_id
        )
        for candidate_id in request.rows:
            for seed in request.seeds:
                stem = _result_stem(fixture_id, candidate_id, seed)
                result = json.loads(
                    (root / "results" / f"{stem}.json").read_text(
                        encoding="utf-8"
                    )
                )
                if result.get("schema_version") != expected_fit_schema:
                    raise ValueError(
                        f"Unexpected sealed fit schema for {stem}."
                    )
                if prospective:
                    validate_version_binding(
                        result,
                        contract_adapter,
                        expected_schema=FIT_SCHEMA,
                    )
                    validate_version_binding(
                        result.get("latent_integrity"),
                        contract_adapter,
                        expected_schema=LATENT_INTEGRITY_ASSESSMENT_SCHEMA,
                    )
                for name, expected in (
                    ("fixture_id", fixture_id),
                    ("candidate_id", candidate_id),
                    ("training_seed", seed),
                    ("data_digest", fixture_manifest["data_digest"]),
                    (
                        "source_data_digest",
                        fixture_manifest["source_data_digest"],
                    ),
                    (
                        "state_annotation_digest",
                        fixture_manifest["state_annotation_digest"],
                    ),
                    ("split_digest", fixture_manifest["split_digest"]),
                    ("n_cells", fixture_manifest["n_cells"]),
                    ("n_genes", fixture_manifest["n_genes"]),
                    ("n_proteins", fixture_manifest["n_proteins"]),
                    (
                        "n_train_cells",
                        fixture_manifest["n_train_cells"],
                    ),
                    (
                        "n_validation_cells",
                        fixture_manifest["n_validation_cells"],
                    ),
                    (
                        "latent_dimension",
                        fixture_manifest["latent_dimension"],
                    ),
                    (
                        "learning_rate",
                        fixture_manifest["learning_rate"],
                    ),
                    (
                        "evaluation_seed",
                        fixture_manifest["evaluation_seed"],
                    ),
                ):
                    if result.get(name) != expected:
                        raise ValueError(
                            f"Sealed result {name} drifted for {stem}."
                        )
                validate_metric_payload(
                    _ordered_metric_payload(
                        result["metrics"],
                        contract_adapter=contract_adapter,
                    ),
                    candidate_id=candidate_id,
                    lifecycle="per_fit",
                    contract_adapter=contract_adapter,
                    required_no_call_reasons=result.get(
                        "metric_no_call_reasons",
                        {},
                    ),
                )
                expected_worker_path = f"workers/{stem}.json"
                if (
                    result.get("worker_manifest_relative_path")
                    != expected_worker_path
                ):
                    raise ValueError(
                        f"Sealed worker path drifted for {stem}."
                    )
                worker_manifest = json.loads(
                    (root / expected_worker_path).read_text(
                        encoding="utf-8"
                    )
                )
                _validate_fit_worker_manifest(
                    worker_manifest,
                    fixture_id=fixture_id,
                    candidate_id=candidate_id,
                    seed=seed,
                    code_digest=code_manifest["code_digest"],
                    contract_adapter=contract_adapter,
                )
                if not (root / "worker-logs" / f"{stem}.log").is_file():
                    raise ValueError(
                        f"Sealed worker log is missing for {stem}."
                    )
                checkpoint_root = root / result[
                    "checkpoint_relative_path"
                ]
                checkpoint_artifact = (
                    checkpoint_root
                    / result["best_checkpoint_identity"]["artifact_name"]
                )
                if not checkpoint_artifact.is_dir():
                    raise ValueError(
                        f"Sealed checkpoint is missing: {checkpoint_artifact}."
                    )
                _validate_recomputed_fit_identity(
                    result,
                    artifact_state_digest=(
                        _checkpoint_artifact_state_digest(
                            checkpoint_artifact
                        )
                    ),
                    contract_adapter=contract_adapter,
                )
                representations[
                    (candidate_id, fixture_id, seed)
                ] = _read_representations(
                    root / result["representation_relative_path"],
                    candidate_id=candidate_id,
                )
                results.append(result)

    aggregate = json.loads(
        (root / "aggregate.json").read_text(encoding="utf-8")
    )
    if request.authoritative_full_grid:
        if prospective:
            validate_version_binding(
                aggregate,
                contract_adapter,
                expected_schema="mrtotalvi-convergence-aggregate-v3",
            )
            recomputed = aggregate_diagnosis_grid_v2(
                results,
                representations=representations,
                contract_adapter=contract_adapter,
            )
        else:
            recomputed = aggregate_diagnosis_grid(
                results,
                representations=representations,
                contract_adapter=contract_adapter,
            )
        if _strict_jsonable(recomputed) != aggregate:
            raise ValueError("Sealed convergence aggregate is not reproducible.")
        if manifest.status != "complete":
            raise ValueError(
                "Authoritative convergence run must have complete status."
            )
    else:
        if prospective:
            validate_version_binding(
                aggregate,
                contract_adapter,
                expected_schema=PARTIAL_AGGREGATE_SCHEMA,
            )
            recomputed = _partial_aggregate(
                results,
                request=request,
                contract_adapter=contract_adapter,
            )
        else:
            if (
                aggregate.get("schema_version")
                != LEGACY_PARTIAL_AGGREGATE_SCHEMA
            ):
                raise ValueError(
                    "Unexpected historical partial aggregate schema."
                )
            recomputed = _partial_aggregate_v1(
                results,
                request=request,
            )
        if _strict_jsonable(recomputed) != aggregate:
            raise ValueError("Sealed partial aggregate is not reproducible.")
        if manifest.status != "inconclusive":
            raise ValueError(
                "Non-authoritative convergence probe must be inconclusive."
            )
    return {
        "run_id": manifest.run_id,
        "status": manifest.status,
        "authoritative_full_grid": request.authoritative_full_grid,
        "n_fits": len(results),
        "aggregate_schema_version": aggregate["schema_version"],
    }


def verify_convergence_run_against_current_repository(
    run_dir: str | Path,
) -> dict[str, object]:
    """Verify sealed evidence, then separately require current-source identity."""
    verified = verify_convergence_run(run_dir)
    root = Path(run_dir).resolve()
    code_manifest = json.loads(
        (root / "code-manifest.json").read_text(encoding="utf-8")
    )
    _verify_live_sources_against_manifest(code_manifest)
    return {**verified, "current_repository_compatible": True}


def main() -> None:
    """CLI entrypoint."""
    run_convergence_diagnosis(_parse_args())


if __name__ == "__main__":
    main()
