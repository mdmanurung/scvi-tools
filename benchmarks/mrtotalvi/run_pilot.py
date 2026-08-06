"""Run an immutable CPU MrTotalVI synthetic mechanism benchmark.

This entrypoint produces pilot-cache evidence only. It cannot select a model,
validate biology, estimate publication FDR, or resolve the canonical data gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .aggregate import aggregate_fixture_results
from .config import candidate_configs
from .manifest import (
    ArtifactRecord,
    RunManifest,
    make_run_id,
    sha256_file,
    verify_run_manifest,
)
from .runner import FixtureRunConfig, run_candidate_fixture
from .simulation import SCENARIO_NAMES, ScenarioConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise argparse.ArgumentTypeError("value must contain unique comma-separated items.")
    return items


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in _parse_csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be integers.") from error
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be nonnegative.")
    return seeds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".scratch/mrtotalvi-v2/validation-runs"),
    )
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default="mixed")
    parser.add_argument(
        "--candidates",
        type=_parse_csv,
        default=tuple(candidate_configs()),
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=(0, 1, 2))
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--n-donors", type=int, default=3)
    parser.add_argument("--cells-per-sample", type=int, default=48)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--n-genes", type=int, default=24)
    parser.add_argument("--n-proteins", type=int, default=8)
    parser.add_argument("--n-latent", type=int, default=4)
    parser.add_argument("--n-hidden", type=int, default=32)
    parser.add_argument("--n-prior-components", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _strict_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _strict_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_strict_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _strict_jsonable(value.item())
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _strict_jsonable(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _payload_digest(payload) -> str:
    encoded = json.dumps(
        _strict_jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _code_manifest() -> dict:
    paths = sorted(
        [
            *REPO_ROOT.glob("benchmarks/mrtotalvi/*.py"),
            *REPO_ROOT.glob("src/scvi/external/mrtotalvi/*.py"),
        ]
    )
    files = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in paths
    }
    payload = {
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status_sha256": hashlib.sha256(
            _git("status", "--short").encode()
        ).hexdigest(),
        "files": files,
    }
    payload["code_digest"] = _payload_digest(payload)
    return payload


def _environment_manifest() -> dict:
    packages = {}
    for package in (
        "anndata",
        "lightning",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "xarray",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }
    except ImportError:
        torch_info = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "torch": torch_info,
        "scientific_scope": (
            "CPU synthetic mechanism pilot; no canonical or biological input"
        ),
    }


def _assessment(
    aggregate: dict,
    *,
    scenario: str,
    candidates: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict:
    per_candidate = aggregate["candidates"]
    hypotheses = {}
    centered = [
        result["metrics"]["centering_max_abs"]
        for candidate in candidates
        if candidate in {"C2", "C3", "C4"}
        for result in per_candidate[candidate]["per_seed"].values()
    ]
    if centered:
        hypotheses["H1_centering"] = {
            "verdict": (
                "mechanism-supported"
                if all(value <= 1e-6 for value in centered)
                else "mechanism-not-supported"
            ),
            "maximum_absolute_error": max(centered),
        }

    if {"C2", "C4"}.issubset(candidates):
        leakage_deltas = [
            per_candidate["C4"]["per_seed"][str(seed)]["metrics"][
                "sample_balanced_accuracy_within_state"
            ]
            - per_candidate["C2"]["per_seed"][str(seed)]["metrics"][
                "sample_balanced_accuracy_within_state"
            ]
            for seed in seeds
        ]
        hypotheses["H3_sample_blind_leakage"] = {
            "verdict": (
                "mechanism-supported"
                if all(delta <= 0.0 for delta in leakage_deltas)
                else (
                    "mechanism-not-supported"
                    if all(delta >= 0.0 for delta in leakage_deltas)
                    else "inconclusive"
                )
            ),
            "per_seed_C4_minus_C2": leakage_deltas,
            "lower_is_better": True,
        }

    if {"C2", "C3"}.issubset(candidates) and scenario == "unequal_cells":
        elbo_sd_deltas = [
            per_candidate["C3"]["per_seed"][str(seed)]["metrics"][
                "heldout_sample_elbo_sd"
            ]
            - per_candidate["C2"]["per_seed"][str(seed)]["metrics"][
                "heldout_sample_elbo_sd"
            ]
            for seed in seeds
        ]
        hypotheses["H5_sample_equal_weighting"] = {
            "verdict": (
                "mechanism-supported"
                if all(delta <= 0.0 for delta in elbo_sd_deltas)
                else (
                    "mechanism-not-supported"
                    if all(delta >= 0.0 for delta in elbo_sd_deltas)
                    else "inconclusive"
                )
            ),
            "per_seed_C3_minus_C2_sample_elbo_sd": elbo_sd_deltas,
            "lower_is_better": True,
            "limitation": (
                "held-out per-sample ELBO dispersion is an engineering proxy, "
                "not a DA calibration endpoint"
            ),
        }

    return {
        "schema_version": "mrtotalvi-pilot-assessment-v1",
        "scenario": scenario,
        "hypotheses": hypotheses,
        "overall_verdict": "inconclusive",
        "promotion": "prohibited",
        "limitations": [
            (
                f"small CPU fixture and {len(seeds)} descriptive "
                "training seed(s)"
            ),
            "no Milo replicate-level DA inference",
            "no publication FDR or power estimate",
            "no canonical human or locked macaque input",
            "no candidate selection or default change",
        ],
    }


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
        scientific_scope=(
            "synthetic mechanism pilot; not biological validation; "
            "not candidate-selection or promotion evidence"
        ),
        status=status,
        artifacts=_artifact_records(run_dir),
    )
    _write_json(run_dir / "run-manifest.json", manifest.to_dict())
    verify_run_manifest(run_dir / "run-manifest.json")


def run_pilot(args: argparse.Namespace) -> Path:
    """Execute and atomically seal one explicit candidate-by-seed grid."""
    candidates = tuple(args.candidates)
    unknown = sorted(set(candidates) - set(candidate_configs()))
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}.")
    seeds = tuple(args.seeds)
    scenario_config = ScenarioConfig(
        scenario=args.scenario,
        n_donors=args.n_donors,
        cells_per_sample=args.cells_per_sample,
        n_states=args.n_states,
        n_genes=args.n_genes,
        n_proteins=args.n_proteins,
        latent_truth_dim=args.n_latent,
    )
    execution_config = {
        "scenario": asdict(scenario_config),
        "candidates": list(candidates),
        "seeds": list(seeds),
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "n_latent": args.n_latent,
        "n_hidden": args.n_hidden,
        "n_prior_components": args.n_prior_components,
        "candidate_axes": {
            name: candidate_configs()[name].model_axes()
            for name in candidates
        },
        "scientific_scope": (
            "synthetic mechanism pilot; not biological validation; "
            "not candidate-selection evidence"
        ),
    }
    code_manifest = _code_manifest()
    code_digest = code_manifest["code_digest"]
    config_digest = _payload_digest(execution_config)
    data_digest = _payload_digest(
        {
            "generator_sha256": code_manifest["files"][
                "benchmarks/mrtotalvi/simulation.py"
            ],
            "scenario": asdict(scenario_config),
            "seeds": list(seeds),
        }
    )
    created_at = datetime.now(UTC).replace(microsecond=0)
    run_id = make_run_id(
        timestamp=created_at,
        code_digest=code_digest,
        config_digest=config_digest,
        data_digest=data_digest,
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / run_id
    failed_destination = output_root / f"{run_id}-failed"
    temporary = output_root / f".{run_id}.tmp-{os.getpid()}"
    for path in (destination, failed_destination, temporary):
        if path.exists():
            raise FileExistsError(f"Refusing existing benchmark destination {path}.")
    temporary.mkdir()

    try:
        _write_json(temporary / "configuration.json", execution_config)
        _write_json(temporary / "code-manifest.json", code_manifest)
        _write_json(temporary / "environment.json", _environment_manifest())
        results = []
        for candidate in candidates:
            for seed in seeds:
                print(
                    json.dumps(
                        {
                            "event": "run_start",
                            "candidate": candidate,
                            "scenario": args.scenario,
                            "seed": seed,
                        }
                    ),
                    flush=True,
                )
                result = run_candidate_fixture(
                    FixtureRunConfig(
                        candidate=candidate,
                        scenario=scenario_config,
                        seed=seed,
                        max_epochs=args.max_epochs,
                        batch_size=args.batch_size,
                        n_latent=args.n_latent,
                        n_hidden=args.n_hidden,
                        n_prior_components=args.n_prior_components,
                    )
                )
                results.append(result)
                _write_json(
                    temporary / "results" / f"{candidate}-seed{seed}.json",
                    result,
                )
                print(
                    json.dumps(
                        {
                            "event": "run_complete",
                            "candidate": candidate,
                            "scenario": args.scenario,
                            "seed": seed,
                            "wall_seconds": result["wall_seconds"],
                        }
                    ),
                    flush=True,
                )
        aggregate = aggregate_fixture_results(
            results,
            expected_candidates=candidates,
            expected_seeds=seeds,
            scenario=args.scenario,
        )
        _write_json(temporary / "aggregate.json", aggregate)
        _write_json(
            temporary / "assessment.json",
            _assessment(
                aggregate,
                scenario=args.scenario,
                candidates=candidates,
                seeds=seeds,
            ),
        )
        _seal_manifest(
            temporary,
            run_id=run_id,
            created_at=created_at,
            code_digest=code_digest,
            config_digest=config_digest,
            data_digest=data_digest,
            status="complete",
        )
        temporary.rename(destination)
        verify_run_manifest(destination / "run-manifest.json")
        print(json.dumps({"event": "sealed", "path": str(destination)}), flush=True)
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
        temporary.rename(failed_destination)
        raise


def main() -> None:
    """CLI entrypoint."""
    run_pilot(_parse_args())


if __name__ == "__main__":
    main()
