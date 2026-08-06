"""Run one RDX-03 fit in a fresh process and persist worker-local evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from .convergence_runner import prepare_diagnosis_fixture, run_diagnosis_fit
from .metric_schema import validate_metric_dictionary
from .run_convergence_diagnosis import (
    CODE_MANIFEST_SCHEMA,
    REPO_ROOT,
    _validate_diagnostic_estimator_configuration,
    _validate_runtime_import_identity,
    _verify_live_sources_against_manifest,
    _write_representations,
)
from .run_pilot import _write_json
from .versioning import (
    PROSPECTIVE_EXECUTION_SCHEMA_V2,
    resolve_redesign_contract_adapter,
    validate_version_binding,
    version_binding_fields,
)

WORKER_SCHEMA = "mrtotalvi-convergence-fit-worker-v3"
METRIC_DICTIONARY_SCHEMA = "mrtotalvi-redesign-metric-dictionary-v3"
MEMORY_SCOPE = "fresh_process_per_fit"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--configuration", required=True, type=Path)
    parser.add_argument(
        "--redesign-run-contract",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--latent-integrity-policy",
        required=True,
        type=Path,
    )
    parser.add_argument("--metric-dictionary", required=True, type=Path)
    parser.add_argument("--code-manifest", required=True, type=Path)
    return parser.parse_args()


def _read_sealed_json(path: Path, *, name: str) -> dict:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Sealed {name} must be a JSON object.")
    return payload


def run_fit_worker(args: argparse.Namespace) -> Path:
    """Execute exactly one fit without inheriting another fit's RSS history."""
    output = args.output_dir.resolve()
    configuration = _read_sealed_json(
        args.configuration,
        name="configuration",
    )
    run_contract = _read_sealed_json(
        args.redesign_run_contract,
        name="redesign run contract",
    )
    policy = _read_sealed_json(
        args.latent_integrity_policy,
        name="latent-integrity policy",
    )
    metric_dictionary = _read_sealed_json(
        args.metric_dictionary,
        name="metric dictionary",
    )
    code_manifest = _read_sealed_json(
        args.code_manifest,
        name="code manifest",
    )
    _validate_runtime_import_identity(code_manifest.get("runtime_imports"))
    _verify_live_sources_against_manifest(code_manifest)
    adapter = resolve_redesign_contract_adapter(
        configuration,
        run_contract_payload=run_contract,
        latent_integrity_policy_payload=policy,
    )
    validate_version_binding(
        configuration,
        adapter,
        expected_schema=PROSPECTIVE_EXECUTION_SCHEMA_V2,
    )
    _validate_diagnostic_estimator_configuration(
        configuration.get("diagnostic_estimators")
    )
    validate_version_binding(
        code_manifest,
        adapter,
        expected_schema=CODE_MANIFEST_SCHEMA,
    )
    validate_version_binding(
        metric_dictionary,
        adapter,
        expected_schema=METRIC_DICTIONARY_SCHEMA,
    )
    validate_metric_dictionary(
        metric_dictionary,
        contract_adapter=adapter,
    )

    output.mkdir(parents=False, exist_ok=False)
    request = {
        "schema_version": WORKER_SCHEMA,
        **version_binding_fields(adapter),
        "fixture_id": args.fixture,
        "candidate_id": args.candidate,
        "training_seed": args.seed,
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "python_executable": str(Path(sys.executable).resolve()),
        "memory_scope": MEMORY_SCOPE,
        "code_digest": code_manifest["code_digest"],
    }
    validate_version_binding(
        request,
        adapter,
        expected_schema=WORKER_SCHEMA,
    )
    _write_json(output / "worker-manifest.json", request)
    try:
        fixture = prepare_diagnosis_fixture(
            args.fixture,
            repo_root=REPO_ROOT,
        )
        result, representations = run_diagnosis_fit(
            fixture,
            candidate_id=args.candidate,
            training_seed=args.seed,
            checkpoint_dir=output / "checkpoint",
            contract_adapter=adapter,
        )
        _verify_live_sources_against_manifest(code_manifest)
        _write_json(output / "result.json", result)
        _write_representations(
            output / "representation.npz",
            representations,
        )
        _verify_live_sources_against_manifest(code_manifest)
    except BaseException as error:
        _write_json(
            output / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(
        json.dumps(
            {
                "event": "worker_complete",
                "fixture_id": args.fixture,
                "candidate_id": args.candidate,
                "training_seed": args.seed,
            }
        ),
        flush=True,
    )
    return output


def main() -> None:
    """CLI entrypoint."""
    run_fit_worker(_parse_args())


if __name__ == "__main__":
    main()
