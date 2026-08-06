"""Prospective v3 contracts for one isolated RDX-03 fit worker."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pytest
from benchmarks.mrtotalvi.latent_integrity import latent_integrity_policy_v2
from benchmarks.mrtotalvi.metric_schema import build_metric_dictionary_v3
from benchmarks.mrtotalvi.redesign_contract import redesign_run_contract_v2
from benchmarks.mrtotalvi.run_convergence_diagnosis import (
    CODE_MANIFEST_SCHEMA,
    _diagnostic_estimator_configuration,
    _runtime_import_identity,
    _verify_live_sources_against_manifest,
)
from benchmarks.mrtotalvi.run_convergence_fit import (
    MEMORY_SCOPE,
    WORKER_SCHEMA,
    _parse_args,
    run_fit_worker,
)
from benchmarks.mrtotalvi.versioning import (
    PROSPECTIVE_EXECUTION_SCHEMA_V2,
    prospective_redesign_contract_adapter,
    version_binding_fields,
)


def _write_json(path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _sealed_args(tmp_path) -> tuple[argparse.Namespace, object]:
    adapter = prospective_redesign_contract_adapter()
    bindings = version_binding_fields(adapter)
    payloads = {
        "configuration": {
            "schema_version": PROSPECTIVE_EXECUTION_SCHEMA_V2,
            **bindings,
            "diagnostic_estimators": _diagnostic_estimator_configuration(),
        },
        "redesign_run_contract": redesign_run_contract_v2().to_dict(),
        "latent_integrity_policy": latent_integrity_policy_v2(),
        "metric_dictionary": build_metric_dictionary_v3(
            contract_adapter=adapter,
        ),
        "code_manifest": {
            "schema_version": CODE_MANIFEST_SCHEMA,
            **bindings,
            "runtime_imports": _runtime_import_identity(),
            "files": {},
            "code_digest": "a" * 64,
        },
    }
    paths = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name.replace('_', '-')}.json"
        _write_json(path, payload)
        paths[name] = path
    return (
        argparse.Namespace(
            fixture="mixed",
            candidate="B1",
            seed=0,
            output_dir=tmp_path / "worker-output",
            **paths,
        ),
        adapter,
    )


def test_worker_resolves_sealed_v2_inputs_before_mocked_fit(
    tmp_path,
    monkeypatch,
):
    """The worker emits v3 bindings and retains every source-snapshot check."""
    args, expected_adapter = _sealed_args(tmp_path)
    source_checks = []
    fit_calls = []
    fixture = object()

    def verify_sources(payload):
        source_checks.append(payload)
        _verify_live_sources_against_manifest(payload)

    def run_fit(
        received_fixture,
        *,
        candidate_id,
        training_seed,
        checkpoint_dir,
        contract_adapter,
    ):
        fit_calls.append(
            {
                "fixture": received_fixture,
                "candidate_id": candidate_id,
                "training_seed": training_seed,
                "checkpoint_dir": checkpoint_dir,
                "adapter": contract_adapter,
            }
        )
        return (
            {"schema_version": "mrtotalvi-convergence-fit-v3"},
            {
                "factual_z": {
                    "cell_ids": np.asarray(["cell-0", "cell-1"]),
                    "values": np.asarray([[0.0], [1.0]]),
                }
            },
        )

    monkeypatch.setattr(
        "benchmarks.mrtotalvi.run_convergence_fit."
        "_verify_live_sources_against_manifest",
        verify_sources,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.run_convergence_fit."
        "prepare_diagnosis_fixture",
        lambda *_args, **_kwargs: fixture,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.run_convergence_fit.run_diagnosis_fit",
        run_fit,
    )

    output = run_fit_worker(args)

    assert len(source_checks) == 3
    assert len(fit_calls) == 1
    assert fit_calls[0]["fixture"] is fixture
    assert fit_calls[0]["adapter"].integrity_version == "v2"
    assert version_binding_fields(fit_calls[0]["adapter"]) == (
        version_binding_fields(expected_adapter)
    )
    manifest = json.loads(
        (output / "worker-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == {
        "schema_version": "mrtotalvi-convergence-fit-worker-v3",
        **version_binding_fields(expected_adapter),
        "fixture_id": "mixed",
        "candidate_id": "B1",
        "training_seed": 0,
        "process_id": manifest["process_id"],
        "parent_process_id": manifest["parent_process_id"],
        "python_executable": manifest["python_executable"],
        "memory_scope": MEMORY_SCOPE,
        "code_digest": "a" * 64,
    }
    assert WORKER_SCHEMA == "mrtotalvi-convergence-fit-worker-v3"


def test_worker_cli_requires_every_sealed_input_path(monkeypatch):
    """No worker can infer a prospective contract from live defaults."""
    arguments = [
        "run-convergence-fit",
        "--fixture",
        "mixed",
        "--candidate",
        "B1",
        "--seed",
        "0",
        "--output-dir",
        "output",
        "--configuration",
        "configuration.json",
        "--redesign-run-contract",
        "redesign-run-contract.json",
        "--latent-integrity-policy",
        "latent-integrity-policy.json",
        "--metric-dictionary",
        "metric-dictionary.json",
        "--code-manifest",
        "code-manifest.json",
    ]
    required = (
        "--configuration",
        "--redesign-run-contract",
        "--latent-integrity-policy",
        "--metric-dictionary",
        "--code-manifest",
    )

    for option in required:
        position = arguments.index(option)
        missing = [
            *arguments[:position],
            *arguments[position + 2 :],
        ]
        monkeypatch.setattr(sys, "argv", missing)
        with pytest.raises(SystemExit):
            _parse_args()

    monkeypatch.setattr(sys, "argv", arguments)
    parsed = _parse_args()
    assert parsed.configuration.name == "configuration.json"
    assert parsed.redesign_run_contract.name == "redesign-run-contract.json"
    assert parsed.latent_integrity_policy.name == (
        "latent-integrity-policy.json"
    )
    assert parsed.metric_dictionary.name == "metric-dictionary.json"
    assert parsed.code_manifest.name == "code-manifest.json"


@pytest.mark.parametrize(
    "invalid_artifact",
    ["configuration", "metric_dictionary", "code_manifest"],
)
def test_invalid_sealed_input_stops_before_output_or_fit(
    tmp_path,
    monkeypatch,
    invalid_artifact,
):
    """Execution or dictionary drift fails before fixture/model work starts."""
    args, _ = _sealed_args(tmp_path)
    path = getattr(args, invalid_artifact)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if invalid_artifact in {"configuration", "code_manifest"}:
        payload["latent_integrity_policy_digest"] = "0" * 64
    else:
        payload["metrics"].popitem()
    _write_json(path, payload)
    calls = []

    monkeypatch.setattr(
        "benchmarks.mrtotalvi.run_convergence_fit."
        "prepare_diagnosis_fixture",
        lambda *_args, **_kwargs: calls.append("fixture"),
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.run_convergence_fit.run_diagnosis_fit",
        lambda *_args, **_kwargs: calls.append("fit"),
    )

    with pytest.raises(ValueError):
        run_fit_worker(args)

    assert calls == []
    assert not args.output_dir.exists()


def test_worker_rejects_diagnostic_estimator_drift_before_output_or_fit(
    tmp_path,
    monkeypatch,
):
    args, _ = _sealed_args(tmp_path)
    configuration = json.loads(
        args.configuration.read_text(encoding="utf-8")
    )
    configuration["diagnostic_estimators"]["factual_z_posterior_scale"][
        "mrtotalvi"
    ]["draws"] = 31
    _write_json(args.configuration, configuration)
    calls = []
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.run_convergence_fit.prepare_diagnosis_fixture",
        lambda *_args, **_kwargs: calls.append("fixture"),
    )

    with pytest.raises(ValueError, match="estimator configuration"):
        run_fit_worker(args)

    assert calls == []
    assert not args.output_dir.exists()
