from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path
from scripts.validate_usage_readiness import (
    ACCEPTANCE_INPUT_PATHS,
    MANDATORY_ACCEPTANCE_CHECKS,
    ROOT,
    ContractError,
    _protocol_run_grid,
    _receipt_digest,
    _record_digest_from_hex,
    _validate_frozen_protocol_artifact,
    _validate_protocol_run_grid,
    _validate_terminal_outputs,
    load_json,
    sha256_file,
    validate_artifact_bundle,
    validate_matrix,
    validate_protocol,
    validate_repository,
    validate_terminal_manifest,
)


@pytest.fixture
def matrix() -> dict:
    return load_json(ROOT / "docs/artifacts/usage-readiness-matrix-v1.json")


@pytest.fixture
def passed_artifact_bundle() -> tuple[dict, dict, dict]:
    manifest = load_json(ROOT / "docs/artifacts/cytoanvi-0.2.0/manifest.json")
    source_commit = "1" * 40
    wheel_sha = "2" * 64
    record_sha = "3" * 64
    manifest["source"].update({"commit": source_commit, "tree": "4" * 40, "clean": True})
    manifest["build"].update(
        {
            "status": "passed",
            "attempt_id": "fixture-attempt",
            "candidate_dir": "/tmp/fixture-candidate",
            "pid": 123,
            "command": "python -m build",
            "python": "/usr/bin/python",
            "environment_digest": "5" * 64,
            "started_at": "2026-08-07T00:00:00Z",
            "completed_at": "2026-08-07T00:01:00Z",
        }
    )
    manifest["dependency_authority"].update(
        {
            "status": "verified",
            "kind": "hashed_lock_and_wheelhouse",
            "path": "/authority",
            "sha256": "6" * 64,
        }
    )
    manifest["wheel"].update(
        {
            "path": "/tmp/cytoanvi-0.2.0-py3-none-any.whl",
            "filename": "cytoanvi-0.2.0-py3-none-any.whl",
            "sha256": wheel_sha,
            "size_bytes": 1234,
            "metadata_name": "cytoanvi",
            "metadata_version": "0.2.0",
        }
    )
    wheel_paths = ["cytoanvi/__init__.py", "cytoanvi-0.2.0.dist-info/RECORD"]
    source_paths = ["cytoanvi/__init__.py"]
    manifest["inventory"].update(
        {
            "details_sha256": "b" * 64,
            "record_sha256": record_sha,
            "record_entries": 2,
            "wheel_files": wheel_paths,
            "source_files": source_paths,
            "acceptance_inputs": [
                {"path": path, "sha256": "a" * 64, "size_bytes": 1}
                for path in sorted(ACCEPTANCE_INPUT_PATHS)
            ],
            "source_vs_wheel": "match",
        }
    )
    manifest["installed_acceptance"].update(
        {
            "status": "passed",
            "acceptance_attempt_id": "fixture-acceptance-attempt",
            "receipt": "docs/artifacts/cytoanvi-0.2.0/installed-acceptance.json",
            "receipt_sha256": "c" * 64,
            "blocker": None,
        }
    )
    inventory = {
        "schema_version": "cytoanvi-artifact-inventory-v1",
        "source_commit": source_commit,
        "wheel_sha256": wheel_sha,
        "record": {
            "path": "cytoanvi-0.2.0.dist-info/RECORD",
            "sha256": record_sha,
            "lines": [
                f"cytoanvi/__init__.py,{_record_digest_from_hex('7' * 64)},1",
                "cytoanvi-0.2.0.dist-info/RECORD,,",
            ],
        },
        "wheel_files": [
            {"path": "cytoanvi/__init__.py", "sha256": "7" * 64, "size_bytes": 1},
            {
                "path": "cytoanvi-0.2.0.dist-info/RECORD",
                "sha256": record_sha,
                "size_bytes": 2,
            },
        ],
        "source_files": [{"path": "cytoanvi/__init__.py", "sha256": "7" * 64, "size_bytes": 1}],
        "acceptance_inputs": [
            {"path": path, "sha256": "a" * 64, "size_bytes": 1}
            for path in sorted(ACCEPTANCE_INPUT_PATHS)
        ],
    }
    receipt = {
        "schema_version": "cytoanvi-installed-acceptance-v1",
        "status": "passed",
        "acceptance_attempt_id": "fixture-acceptance-attempt",
        "artifact": {
            "version": "0.2.0",
            "wheel_sha256": wheel_sha,
            "source_commit": source_commit,
        },
        "dependency_authority": {
            "path": "/authority",
            "sha256": "6" * 64,
            "verified": True,
        },
        "execution": {
            "command": "accept fixture",
            "started_at": "2026-08-07T00:02:00Z",
            "completed_at": "2026-08-07T00:03:00Z",
            "runtime_seconds": 60,
            "exit_status": 0,
            "cwd": "/tmp/accept",
        },
        "isolation": {
            "outside_checkout": True,
            "pythonpath_unset": True,
            "editable_installs_absent": True,
            "scvi_tools_absent": True,
            "fresh_environment": True,
        },
        "checks": [
            {"id": check_id, "status": "passed", "detail": "fixture"}
            for check_id in sorted(MANDATORY_ACCEPTANCE_CHECKS)
        ],
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = _receipt_digest(receipt)
    return manifest, inventory, receipt


def _reseal(receipt: dict) -> None:
    receipt["receipt_sha256"] = _receipt_digest(receipt)


def _fixture_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _complete_frozen_capability(protocol: dict, index: int = 0) -> dict:
    capability = protocol["capabilities"][index]
    capability.update(
        {
            "freeze_status": "frozen",
            "cohorts": [
                {
                    "id": "fixture-independent-cohort-v1",
                    "role": "external_validation",
                    "manifest_sha256": _fixture_digest("fixture cohort manifest"),
                }
            ],
            "biological_unit": "donor",
            "split_and_leakage_boundary": {
                "split_id": "fixture-donor-split-v1",
                "split_sha256": _fixture_digest("fixture donor split"),
                "group_key": "donor_id",
                "leakage_boundary": "donor-disjoint train and evaluation partitions",
            },
            "rng_streams": {
                "training_seeds": [0, 1, 2],
                "streams": {"split": 101, "training": 202, "uncertainty": 303},
            },
            "compute_budget": {
                "budget_id": "fixture-equal-budget-v1",
                "evaluation_grid": ["arm"],
                "maximum_walltime_seconds": 3600,
                "accelerator": "gpu",
                "devices": 1,
            },
            "representation_semantics": ["frozen fixture representation"],
            "primary_endpoint": {
                "name": "donor-level held-out performance",
                "metric": "macro_f1",
                "aggregation_unit": "donor",
                "direction": "maximize",
            },
            "numeric_margin": {
                "kind": "noninferiority",
                "operator": ">=",
                "value": -0.02,
                "unit": "macro_f1_difference",
            },
            "donor_level_uncertainty": {
                "method": "donor_bootstrap",
                "biological_unit": "donor",
                "confidence_level": 0.95,
                "resamples": 1000,
            },
            "multiplicity": {
                "method": "holm",
                "family": "frozen primary endpoint family",
                "alpha": 0.05,
            },
            "controls": {
                "positive": ["prespecified known-signal control"],
                "negative": ["prespecified label-permutation control"],
            },
            "no_call_policy": {
                "policy_id": "fixture-no-call-v1",
                "conditions": ["terminal grid or uncertainty contract is incomplete"],
                "action": "no_call",
            },
            "immutable_outputs": {
                "files": [
                    {
                        "path": "results/fixture-terminal.json",
                        "schema": "docs/artifacts/schemas/terminal-run-manifest.schema.json",
                        "schema_sha256": sha256_file(
                            ROOT / "docs/artifacts/schemas/terminal-run-manifest.schema.json"
                        ),
                    }
                ]
            },
            "pre_run_review": "approved",
            "post_run_review": "not_run",
            "blockers": [],
        }
    )
    return capability


def _approve_protocol_capability(protocol: dict, capability: dict) -> None:
    protocol["independent_review"].update(
        {
            "status": "approved",
            "reviewer": "Independent Reviewer",
            "date": "2026-08-07",
            "independence_rule": "Reviewer did not develop or execute the evaluated capability.",
            "capability_reviews": [
                {
                    "capability_id": capability["capability_id"],
                    "stage": "pre_run",
                    "status": "approved",
                    "reviewer": "Independent Reviewer",
                    "date": "2026-08-07",
                    "independence_attestation": "No development or execution role.",
                }
            ],
        }
    )


def test_repository_contracts_validate() -> None:
    checked = validate_repository()
    assert "docs/artifacts/usage-readiness-matrix-v1.json" in checked


def test_matrix_rejects_omitted_or_duplicate_rows(matrix: dict) -> None:
    missing = copy.deepcopy(matrix)
    missing["rows"].pop()
    with pytest.raises(ContractError, match="Schema validation|row mismatch"):
        validate_matrix(missing)

    duplicate = copy.deepcopy(matrix)
    duplicate["rows"][-1] = copy.deepcopy(duplicate["rows"][0])
    with pytest.raises(ContractError, match="duplicate"):
        validate_matrix(duplicate)


def test_matrix_rejects_artifact_mismatch(matrix: dict) -> None:
    matrix["rows"][0]["artifact_identity"]["source_commit"] = "0" * 40
    with pytest.raises(ContractError, match="artifact identity mismatch"):
        validate_matrix(matrix)


def test_matrix_rejects_global_manifest_mismatch(matrix: dict) -> None:
    matrix["artifact"]["source_commit"] = "0" * 40
    for row in matrix["rows"]:
        row["artifact_identity"]["source_commit"] = "0" * 40
    with pytest.raises(ContractError, match="differs from the artifact manifest"):
        validate_matrix(matrix)


def test_matrix_rejects_missing_negative_result_evidence(matrix: dict) -> None:
    tta = next(row for row in matrix["rows"] if row["capability_id"] == "cytoanvi.tta_ood")
    tta["evidence_links"] = []
    with pytest.raises(ContractError, match="negative-result evidence"):
        validate_matrix(matrix)


def test_matrix_rejects_missing_or_external_evidence(matrix: dict) -> None:
    core = next(row for row in matrix["rows"] if row["capability_id"] == "cytoanvi.core")
    core["evidence_links"] = ["docs/does-not-exist.md"]
    with pytest.raises(ContractError, match="missing or non-regular evidence link"):
        validate_matrix(matrix)

    core["evidence_links"] = ["../outside-repository.md"]
    with pytest.raises(ContractError, match="outside the repository"):
        validate_matrix(matrix)

    core["evidence_links"] = [".living/local-review.md"]
    with pytest.raises(ContractError, match="local-only evidence"):
        validate_matrix(matrix)


def test_matrix_rejects_unsigned_or_nonterminal_promotion(matrix: dict, tmp_path: Path) -> None:
    manifest_path = tmp_path / "docs/artifacts/cytoanvi-0.2.0/manifest.json"
    manifest_path.parent.mkdir(parents=True)

    def write_matching_manifest() -> None:
        manifest_path.write_text(
            json.dumps(
                {
                    "candidate": {"version": matrix["artifact"]["version"]},
                    "wheel": {"sha256": matrix["artifact"]["wheel_sha256"]},
                    "source": {"commit": matrix["artifact"]["source_commit"]},
                    "installed_acceptance": {"status": matrix["artifact"]["installed_acceptance"]},
                }
            )
        )

    for row in matrix["rows"]:
        for raw_link in row["evidence_links"]:
            evidence_path = tmp_path / raw_link
            if evidence_path != manifest_path:
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                evidence_path.write_text("fixture evidence\n")
    write_matching_manifest()

    core = next(row for row in matrix["rows"] if row["capability_id"] == "cytoanvi.core")
    core["decision"] = "conditional-go"
    with pytest.raises(ContractError, match="installed acceptance"):
        validate_matrix(matrix, tmp_path)

    matrix["artifact"]["installed_acceptance"] = "passed"
    write_matching_manifest()
    with pytest.raises(ContractError, match="terminal P2 evidence"):
        validate_matrix(matrix, tmp_path)

    core["scientific_result"] = "terminal_positive"
    with pytest.raises(ContractError, match="human signature"):
        validate_matrix(matrix, tmp_path)


def test_passed_artifact_bundle_is_cross_bound(
    passed_artifact_bundle: tuple[dict, dict, dict],
) -> None:
    validate_artifact_bundle(*passed_artifact_bundle)


def test_passed_acceptance_requires_successful_complete_build(
    passed_artifact_bundle: tuple[dict, dict, dict],
) -> None:
    manifest, inventory, receipt = passed_artifact_bundle
    manifest["build"]["status"] = "not_built"
    with pytest.raises(ContractError, match="complete successful build state"):
        validate_artifact_bundle(manifest, inventory, receipt)


def test_passed_build_rejects_missing_or_mismatched_inventory(
    passed_artifact_bundle: tuple[dict, dict, dict],
) -> None:
    manifest, inventory, receipt = passed_artifact_bundle
    manifest["source"]["tree"] = None
    with pytest.raises(ContractError, match="null or empty"):
        validate_artifact_bundle(manifest, inventory, receipt)

    manifest["source"]["tree"] = "4" * 40
    manifest["inventory"]["wheel_files"] = list(reversed(manifest["inventory"]["wheel_files"]))
    with pytest.raises(ContractError, match="Wheel file list differs"):
        validate_artifact_bundle(manifest, inventory, receipt)

    manifest["inventory"]["wheel_files"] = [entry["path"] for entry in inventory["wheel_files"]]
    del inventory["record"]
    with pytest.raises(ContractError, match="Schema validation"):
        validate_artifact_bundle(manifest, inventory, receipt)


def test_passed_build_rejects_payload_or_record_byte_mismatch(
    passed_artifact_bundle: tuple[dict, dict, dict],
) -> None:
    manifest, inventory, receipt = passed_artifact_bundle
    inventory["source_files"][0]["sha256"] = "8" * 64
    with pytest.raises(ContractError, match="changed=.*cytoanvi/__init__.py"):
        validate_artifact_bundle(manifest, inventory, receipt)

    inventory["source_files"][0]["sha256"] = "7" * 64
    inventory["record"]["lines"][0] = "cytoanvi/__init__.py,sha256=wrong,1"
    with pytest.raises(ContractError, match="RECORD SHA-256 differs"):
        validate_artifact_bundle(manifest, inventory, receipt)


def test_passed_receipt_rejects_cross_state_failures(
    passed_artifact_bundle: tuple[dict, dict, dict],
) -> None:
    manifest, inventory, receipt = passed_artifact_bundle

    broken = copy.deepcopy(receipt)
    broken["artifact"]["source_commit"] = "9" * 40
    _reseal(broken)
    with pytest.raises(ContractError, match="does not bind the artifact tuple"):
        validate_artifact_bundle(manifest, inventory, broken)

    broken = copy.deepcopy(receipt)
    broken["dependency_authority"]["verified"] = False
    _reseal(broken)
    with pytest.raises(ContractError, match="verified dependency authority"):
        validate_artifact_bundle(manifest, inventory, broken)

    broken = copy.deepcopy(receipt)
    broken["isolation"]["outside_checkout"] = False
    _reseal(broken)
    with pytest.raises(ContractError, match="false isolation"):
        validate_artifact_bundle(manifest, inventory, broken)

    broken = copy.deepcopy(receipt)
    broken["execution"]["exit_status"] = 1
    _reseal(broken)
    with pytest.raises(ContractError, match="exit status 0"):
        validate_artifact_bundle(manifest, inventory, broken)

    broken = copy.deepcopy(receipt)
    broken["checks"] = broken["checks"][1:]
    _reseal(broken)
    with pytest.raises(ContractError, match="omits mandatory checks"):
        validate_artifact_bundle(manifest, inventory, broken)

    broken = copy.deepcopy(receipt)
    broken["receipt_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="digest is invalid"):
        validate_artifact_bundle(manifest, inventory, broken)

    broken = copy.deepcopy(receipt)
    broken["status"] = "failed"
    _reseal(broken)
    with pytest.raises(ContractError, match="status differs"):
        validate_artifact_bundle(manifest, inventory, broken)


def test_post_build_nonpassing_receipt_must_bind_artifact_and_digest(
    passed_artifact_bundle: tuple[dict, dict, dict],
) -> None:
    manifest, inventory, receipt = passed_artifact_bundle
    manifest["installed_acceptance"].update(
        {
            "status": "blocked_dependency_authority",
            "blocker": "fixture blocker",
        }
    )
    receipt["status"] = "blocked_dependency_authority"
    receipt["artifact"].update({"wheel_sha256": None, "source_commit": None})
    _reseal(receipt)
    with pytest.raises(ContractError, match="does not bind the artifact tuple"):
        validate_artifact_bundle(manifest, inventory, receipt)

    receipt["artifact"].update(
        {
            "wheel_sha256": manifest["wheel"]["sha256"],
            "source_commit": manifest["source"]["commit"],
        }
    )
    receipt["receipt_sha256"] = None
    with pytest.raises(ContractError, match="has no valid digest"):
        validate_artifact_bundle(manifest, inventory, receipt)


def test_terminal_manifest_rejects_partial_grid(tmp_path: Path) -> None:
    output = tmp_path / "terminal-output.json"
    output.write_bytes(b"fixture\n")
    manifest = {
        "schema_version": "usage-readiness-terminal-run-v1",
        "run_id": "fixture",
        "capability_id": "cytoanvi.core",
        "protocol_sha256": "1" * 64,
        "artifact": {"version": "0.2.0", "wheel_sha256": "2" * 64, "source_commit": "3" * 40},
        "expected_grid": ["seed-0", "seed-1", "seed-2"],
        "observed_grid": ["seed-0", "seed-1"],
        "status": "terminal_success",
        "terminal_evidence": {
            "execution_backend": "scheduler",
            "backend_justification": None,
            "process_exit": 0,
            "scheduler_accounting": {"state": "COMPLETED"},
            "started_at": "2026-08-07T00:00:00Z",
            "completed_at": "2026-08-07T01:00:00Z",
        },
        "outputs": [
            {
                "path": str(output),
                "sha256": sha256_file(output),
                "size_bytes": output.stat().st_size,
            }
        ],
        "negative_results_retained": True,
    }
    with pytest.raises(ContractError, match="incomplete or extra"):
        validate_terminal_manifest(manifest)


def test_terminal_manifest_requires_backend_evidence(tmp_path: Path) -> None:
    output = tmp_path / "terminal-output.json"
    output.write_bytes(b"fixture\n")
    manifest = {
        "schema_version": "usage-readiness-terminal-run-v1",
        "run_id": "fixture",
        "capability_id": "cytoanvi.core",
        "protocol_sha256": "1" * 64,
        "artifact": {"version": "0.2.0", "wheel_sha256": "2" * 64, "source_commit": "3" * 40},
        "expected_grid": ["seed-0"],
        "observed_grid": ["seed-0"],
        "status": "terminal_success",
        "terminal_evidence": {
            "execution_backend": "scheduler",
            "backend_justification": None,
            "process_exit": 0,
            "scheduler_accounting": {},
            "started_at": "2026-08-07T00:00:00Z",
            "completed_at": "2026-08-07T01:00:00Z",
        },
        "outputs": [
            {
                "path": str(output),
                "sha256": sha256_file(output),
                "size_bytes": output.stat().st_size,
            }
        ],
        "negative_results_retained": True,
    }
    with pytest.raises(ContractError, match="Schema validation|scheduler accounting"):
        validate_terminal_manifest(manifest)

    manifest["terminal_evidence"].update(
        {
            "execution_backend": "local",
            "backend_justification": None,
            "scheduler_accounting": None,
        }
    )
    with pytest.raises(ContractError, match="backend justification"):
        validate_terminal_manifest(manifest)


def test_protocol_rejects_frozen_capability_without_both_controls() -> None:
    protocol = load_json(ROOT / "benchmarks/cytoanvi/usage_readiness_contract_v1.json")
    protocol["artifact"].update(
        {"wheel_sha256": _fixture_digest("wheel"), "source_commit": "2" * 40}
    )
    capability = _complete_frozen_capability(protocol)
    capability["controls"]["positive"] = []
    _approve_protocol_capability(protocol, capability)
    with pytest.raises(ContractError, match="Schema validation|positive or negative controls"):
        validate_protocol(protocol)


def test_protocol_approved_review_requires_named_dated_authority_and_receipt() -> None:
    protocol = load_json(ROOT / "benchmarks/mrtotalvi/usage_readiness_contract_v1.json")
    protocol["independent_review"].update(
        {
            "status": "approved",
            "reviewer": None,
            "date": None,
            "independence_rule": None,
            "capability_reviews": [],
        }
    )
    with pytest.raises(ContractError, match="Schema validation"):
        validate_protocol(protocol)

    protocol = load_json(ROOT / "benchmarks/mrtotalvi/usage_readiness_contract_v1.json")
    protocol["artifact"].update(
        {"wheel_sha256": _fixture_digest("wheel"), "source_commit": "2" * 40}
    )
    _complete_frozen_capability(protocol)
    protocol["independent_review"].update(
        {
            "status": "approved",
            "reviewer": "Independent Reviewer",
            "date": "2026-08-07",
            "independence_rule": "Reviewer did not develop or execute the evaluated capability.",
            "capability_reviews": [
                {
                    "capability_id": "mrtotalvi.embeddings",
                    "stage": "pre_run",
                    "status": "approved",
                    "reviewer": "Independent Reviewer",
                    "date": "2026-08-07",
                    "independence_attestation": "No development or execution role.",
                }
            ],
        }
    )
    with pytest.raises(ContractError, match="lacks one approved review receipt"):
        validate_protocol(protocol)


def test_terminal_grid_is_exact_protocol_arm_seed_cross_product() -> None:
    entry = {
        "capability_id": "mrtotalvi.core",
        "rng_streams": {"training_seeds": [0, 1, 2]},
        "compute_budget": {"evaluation_grid": ["B0", "B1"]},
    }
    required = [
        "B0-seed-0",
        "B0-seed-1",
        "B0-seed-2",
        "B1-seed-0",
        "B1-seed-1",
        "B1-seed-2",
    ]
    assert _protocol_run_grid(entry) == required
    terminal = {"expected_grid": ["B0-seed-0"], "observed_grid": ["B0-seed-0"]}
    with pytest.raises(ContractError, match="does not exactly match"):
        _validate_protocol_run_grid(terminal, entry)

    terminal.update({"expected_grid": required, "observed_grid": required})
    _validate_protocol_run_grid(terminal, entry)


def test_protocol_rejects_unreviewed_frozen_capability() -> None:
    protocol_path = ROOT / "benchmarks/cytoanvi/usage_readiness_contract_v1.json"
    if not protocol_path.exists():
        pytest.skip("protocol is added in the P2 governance step")
    protocol = json.loads(protocol_path.read_text())
    protocol["artifact"].update(
        {"wheel_sha256": _fixture_digest("wheel"), "source_commit": "2" * 40}
    )
    capability = _complete_frozen_capability(protocol)
    capability["pre_run_review"] = "pending"
    with pytest.raises(ContractError, match="Schema validation|pre-run review"):
        validate_protocol(protocol)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("cohorts", [{}]),
        ("split_and_leakage_boundary", {}),
        ("rng_streams", {"training_seeds": [0, 1, 2], "streams": {}}),
        ("primary_endpoint", {}),
        ("numeric_margin", {}),
        ("donor_level_uncertainty", {}),
        ("multiplicity", {}),
        ("no_call_policy", {}),
        ("immutable_outputs", {}),
    ],
)
def test_protocol_rejects_structural_placeholders_in_frozen_capability(
    field: str, replacement: object
) -> None:
    protocol = load_json(ROOT / "benchmarks/cytoanvi/usage_readiness_contract_v1.json")
    protocol["artifact"].update(
        {"wheel_sha256": _fixture_digest("wheel"), "source_commit": "2" * 40}
    )
    capability = _complete_frozen_capability(protocol)
    _approve_protocol_capability(protocol, capability)
    capability[field] = replacement
    with pytest.raises(ContractError, match="Schema validation|Frozen protocol"):
        validate_protocol(protocol)


def test_protocol_rejects_semantic_placeholders_and_unsealed_output_schema() -> None:
    protocol = load_json(ROOT / "benchmarks/cytoanvi/usage_readiness_contract_v1.json")
    protocol["artifact"].update(
        {"wheel_sha256": _fixture_digest("wheel"), "source_commit": "2" * 40}
    )
    capability = _complete_frozen_capability(protocol)
    _approve_protocol_capability(protocol, capability)

    capability["cohorts"][0]["id"] = "placeholder"
    with pytest.raises(ContractError, match="placeholder cohort id"):
        validate_protocol(protocol)

    capability = _complete_frozen_capability(protocol)
    capability["rng_streams"]["streams"] = {
        "split": 101,
        "training": 101,
        "uncertainty": 303,
    }
    with pytest.raises(ContractError, match="reuses an RNG seed"):
        validate_protocol(protocol)

    capability["rng_streams"]["streams"]["training"] = 202
    capability["immutable_outputs"]["files"][0]["schema_sha256"] = _fixture_digest(
        "wrong schema"
    )
    with pytest.raises(ContractError, match="output schema hash is not exact"):
        validate_protocol(protocol)


@pytest.mark.parametrize(
    ("field", "key"),
    [
        ("numeric_margin", "value"),
        ("donor_level_uncertainty", "confidence_level"),
        ("multiplicity", "alpha"),
    ],
)
def test_protocol_rejects_nonfinite_frozen_numeric_authority(field: str, key: str) -> None:
    protocol = load_json(ROOT / "benchmarks/cytoanvi/usage_readiness_contract_v1.json")
    protocol["artifact"].update(
        {"wheel_sha256": _fixture_digest("wheel"), "source_commit": "2" * 40}
    )
    capability = _complete_frozen_capability(protocol)
    _approve_protocol_capability(protocol, capability)
    capability[field][key] = float("nan")
    with pytest.raises(ContractError, match="non-finite"):
        validate_protocol(protocol)


def test_terminal_outputs_match_frozen_paths_and_sealed_json_schema(tmp_path: Path) -> None:
    schema_path = tmp_path / "schemas/fixture-output.schema.json"
    schema_path.parent.mkdir(parents=True)
    schema_path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "required": ["ok"],
                "properties": {"ok": {"const": True}},
            }
        )
    )
    output_path = tmp_path / "results/fixture.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"ok": true}\n')
    relative_output = output_path.relative_to(tmp_path).as_posix()
    entry = {
        "capability_id": "cytoanvi.core",
        "immutable_outputs": {
            "files": [
                {
                    "path": relative_output,
                    "schema": schema_path.relative_to(tmp_path).as_posix(),
                    "schema_sha256": sha256_file(schema_path),
                }
            ]
        },
    }
    terminal = {"outputs": [{"path": relative_output}]}
    _validate_terminal_outputs(terminal, entry, tmp_path)

    terminal["outputs"][0]["path"] = "results/drifted.json"
    with pytest.raises(ContractError, match="paths differ from the frozen protocol"):
        _validate_terminal_outputs(terminal, entry, tmp_path)

    terminal["outputs"][0]["path"] = relative_output
    output_path.write_text('{"unexpected": true}\n')
    with pytest.raises(ContractError, match="Schema validation"):
        _validate_terminal_outputs(terminal, entry, tmp_path)


def test_frozen_protocol_artifact_binds_successful_installed_manifest(
    passed_artifact_bundle: tuple[dict, dict, dict],
) -> None:
    manifest, _inventory, _receipt = passed_artifact_bundle
    protocol = load_json(ROOT / "benchmarks/cytoanvi/usage_readiness_contract_v1.json")
    protocol["capabilities"][0]["freeze_status"] = "frozen"
    protocol["artifact"] = {
        "version": manifest["candidate"]["version"],
        "wheel_sha256": manifest["wheel"]["sha256"],
        "source_commit": manifest["source"]["commit"],
    }
    _validate_frozen_protocol_artifact(protocol, manifest)

    protocol["artifact"]["wheel_sha256"] = _fixture_digest("stale wheel")
    with pytest.raises(ContractError, match="differs from the accepted artifact"):
        _validate_frozen_protocol_artifact(protocol, manifest)

    protocol["artifact"]["wheel_sha256"] = manifest["wheel"]["sha256"]
    manifest["build"]["status"] = "not_built"
    with pytest.raises(ContractError, match="successful build and passed installed acceptance"):
        _validate_frozen_protocol_artifact(protocol, manifest)
