#!/usr/bin/env python3
"""Validate usage-readiness schemas, protocols, manifests, and capability decisions."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from jsonschema import Draft202012Validator, FormatChecker

if TYPE_CHECKING:
    from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs" / "artifacts" / "schemas"
CANONICAL_MANIFEST_PATH = "docs/artifacts/cytoanvi-0.2.0/manifest.json"
CANONICAL_INVENTORY_PATH = "docs/artifacts/cytoanvi-0.2.0/inventory.json"
CANONICAL_RECEIPT_PATH = "docs/artifacts/cytoanvi-0.2.0/installed-acceptance.json"
CANONICAL_CLAIM_PATH = (
    "docs/artifacts/cytoanvi-0.2.0/.cytoanvi-0.2.0-build-ordinal-1.claim.json"
)

CAPABILITY_IDS = (
    "cytoanvi.core",
    "cytoanvi.mapping.same_panel",
    "cytoanvi.mapping.panel_divergent",
    "cytoanvi.hierarchy",
    "cytoanvi.integration_clustering",
    "cytoanvi.tta_ood",
    "cytoanvi.continual",
    "cytoanvi.mapqc",
    "cytoanvi.artifact",
    "mrtotalvi.core",
    "mrtotalvi.embeddings",
    "mrtotalvi.prior_choice",
    "mrtotalvi.label_supervision",
    "mrtotalvi.da",
    "mrtotalvi.legacy_de",
    "mrtotalvi.centered_v2",
    "mrtotalvi.streaming",
    "mrtotalvi.new_sample_inference",
    "mrtotalvi.artifact",
)

MANDATORY_ACCEPTANCE_CHECKS = frozenset(
    {
        "dependency_authority",
        "locked_dependencies",
        "pip_check",
        "namespace_identity",
        "installed_inventory",
        "cytoanvi_core",
        "tree_arches",
        "mrtotalvi_core",
    }
)

ACCEPTANCE_INPUT_PATHS = frozenset(
    {
        "scripts/accept_usage_readiness_wheel",
        "scripts/usage_readiness_installed_smoke.py",
        "vignettes/cytoanvi_treearches_synthetic.py",
    }
)


class ContractError(ValueError):
    """Raised when a cross-file readiness invariant is violated."""


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and retain a useful path in failures."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    """Hash a regular file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_against_schema(instance: dict[str, Any], schema_path: Path) -> None:
    """Validate one object against a strict Draft 2020-12 schema."""
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.absolute_path))
    if errors:
        rendered = "; ".join(
            f"/{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
            for error in errors
        )
        raise ContractError(f"Schema validation failed against {schema_path.name}: {rendered}")


def _artifact_tuple(artifact: dict[str, Any]) -> tuple[Any, Any, Any]:
    return artifact.get("version"), artifact.get("wheel_sha256"), artifact.get("source_commit")


def _receipt_digest(receipt: dict[str, Any]) -> str:
    payload = dict(receipt)
    payload["receipt_sha256"] = None
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _record_digest_from_hex(hex_digest: str) -> str:
    encoded = base64.urlsafe_b64encode(bytes.fromhex(hex_digest)).decode().rstrip("=")
    return f"sha256={encoded}"


def _validate_record_inventory(inventory: dict[str, Any]) -> None:
    """Bind every stored RECORD row to the detailed wheel inventory."""
    record = inventory["record"]
    rows = list(csv.reader(record["lines"]))
    if any(len(row) != 3 for row in rows):
        raise ContractError(
            "Wheel RECORD inventory does not contain exactly three columns per row"
        )
    paths = [row[0] for row in rows]
    if len(paths) != len(set(paths)):
        raise ContractError("Wheel RECORD inventory contains duplicate paths")
    wheel_by_path = {entry["path"]: entry for entry in inventory["wheel_files"]}
    dist_info_prefix = "cytoanvi-0.2.0.dist-info/"
    unexpected = sorted(
        path
        for path in wheel_by_path
        if not path.startswith(("cytoanvi/", "scvi/", dist_info_prefix))
    )
    if unexpected:
        raise ContractError(f"Wheel inventory contains non-owned top-level payloads: {unexpected}")
    if set(paths) != set(wheel_by_path):
        missing = sorted(set(wheel_by_path) - set(paths))
        extra = sorted(set(paths) - set(wheel_by_path))
        raise ContractError(
            "Wheel RECORD path set differs from detailed inventory: "
            f"missing={missing}, extra={extra}"
        )
    if not record["path"].endswith(".dist-info/RECORD"):
        raise ContractError("Wheel RECORD path is not a distribution RECORD")
    for path, encoded_hash, encoded_size in rows:
        if path == record["path"]:
            if encoded_hash or encoded_size:
                raise ContractError("Wheel RECORD self-entry does not omit hash and size")
            continue
        entry = wheel_by_path[path]
        if encoded_hash != _record_digest_from_hex(entry["sha256"]):
            raise ContractError(f"Wheel RECORD SHA-256 differs for {path}")
        if encoded_size != str(entry["size_bytes"]):
            raise ContractError(f"Wheel RECORD size differs for {path}")


def validate_artifact_bundle(
    manifest: dict[str, Any],
    inventory: dict[str, Any],
    receipt: dict[str, Any],
    root: Path | None = None,
) -> None:
    """Validate cross-file artifact identity and acceptance-state invariants."""
    validate_against_schema(manifest, SCHEMA_DIR / "artifact-manifest.schema.json")
    validate_against_schema(inventory, SCHEMA_DIR / "artifact-inventory.schema.json")
    validate_against_schema(receipt, SCHEMA_DIR / "installed-acceptance-receipt.schema.json")
    if manifest["inventory"]["details_path"] != CANONICAL_INVENTORY_PATH:
        raise ContractError("Artifact manifest does not use the canonical inventory path")
    if manifest["installed_acceptance"]["receipt"] != CANONICAL_RECEIPT_PATH:
        raise ContractError("Artifact manifest does not use the canonical receipt path")
    if manifest["build"]["claim_path"] != CANONICAL_CLAIM_PATH:
        raise ContractError("Artifact manifest does not use the canonical build claim path")
    if root is not None:
        inventory_path = root / CANONICAL_INVENTORY_PATH
        receipt_path = root / CANONICAL_RECEIPT_PATH
        for label, path in (("inventory", inventory_path), ("receipt", receipt_path)):
            if path.is_symlink() or not path.is_file():
                raise ContractError(f"Canonical artifact {label} is missing or non-regular")
        if manifest["inventory"]["details_sha256"] is not None and sha256_file(
            inventory_path
        ) != manifest["inventory"]["details_sha256"]:
            raise ContractError("Canonical inventory file hash differs from the manifest")
        if manifest["installed_acceptance"]["receipt_sha256"] is not None and sha256_file(
            receipt_path
        ) != manifest["installed_acceptance"]["receipt_sha256"]:
            raise ContractError("Canonical receipt file hash differs from the manifest")
    if manifest["build"]["status"] == "passed":
        source = manifest["source"]
        build = manifest["build"]
        wheel = manifest["wheel"]
        inventory_summary = manifest["inventory"]
        missing_source = [key for key in ("commit", "tree") if not source[key]]
        missing_build = [
            key
            for key in (
                "attempt_id",
                "candidate_dir",
                "pid",
                "command",
                "python",
                "environment_digest",
                "started_at",
                "completed_at",
            )
            if not build[key]
        ]
        missing_wheel = [
            key
            for key in (
                "path",
                "filename",
                "sha256",
                "size_bytes",
                "metadata_name",
                "metadata_version",
            )
            if not wheel[key]
        ]
        if missing_source or missing_build or missing_wheel:
            raise ContractError(
                "Passed build has null or empty identity/evidence fields: "
                f"source={missing_source}, build={missing_build}, wheel={missing_wheel}"
            )
        if source["commit"] == "0" * 40 or source["tree"] == "0" * 40:
            raise ContractError("Passed build uses a zero source commit or tree sentinel")
        if manifest["source"]["clean"] is not True:
            raise ContractError("Passed build does not record a clean source tree")
        if wheel["metadata_name"] != "cytoanvi" or wheel["metadata_version"] != "0.2.0":
            raise ContractError("Passed build wheel metadata does not identify cytoanvi 0.2.0")
        if Path(wheel["path"]).name != wheel["filename"]:
            raise ContractError("Passed build wheel path and filename differ")
        if inventory_summary["source_vs_wheel"] != "match":
            raise ContractError("Passed build does not have matching source/wheel inventories")
        if not inventory_summary["details_sha256"]:
            raise ContractError("Passed build does not bind the canonical inventory file hash")
        if not manifest["installed_acceptance"]["receipt_sha256"]:
            raise ContractError("Passed build does not bind the canonical receipt file hash")
        if inventory["source_commit"] != source["commit"]:
            raise ContractError("Inventory source commit differs from artifact manifest")
        if inventory["wheel_sha256"] != wheel["sha256"]:
            raise ContractError("Inventory wheel SHA differs from artifact manifest")
        if inventory_summary["record_sha256"] != inventory["record"]["sha256"]:
            raise ContractError("RECORD digest differs between inventory and artifact manifest")
        if inventory_summary["record_entries"] != len(inventory["record"]["lines"]):
            raise ContractError(
                "RECORD entry count differs between inventory and artifact manifest"
            )
        if not inventory["record"]["lines"]:
            raise ContractError("Passed build has an empty wheel RECORD inventory")

        detailed_wheel_paths = [entry["path"] for entry in inventory["wheel_files"]]
        detailed_source_paths = [entry["path"] for entry in inventory["source_files"]]
        if not detailed_wheel_paths or not detailed_source_paths:
            raise ContractError("Passed build has an empty source or wheel file inventory")
        if inventory_summary["wheel_files"] != detailed_wheel_paths:
            raise ContractError("Wheel file list differs between inventory and artifact manifest")
        if inventory_summary["source_files"] != detailed_source_paths:
            raise ContractError("Source file list differs between inventory and artifact manifest")
        if len(detailed_wheel_paths) != len(set(detailed_wheel_paths)) or len(
            detailed_source_paths
        ) != len(set(detailed_source_paths)):
            raise ContractError("Passed build inventory contains duplicate paths")
        acceptance_inputs = inventory["acceptance_inputs"]
        acceptance_paths = [entry["path"] for entry in acceptance_inputs]
        if set(acceptance_paths) != ACCEPTANCE_INPUT_PATHS or len(acceptance_paths) != len(
            ACCEPTANCE_INPUT_PATHS
        ):
            raise ContractError(
                "Passed build acceptance-input path set differs from the mandatory harness inputs"
            )
        if inventory_summary["acceptance_inputs"] != acceptance_inputs:
            raise ContractError(
                "Acceptance-input path/hash/size inventory differs from the artifact manifest"
            )
        if root is not None:
            for entry in acceptance_inputs:
                relative = Path(entry["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ContractError("Acceptance input path escapes the repository")
                path = root / relative
                if path.is_symlink() or not path.is_file():
                    raise ContractError(
                        f"Acceptance input is missing or non-regular: {entry['path']}"
                    )
                if (
                    sha256_file(path) != entry["sha256"]
                    or path.stat().st_size != entry["size_bytes"]
                ):
                    raise ContractError(
                        f"Acceptance input drifted after the candidate build: {entry['path']}"
                    )
        _validate_record_inventory(inventory)
        record_files = [
            entry
            for entry in inventory["wheel_files"]
            if entry["path"] == inventory["record"]["path"]
        ]
        if len(record_files) != 1 or record_files[0]["sha256"] != inventory["record"]["sha256"]:
            raise ContractError("Detailed wheel inventory does not bind the recorded RECORD file")
        source_by_path = {
            entry["path"]: (entry["sha256"], entry["size_bytes"])
            for entry in inventory["source_files"]
        }
        package_wheel_by_path = {
            entry["path"]: (entry["sha256"], entry["size_bytes"])
            for entry in inventory["wheel_files"]
            if entry["path"].startswith(("cytoanvi/", "scvi/"))
        }
        if package_wheel_by_path != source_by_path:
            missing = sorted(set(source_by_path) - set(package_wheel_by_path))
            extra = sorted(set(package_wheel_by_path) - set(source_by_path))
            changed = sorted(
                path
                for path in set(source_by_path) & set(package_wheel_by_path)
                if source_by_path[path] != package_wheel_by_path[path]
            )
            raise ContractError(
                "Detailed source and wheel package inventories differ: "
                f"missing={missing}, extra={extra}, changed={changed}"
            )
    acceptance = manifest["installed_acceptance"]["status"]
    if receipt["status"] != acceptance:
        raise ContractError("Installed-acceptance receipt status differs from artifact manifest")
    if (
        receipt["acceptance_attempt_id"]
        != manifest["installed_acceptance"]["acceptance_attempt_id"]
    ):
        raise ContractError("Installed-acceptance attempt identity differs across files")
    if receipt["receipt_sha256"] is not None and receipt["receipt_sha256"] != _receipt_digest(
        receipt
    ):
        raise ContractError("Installed-acceptance receipt digest is invalid")
    if manifest["build"]["status"] == "passed":
        if _artifact_tuple(receipt["artifact"]) != (
            manifest["candidate"]["version"],
            manifest["wheel"]["sha256"],
            manifest["source"]["commit"],
        ):
            raise ContractError("Post-build acceptance receipt does not bind the artifact tuple")
        if receipt["receipt_sha256"] is None:
            raise ContractError("Post-build acceptance receipt has no valid digest")
    if acceptance == "passed":
        if manifest["build"]["status"] != "passed":
            raise ContractError(
                "Passed installed acceptance requires a complete successful build state"
            )
        if not receipt["acceptance_attempt_id"]:
            raise ContractError("Passed installed acceptance lacks an attempt identity")
        if _artifact_tuple(receipt["artifact"]) != (
            manifest["candidate"]["version"],
            manifest["wheel"]["sha256"],
            manifest["source"]["commit"],
        ):
            raise ContractError("Acceptance receipt artifact differs from artifact manifest")
        if receipt["receipt_sha256"] is None:
            raise ContractError("Passed installed acceptance has no receipt digest")
        authority = receipt["dependency_authority"]
        manifest_authority = manifest["dependency_authority"]
        if authority["verified"] is not True or not authority["path"] or not authority["sha256"]:
            raise ContractError("Passed installed acceptance lacks verified dependency authority")
        if manifest_authority != {
            "status": "verified",
            "kind": "hashed_lock_and_wheelhouse",
            "path": authority["path"],
            "sha256": authority["sha256"],
        }:
            raise ContractError("Acceptance dependency authority differs from artifact manifest")
        if not all(receipt["isolation"].values()):
            raise ContractError("Passed installed acceptance has a false isolation assertion")
        if receipt["execution"]["exit_status"] != 0:
            raise ContractError("Passed installed acceptance does not have exit status 0")
        check_ids = [check["id"] for check in receipt["checks"]]
        if len(check_ids) != len(set(check_ids)):
            raise ContractError("Passed installed acceptance contains duplicate check IDs")
        missing_checks = sorted(MANDATORY_ACCEPTANCE_CHECKS - set(check_ids))
        if missing_checks:
            raise ContractError(
                f"Passed installed acceptance omits mandatory checks: {missing_checks}"
            )
        if any(check["status"] != "passed" for check in receipt["checks"]):
            raise ContractError("Passed installed acceptance contains a non-passing check")
        installed_acceptance = manifest["installed_acceptance"]
        if not installed_acceptance["receipt"] or installed_acceptance["blocker"] is not None:
            raise ContractError(
                "Passed artifact manifest lacks a receipt path or retains a blocker"
            )


def validate_terminal_manifest(
    manifest: dict[str, Any],
    expected_artifact: dict[str, Any] | None = None,
    root: Path = ROOT,
) -> None:
    """Reject partial terminal grids and artifact drift."""
    validate_against_schema(manifest, SCHEMA_DIR / "terminal-run-manifest.schema.json")
    status = manifest["status"]
    if status in {"terminal_success", "terminal_negative", "terminal_inconclusive"}:
        if set(manifest["expected_grid"]) != set(manifest["observed_grid"]):
            raise ContractError("Terminal scientific result has an incomplete or extra run grid")
        if not manifest["negative_results_retained"]:
            raise ContractError("Terminal manifest does not retain negative results")
        if manifest["terminal_evidence"]["process_exit"] != 0:
            raise ContractError("Terminal scientific result does not have process exit 0")
        output_paths = [output["path"] for output in manifest["outputs"]]
        if not output_paths or len(output_paths) != len(set(output_paths)):
            raise ContractError("Terminal scientific result lacks unique immutable outputs")
        for output in manifest["outputs"]:
            raw_path = Path(output["path"])
            if not raw_path.is_absolute() and ".." in raw_path.parts:
                raise ContractError("Terminal output path escapes its evidence root")
            output_path = raw_path if raw_path.is_absolute() else root / raw_path
            if output_path.is_symlink() or not output_path.is_file():
                raise ContractError(f"Terminal output is missing or non-regular: {raw_path}")
            if (
                sha256_file(output_path) != output["sha256"]
                or output_path.stat().st_size != output["size_bytes"]
            ):
                raise ContractError(f"Terminal output hash/size mismatch: {raw_path}")
    if status.startswith("terminal_"):
        evidence = manifest["terminal_evidence"]
        if (
            evidence["process_exit"] is None
            or evidence["started_at"] is None
            or evidence["completed_at"] is None
        ):
            raise ContractError(
                "Terminal scientific result lacks process-exit/completion evidence"
            )
        if evidence["execution_backend"] == "scheduler":
            accounting = evidence["scheduler_accounting"]
            if not isinstance(accounting, dict) or not accounting:
                raise ContractError(
                    "Scheduler terminal result lacks scheduler accounting evidence"
                )
            if accounting.get("state") != "COMPLETED":
                raise ContractError(
                    "Scheduler terminal result does not record state COMPLETED"
                )
        elif not evidence["backend_justification"]:
            raise ContractError("Local terminal result lacks an explicit backend justification")
    if expected_artifact is not None and _artifact_tuple(manifest["artifact"]) != _artifact_tuple(
        expected_artifact
    ):
        raise ContractError("Terminal manifest artifact identity differs from the matrix")


def _protocol_run_grid(entry: dict[str, Any]) -> list[str]:
    """Derive canonical run IDs from a frozen arm grid crossed with training seeds."""
    compute_budget = entry.get("compute_budget")
    evaluation_grid = (
        compute_budget.get("evaluation_grid") if isinstance(compute_budget, dict) else None
    )
    if (
        not isinstance(evaluation_grid, list)
        or not evaluation_grid
        or any(not isinstance(arm, str) or not arm.strip() for arm in evaluation_grid)
        or len(evaluation_grid) != len(set(evaluation_grid))
    ):
        raise ContractError(
            f"Frozen capability {entry['capability_id']} lacks a unique nonempty "
            "compute_budget.evaluation_grid"
        )
    if any("-seed-" in arm for arm in evaluation_grid):
        raise ContractError(
            f"Frozen capability {entry['capability_id']} evaluation arms must not embed seeds"
        )
    seeds = entry["rng_streams"]["training_seeds"]
    return [f"{arm}-seed-{seed}" for arm in evaluation_grid for seed in seeds]


def _validate_protocol_run_grid(
    terminal: dict[str, Any], entry: dict[str, Any]
) -> None:
    """Bind terminal expected and observed runs to the reviewed protocol cross-product."""
    required = _protocol_run_grid(entry)
    if terminal["expected_grid"] != required or terminal["observed_grid"] != required:
        raise ContractError(
            f"{entry['capability_id']} terminal run grid does not exactly match the "
            "frozen protocol evaluation_grid x training_seeds"
        )


def _require_capability_review(
    protocol: dict[str, Any], entry: dict[str, Any], stage: str
) -> None:
    """Require one named independent receipt for an approved capability review stage."""
    authority = protocol["independent_review"]
    if authority["status"] != "approved":
        raise ContractError(
            f"{entry['capability_id']} {stage} approval lacks approved independent authority"
        )
    matching = [
        receipt
        for receipt in authority["capability_reviews"]
        if receipt["capability_id"] == entry["capability_id"] and receipt["stage"] == stage
    ]
    if len(matching) != 1 or matching[0]["status"] != "approved":
        raise ContractError(
            f"{entry['capability_id']} {stage} approval lacks one approved review receipt"
        )
    if matching[0]["reviewer"] != authority["reviewer"]:
        raise ContractError(
            f"{entry['capability_id']} {stage} receipt is not bound to the independent reviewer"
        )


def _require_nonplaceholder(label: str, value: str) -> None:
    """Reject syntactically nonempty but unresolved frozen-protocol identifiers."""
    unresolved = {"none", "null", "placeholder", "tbd", "todo", "unknown", "unfrozen", "unset"}
    if value.strip().lower() in unresolved:
        raise ContractError(f"Frozen protocol has placeholder {label}: {value!r}")


def _require_exact_digest(label: str, value: str) -> None:
    """Reject obvious placeholder digests after schema-level shape validation."""
    if len(set(value)) == 1:
        raise ContractError(f"Frozen protocol has placeholder {label} digest")


def _validate_frozen_capability(entry: dict[str, Any], root: Path) -> None:
    """Semantically bind all authority needed before a scientific run can be frozen."""
    capability_id = entry["capability_id"]
    cohort_ids: list[str] = []
    for cohort in entry["cohorts"]:
        _require_nonplaceholder("cohort id", cohort["id"])
        _require_nonplaceholder("cohort role", cohort["role"])
        _require_exact_digest("cohort manifest", cohort["manifest_sha256"])
        cohort_ids.append(cohort["id"])
    if len(cohort_ids) != len(set(cohort_ids)):
        raise ContractError(f"Frozen capability {capability_id} has duplicate cohort IDs")

    split = entry["split_and_leakage_boundary"]
    for label in ("split_id", "group_key", "leakage_boundary"):
        _require_nonplaceholder(label, split[label])
    _require_exact_digest("split", split["split_sha256"])

    streams = entry["rng_streams"]["streams"]
    if len(streams.values()) != len(set(streams.values())):
        raise ContractError(
            f"Frozen capability {capability_id} reuses an RNG seed across named streams"
        )
    _require_nonplaceholder("compute budget id", entry["compute_budget"]["budget_id"])
    _protocol_run_grid(entry)

    endpoint = entry["primary_endpoint"]
    for label in ("name", "metric", "aggregation_unit"):
        _require_nonplaceholder(f"endpoint {label}", endpoint[label])
    _require_nonplaceholder("numeric-margin unit", entry["numeric_margin"]["unit"])
    if not math.isfinite(entry["numeric_margin"]["value"]):
        raise ContractError(f"Frozen capability {capability_id} has a non-finite numeric margin")
    uncertainty = entry["donor_level_uncertainty"]
    _require_nonplaceholder("uncertainty method", uncertainty["method"])
    _require_nonplaceholder("uncertainty biological unit", uncertainty["biological_unit"])
    if not math.isfinite(uncertainty["confidence_level"]):
        raise ContractError(
            f"Frozen capability {capability_id} has a non-finite confidence level"
        )
    multiplicity = entry["multiplicity"]
    _require_nonplaceholder("multiplicity method", multiplicity["method"])
    _require_nonplaceholder("multiplicity family", multiplicity["family"])
    if not math.isfinite(multiplicity["alpha"]):
        raise ContractError(f"Frozen capability {capability_id} has a non-finite alpha")

    controls = entry["controls"]
    for kind in ("positive", "negative"):
        for control in controls[kind]:
            _require_nonplaceholder(f"{kind} control", control)
    no_call = entry["no_call_policy"]
    _require_nonplaceholder("no-call policy id", no_call["policy_id"])
    for condition in no_call["conditions"]:
        _require_nonplaceholder("no-call condition", condition)

    output_paths: list[str] = []
    for output in entry["immutable_outputs"]["files"]:
        output_path = Path(output["path"])
        if output_path.is_absolute() or ".." in output_path.parts:
            raise ContractError(
                f"Frozen capability {capability_id} has an output path outside its evidence root"
            )
        schema_relative = Path(output["schema"])
        if schema_relative.is_absolute() or ".." in schema_relative.parts:
            raise ContractError(
                f"Frozen capability {capability_id} has an external output schema path"
            )
        schema_path = root / schema_relative
        if schema_path.is_symlink() or not schema_path.is_file():
            raise ContractError(
                f"Frozen capability {capability_id} output schema is missing or non-regular"
            )
        if sha256_file(schema_path) != output["schema_sha256"]:
            raise ContractError(
                f"Frozen capability {capability_id} output schema hash is not exact"
            )
        output_paths.append(output["path"])
    if len(output_paths) != len(set(output_paths)):
        raise ContractError(f"Frozen capability {capability_id} has duplicate output paths")


def validate_protocol(protocol: dict[str, Any], root: Path = ROOT) -> None:
    """Validate a P2 contract and its freeze authority."""
    validate_against_schema(protocol, SCHEMA_DIR / "scientific-protocol.schema.json")
    ids = [entry["capability_id"] for entry in protocol["capabilities"]]
    if len(ids) != len(set(ids)):
        raise ContractError("Protocol contains duplicate capability IDs")
    review = protocol["independent_review"]
    review_keys = [
        (receipt["capability_id"], receipt["stage"])
        for receipt in review["capability_reviews"]
    ]
    if len(review_keys) != len(set(review_keys)):
        raise ContractError("Protocol contains duplicate capability review receipts")
    unknown_reviews = sorted(
        capability_id for capability_id, _stage in review_keys if capability_id not in ids
    )
    if unknown_reviews:
        raise ContractError(
            f"Protocol contains review receipts for unknown capabilities: {unknown_reviews}"
        )
    if review["capability_reviews"] and review["status"] != "approved":
        raise ContractError("Protocol has capability review receipts without approved authority")
    prefix = f"{protocol['model']}."
    if any(not capability_id.startswith(prefix) for capability_id in ids):
        raise ContractError("Protocol contains a capability from the wrong model")
    expected_ids = {
        capability_id for capability_id in CAPABILITY_IDS if capability_id.startswith(prefix)
    }
    if set(ids) != expected_ids or len(ids) != len(expected_ids):
        raise ContractError(
            f"Protocol capability set mismatch: missing={sorted(expected_ids - set(ids))}, "
            f"extra={sorted(set(ids) - expected_ids)}"
        )
    if any(entry["freeze_status"] == "frozen" for entry in protocol["capabilities"]):
        artifact = protocol["artifact"]
        if not artifact["wheel_sha256"] or not artifact["source_commit"]:
            raise ContractError("Frozen capability lacks exact artifact identity")
    for entry in protocol["capabilities"]:
        seeds = set(entry["rng_streams"]["training_seeds"])
        if not {0, 1, 2}.issubset(seeds):
            raise ContractError(
                f"{entry['capability_id']} does not freeze minimum seeds [0, 1, 2]"
            )
        if entry["freeze_status"] == "frozen":
            required_frozen = (
                "cohorts",
                "biological_unit",
                "split_and_leakage_boundary",
                "compute_budget",
                "representation_semantics",
                "primary_endpoint",
                "numeric_margin",
                "donor_level_uncertainty",
                "multiplicity",
                "no_call_policy",
                "immutable_outputs",
            )
            missing = [key for key in required_frozen if not entry[key]]
            if missing or entry["blockers"]:
                raise ContractError(
                    f"Frozen capability {entry['capability_id']} has unresolved fields: {missing}"
                )
            _validate_frozen_capability(entry, root)
            controls = entry["controls"]
            if not controls["positive"] or not controls["negative"]:
                raise ContractError(
                    f"Frozen capability {entry['capability_id']} lacks positive or negative "
                    "controls"
                )
            if entry["pre_run_review"] != "approved":
                raise ContractError(
                    f"Frozen capability {entry['capability_id']} lacks pre-run review"
                )
            _require_capability_review(protocol, entry, "pre_run")
            if entry["post_run_review"] == "approved":
                _require_capability_review(protocol, entry, "post_run")
    if protocol["status"] == "frozen":
        if any(entry["freeze_status"] != "frozen" for entry in protocol["capabilities"]):
            raise ContractError("Protocol is frozen while at least one capability is not frozen")
        if protocol["independent_review"]["status"] != "approved":
            raise ContractError("Frozen protocol lacks independent approval")


def _validate_frozen_protocol_artifact(protocol: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Bind every pre-run frozen capability to the accepted canonical artifact."""
    if not any(entry["freeze_status"] == "frozen" for entry in protocol["capabilities"]):
        return
    if (
        manifest["build"]["status"] != "passed"
        or manifest["installed_acceptance"]["status"] != "passed"
    ):
        raise ContractError(
            "Frozen protocol requires a successful build and passed installed acceptance"
        )
    expected = {
        "version": manifest["candidate"]["version"],
        "wheel_sha256": manifest["wheel"]["sha256"],
        "source_commit": manifest["source"]["commit"],
    }
    if protocol["artifact"] != expected:
        raise ContractError("Frozen protocol artifact differs from the accepted artifact manifest")


def _validate_terminal_outputs(
    terminal: dict[str, Any], entry: dict[str, Any], root: Path
) -> None:
    """Bind terminal files to frozen paths and validate JSON bytes against sealed schemas."""
    frozen_outputs = entry["immutable_outputs"]["files"]
    frozen_paths = [output["path"] for output in frozen_outputs]
    terminal_paths = [output["path"] for output in terminal["outputs"]]
    if terminal_paths != frozen_paths:
        raise ContractError(
            f"{entry['capability_id']} terminal output paths differ from the frozen protocol"
        )
    for output in frozen_outputs:
        output_path = root / output["path"]
        if output_path.suffix != ".json":
            continue
        try:
            instance = load_json(output_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(
                f"{entry['capability_id']} terminal JSON output is unreadable: {output['path']}"
            ) from exc
        validate_against_schema(instance, root / output["schema"])


def _protocol_path(capability_id: str, root: Path) -> Path:
    """Return the sole machine protocol authorized for a capability namespace."""
    model = capability_id.split(".", maxsplit=1)[0]
    relative = {
        "cytoanvi": "benchmarks/cytoanvi/usage_readiness_contract_v1.json",
        "mrtotalvi": "benchmarks/mrtotalvi/usage_readiness_contract_v1.json",
    }.get(model)
    if relative is None:
        raise ContractError(f"No protocol namespace for capability {capability_id}")
    return root / relative


def _validate_terminal_row(
    row: dict[str, Any], global_artifact: dict[str, Any], root: Path
) -> None:
    """Bind one terminal matrix row to its exact reviewed protocol and run evidence."""
    if row["engineering_execution"] != "installed_passed":
        raise ContractError(
            f"{row['capability_id']} has terminal science without installed engineering pass"
        )
    protocol_path = _protocol_path(row["capability_id"], root)
    if protocol_path.is_symlink() or not protocol_path.is_file():
        raise ContractError(f"{row['capability_id']} lacks a regular machine protocol")
    protocol = load_json(protocol_path)
    validate_protocol(protocol, root)
    if _artifact_tuple(protocol["artifact"]) != _artifact_tuple(global_artifact):
        raise ContractError(f"{row['capability_id']} protocol has an artifact mismatch")
    entries = [
        entry
        for entry in protocol["capabilities"]
        if entry["capability_id"] == row["capability_id"]
    ]
    if len(entries) != 1:
        raise ContractError(f"{row['capability_id']} lacks one exact protocol entry")
    entry = entries[0]
    if (
        entry["freeze_status"] != "frozen"
        or entry["pre_run_review"] != "approved"
        or entry["post_run_review"] != "approved"
        or protocol["independent_review"]["status"] != "approved"
    ):
        raise ContractError(
            f"{row['capability_id']} terminal evidence lacks frozen independent pre/post review"
        )

    terminal_manifests = []
    for raw_link in row["evidence_links"]:
        link = root / raw_link
        if link.suffix != ".json":
            continue
        candidate = load_json(link)
        if candidate.get("schema_version") == "usage-readiness-terminal-run-v1":
            terminal_manifests.append(candidate)
    matching = [
        manifest
        for manifest in terminal_manifests
        if manifest["capability_id"] == row["capability_id"]
    ]
    if len(matching) != 1:
        raise ContractError(
            f"{row['capability_id']} lacks one exact matching terminal run manifest"
        )
    terminal = matching[0]
    validate_terminal_manifest(terminal, global_artifact, root)
    _validate_protocol_run_grid(terminal, entry)
    _validate_terminal_outputs(terminal, entry, root)
    expected_status = {
        "terminal_positive": "terminal_success",
        "terminal_negative": "terminal_negative",
        "terminal_inconclusive": "terminal_inconclusive",
    }[row["scientific_result"]]
    if terminal["status"] != expected_status:
        raise ContractError(
            f"{row['capability_id']} matrix result differs from terminal run status"
        )
    if terminal["protocol_sha256"] != sha256_file(protocol_path):
        raise ContractError(
            f"{row['capability_id']} terminal run does not bind the exact protocol hash"
        )
    if row["decision"] in {"go", "conditional-go"} and terminal["terminal_evidence"][
        "execution_backend"
    ] != "scheduler":
        raise ContractError(
            f"{row['capability_id']} promotion lacks scheduler terminal evidence"
        )


def validate_matrix(matrix: dict[str, Any], root: Path = ROOT) -> None:
    """Validate exact rows and refuse unsigned or non-terminal promotion."""
    validate_against_schema(matrix, SCHEMA_DIR / "capability-decision-matrix.schema.json")
    manifest_path = root / "docs/artifacts/cytoanvi-0.2.0/manifest.json"
    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        expected_artifact = {
            "version": manifest["candidate"]["version"],
            "wheel_sha256": manifest["wheel"]["sha256"],
            "source_commit": manifest["source"]["commit"],
            "installed_acceptance": manifest["installed_acceptance"]["status"],
        }
        if matrix["artifact"] != expected_artifact:
            raise ContractError(
                "Capability matrix global artifact identity differs from the artifact manifest"
            )
    rows = matrix["rows"]
    observed_ids = [row["capability_id"] for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise ContractError("Capability matrix contains duplicate rows")
    missing = sorted(set(CAPABILITY_IDS) - set(observed_ids))
    extra = sorted(set(observed_ids) - set(CAPABILITY_IDS))
    if missing or extra or len(rows) != len(CAPABILITY_IDS):
        raise ContractError(f"Capability matrix row mismatch: missing={missing}, extra={extra}")

    global_artifact = matrix["artifact"]
    for row in rows:
        if _artifact_tuple(row["artifact_identity"]) != _artifact_tuple(global_artifact):
            raise ContractError(f"{row['capability_id']} has an artifact identity mismatch")
        if (
            row["engineering_execution"] == "installed_passed"
            and global_artifact["installed_acceptance"] != "passed"
        ):
            raise ContractError(
                f"{row['capability_id']} claims installed pass without global installed acceptance"
            )
        if (
            row["scientific_result"] in {"historical_negative", "terminal_negative"}
            and not row["evidence_links"]
        ):
            raise ContractError(f"{row['capability_id']} omits negative-result evidence")
        for raw_link in row["evidence_links"]:
            relative_link = Path(raw_link)
            if relative_link.is_absolute() or ".." in relative_link.parts:
                raise ContractError(
                    f"{row['capability_id']} has an evidence link outside the repository"
                )
            if relative_link.parts and relative_link.parts[0] in {".living", ".scratch"}:
                raise ContractError(
                    f"{row['capability_id']} depends on local-only evidence: {raw_link}"
                )
            link = root / relative_link
            if link.is_symlink() or not link.is_file():
                raise ContractError(
                    f"{row['capability_id']} has a missing or non-regular evidence link: "
                    f"{raw_link}"
                )
        if row["decision"] in {"go", "conditional-go"}:
            if global_artifact["installed_acceptance"] != "passed":
                raise ContractError(
                    f"{row['capability_id']} is promoted without installed acceptance"
                )
            if row["scientific_result"] != "terminal_positive":
                raise ContractError(
                    f"{row['capability_id']} is promoted without terminal P2 evidence"
                )
            if not row["approver"] or not row["approval_date"]:
                raise ContractError(
                    f"{row['capability_id']} is promoted without a human signature"
                )

        if row["scientific_result"].startswith("terminal_"):
            _validate_terminal_row(row, global_artifact, root)


def validate_repository(root: Path = ROOT) -> list[str]:
    """Validate every tracked readiness contract in the repository."""
    checked: list[str] = []
    for schema_path in sorted((root / "docs" / "artifacts" / "schemas").glob("*.json")):
        schema = load_json(schema_path)
        Draft202012Validator.check_schema(schema)
        checked.append(str(schema_path.relative_to(root)))

    manifest_path = root / "docs" / "artifacts" / "cytoanvi-0.2.0" / "manifest.json"
    inventory_path = root / "docs" / "artifacts" / "cytoanvi-0.2.0" / "inventory.json"
    receipt_path = root / "docs" / "artifacts" / "cytoanvi-0.2.0" / "installed-acceptance.json"
    manifest = load_json(manifest_path)
    inventory = load_json(inventory_path)
    receipt = load_json(receipt_path)
    validate_artifact_bundle(manifest, inventory, receipt, root)
    checked.append(str(manifest_path.relative_to(root)))
    checked.append(str(inventory_path.relative_to(root)))
    checked.append(str(receipt_path.relative_to(root)))

    matrix_path = root / "docs" / "artifacts" / "usage-readiness-matrix-v1.json"
    validate_matrix(load_json(matrix_path), root)
    checked.append(str(matrix_path.relative_to(root)))

    protocol_paths = (
        root / "benchmarks" / "cytoanvi" / "usage_readiness_contract_v1.json",
        root / "benchmarks" / "mrtotalvi" / "usage_readiness_contract_v1.json",
    )
    for path in protocol_paths:
        if path.exists():
            protocol = load_json(path)
            validate_protocol(protocol, root)
            _validate_frozen_protocol_artifact(protocol, manifest)
            checked.append(str(path.relative_to(root)))
    return checked


def main() -> int:
    """Validate the complete cross-file usage-readiness repository state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    checked = validate_repository(args.root.resolve())
    print(f"usage-readiness contracts valid: {len(checked)} files")
    for path in checked:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
