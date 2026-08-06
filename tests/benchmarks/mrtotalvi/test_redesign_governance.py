"""Immutable governance evidence for the MrTotalVI redesign."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime

import pytest
from benchmarks.mrtotalvi.governance import (
    REQUIRED_GOVERNANCE_ROLES,
    _rename_no_replace,
    build_redesign_governance_manifest,
    verify_redesign_governance_manifest,
    verify_redesign_governance_run,
    write_redesign_governance_run,
)


def _inventory(root):
    inventory = {}
    for role in REQUIRED_GOVERNANCE_ROLES:
        if role == "old_run":
            old_run = root / "old-run"
            old_run.mkdir()
            artifact = old_run / "artifact.txt"
            artifact.write_text("old evidence\n", encoding="utf-8")
            checksum = old_run / "checksums.sha256"
            checksum.write_text(
                f"{_sha256(artifact)}  artifact.txt\n",
                encoding="utf-8",
            )
            inventory[role] = ["old-run/checksums.sha256"]
            continue
        path = root / f"{role}.txt"
        path.write_text(f"{role}\n", encoding="utf-8")
        inventory[role] = [path.name]
    return inventory


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_governance_manifest_freezes_exact_roles_config_and_file_hashes(tmp_path):
    """Every prerequisite stays explicit and hash-verified."""
    inventory = _inventory(tmp_path)
    created_at = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
    manifest = build_redesign_governance_manifest(
        repository_root=tmp_path,
        base_commit="d8c8e997a67997a53f55923eb3ab14e6cf06f94c",
        created_at=created_at,
        inventory=inventory,
    )

    assert {record.role for record in manifest.files} == set(
        REQUIRED_GOVERNANCE_ROLES
    )
    assert [config["candidate_id"] for config in manifest.candidate_configs] == [
        "B0",
        "B1",
        "B2",
        "B3",
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
    ]

    run_dir = write_redesign_governance_run(
        repository_root=tmp_path,
        output_root=tmp_path / "runs",
        manifest=manifest,
    )
    manifest_path = run_dir / "governance-manifest.json"
    verified = verify_redesign_governance_manifest(
        manifest_path,
        repository_root=tmp_path,
    )
    assert verified == manifest
    assert (
        verify_redesign_governance_run(
            run_dir,
            repository_root=tmp_path,
        )
        == manifest
    )
    assert not (tmp_path / "runs" / "latest").exists()
    checksums = (run_dir / "checksums.sha256").read_text(encoding="utf-8")
    assert "  governance-manifest.json\n" in checksums
    assert "  redesign-run-contract.json\n" in checksums

    with pytest.raises(FileExistsError):
        write_redesign_governance_run(
            repository_root=tmp_path,
            output_root=tmp_path / "runs",
            manifest=manifest,
        )

    (tmp_path / "package_source.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_redesign_governance_manifest(
            manifest_path,
            repository_root=tmp_path,
        )


def test_governance_recursively_verifies_old_checksum_inventories(tmp_path):
    """Hashing an old manifest cannot hide drift in its declared artifacts."""
    inventory = _inventory(tmp_path)
    manifest = build_redesign_governance_manifest(
        repository_root=tmp_path,
        base_commit="d8c8e997a67997a53f55923eb3ab14e6cf06f94c",
        created_at=datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        inventory=inventory,
    )
    run_dir = write_redesign_governance_run(
        repository_root=tmp_path,
        output_root=tmp_path / "runs",
        manifest=manifest,
    )
    (tmp_path / "old-run" / "artifact.txt").write_text(
        "silently drifted\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Nested checksum mismatch"):
        verify_redesign_governance_run(
            run_dir,
            repository_root=tmp_path,
        )


def test_governance_run_replays_registered_pre_diagnosis_v1_contract(
    tmp_path,
):
    """Historical governance remains verifiable without upgrading its contract."""
    inventory = _inventory(tmp_path)
    manifest = build_redesign_governance_manifest(
        repository_root=tmp_path,
        base_commit="d8c8e997a67997a53f55923eb3ab14e6cf06f94c",
        created_at=datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        inventory=inventory,
    )
    run_dir = write_redesign_governance_run(
        repository_root=tmp_path,
        output_root=tmp_path / "runs",
        manifest=manifest,
    )
    contract_path = run_dir / "redesign-run-contract.json"
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    del payload["diagnosis"]
    contract_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = run_dir / "governance-manifest.json"
    (run_dir / "checksums.sha256").write_text(
        f"{_sha256(manifest_path)}  governance-manifest.json\n"
        f"{_sha256(contract_path)}  redesign-run-contract.json\n",
        encoding="utf-8",
    )

    assert (
        verify_redesign_governance_run(
            run_dir,
            repository_root=tmp_path,
        )
        == manifest
    )


def test_atomic_rename_never_replaces_an_existing_empty_directory(tmp_path):
    """The final no-replace operation closes the existence-check race."""
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    new_file = source / "new.txt"
    new_file.write_text("new\n", encoding="utf-8")
    (source / "checksums.sha256").write_text(
        f"{_sha256(new_file)}  new.txt\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        _rename_no_replace(source, destination)
    assert destination.is_dir()
    assert not list(destination.iterdir())
    assert (source / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert os.path.isdir(source)


def test_governance_manifest_rejects_schema_drift_and_incomplete_roles(tmp_path):
    """Unknown fields and an incomplete evidence inventory fail closed."""
    inventory = _inventory(tmp_path)
    manifest = build_redesign_governance_manifest(
        repository_root=tmp_path,
        base_commit="d8c8e997a67997a53f55923eb3ab14e6cf06f94c",
        created_at=datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        inventory=inventory,
    )
    run_dir = write_redesign_governance_run(
        repository_root=tmp_path,
        output_root=tmp_path / "runs",
        manifest=manifest,
    )
    manifest_path = run_dir / "governance-manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    payload["undeclared_axis"] = True
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown governance fields"):
        verify_redesign_governance_manifest(
            manifest_path,
            repository_root=tmp_path,
        )

    incomplete = dict(inventory)
    incomplete.pop("old_run")
    with pytest.raises(ValueError, match="roles"):
        build_redesign_governance_manifest(
            repository_root=tmp_path,
            base_commit="d8c8e997a67997a53f55923eb3ab14e6cf06f94c",
            created_at=datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
            inventory=incomplete,
        )
