"""Immutable MrTotalVI benchmark manifest behavior."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from benchmarks.mrtotalvi import (
    ArtifactRecord,
    RunManifest,
    make_run_id,
    sha256_file,
    verify_run_manifest,
)


def _write_manifest(run_dir, *, tier="pilot_cache"):
    artifact = run_dir / "metrics.json"
    artifact.write_text('{"metric": 1.0}\n', encoding="utf-8")
    created = datetime(2026, 7, 26, 9, 30, tzinfo=UTC)
    code_digest = "a" * 64
    config_digest = "b" * 64
    data_digest = "c" * 64
    manifest = RunManifest(
        schema_version="mrtotalvi-benchmark-run-v1",
        run_id=make_run_id(
            timestamp=created,
            code_digest=code_digest,
            config_digest=config_digest,
            data_digest=data_digest,
        ),
        created_at=created.isoformat(),
        code_digest=code_digest,
        config_digest=config_digest,
        data_digest=data_digest,
        evidence_tier=tier,
        scientific_scope="synthetic mechanism pilot",
        status="complete",
        artifacts=(
            ArtifactRecord(
                path="metrics.json",
                sha256=sha256_file(artifact),
                bytes=artifact.stat().st_size,
                durable_uri=(
                    "s3://example/mrtotalvi/metrics.json"
                    if tier == "publication"
                    else None
                ),
            ),
        ),
    )
    path = run_dir / "run-manifest.json"
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_manifest_verification_rejects_missing_extra_and_hash_mismatched_artifacts(
    tmp_path,
):
    """Aggregation sees exactly the immutable files declared by one run."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = _write_manifest(run_dir)

    verified = verify_run_manifest(manifest_path)
    assert verified.run_id == "20260726T093000Z-aaaaaaaa-bbbbbbbb-cccccccc"

    extra = run_dir / "untracked.txt"
    extra.write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extra artifact"):
        verify_run_manifest(manifest_path)
    extra.unlink()

    artifact = run_dir / "metrics.json"
    artifact.write_text('{"metric": 2.0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_run_manifest(manifest_path)
    artifact.unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        verify_run_manifest(manifest_path)


def test_publication_manifest_requires_durable_artifact_uris(tmp_path):
    """A scratch-only cache cannot be promoted by changing one status string."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = _write_manifest(run_dir, tier="pilot_cache")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["evidence_tier"] = "publication"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="durable_uri"):
        verify_run_manifest(manifest_path)
