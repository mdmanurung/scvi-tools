"""Fail-closed run manifests for MrTotalVI benchmark evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_digest(name: str, value: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal.") from error


def make_run_id(
    *,
    timestamp: datetime,
    code_digest: str,
    config_digest: str,
    data_digest: str,
) -> str:
    """Construct the frozen timestamp-plus-three-digests run identifier."""
    for name, value in (
        ("code_digest", code_digest),
        ("config_digest", config_digest),
        ("data_digest", data_digest),
    ):
        _validate_digest(name, value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware.")
    timestamp_utc = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{timestamp_utc}-{code_digest[:8]}-{config_digest[:8]}-"
        f"{data_digest[:8]}"
    )


@dataclass(frozen=True)
class ArtifactRecord:
    """One exact artifact belonging to a run."""

    path: str
    sha256: str
    bytes: int
    durable_uri: str | None = None

    def __post_init__(self) -> None:
        """Validate the relative path, digest, and size."""
        relative = PurePosixPath(self.path)
        if relative.is_absolute() or ".." in relative.parts or self.path in {"", "."}:
            raise ValueError("artifact path must be a non-empty relative path.")
        _validate_digest("artifact sha256", self.sha256)
        if self.bytes < 0:
            raise ValueError("artifact bytes must be nonnegative.")

    @classmethod
    def from_dict(cls, payload: dict) -> ArtifactRecord:
        """Parse one artifact without accepting unknown fields."""
        expected = {"path", "sha256", "bytes", "durable_uri"}
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"Unknown artifact fields: {sorted(unknown)}")
        return cls(
            path=payload["path"],
            sha256=payload["sha256"],
            bytes=payload["bytes"],
            durable_uri=payload.get("durable_uri"),
        )


@dataclass(frozen=True)
class RunManifest:
    """Immutable identity and exact artifact inventory for one run."""

    schema_version: str
    run_id: str
    created_at: str
    code_digest: str
    config_digest: str
    data_digest: str
    evidence_tier: Literal["pilot_cache", "publication"]
    scientific_scope: str
    status: Literal["complete", "failed", "inconclusive", "blocked"]
    artifacts: tuple[ArtifactRecord, ...]

    def to_dict(self) -> dict:
        """Return a strict JSON-compatible mapping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> RunManifest:
        """Parse a run manifest without silently accepting schema drift."""
        expected = {
            "schema_version",
            "run_id",
            "created_at",
            "code_digest",
            "config_digest",
            "data_digest",
            "evidence_tier",
            "scientific_scope",
            "status",
            "artifacts",
        }
        missing = expected - set(payload)
        unknown = set(payload) - expected
        if missing:
            raise ValueError(f"Manifest fields missing: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Unknown manifest fields: {sorted(unknown)}")
        artifacts = payload["artifacts"]
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("Manifest artifacts must be a non-empty list.")
        return cls(
            schema_version=payload["schema_version"],
            run_id=payload["run_id"],
            created_at=payload["created_at"],
            code_digest=payload["code_digest"],
            config_digest=payload["config_digest"],
            data_digest=payload["data_digest"],
            evidence_tier=payload["evidence_tier"],
            scientific_scope=payload["scientific_scope"],
            status=payload["status"],
            artifacts=tuple(ArtifactRecord.from_dict(item) for item in artifacts),
        )


def verify_run_manifest(path: str | Path) -> RunManifest:
    """Verify identity, policy, and exact on-disk artifacts for one run."""
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = RunManifest.from_dict(payload)
    if manifest.schema_version != "mrtotalvi-benchmark-run-v1":
        raise ValueError(f"Unsupported schema_version {manifest.schema_version!r}.")
    if manifest.evidence_tier not in {"pilot_cache", "publication"}:
        raise ValueError(f"Unsupported evidence_tier {manifest.evidence_tier!r}.")
    if manifest.status not in {"complete", "failed", "inconclusive", "blocked"}:
        raise ValueError(f"Unsupported status {manifest.status!r}.")

    created = datetime.fromisoformat(manifest.created_at)
    expected_id = make_run_id(
        timestamp=created,
        code_digest=manifest.code_digest,
        config_digest=manifest.config_digest,
        data_digest=manifest.data_digest,
    )
    if manifest.run_id != expected_id:
        raise ValueError(
            f"run_id mismatch: expected {expected_id!r}, got {manifest.run_id!r}."
        )

    if manifest.evidence_tier == "publication":
        missing_uri = [
            artifact.path
            for artifact in manifest.artifacts
            if not artifact.durable_uri
        ]
        if missing_uri:
            raise ValueError(
                "Publication artifacts require durable_uri values; missing for "
                f"{missing_uri}."
            )

    root = manifest_path.parent
    declared = {artifact.path for artifact in manifest.artifacts}
    if len(declared) != len(manifest.artifacts):
        raise ValueError("Manifest contains duplicate artifact paths.")
    actual = {
        file.relative_to(root).as_posix()
        for file in root.rglob("*")
        if file.is_file() and file.resolve() != manifest_path
    }
    missing = sorted(declared - actual)
    if missing:
        raise ValueError(f"Run has missing artifact files: {missing}.")
    extra = sorted(actual - declared)
    if extra:
        raise ValueError(f"Run has extra artifact files: {extra}.")

    for artifact in manifest.artifacts:
        artifact_path = root / artifact.path
        observed_digest = sha256_file(artifact_path)
        if observed_digest != artifact.sha256:
            raise ValueError(
                f"Artifact hash mismatch for {artifact.path!r}: "
                f"expected {artifact.sha256}, got {observed_digest}."
            )
        observed_bytes = artifact_path.stat().st_size
        if observed_bytes != artifact.bytes:
            raise ValueError(
                f"Artifact byte-size mismatch for {artifact.path!r}: "
                f"expected {artifact.bytes}, got {observed_bytes}."
            )
    return manifest
