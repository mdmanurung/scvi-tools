"""Fail-closed governance manifests for the MrTotalVI redesign."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from .manifest import sha256_file, verify_run_manifest
from .redesign_contract import (
    redesign_candidate_configs,
    redesign_config_digest,
    redesign_run_contract,
    validate_redesign_candidate,
)
from .versioning import registered_redesign_run_contract_v1

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal


REQUIRED_GOVERNANCE_ROLES = (
    "packet",
    "package_source",
    "regression_fixture",
    "old_run",
    "baseline_evidence",
    "metric_dictionary",
)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def _validate_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal.") from error


def _validate_created_at(value: str) -> datetime:
    try:
        created_at = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("created_at must be an ISO-8601 datetime.") from error
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware.")
    return created_at


@dataclass(frozen=True)
class GovernanceFileRecord:
    """One repository file frozen by role, digest, and byte size."""

    role: Literal[
        "packet",
        "package_source",
        "regression_fixture",
        "old_run",
        "baseline_evidence",
        "metric_dictionary",
    ]
    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        """Validate the role, relative path, digest, and byte size."""
        if self.role not in REQUIRED_GOVERNANCE_ROLES:
            raise ValueError(f"Unknown governance role {self.role!r}.")
        relative = PurePosixPath(self.path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or self.path in {"", "."}
        ):
            raise ValueError(
                "governance file path must be a non-empty relative path."
            )
        _validate_sha256("governance file sha256", self.sha256)
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, int)
            or self.bytes < 0
        ):
            raise ValueError(
                "governance file bytes must be a nonnegative integer."
            )

    @classmethod
    def from_dict(cls, payload: dict) -> GovernanceFileRecord:
        """Parse one file record without accepting schema drift."""
        if not isinstance(payload, dict):
            raise ValueError("Governance file record must be a mapping.")
        expected = {field.name for field in fields(cls)}
        missing = expected - set(payload)
        unknown = set(payload) - expected
        if missing:
            raise ValueError(
                f"Governance file fields missing: {sorted(missing)}"
            )
        if unknown:
            raise ValueError(
                f"Unknown governance file fields: {sorted(unknown)}"
            )
        return cls(**payload)


@dataclass(frozen=True)
class RedesignGovernanceManifest:
    """Exact redesign configuration and prerequisite hash inventory."""

    schema_version: str
    base_commit: str
    created_at: str
    config_digest: str
    candidate_configs: tuple[dict[str, str], ...]
    files: tuple[GovernanceFileRecord, ...]

    def __post_init__(self) -> None:
        """Validate schema, candidates, roles, and configuration identity."""
        if self.schema_version != "mrtotalvi-redesign-governance-v1":
            raise ValueError(
                f"Unsupported governance schema {self.schema_version!r}."
            )
        if not isinstance(self.base_commit, str) or not _COMMIT_PATTERN.fullmatch(
            self.base_commit
        ):
            raise ValueError(
                "base_commit must be a lowercase 40-character Git digest."
            )
        _validate_created_at(self.created_at)
        _validate_sha256("config_digest", self.config_digest)

        expected_configs = redesign_candidate_configs()
        observed_ids = []
        for payload in self.candidate_configs:
            config = validate_redesign_candidate(payload)
            observed_ids.append(config.candidate_id)
        if observed_ids != list(expected_configs):
            raise ValueError(
                "candidate_configs must contain exact ordered B0-B3/D0-D5."
            )
        if self.config_digest != redesign_config_digest():
            raise ValueError(
                "config_digest does not match the frozen candidate contract."
            )

        roles = {record.role for record in self.files}
        if roles != set(REQUIRED_GOVERNANCE_ROLES):
            raise ValueError(
                "Governance file roles must equal "
                f"{list(REQUIRED_GOVERNANCE_ROLES)}; got {sorted(roles)}."
            )
        paths = [record.path for record in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Governance file paths must be unique.")
        metric_records = [
            record
            for record in self.files
            if record.role == "metric_dictionary"
        ]
        if len(metric_records) != 1:
            raise ValueError(
                "Governance requires exactly one metric_dictionary file."
            )

    def to_dict(self) -> dict:
        """Return the strict JSON-compatible governance payload."""
        return {
            "schema_version": self.schema_version,
            "base_commit": self.base_commit,
            "created_at": self.created_at,
            "config_digest": self.config_digest,
            "candidate_configs": [
                dict(config) for config in self.candidate_configs
            ],
            "files": [asdict(record) for record in self.files],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> RedesignGovernanceManifest:
        """Parse one governance manifest without accepting unknown fields."""
        if not isinstance(payload, dict):
            raise ValueError("Governance manifest must be a mapping.")
        expected = {field.name for field in fields(cls)}
        missing = expected - set(payload)
        unknown = set(payload) - expected
        if missing:
            raise ValueError(f"Governance fields missing: {sorted(missing)}")
        if unknown:
            raise ValueError(f"Unknown governance fields: {sorted(unknown)}")
        configs = payload["candidate_configs"]
        records = payload["files"]
        if not isinstance(configs, list):
            raise ValueError("candidate_configs must be a list.")
        if not isinstance(records, list) or not records:
            raise ValueError("Governance files must be a non-empty list.")
        return cls(
            schema_version=payload["schema_version"],
            base_commit=payload["base_commit"],
            created_at=payload["created_at"],
            config_digest=payload["config_digest"],
            candidate_configs=tuple(configs),
            files=tuple(
                GovernanceFileRecord.from_dict(record) for record in records
            ),
        )


def _resolve_repository_file(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(
            f"Governance file escapes repository root: {relative_path!r}."
        )
    if not path.is_file():
        raise ValueError(
            f"Governance file is missing or not regular: {relative_path!r}."
        )
    return path


def _verify_checksum_inventory(path: Path) -> None:
    """Verify every relative file declared by a SHA-256 inventory."""
    declared = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        digest, separator, relative_path = line.partition("  ")
        if (
            not separator
            or not relative_path
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"Malformed checksum inventory line {line_number} in {path}."
            )
        relative = PurePosixPath(relative_path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative_path in {"", "."}
        ):
            raise ValueError(
                f"Unsafe checksum path {relative_path!r} in {path}."
            )
        if relative_path in declared:
            raise ValueError(
                f"Duplicate checksum path {relative_path!r} in {path}."
            )
        declared.add(relative_path)
        artifact = (path.parent / relative_path).resolve()
        if (
            not artifact.is_relative_to(path.parent.resolve())
            or not artifact.is_file()
        ):
            raise ValueError(
                f"Nested checksum artifact missing: {relative_path!r}."
            )
        observed = sha256_file(artifact)
        if observed != digest:
            raise ValueError(
                f"Nested checksum mismatch for {relative_path!r}: "
                f"expected {digest}, got {observed}."
            )
    if not declared:
        raise ValueError(f"Checksum inventory is empty: {path}.")


def _verify_nested_old_evidence(record: GovernanceFileRecord, path: Path) -> None:
    """Re-read artifacts referenced by recognized old evidence inventories."""
    if record.role != "old_run":
        return
    if path.name == "run-manifest.json":
        verify_run_manifest(path)
    elif path.name == "checksums.sha256":
        _verify_checksum_inventory(path)


def _rename_no_replace(source: str | Path, destination: str | Path) -> None:
    """Publish a directory while refusing every existing target."""
    source_path = Path(source)
    destination_path = Path(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    unsupported_errors = {
        errno.EINVAL,
        errno.ENOSYS,
        errno.EOPNOTSUPP,
    }
    if renameat2 is not None:
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source_path),
            -100,
            os.fsencode(destination_path),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                str(destination_path),
            )
        if error_number not in unsupported_errors:
            raise OSError(
                error_number,
                os.strerror(error_number),
                str(destination_path),
            )

    children = tuple(source_path.iterdir())
    if not children or any(
        not child.is_file() or child.is_symlink() for child in children
    ):
        raise RuntimeError(
            "No-replace fallback supports only non-empty regular-file runs."
        )
    checksums = [child for child in children if child.name == "checksums.sha256"]
    if len(checksums) != 1:
        raise RuntimeError(
            "No-replace fallback requires one checksums.sha256 completion file."
        )
    ordered = sorted(
        children,
        key=lambda child: (child.name == "checksums.sha256", child.name),
    )

    destination_path.mkdir()
    linked = []
    try:
        for child in ordered:
            linked_path = destination_path / child.name
            os.link(child, linked_path, follow_symlinks=False)
            linked.append(linked_path)
        directory_fd = os.open(
            destination_path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        for child in children:
            child.unlink()
        source_path.rmdir()
    except Exception:
        for linked_path in reversed(linked):
            if linked_path.exists():
                linked_path.unlink()
        if destination_path.exists():
            destination_path.rmdir()
        raise


def build_redesign_governance_manifest(
    *,
    repository_root: str | Path,
    base_commit: str,
    created_at: datetime,
    inventory: Mapping[str, Sequence[str]],
) -> RedesignGovernanceManifest:
    """Hash an explicit, complete governance inventory."""
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise ValueError("repository_root must be an existing directory.")
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware.")
    roles = set(inventory)
    if roles != set(REQUIRED_GOVERNANCE_ROLES):
        raise ValueError(
            "Governance inventory roles must equal "
            f"{list(REQUIRED_GOVERNANCE_ROLES)}; got {sorted(roles)}."
        )

    records = []
    for role in REQUIRED_GOVERNANCE_ROLES:
        role_paths = inventory[role]
        if not isinstance(role_paths, Sequence) or isinstance(
            role_paths, (str, bytes)
        ):
            raise ValueError(f"Inventory role {role!r} must be a sequence.")
        if not role_paths:
            raise ValueError(f"Inventory role {role!r} cannot be empty.")
        for relative_path in sorted(role_paths):
            if not isinstance(relative_path, str):
                raise ValueError("Governance paths must be strings.")
            path = _resolve_repository_file(root, relative_path)
            records.append(
                GovernanceFileRecord(
                    role=role,
                    path=relative_path,
                    sha256=sha256_file(path),
                    bytes=path.stat().st_size,
                )
            )

    configs = redesign_candidate_configs()
    return RedesignGovernanceManifest(
        schema_version="mrtotalvi-redesign-governance-v1",
        base_commit=base_commit,
        created_at=created_at.isoformat(),
        config_digest=redesign_config_digest(),
        candidate_configs=tuple(
            configs[candidate_id].to_dict() for candidate_id in configs
        ),
        files=tuple(records),
    )


def verify_redesign_governance_manifest(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> RedesignGovernanceManifest:
    """Verify the strict manifest and every referenced repository file."""
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = RedesignGovernanceManifest.from_dict(payload)
    root = Path(repository_root).resolve()
    for record in manifest.files:
        file_path = _resolve_repository_file(root, record.path)
        observed_digest = sha256_file(file_path)
        if observed_digest != record.sha256:
            raise ValueError(
                f"Governance hash mismatch for {record.path!r}: "
                f"expected {record.sha256}, got {observed_digest}."
            )
        observed_bytes = file_path.stat().st_size
        if observed_bytes != record.bytes:
            raise ValueError(
                f"Governance byte-size mismatch for {record.path!r}: "
                f"expected {record.bytes}, got {observed_bytes}."
            )
        _verify_nested_old_evidence(record, file_path)
    return manifest


def _manifest_digest(manifest: RedesignGovernanceManifest) -> str:
    payload = json.dumps(
        manifest.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_redesign_governance_run(
    *,
    repository_root: str | Path,
    output_root: str | Path,
    manifest: RedesignGovernanceManifest,
) -> Path:
    """Atomically seal one new governance directory without a latest pointer."""
    root = Path(repository_root).resolve()
    for record in manifest.files:
        file_path = _resolve_repository_file(root, record.path)
        if (
            sha256_file(file_path) != record.sha256
            or file_path.stat().st_size != record.bytes
        ):
            raise ValueError(
                f"Cannot seal governance after drift in {record.path!r}."
            )
        _verify_nested_old_evidence(record, file_path)

    created_at = _validate_created_at(manifest.created_at)
    timestamp = created_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest_digest = _manifest_digest(manifest)
    run_id = f"{timestamp}-{manifest_digest[:12]}"
    parent = Path(output_root)
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / run_id
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"Governance run already exists: {run_dir}.")

    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=parent))
    try:
        manifest_path = temporary / "governance-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_contract_path = temporary / "redesign-run-contract.json"
        run_contract_path.write_text(
            json.dumps(
                redesign_run_contract().to_dict(),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "checksums.sha256").write_text(
            f"{sha256_file(manifest_path)}  governance-manifest.json\n"
            f"{sha256_file(run_contract_path)}  redesign-run-contract.json\n",
            encoding="utf-8",
        )
        _rename_no_replace(temporary, run_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return run_dir


def verify_redesign_governance_run(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> RedesignGovernanceManifest:
    """Verify exact run contents, checksums, contract, and nested evidence."""
    run_dir = Path(path).resolve()
    expected_files = {
        "checksums.sha256",
        "governance-manifest.json",
        "redesign-run-contract.json",
    }
    actual_files = {
        child.name for child in run_dir.iterdir() if child.is_file()
    }
    if actual_files != expected_files:
        raise ValueError(
            "Governance run files must equal "
            f"{sorted(expected_files)}; got {sorted(actual_files)}."
        )
    if any(child.is_dir() for child in run_dir.iterdir()):
        raise ValueError("Governance run cannot contain subdirectories.")
    _verify_checksum_inventory(run_dir / "checksums.sha256")
    contract_payload = json.loads(
        (run_dir / "redesign-run-contract.json").read_text(encoding="utf-8")
    )
    registered_redesign_run_contract_v1(contract_payload)
    return verify_redesign_governance_manifest(
        run_dir / "governance-manifest.json",
        repository_root=repository_root,
    )
