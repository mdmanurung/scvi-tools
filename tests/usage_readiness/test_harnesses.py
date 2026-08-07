from __future__ import annotations

import base64
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]


def _load_extensionless(name: str, path: Path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


BUILD = _load_extensionless(
    "build_usage_readiness_wheel", ROOT / "scripts/build_usage_readiness_wheel"
)
ACCEPT = _load_extensionless(
    "accept_usage_readiness_wheel", ROOT / "scripts/accept_usage_readiness_wheel"
)
SMOKE = _load_extensionless(
    "usage_readiness_installed_smoke", ROOT / "scripts/usage_readiness_installed_smoke.py"
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_hash(value: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).decode().rstrip("=")
    return f"sha256={encoded}"


def _git(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(["git", *command], cwd=cwd, check=True, capture_output=True)


def test_build_requires_independent_reconstructed_source(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    primary.mkdir()
    _git(["init"], primary)
    (primary / "tracked.txt").write_text("committed\n")
    _git(["add", "tracked.txt"], primary)
    _git(
        [
            "-c",
            "user.name=Usage Readiness",
            "-c",
            "user.email=usage-readiness@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        primary,
    )

    reconstructed = tmp_path / "reconstructed"
    _git(["clone", "--no-hardlinks", str(primary), str(reconstructed)])
    BUILD.verify_reconstructed_source_provenance(reconstructed, primary)

    with pytest.raises(RuntimeError, match="distinct reconstruction"):
        BUILD.verify_reconstructed_source_provenance(primary, primary)

    linked = tmp_path / "linked-worktree"
    _git(["worktree", "add", "--detach", str(linked), "HEAD"], primary)
    with pytest.raises(RuntimeError, match="independent Git directory"):
        BUILD.verify_reconstructed_source_provenance(linked, primary)


def test_installed_smoke_import_is_numpy_independent() -> None:
    assert "np" not in vars(SMOKE)


def test_build_attempt_is_claimed_before_backend_and_cannot_repeat(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "build": {
            "status": "not_built",
            "attempt_id": None,
            "candidate_dir": None,
            "pid": None,
            "command": None,
            "python": None,
            "environment_digest": None,
            "started_at": None,
            "completed_at": None,
        }
    }
    manifest_path.write_text(json.dumps(manifest))
    first_output = tmp_path / "output-one"
    second_output = tmp_path / "output-two"
    first_output.mkdir()
    second_output.mkdir()
    claim_path = tmp_path / BUILD.CLAIM_FILENAME
    candidate_dir, attempt_id = BUILD.claim_build_attempt(
        output_dir=first_output,
        claim_path=claim_path,
        manifest_path=manifest_path,
        manifest=manifest,
        command="python -m build",
        started="2026-08-07T00:00:00Z",
        source_commit="1" * 40,
    )
    recorded = json.loads(manifest_path.read_text())
    assert candidate_dir.is_dir()
    assert claim_path.is_file()
    assert recorded["build"]["status"] == "building"
    assert recorded["build"]["attempt_id"] == attempt_id

    second_manifest_path = tmp_path / "copied-manifest.json"
    second_manifest = copy.deepcopy(manifest)
    second_manifest["build"]["status"] = "not_built"
    second_manifest_path.write_text(json.dumps(second_manifest))
    with pytest.raises(RuntimeError, match="prior or interrupted"):
        BUILD.claim_build_attempt(
            output_dir=second_output,
            claim_path=claim_path,
            manifest_path=second_manifest_path,
            manifest=second_manifest,
            command="python -m build",
            started="2026-08-07T00:01:00Z",
            source_commit="1" * 40,
        )


def test_build_attempt_rejects_orphan_candidate_directory(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = {"build": {"status": "not_built"}}
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / ".cytoanvi-0.2.0-candidate-orphan").mkdir()
    with pytest.raises(RuntimeError, match="prior or interrupted"):
        BUILD.claim_build_attempt(
            output_dir=tmp_path,
            claim_path=tmp_path / BUILD.CLAIM_FILENAME,
            manifest_path=manifest_path,
            manifest=manifest,
            command="python -m build",
            started="2026-08-07T00:00:00Z",
            source_commit="1" * 40,
        )


def test_wheel_record_and_source_payload_are_byte_verified(tmp_path: Path) -> None:
    package = b"__version__ = '0.2.0'\n"
    metadata = b"Metadata-Version: 2.4\nName: cytoanvi\nVersion: 0.2.0\n"
    record_path = "cytoanvi-0.2.0.dist-info/RECORD"
    record = (
        f"cytoanvi/__init__.py,{_record_hash(package)},{len(package)}\n"
        f"cytoanvi-0.2.0.dist-info/METADATA,{_record_hash(metadata)},{len(metadata)}\n"
        f"{record_path},,\n"
    ).encode()
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("cytoanvi/__init__.py", package)
        archive.writestr("cytoanvi-0.2.0.dist-info/METADATA", metadata)
        archive.writestr(record_path, record)
    wheel_files, _record, _metadata = BUILD.wheel_inventory(wheel)
    source_files = [
        {"path": "cytoanvi/__init__.py", "sha256": _sha(package), "size_bytes": len(package)}
    ]
    assert BUILD.source_wheel_mismatches(source_files, wheel_files) == ([], [], [])
    changed_source = copy.deepcopy(source_files)
    changed_source[0]["sha256"] = "0" * 64
    assert BUILD.source_wheel_mismatches(changed_source, wheel_files)[2] == [
        "cytoanvi/__init__.py"
    ]

    bad_record = record.replace(_record_hash(package).encode(), b"sha256=wrong")
    bad_wheel = tmp_path / "bad.whl"
    with zipfile.ZipFile(bad_wheel, "w") as archive:
        archive.writestr("cytoanvi/__init__.py", package)
        archive.writestr("cytoanvi-0.2.0.dist-info/METADATA", metadata)
        archive.writestr(record_path, bad_record)
    with pytest.raises(RuntimeError, match="RECORD SHA-256 mismatch"):
        BUILD.wheel_inventory(bad_wheel)


@pytest.mark.parametrize("unexpected", ["sitecustomize.py", "cytoanvi-0.2.0.data/scripts/tool"])
def test_wheel_rejects_nonowned_top_level_payloads(tmp_path: Path, unexpected: str) -> None:
    wheel = tmp_path / "unexpected.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(unexpected, b"payload")
    with pytest.raises(RuntimeError, match="non-owned top-level payload"):
        BUILD.wheel_inventory(wheel)


def test_authority_rejects_platform_or_glibc_mismatch(tmp_path: Path) -> None:
    authority = {
        "schema_version": "cytoanvi-dependency-authority-v1",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": "linux-x86_64-glibc999",
        "requirements_file": "requirements.lock",
        "requirements_sha256": "0" * 64,
        "wheelhouse": "wheelhouse",
        "wheel_inventory": [{"filename": "fixture.whl", "sha256": "0" * 64}],
    }
    (tmp_path / "authority.json").write_text(json.dumps(authority))
    with pytest.raises(ValueError, match="runtime platform"):
        ACCEPT.verify_authority(tmp_path)


def _write_authority(root: Path, filename: str, target: Path) -> None:
    requirements = root / "requirements.lock"
    lock = f"fixture-dependency==1.0 --hash=sha256:{'1' * 64}\n"
    requirements.write_text(lock)
    wheelhouse = root / "wheelhouse"
    wheelhouse.mkdir()
    if target.parent == wheelhouse:
        target.write_bytes(b"fixture")
    libc_name, libc_version = platform.libc_ver()
    authority = {
        "schema_version": "cytoanvi-dependency-authority-v1",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": f"{platform.system().lower()}-{platform.machine()}-{libc_name}{libc_version}",
        "requirements_file": requirements.name,
        "requirements_sha256": _sha(lock.encode()),
        "wheelhouse": wheelhouse.name,
        "wheel_inventory": [{"filename": filename, "sha256": _sha(b"fixture")}],
    }
    (root / "authority.json").write_text(json.dumps(authority))


def test_authority_lock_grammar_accepts_only_hashed_pins() -> None:
    valid = (
        f"# generated exact lock\nalpha-pkg==1.2.3 \\\n"
        f"  --hash=sha256:{'1' * 64} \\\n"
        f"  --hash=sha256:{'2' * 64}\n"
        f"beta_pkg[extra]==2.0rc1 --hash=sha256:{'3' * 64}\n"
    )
    assert ACCEPT.parse_requirements_lock(valid) == [
        ("alpha-pkg", "1.2.3", (f"--hash=sha256:{'1' * 64}", f"--hash=sha256:{'2' * 64}")),
        ("beta-pkg", "2.0rc1", (f"--hash=sha256:{'3' * 64}",)),
    ]

    malicious = (
        "--find-links https://example.invalid/wheels",
        "--index-url https://example.invalid/simple",
        "--extra-index-url https://example.invalid/simple",
        "-r nested-requirements.txt",
        "-c constraints.txt",
        "-e .",
        "../local.whl",
        f"alpha @ https://example.invalid/alpha.whl --hash=sha256:{'1' * 64}",
        f"alpha==1.0 --hash=sha256:{'1' * 64} --find-links=/tmp/wheels",
        "alpha==1.0",
        f"alpha>=1.0 --hash=sha256:{'1' * 64}",
        f"alpha==1.0 ; python_version > '3.12' --hash=sha256:{'1' * 64}",
    )
    for record in malicious:
        with pytest.raises(ValueError, match="requirements.lock"):
            ACCEPT.parse_requirements_lock(record + "\n")


def test_authority_rejects_nonwheel_traversal_and_symlink_entries(tmp_path: Path) -> None:
    sdist_root = tmp_path / "sdist"
    sdist_root.mkdir()
    _write_authority(sdist_root, "fixture.tar.gz", sdist_root / "wheelhouse/fixture.tar.gz")
    with pytest.raises(ValueError, match="basename-only .whl"):
        ACCEPT.verify_authority(sdist_root)

    traversal_root = tmp_path / "traversal"
    traversal_root.mkdir()
    _write_authority(traversal_root, "../fixture.whl", traversal_root / "wheelhouse/fixture.whl")
    with pytest.raises(ValueError, match="basename-only .whl"):
        ACCEPT.verify_authority(traversal_root)

    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    outside = tmp_path / "outside.whl"
    outside.write_bytes(b"fixture")
    _write_authority(symlink_root, "linked.whl", tmp_path / "unused")
    (symlink_root / "wheelhouse/linked.whl").symlink_to(outside)
    with pytest.raises(ValueError, match="regular non-symlink"):
        ACCEPT.verify_authority(symlink_root)


def test_authority_rejects_symlinked_lock_or_wheelhouse(tmp_path: Path) -> None:
    lock_root = tmp_path / "lock-link"
    lock_root.mkdir()
    outside_lock = tmp_path / "outside.lock"
    outside_lock.write_text("")
    (lock_root / "requirements.lock").symlink_to(outside_lock)
    wheelhouse = lock_root / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "fixture.whl"
    wheel.write_bytes(b"fixture")
    libc_name, libc_version = platform.libc_ver()
    authority = {
        "schema_version": "cytoanvi-dependency-authority-v1",
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": (
            f"{platform.system().lower()}-{platform.machine()}-{libc_name}{libc_version}"
        ),
        "requirements_file": "requirements.lock",
        "requirements_sha256": _sha(b""),
        "wheelhouse": "wheelhouse",
        "wheel_inventory": [{"filename": wheel.name, "sha256": _sha(b"fixture")}],
    }
    (lock_root / "authority.json").write_text(json.dumps(authority))
    with pytest.raises(ValueError, match="contains a symlink"):
        ACCEPT.verify_authority(lock_root)

    wheelhouse_root = tmp_path / "wheelhouse-link"
    wheelhouse_root.mkdir()
    (wheelhouse_root / "requirements.lock").write_text("")
    outside_wheelhouse = tmp_path / "outside-wheelhouse"
    outside_wheelhouse.mkdir()
    (outside_wheelhouse / "fixture.whl").write_bytes(b"fixture")
    (wheelhouse_root / "wheelhouse").symlink_to(outside_wheelhouse)
    (wheelhouse_root / "authority.json").write_text(json.dumps(authority))
    with pytest.raises(ValueError, match="contains a symlink"):
        ACCEPT.verify_authority(wheelhouse_root)


def _installed_inventory_fixture() -> tuple[list[dict], dict, str]:
    wheel_sha = "9" * 64
    dist_info = "cytoanvi-0.2.0.dist-info"
    payloads = {
        "cytoanvi/__init__.py": b"payload\n",
        f"{dist_info}/METADATA": b"Name: cytoanvi\nVersion: 0.2.0\n",
        f"{dist_info}/INSTALLER": b"pip\n",
        f"{dist_info}/REQUESTED": b"",
        f"{dist_info}/direct_url.json": json.dumps(
            {"archive_info": {"hashes": {"sha256": wheel_sha}}, "url": "file:///tmp/wheel"}
        ).encode(),
        f"cytoanvi/__pycache__/__init__.{sys.implementation.cache_tag}.pyc": b"bytecode",
    }
    record_path = f"{dist_info}/RECORD"
    record_lines = []
    for path, value in payloads.items():
        if "/__pycache__/" in path:
            record_lines.append(f"{path},,")
        else:
            record_lines.append(f"{path},{_record_hash(value)},{len(value)}")
    record_lines.append(f"{record_path},,")
    record_bytes = ("\n".join(record_lines) + "\n").encode()
    installed_files = [
        {"path": path, "sha256": _sha(value), "size_bytes": len(value)}
        for path, value in sorted(payloads.items())
    ]
    installed_files.append(
        {"path": record_path, "sha256": _sha(record_bytes), "size_bytes": len(record_bytes)}
    )
    sealed_files = [
        {
            "path": "cytoanvi/__init__.py",
            "sha256": _sha(payloads["cytoanvi/__init__.py"]),
            "size_bytes": len(payloads["cytoanvi/__init__.py"]),
        },
        {
            "path": f"{dist_info}/METADATA",
            "sha256": _sha(payloads[f"{dist_info}/METADATA"]),
            "size_bytes": len(payloads[f"{dist_info}/METADATA"]),
        },
        {"path": record_path, "sha256": "8" * 64, "size_bytes": 100},
    ]
    identity = {
        "installed_inventory": {
            "files": installed_files,
            "record": {"path": record_path, "lines": record_lines},
            "pip_generated": {
                "INSTALLER": "pip\n",
                "REQUESTED": "",
                "direct_url.json": {
                    "archive_info": {"hashes": {"sha256": wheel_sha}},
                    "url": "file:///tmp/wheel",
                },
            },
        }
    }
    return sealed_files, identity, wheel_sha


def test_installed_inventory_allows_only_verified_pip_outputs() -> None:
    sealed, identity, wheel_sha = _installed_inventory_fixture()
    assert ACCEPT.validate_installed_inventory(sealed, identity, wheel_sha) == 7

    changed = copy.deepcopy(identity)
    payload = next(
        entry
        for entry in changed["installed_inventory"]["files"]
        if entry["path"] == "cytoanvi/__init__.py"
    )
    payload["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="payload hash/size mismatch"):
        ACCEPT.validate_installed_inventory(sealed, changed, wheel_sha)

    unexpected = copy.deepcopy(identity)
    unexpected["installed_inventory"]["files"].append(
        {"path": "cytoanvi/unsealed.bin", "sha256": "0" * 64, "size_bytes": 1}
    )
    with pytest.raises(RuntimeError, match="unexpected generated files"):
        ACCEPT.validate_installed_inventory(sealed, unexpected, wheel_sha)

    wrong_direct_url = copy.deepcopy(identity)
    wrong_direct_url["installed_inventory"]["pip_generated"]["direct_url.json"]["archive_info"][
        "hashes"
    ]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="does not bind the wheel"):
        ACCEPT.validate_installed_inventory(sealed, wrong_direct_url, wheel_sha)


def test_installed_inventory_allows_only_exact_verified_console_script() -> None:
    sealed, identity, wheel_sha = _installed_inventory_fixture()
    installed = identity["installed_inventory"]
    name = "cytoanvi-install-skills"
    script_path = f"../../../bin/{name}"
    target = "scvi._skills.install:main"
    script = b"sealed launcher bytes"
    script_entry = {
        "path": script_path,
        "sha256": _sha(script),
        "size_bytes": len(script),
    }
    installed["files"].append(script_entry)
    installed["record"]["lines"].insert(-1, f"{script_path},{_record_hash(script)},{len(script)}")
    record = ("\n".join(installed["record"]["lines"]) + "\n").encode()
    record_entry = next(
        entry for entry in installed["files"] if entry["path"].endswith(".dist-info/RECORD")
    )
    record_entry.update({"sha256": _sha(record), "size_bytes": len(record)})
    installed["pip_generated"]["console_scripts"] = [
        {"name": name, "target": target, **script_entry}
    ]
    expected = {name: script_path}
    targets = {name: target}
    assert ACCEPT.validate_installed_inventory(
        sealed, identity, wheel_sha, expected, targets
    ) == 8

    bad_record = copy.deepcopy(identity)
    bad_record["installed_inventory"]["record"]["lines"][-2] = (
        f"{script_path},sha256=wrong,{len(script)}"
    )
    with pytest.raises(RuntimeError, match="RECORD SHA-256 mismatch"):
        ACCEPT.validate_installed_inventory(sealed, bad_record, wheel_sha, expected, targets)

    wrong_path = copy.deepcopy(identity)
    wrong_path["installed_inventory"]["pip_generated"]["console_scripts"][0]["path"] = (
        "../../../outside/cytoanvi-install-skills"
    )
    with pytest.raises(RuntimeError, match="path differs from fresh venv"):
        ACCEPT.validate_installed_inventory(sealed, wrong_path, wheel_sha, expected, targets)

    wrong_target = copy.deepcopy(identity)
    wrong_target["installed_inventory"]["pip_generated"]["console_scripts"][0]["target"] = (
        "unsealed.module:main"
    )
    with pytest.raises(RuntimeError, match="target differs from the wheel"):
        ACCEPT.validate_installed_inventory(
            sealed, wrong_target, wheel_sha, expected, targets
        )


def test_console_script_names_and_targets_are_derived_from_wheel(tmp_path: Path) -> None:
    wheel = tmp_path / "fixture.whl"
    entry_points = "[console_scripts]\ncytoanvi-install-skills = scvi._skills.install:main\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("cytoanvi-0.2.0.dist-info/entry_points.txt", entry_points)
    assert ACCEPT.wheel_console_scripts(wheel) == ACCEPT.EXPECTED_CONSOLE_SCRIPTS
    assert BUILD.wheel_console_scripts(wheel) == BUILD.EXPECTED_CONSOLE_SCRIPTS

    wrong = tmp_path / "wrong.whl"
    with zipfile.ZipFile(wrong, "w") as archive:
        archive.writestr(
            "cytoanvi-0.2.0.dist-info/entry_points.txt",
            "[console_scripts]\ncytoanvi-install-skills = unsealed.module:main\n",
        )
    with pytest.raises(ValueError, match="frozen source contract"):
        ACCEPT.wheel_console_scripts(wrong)
    with pytest.raises(RuntimeError, match="frozen source contract"):
        BUILD.wheel_console_scripts(wrong)


def test_installed_record_traversal_is_limited_to_fresh_venv_console_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "venv"
    scripts_dir = environment / "bin"
    site_root = environment / "lib/python3.13/site-packages"
    dist_info = site_root / "cytoanvi-0.2.0.dist-info"
    cyto = site_root / "cytoanvi"
    scvi = site_root / "scvi"
    for path in (scripts_dir, dist_info, cyto, scvi):
        path.mkdir(parents=True, exist_ok=True)
    python = scripts_dir / "python"
    python.write_text("")
    monkeypatch.setattr(SMOKE.sys, "executable", str(python))

    target = "scvi._skills.install:main"
    wrapper = f"""#!{python}
# -*- coding: utf-8 -*-
import re
import sys
from scvi._skills.install import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])
    sys.exit(main())
""".encode()
    payloads = {
        "cytoanvi/__init__.py": b"",
        "scvi/__init__.py": b"",
        "../../../bin/cytoanvi-install-skills": wrapper,
    }
    for relative, value in payloads.items():
        path = (site_root / relative).resolve()
        path.write_bytes(value)
    record_relative = "cytoanvi-0.2.0.dist-info/RECORD"
    record_lines = [
        f"{relative},{_record_hash(value)},{len(value)}" for relative, value in payloads.items()
    ]
    record_lines.append(f"{record_relative},,")
    (site_root / record_relative).write_text("\n".join(record_lines) + "\n")
    distribution = SimpleNamespace(_path=dist_info, locate_file=lambda _path: site_root)
    inventory = SMOKE.installed_distribution_inventory(
        distribution, {"cytoanvi-install-skills": target}
    )
    assert inventory["pip_generated"]["console_scripts"][0]["path"] == (
        "../../../bin/cytoanvi-install-skills"
    )

    record_lines.insert(-1, "../../../../outside-script,sha256=wrong,1")
    (site_root / record_relative).write_text("\n".join(record_lines) + "\n")
    with pytest.raises(RuntimeError, match="escapes site-packages"):
        SMOKE.installed_distribution_inventory(
            distribution, {"cytoanvi-install-skills": target}
        )

    record_lines.pop(-2)
    malicious = scripts_dir / "cytoanvi-install-skills"
    malicious.write_text(f"#!{python}\nimport sys\nsys.exit(99)\n")
    malicious_bytes = malicious.read_bytes()
    script_row = next(
        index
        for index, row in enumerate(record_lines)
        if row.startswith("../../../bin/cytoanvi-install-skills,")
    )
    record_lines[script_row] = (
        "../../../bin/cytoanvi-install-skills,"
        f"{_record_hash(malicious_bytes)},{len(malicious_bytes)}"
    )
    (site_root / record_relative).write_text("\n".join(record_lines) + "\n")
    with pytest.raises(RuntimeError, match="wrapper differs from the sealed target"):
        SMOKE.installed_distribution_inventory(
            distribution, {"cytoanvi-install-skills": target}
        )


def test_acceptance_inputs_reject_post_build_harness_drift(tmp_path: Path) -> None:
    entries = []
    for index, relative in enumerate(sorted(ACCEPT.ACCEPTANCE_INPUT_PATHS)):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        value = f"fixture-{index}\n".encode()
        path.write_bytes(value)
        entries.append({"path": relative, "sha256": _sha(value), "size_bytes": len(value)})
    ACCEPT.verify_acceptance_inputs(tmp_path, entries)

    drifted = tmp_path / "scripts/usage_readiness_installed_smoke.py"
    drifted.write_text("post-build drift\n")
    with pytest.raises(ValueError, match="drifted after the candidate build"):
        ACCEPT.verify_acceptance_inputs(tmp_path, entries)


def test_acceptance_subprocess_environment_drops_host_package_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "CONDA_PREFIX",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PIP_CONSTRAINT",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        "PIP_INDEX_URL",
        "PIP_NO_BINARY",
        "PIP_ONLY_BINARY",
        "PIP_REQUIREMENT",
        "PIP_TARGET",
        "PIP_USER",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "VIRTUAL_ENV",
    ):
        monkeypatch.setenv(name, "poisoned-host-value")
    monkeypatch.setenv("PATH", "/poisoned/host/path")
    environment = ACCEPT.isolated_subprocess_environment(tmp_path)
    assert environment["PIP_CONFIG_FILE"] == ACCEPT.os.devnull
    assert environment["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert environment["PATH"] == ACCEPT.os.defpath
    assert not any(
        name in environment
        for name in (
            "CONDA_PREFIX",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "LD_AUDIT",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "PIP_CONSTRAINT",
            "PIP_EXTRA_INDEX_URL",
            "PIP_FIND_LINKS",
            "PIP_INDEX_URL",
            "PIP_NO_BINARY",
            "PIP_ONLY_BINARY",
            "PIP_REQUIREMENT",
            "PIP_TARGET",
            "PIP_USER",
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONUSERBASE",
            "VIRTUAL_ENV",
        )
    )
