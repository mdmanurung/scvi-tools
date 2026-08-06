"""Authenticate and assess the immutable pre-v2 numerical oracles.

The CLI writes one canonical JSON assessment to stdout. Oracle inventory and
run-manifest authority are verified before importing model code, loading a
checkpoint, or opening an NPZ payload.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

EXPECTED_CHECKSUM_MANIFEST_SHA256 = (
    "fe0f8a8ff48098579dd398e70e1f3513c02da54a6708c384f45c97592e33ea38"
)
EXPECTED_SOURCE_COMMIT = "d8c8e997a67997a53f55923eb3ab14e6cf06f94c"
EXPECTED_PORTABILITY_POLICY_SHA256 = (
    "385fd0e959001dd7ed44437b31d1f055fb82ba3bf74014fefa221559d865ef54"
)
INTENTIONAL_PARENT_PATH = "../generate_legacy_oracle.py"
ORACLE_MODEL_NAMES = ("mrtotalvi", "mrmultivi")
PORTABILITY_POLICY_PATH = Path(__file__).with_name("cpu_portability_policy.json")
DEFAULT_ORACLE_ROOT = Path(__file__).with_name("d8c8e997")

_CHECKSUM_LINE = re.compile(r"^(?P<digest>[0-9a-f]{64})  (?P<path>[^\r\n]+)$")
_THREAD_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TORCH_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def canonical_json_bytes(value: object) -> bytes:
    """Return strict, sorted, whitespace-free JSON with one terminal newline."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 of the canonical JSON representation."""
    return _canonical_digest(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        value[key] = item
    return value


def _load_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return value


def _load_json_file(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return _load_json_bytes(path.read_bytes(), label=label)


def _parse_checksum_records(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Oracle checksum manifest is not UTF-8.") from exc
    if not text.endswith("\n"):
        raise ValueError("Oracle checksum manifest must end with one newline.")

    records: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"Malformed checksum record on line {line_number}.")
        relative = match.group("path")
        if relative in records:
            raise ValueError(f"Duplicate checksum path: {relative!r}.")
        if "\\" in relative:
            raise ValueError(f"Checksum path must use POSIX separators: {relative!r}.")
        parsed = PurePosixPath(relative)
        if parsed.is_absolute():
            raise ValueError(f"Absolute checksum path is forbidden: {relative!r}.")
        if ".." in parsed.parts and relative != INTENTIONAL_PARENT_PATH:
            raise ValueError(f"Parent checksum path is forbidden: {relative!r}.")
        if "." in parsed.parts or not parsed.parts:
            raise ValueError(f"Non-canonical checksum path: {relative!r}.")
        records[relative] = match.group("digest")

    if [path for path in records if path.startswith("../")] != [INTENTIONAL_PARENT_PATH]:
        raise ValueError(
            "The checksum manifest must contain only the intentional parent generator path."
        )
    return records


def _actual_inventory_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(f"Symlink is forbidden in oracle inventory: {relative!r}.")
            paths.add(relative)
    return paths


def verify_oracle_inventory(
    oracle_root: Path,
    *,
    expected_manifest_sha256: str = EXPECTED_CHECKSUM_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Verify exact paths and bytes in an immutable oracle tree.

    ``expected_manifest_sha256`` exists for semantic unit tests. Production and
    CLI callers use the pinned default and cannot obtain it from the oracle.
    """
    root = Path(oracle_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Oracle root must be a regular directory: {root}")
    checksum_path = root / "checksums.sha256"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise ValueError("Oracle checksum manifest must be a regular file.")
    raw_manifest = checksum_path.read_bytes()
    actual_manifest_sha256 = _sha256_bytes(raw_manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "Oracle checksum-manifest digest mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}."
        )
    records = _parse_checksum_records(raw_manifest)

    internal_files = {path for path in records if path != INTENTIONAL_PARENT_PATH}
    expected_paths = {"checksums.sha256", *internal_files}
    for relative in internal_files:
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            expected_paths.add(parent.as_posix())
            parent = parent.parent
    actual_paths = _actual_inventory_paths(root)
    missing_paths = sorted(expected_paths - actual_paths)
    extra_paths = sorted(actual_paths - expected_paths)
    if missing_paths or extra_paths:
        raise ValueError(
            "Oracle inventory paths differ from the checksum authority: "
            f"missing={missing_paths}, extra={extra_paths}."
        )

    for relative, expected_digest in records.items():
        path = (
            root.parent / "generate_legacy_oracle.py"
            if relative == INTENTIONAL_PARENT_PATH
            else root / relative
        )
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Oracle checksum target must be a regular file: {relative!r}.")
        actual_digest = _sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Oracle checksum mismatch for {relative!r}: "
                f"expected {expected_digest}, got {actual_digest}."
            )

    environment = _load_json_file(
        root / "environment_manifest.json",
        label="oracle environment manifest",
    )
    if environment.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise ValueError("Oracle environment source commit is not authoritative.")
    return {
        "checksum_manifest_sha256": actual_manifest_sha256,
        "external_paths": [INTENTIONAL_PARENT_PATH],
        "source_commit": environment["source_commit"],
        "verified_file_count": len(records),
        "verified_paths": sorted(records),
    }


def _validate_policy(policy: Mapping[str, Any]) -> None:
    expected_top_level = {
        "decision_use",
        "derivation",
        "execution",
        "oracle",
        "requirements",
        "schema",
        "tolerances",
    }
    if set(policy) != expected_top_level:
        raise ValueError("CPU-portability policy has unexpected top-level fields.")
    if policy["schema"] != "mrtotalvi-pre-v2-oracle-cpu-portability-v1":
        raise ValueError("Unsupported CPU-portability policy schema.")
    if policy["oracle"] != {
        "checksum_manifest_sha256": EXPECTED_CHECKSUM_MANIFEST_SHA256,
        "source_commit": EXPECTED_SOURCE_COMMIT,
    }:
        raise ValueError("CPU-portability policy is not bound to the exact oracle.")
    if policy["execution"] != {"device": "cpu", "dtype": "float32"}:
        raise ValueError("CPU-portability policy must be float32 and CPU-only.")
    if policy["decision_use"] != {"promotion": False, "selection": False}:
        raise ValueError("CPU-portability evidence cannot select or promote a method.")
    if policy["requirements"] != {
        "array_keys": "exact",
        "finite": "required",
        "gradient_manifest": "exact_keys_shapes_and_presence",
        "shapes": "exact",
        "state_manifest": "exact_keys_dtypes_and_shapes",
    }:
        raise ValueError("CPU-portability policy requirements drifted.")
    strict = policy["tolerances"].get("strict")
    portable = policy["tolerances"].get("portable")
    if strict != {"atol": 1e-7, "rtol": 1e-6}:
        raise ValueError("Strict tolerances must remain oracle-manifest tolerances.")
    if portable != {"atol": 5e-6, "rtol": 1e-6}:
        raise ValueError("Portable tolerances drifted.")
    derivation = policy["derivation"]
    if derivation != {
        "observed_floor_requiring_max_abs_delta": 3.814697265625e-6,
        "observed_overall_max_abs_delta": 7.62939453125e-6,
        "portable_atol_ceiling": 5e-6,
        "rule": (
            "absolute_floor_is_the_decimal_ceiling_above_near_zero_cpu_float32_"
            "residuals;the_larger_loss_delta_is_covered_by_the_shared_relative_term"
        ),
    }:
        raise ValueError("Portable tolerance derivation or ceiling drifted.")
    if portable["atol"] > derivation["portable_atol_ceiling"]:
        raise ValueError("Portable absolute tolerance exceeds its derivation ceiling.")
    if derivation["observed_floor_requiring_max_abs_delta"] > portable["atol"]:
        raise ValueError("Portable absolute tolerance does not cover its derivation datum.")


def load_portability_policy(path: Path) -> tuple[dict[str, Any], bytes, str]:
    """Load a canonical policy and return its payload, bytes, and SHA-256."""
    policy_path = Path(path)
    if policy_path.is_symlink() or not policy_path.is_file():
        raise ValueError(f"CPU-portability policy must be a regular file: {policy_path}")
    raw = policy_path.read_bytes()
    policy_sha256 = _sha256_bytes(raw)
    if policy_sha256 != EXPECTED_PORTABILITY_POLICY_SHA256:
        raise ValueError(
            "CPU-portability policy digest mismatch: "
            f"expected {EXPECTED_PORTABILITY_POLICY_SHA256}, got {policy_sha256}."
        )
    policy = _load_json_bytes(raw, label="CPU-portability policy")
    if raw != canonical_json_bytes(policy):
        raise ValueError("CPU-portability policy is not canonical JSON.")
    _validate_policy(policy)
    return policy, raw, policy_sha256


def _expected_run_manifest(model_name: str) -> dict[str, Any]:
    return {
        "batch_size": 7,
        "forward_seed": 9817,
        "kl_weight": 0.73,
        "model_seed": 7301,
        "model_type": "MrTotalVI" if model_name == "mrtotalvi" else "MrMultiVI",
        "oracle_stage": "post_checkpoint_roundtrip",
        "protein_reconstruction_weight": 0.61 if model_name == "mrtotalvi" else None,
        "sample_order": ["sample_0", "sample_1", "sample_2"],
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "tolerances": {"atol": 1e-7, "rtol": 1e-6},
    }


def _load_run_manifests(oracle_root: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for model_name in ORACLE_MODEL_NAMES:
        manifest = _load_json_file(
            oracle_root / model_name / "run_manifest.json",
            label=f"{model_name} run manifest",
        )
        expected = _expected_run_manifest(model_name)
        if manifest != expected:
            raise ValueError(
                f"{model_name} run manifest drifted from its exact source/seed/weight contract."
            )
        manifests[model_name] = manifest
    return manifests


def _array_category(key: str) -> str:
    if key == "gradient.full":
        return "gradient_full"
    if key.startswith("gradient."):
        return "gradient"
    if key.startswith("reconstruction_loss."):
        return "reconstruction_loss"
    if key.startswith("kl_local."):
        return "kl_local"
    if key == "loss":
        return "loss"
    return "forward"


def _comparison_row(
    *,
    key: str,
    actual: Any,
    expected: Any,
    strict: Mapping[str, float],
    portable: Mapping[str, float],
) -> dict[str, Any]:
    import numpy as np

    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.dtype != np.dtype("float32") or expected_array.dtype != np.dtype("float32"):
        raise ValueError(f"{key} must be float32 in both replay and oracle.")
    if actual_array.shape != expected_array.shape:
        raise ValueError(
            f"{key} shape mismatch: actual={actual_array.shape}, oracle={expected_array.shape}."
        )
    actual_finite = np.isfinite(actual_array)
    expected_finite = np.isfinite(expected_array)
    both_finite = actual_finite & expected_finite
    actual64 = actual_array.astype(np.float64)
    expected64 = expected_array.astype(np.float64)
    delta = np.zeros(actual_array.shape, dtype=np.float64)
    np.subtract(actual64, expected64, out=delta, where=both_finite)
    np.abs(delta, out=delta)

    normalized: dict[str, float | None] = {}
    failing: dict[str, int] = {}
    for tolerance_name, tolerance in (("strict", strict), ("portable", portable)):
        denominator = tolerance["atol"] + tolerance["rtol"] * np.abs(expected64)
        normalized_values = np.zeros(actual_array.shape, dtype=np.float64)
        np.divide(delta, denominator, out=normalized_values, where=both_finite)
        normalized[tolerance_name] = (
            float(normalized_values[both_finite].max()) if both_finite.any() else None
        )
        failing[tolerance_name] = int(np.count_nonzero(~both_finite | (delta > denominator)))

    return {
        "category": _array_category(key),
        "dtype": str(actual_array.dtype),
        "failing_count": failing,
        "key": key,
        "max_abs_delta": float(delta[both_finite].max()) if both_finite.any() else None,
        "max_normalized_error": normalized,
        "n": int(actual_array.size),
        "nonfinite": {
            "actual": int(np.count_nonzero(~actual_finite)),
            "oracle": int(np.count_nonzero(~expected_finite)),
        },
        "shape": list(actual_array.shape),
    }


def _validate_state_manifest(module: Any, model_dir: Path) -> None:
    expected_state = _load_json_file(
        model_dir / "state_manifest.json",
        label=f"{model_dir.name} state manifest",
    )
    actual_state = module.state_dict()
    if set(actual_state) != set(expected_state):
        raise ValueError(f"{model_dir.name} state keys differ from the oracle manifest.")
    for key, tensor in actual_state.items():
        expected = expected_state[key]
        if set(expected) != {"dtype", "shape"}:
            raise ValueError(f"Malformed state-manifest row for {key!r}.")
        if expected != {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}:
            raise ValueError(f"State metadata mismatch for {key!r}.")
        if tensor.dtype != __import__("torch").float32 or tensor.device.type != "cpu":
            raise ValueError(f"State tensor {key!r} is not CPU float32.")


def _validate_gradient_manifest(
    module: Any,
    model_dir: Path,
) -> list[tuple[str, Any]]:
    gradient_manifest = _load_json_file(
        model_dir / "gradient_manifest.json",
        label=f"{model_dir.name} gradient manifest",
    )
    parameters = list(module.named_parameters())
    if {name for name, _ in parameters} != set(gradient_manifest):
        raise ValueError(f"{model_dir.name} gradient parameter keys drifted.")
    for name, parameter in parameters:
        expected = gradient_manifest[name]
        if set(expected) != {"present", "selected", "shape"}:
            raise ValueError(f"Malformed gradient-manifest row for {name!r}.")
        if expected["shape"] != list(parameter.shape):
            raise ValueError(f"Gradient shape metadata mismatch for {name!r}.")
        if expected["present"] != (parameter.grad is not None):
            raise ValueError(f"Gradient presence metadata mismatch for {name!r}.")
        if parameter.dtype != __import__("torch").float32 or parameter.device.type != "cpu":
            raise ValueError(f"Parameter {name!r} is not CPU float32.")
    return parameters


def _assess_model(
    *,
    model_name: str,
    run_manifest: dict[str, Any],
    oracle_root: Path,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    from scvi import REGISTRY_KEYS
    from scvi.external import MrMultiVI, MrTotalVI

    model_cls = MrTotalVI if model_name == "mrtotalvi" else MrMultiVI
    model_dir = oracle_root / model_name
    model = model_cls.load(model_dir / "checkpoint", accelerator="cpu")
    module = model.module.eval()
    if list(map(str, model.sample_order)) != run_manifest["sample_order"]:
        raise ValueError(f"{model_name} checkpoint sample order drifted.")
    _validate_state_manifest(module, model_dir)

    batch_size = run_manifest["batch_size"]
    loader = model._make_data_loader(
        adata=model.adata,
        indices=np.arange(batch_size),
        batch_size=batch_size,
    )
    tensors = next(iter(loader))
    if any(
        isinstance(value, torch.Tensor) and value.device.type != "cpu"
        for value in tensors.values()
    ):
        raise ValueError(f"{model_name} data loader produced a non-CPU tensor.")

    loss_kwargs = {"kl_weight": run_manifest["kl_weight"]}
    if run_manifest["protein_reconstruction_weight"] is not None:
        loss_kwargs["pro_recons_weight"] = run_manifest["protein_reconstruction_weight"]
    module.zero_grad(set_to_none=True)
    torch.manual_seed(run_manifest["forward_seed"])
    inference_outputs, _, loss_output = module(tensors, loss_kwargs=loss_kwargs)
    actual_arrays: dict[str, np.ndarray] = {
        "qu.loc": inference_outputs["qu"].loc.detach().cpu().numpy(),
        "qu.scale": inference_outputs["qu"].scale.detach().cpu().numpy(),
        "u": inference_outputs["u"].detach().cpu().numpy(),
        "z_base": inference_outputs["z_base"].detach().cpu().numpy(),
        "eps_raw_legacy": inference_outputs["eps"].detach().cpu().numpy(),
        "z": inference_outputs["z"].detach().cpu().numpy(),
        "loss": loss_output.loss.detach().cpu().numpy(),
    }
    actual_arrays.update(
        {
            f"reconstruction_loss.{key}": value.detach().cpu().numpy()
            for key, value in loss_output.reconstruction_loss.items()
        }
    )
    actual_arrays.update(
        {
            f"kl_local.{key}": value.detach().cpu().numpy()
            for key, value in loss_output.kl_local.items()
        }
    )

    loss_output.loss.backward()
    parameters = _validate_gradient_manifest(module, model_dir)
    gradient_parts: list[np.ndarray] = []
    for parameter_name, parameter in parameters:
        gradient = (
            torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach()
        )
        gradient_array = gradient.cpu().numpy()
        actual_arrays[f"gradient.{parameter_name}"] = gradient_array
        gradient_parts.append(gradient_array.reshape(-1))
    actual_arrays["gradient.full"] = np.concatenate(gradient_parts)

    strict = policy["tolerances"]["strict"]
    portable = policy["tolerances"]["portable"]
    with np.load(model_dir / "oracle_arrays.npz", allow_pickle=False) as expected_arrays:
        expected_keys = set(expected_arrays.files)
        required_keys = {*actual_arrays, "cell_indices"}
        if expected_keys != required_keys:
            raise ValueError(
                f"{model_name} oracle array keys differ: "
                f"missing={sorted(required_keys - expected_keys)}, "
                f"extra={sorted(expected_keys - required_keys)}."
            )
        expected_indices = expected_arrays["cell_indices"]
        actual_indices = tensors[REGISTRY_KEYS.INDICES_KEY].detach().cpu().numpy()
        if (
            expected_indices.dtype != np.dtype("int64")
            or actual_indices.dtype != np.dtype("int64")
            or expected_indices.shape != actual_indices.shape
            or not np.array_equal(expected_indices, actual_indices)
        ):
            raise ValueError(f"{model_name} cell-index replay drifted.")
        rows = [
            _comparison_row(
                key=key,
                actual=actual_arrays[key],
                expected=expected_arrays[key],
                strict=strict,
                portable=portable,
            )
            for key in actual_arrays
        ]

    finite = all(row["nonfinite"] == {"actual": 0, "oracle": 0} for row in rows)
    return {
        "assessed_keys": [row["key"] for row in rows],
        "requirements": {
            "array_keys_exact": True,
            "cell_indices_exact": True,
            "finite": finite,
            "gradient_manifest_exact": True,
            "shapes_exact": True,
            "state_manifest_exact": True,
        },
        "rows": rows,
        "run_manifest": run_manifest,
    }


def _cpu_provenance(torch: Any) -> dict[str, Any]:
    cpu_model = None
    cpu_flags: list[str] = []
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.is_file():
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if not separator:
                continue
            if key.strip() == "model name" and cpu_model is None:
                cpu_model = value.strip()
            elif key.strip() in {"flags", "Features"} and not cpu_flags:
                cpu_flags = sorted(value.split())
            if cpu_model is not None and cpu_flags:
                break
    get_capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    torch_capability = get_capability() if get_capability is not None else None
    return {
        "flags": cpu_flags,
        "flags_count": len(cpu_flags),
        "flags_sha256": _canonical_digest(cpu_flags),
        "machine": platform.machine(),
        "model": cpu_model,
        "processor": platform.processor(),
        "torch_capability": torch_capability,
    }


def _runtime_provenance(torch: Any, np: Any, scvi: Any) -> dict[str, Any]:
    return {
        "cpu": _cpu_provenance(torch),
        "determinism": {
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "execution": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": "cpu",
            "dtype": "float32",
        },
        "runtime": {
            "anndata": importlib.metadata.version("anndata"),
            "executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "scvi_tools": scvi.__version__,
            "torch": torch.__version__,
        },
        "threads": {
            "environment": {key: os.environ.get(key) for key in sorted(_THREAD_ENV)},
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "torch_num_threads": torch.get_num_threads(),
        },
    }


def _require_fixed_process_environment() -> None:
    drift = {
        key: {"expected": expected, "actual": os.environ.get(key)}
        for key, expected in _THREAD_ENV.items()
        if os.environ.get(key) != expected
    }
    if drift:
        raise ValueError(
            "CPU-portability assessment requires the fixed subprocess environment: "
            f"{drift}. Use run_assessment_subprocess()."
        )


def assess_cpu_portability(
    *,
    oracle_root: Path = DEFAULT_ORACLE_ROOT,
    policy_path: Path = PORTABILITY_POLICY_PATH,
) -> dict[str, Any]:
    """Return a raw, digest-bound strict/portable assessment.

    Authentication and both run-manifest validations intentionally precede
    numerical-library imports and every checkpoint/NPZ load.
    """
    root = Path(oracle_root)
    inventory = verify_oracle_inventory(root)
    policy, _, policy_sha256 = load_portability_policy(policy_path)
    run_manifests = _load_run_manifests(root)
    for manifest in run_manifests.values():
        if manifest["tolerances"] != policy["tolerances"]["strict"]:
            raise ValueError("Run-manifest and policy strict tolerances differ.")
    _require_fixed_process_environment()

    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    import scvi

    if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-portability assessment requires CUDA to be masked.")
    models = {
        model_name: _assess_model(
            model_name=model_name,
            run_manifest=run_manifests[model_name],
            oracle_root=root,
            policy=policy,
        )
        for model_name in ORACLE_MODEL_NAMES
    }
    provenance = _runtime_provenance(torch, np, scvi)
    provenance_sha256 = _canonical_digest(provenance)
    assessment_rows_sha256 = _canonical_digest(
        {model_name: models[model_name]["rows"] for model_name in ORACLE_MODEL_NAMES}
    )
    binding_without_digest = {
        "assessment_rows_sha256": assessment_rows_sha256,
        "oracle_checksum_manifest_sha256": inventory["checksum_manifest_sha256"],
        "policy_sha256": policy_sha256,
        "provenance_sha256": provenance_sha256,
        "source_commit": inventory["source_commit"],
    }
    binding = {
        **binding_without_digest,
        "binding_sha256": _canonical_digest(binding_without_digest),
    }

    rows = [row for model in models.values() for row in model["rows"]]
    strict_failures = sum(row["failing_count"]["strict"] for row in rows)
    portable_failures = sum(row["failing_count"]["portable"] for row in rows)
    nonfinite = sum(row["nonfinite"]["actual"] + row["nonfinite"]["oracle"] for row in rows)
    if strict_failures == 0:
        strict_status = "pass"
        strict_classification = "exact_manifest_replay"
    elif portable_failures == 0 and nonfinite == 0:
        strict_status = "host_rounding_mismatches"
        strict_classification = "cpu_float32_host_rounding"
    else:
        strict_status = "out_of_policy"
        strict_classification = "not_cpu_portable"
    return {
        "binding": binding,
        "models": models,
        "provenance": provenance,
        "schema": "mrtotalvi-pre-v2-oracle-cpu-portability-assessment-v1",
        "summary": {
            "assessed_array_count": len(rows),
            "nonfinite_count": nonfinite,
            "portable_failing_element_count": portable_failures,
            "portable_verdict": ("pass" if portable_failures == 0 and nonfinite == 0 else "fail"),
            "strict_diagnostic": {
                "classification": strict_classification,
                "failing_element_count": strict_failures,
                "status": strict_status,
                "tolerances": policy["tolerances"]["strict"],
            },
        },
    }


def run_assessment_subprocess(
    *,
    oracle_root: Path = DEFAULT_ORACLE_ROOT,
    policy_path: Path = PORTABILITY_POLICY_PATH,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Run the raw assessment in a fresh, fixed-thread Python process."""
    repository_root = Path(__file__).parents[4]
    environment = os.environ.copy()
    environment.update(_THREAD_ENV)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository_root / "src"), str(repository_root)]
    )
    command = [
        sys.executable,
        "-m",
        "tests.external.mrtotalvi.legacy_oracle.cpu_portability",
        "--oracle-root",
        str(Path(oracle_root).resolve()),
        "--policy",
        str(Path(policy_path).resolve()),
    ]
    completed = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CPU-portability assessment subprocess failed "
            f"with exit {completed.returncode}:\n"
            f"stdout:\n{completed.stdout.decode(errors='replace')}\n"
            f"stderr:\n{completed.stderr.decode(errors='replace')}"
        )
    assessment = _load_json_bytes(
        completed.stdout,
        label="CPU-portability assessment stdout",
    )
    if completed.stdout != canonical_json_bytes(assessment):
        raise ValueError("CPU-portability assessment stdout is not canonical JSON.")
    return assessment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Assess authenticated pre-v2 oracles under the CPU-portability policy."
    )
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--policy", type=Path, default=PORTABILITY_POLICY_PATH)
    args = parser.parse_args(argv)
    with contextlib.redirect_stdout(sys.stderr):
        assessment = assess_cpu_portability(
            oracle_root=args.oracle_root,
            policy_path=args.policy,
        )
    sys.stdout.buffer.write(canonical_json_bytes(assessment))


if __name__ == "__main__":
    main()
