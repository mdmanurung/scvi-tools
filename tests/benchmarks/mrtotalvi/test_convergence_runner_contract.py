"""Pure execution contracts for the RDX-03 diagnosis runner."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import benchmarks.mrtotalvi.convergence_runner as convergence_runner
import benchmarks.mrtotalvi.run_convergence_diagnosis as convergence_diagnosis
import numpy as np
import pytest
from benchmarks.mrtotalvi.comparator import frozen_optimization_identity
from benchmarks.mrtotalvi.convergence_runner import (
    CANONICAL_HUMAN_CACHE_ENV,
    CANONICAL_HUMAN_CONTRACT_SHA256,
    CANONICAL_HUMAN_H5AD,
    CANONICAL_HUMAN_RUN_ID,
    CANONICAL_HUMAN_SHA256,
    CANONICAL_HUMAN_SPLIT_SHA256,
    _canonical_human_fixture,
    _canonical_human_read_path,
    _mrtotalvi_training_kwargs,
    _state_annotation_digest,
    _validate_evaluation_labels,
)
from benchmarks.mrtotalvi.human_lineage import sha256_lines
from benchmarks.mrtotalvi.manifest import sha256_file
from benchmarks.mrtotalvi.metric_schema import (
    load_metric_dictionary,
    metric_payload_template,
    validate_metric_dictionary,
)
from benchmarks.mrtotalvi.run_convergence_diagnosis import (
    CODE_MANIFEST_SCHEMA,
    FIT_MEMORY_SCOPE,
    FIT_WORKER_SCHEMA,
    LEGACY_CODE_MANIFEST_SCHEMA,
    REPO_ROOT,
    SCIENTIFIC_SCOPE,
    _convergence_code_manifest,
    _diagnostic_estimator_configuration,
    _fit_worker_command,
    _fit_worker_environment,
    _ordered_metric_payload,
    _partial_aggregate_v1,
    _publish_run_no_replace,
    _resolve_execution_contract_adapter,
    _run_isolated_diagnosis_fit,
    _validate_diagnostic_estimator_configuration,
    _validate_fit_worker_manifest,
    _validate_manifest_identity_bindings,
    _validate_recomputed_fit_identity,
    _verify_code_snapshot,
    _verify_live_sources_against_manifest,
    _write_code_snapshot,
    _write_metric_dictionary,
    validate_diagnosis_request,
)
from benchmarks.mrtotalvi.run_pilot import _payload_digest, _write_json
from benchmarks.mrtotalvi.versioning import (
    historical_redesign_contract_adapter,
    prospective_redesign_contract_adapter,
    version_binding_fields,
)


def _write_sealed_annotation_payload(
    path: Path,
    *,
    fixture_path: Path,
    cell_ids: list[str],
    labels: list[str],
    source_sha256: str,
) -> dict:
    records = [
        {"cell_id": cell_id, "label": label}
        for cell_id, label in zip(cell_ids, labels, strict=True)
    ]
    payload = {
        "schema_version": "mrtotalvi-sealed-500-state-annotations-v1",
        "fixture": {
            "fixture_id": "sealed_500",
            "h5ad_path": "sealed/checkpoint/adata.h5ad",
            "h5ad_sha256": sha256_file(fixture_path),
            "ordered_cell_ids_sha256": sha256_lines(cell_ids),
        },
        "annotation": {
            "column": "cell_label_l1p5",
            "source_h5ad_path": "/immutable/source/human_immune_joint.h5ad",
            "source_h5ad_sha256": source_sha256,
            "ordered_labels_sha256": sha256_lines(labels),
            "ordered_selection_sha256": sha256_lines(
                [
                    f"{cell_id}\t{label}"
                    for cell_id, label in zip(cell_ids, labels, strict=True)
                ]
            ),
        },
        "records": records,
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def test_legacy_execution_contract_resolution_uses_stored_digest_only():
    """A sealed v1 execution resolves without consulting the live factory."""
    configuration = {
        "schema_version": "mrtotalvi-convergence-execution-v2",
        "redesign_run_contract_digest": (
            "7cccca9b0b1863a9c345ba570bd39100193a1515ca23462375d46511cdc7402c"
        ),
    }

    adapter = _resolve_execution_contract_adapter(configuration)

    assert adapter.integrity_version == "v1"
    assert adapter.run_contract_digest == configuration[
        "redesign_run_contract_digest"
    ]


def test_prospective_diagnostic_estimators_bind_model_specific_scale_semantics():
    configuration = _diagnostic_estimator_configuration()

    assert configuration == {
        "schema_version": "mrtotalvi-rdx03-diagnostic-estimators-v1",
        "factual_z_posterior_scale": {
            "stock_scvi_totalvi": {
                "estimator": "analytic_qz.scale",
                "aggregation": "mean_over_cells_and_coordinates",
            },
            "mrtotalvi": {
                "estimator": "seeded_monte_carlo_sample_standard_deviation",
                "draws": 32,
                "ddof": 1,
                "rng_stream": "evaluation_seed",
                "aggregation": "mean_over_cells_and_coordinates",
            },
        },
    }
    assert _validate_diagnostic_estimator_configuration(configuration) == (
        configuration
    )
    changed = deepcopy(configuration)
    changed["factual_z_posterior_scale"]["mrtotalvi"]["ddof"] = 0
    with pytest.raises(ValueError, match="estimator configuration"):
        _validate_diagnostic_estimator_configuration(changed)


def test_historical_partial_aggregate_remains_exactly_unextended():
    request = validate_diagnosis_request(
        fixtures=("mixed",),
        rows=("B1",),
        seeds=(0,),
    )
    result = {
        "fixture_id": "mixed",
        "candidate_id": "B1",
        "training_seed": 0,
        "convergence": {"status": "converged"},
        "collapse": {"failed": False},
    }

    assert _partial_aggregate_v1([result], request=request) == {
        "schema_version": "mrtotalvi-convergence-partial-v2",
        "authoritative_full_grid": False,
        "n_fits": 1,
        "requested_fits": 1,
        "fixtures": ["mixed"],
        "rows": ["B1"],
        "training_seeds": [0],
        "fit_failures": [],
        "d0_decision": "not_available_for_non_authoritative_grid",
        "rdx03_completion": "prohibited",
        "scientific_scope": SCIENTIFIC_SCOPE,
    }



def test_mrtotalvi_training_kwargs_preserve_the_frozen_shared_control():
    """MrTotalVI cannot inherit TotalVI defaults that change the comparison."""
    fixture = SimpleNamespace(
        train_indices=np.asarray([0, 1, 2], dtype=np.int64),
        validation_indices=np.asarray([3, 4], dtype=np.int64),
        adata=SimpleNamespace(n_obs=5),
        batch_size=64,
        learning_rate=1e-3,
    )
    callback = object()

    kwargs = _mrtotalvi_training_kwargs(
        fixture,
        callback=callback,
    )

    external_indexing = kwargs.pop("external_indexing")
    np.testing.assert_array_equal(
        external_indexing[0],
        fixture.train_indices,
    )
    np.testing.assert_array_equal(
        external_indexing[1],
        fixture.validation_indices,
    )
    np.testing.assert_array_equal(
        external_indexing[2],
        np.asarray([], dtype=np.int64),
    )
    assert kwargs == {
        "max_epochs": 400,
        "min_epochs": 50,
        "accelerator": "cpu",
        "devices": 1,
        "train_size": None,
        "validation_size": None,
        "shuffle_set_split": False,
        "batch_size": 64,
        "early_stopping": True,
        "early_stopping_monitor": "elbo_validation",
        "early_stopping_mode": "min",
        "early_stopping_patience": 30,
        "early_stopping_min_delta": 0.0,
        "lr": 1e-3,
        "reduce_lr_on_plateau": False,
        "n_steps_kl_warmup": 3,
        "n_epochs_kl_warmup": None,
        "adversarial_classifier": False,
        "plan_kwargs": {
            "optimizer": "Adam",
            "weight_decay": 1e-6,
            "eps": 0.01,
            "gradient_clip_norm": None,
        },
        "check_val_every_n_epoch": 5,
        "callbacks": [callback],
        "enable_checkpointing": True,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }


def test_evaluation_labels_require_two_heldout_cells_per_observed_class():
    """A preregistered metric cannot discover an unevaluable class after fit."""
    validation = np.asarray([2, 3, 4, 5, 6], dtype=np.int64)
    labels = np.asarray(["train", "train", "a", "a", "b", "b", "b"])

    checked = _validate_evaluation_labels(
        labels,
        validation_indices=validation,
        name="state_labels",
    )
    np.testing.assert_array_equal(checked, labels)

    singleton = labels.copy()
    singleton[3] = "b"
    with pytest.raises(ValueError, match="at least two held-out cells"):
        _validate_evaluation_labels(
            singleton,
            validation_indices=validation,
            name="state_labels",
        )


def test_state_annotation_digest_is_order_and_value_sensitive():
    original = _state_annotation_digest(np.asarray(["a", "b", "a"]))

    assert original == _state_annotation_digest(
        np.asarray(["a", "b", "a"])
    )
    assert original != _state_annotation_digest(
        np.asarray(["a", "a", "b"])
    )
    assert original != _state_annotation_digest(
        np.asarray(["a", "b", "changed"])
    )


def test_canonical_human_fixture_is_bound_to_verified_v2_lineage(
    tmp_path,
    monkeypatch,
):
    """RDX-03 consumes only the independently equivalent v2 lineage run."""
    assert CANONICAL_HUMAN_RUN_ID == (
        "20260731T081355Z-991ec740-b50f4e3a-e6ce6542"
    )
    assert CANONICAL_HUMAN_H5AD == Path(
        ".scratch/mrtotalvi-v2-redesign/human-lineage-runs/"
        f"{CANONICAL_HUMAN_RUN_ID}/human-w00-w22.h5ad"
    )
    assert CANONICAL_HUMAN_SHA256 == (
        "37198b29dd5bbd9013969639cd7cbe99f1ed67269a52a17ef78fb994ffabee9b"
    )
    assert CANONICAL_HUMAN_CONTRACT_SHA256 == (
        "b50f4e3a4322b0c2f16337e40508f8eb3cd2b910963cd36096868e383a6bba78"
    )
    assert sha256_file(CANONICAL_HUMAN_H5AD) == CANONICAL_HUMAN_SHA256
    cache_path = tmp_path / "canonical-human.h5ad"
    shutil.copyfile(CANONICAL_HUMAN_H5AD, cache_path)
    monkeypatch.setenv(CANONICAL_HUMAN_CACHE_ENV, str(cache_path))
    assert _canonical_human_read_path(CANONICAL_HUMAN_H5AD) == cache_path

    import anndata as ad

    n_obs = 46_817
    split = np.full(n_obs, "train", dtype="<U7")
    split[37_447:] = "heldout"
    fake_adata = SimpleNamespace(
        shape=(n_obs, 5_000),
        obs={
            "lineage_split": split,
            "cell_label_l2": np.resize(["state_a", "state_b"], n_obs),
            "donor_timepoint": np.resize(["sample_a", "sample_b"], n_obs),
            "batch": np.resize(["batch_a", "batch_b"], n_obs),
        },
        uns={
            "mrtotalvi_human_lineage": {
                "run_id": CANONICAL_HUMAN_RUN_ID,
                "contract_sha256": CANONICAL_HUMAN_CONTRACT_SHA256,
                "factual_human_da": "locked_not_computed_or_inspected",
                "split_assignment_sha256": CANONICAL_HUMAN_SPLIT_SHA256,
            }
        },
    )

    def read_verified_cache(path):
        assert path == cache_path
        return fake_adata

    monkeypatch.setattr(ad, "read_h5ad", read_verified_cache)
    fixture = _canonical_human_fixture(Path("."))
    lineage = fixture.adata.uns["mrtotalvi_human_lineage"]

    assert fixture.adata.shape == (46_817, 5_000)
    assert fixture.source_data_digest == CANONICAL_HUMAN_SHA256
    assert lineage["run_id"] == CANONICAL_HUMAN_RUN_ID
    assert lineage["contract_sha256"] == CANONICAL_HUMAN_CONTRACT_SHA256
    assert (
        lineage["factual_human_da"]
        == "locked_not_computed_or_inspected"
    )


def test_canonical_human_cache_fails_closed_on_relative_or_changed_bytes(
    tmp_path,
    monkeypatch,
):
    """A local read cache cannot substitute different lineage bytes."""
    monkeypatch.setenv(CANONICAL_HUMAN_CACHE_ENV, "relative.h5ad")
    with pytest.raises(ValueError, match="absolute"):
        _canonical_human_read_path(CANONICAL_HUMAN_H5AD)

    changed = tmp_path / "changed.h5ad"
    changed.write_bytes(b"not the authoritative lineage bytes")
    monkeypatch.setenv(CANONICAL_HUMAN_CACHE_ENV, str(changed))
    with pytest.raises(ValueError, match="cache SHA-256"):
        _canonical_human_read_path(CANONICAL_HUMAN_H5AD)

    monkeypatch.setenv(
        CANONICAL_HUMAN_CACHE_ENV,
        str(CANONICAL_HUMAN_H5AD.resolve()),
    )
    with pytest.raises(ValueError, match="fixture SHA-256"):
        _canonical_human_read_path(changed)


def test_sealed_annotation_payload_closes_fixture_and_source_identity(tmp_path):
    fixture_path = tmp_path / "adata.h5ad"
    fixture_path.write_bytes(b"exact sealed fixture")
    cell_ids = ["cell-1", "cell-2", "cell-3"]
    labels = ["CD4 T", "CD8 T", "NK"]
    source_sha256 = "5" * 64
    payload_path = tmp_path / "sealed-500-state-annotations.json"
    _write_sealed_annotation_payload(
        payload_path,
        fixture_path=fixture_path,
        cell_ids=cell_ids,
        labels=labels,
        source_sha256=source_sha256,
    )

    observed = convergence_runner._load_sealed_500_state_annotations(
        payload_path,
        fixture_path=fixture_path,
        expected_cell_ids=tuple(cell_ids),
        expected_payload_sha256=sha256_file(payload_path),
        expected_fixture_sha256=sha256_file(fixture_path),
        expected_source_sha256=source_sha256,
        expected_state_digest=sha256_lines(labels),
    )

    assert observed.tolist() == labels


@pytest.mark.parametrize(
    "drift",
    ["payload_bytes", "fixture_bytes", "label", "order", "missing", "extra"],
)
def test_sealed_annotation_payload_rejects_tamper_order_and_inventory(
    tmp_path,
    drift,
):
    fixture_path = tmp_path / "adata.h5ad"
    fixture_path.write_bytes(b"exact sealed fixture")
    expected_cell_ids = ["cell-1", "cell-2", "cell-3"]
    expected_labels = ["CD4 T", "CD8 T", "NK"]
    observed_cell_ids = list(expected_cell_ids)
    observed_labels = list(expected_labels)
    if drift == "order":
        observed_cell_ids[0], observed_cell_ids[1] = (
            observed_cell_ids[1],
            observed_cell_ids[0],
        )
        observed_labels[0], observed_labels[1] = (
            observed_labels[1],
            observed_labels[0],
        )
    elif drift == "missing":
        observed_cell_ids.pop()
        observed_labels.pop()
    elif drift == "extra":
        observed_cell_ids.append("cell-4")
        observed_labels.append("B")
    source_sha256 = "5" * 64
    payload_path = tmp_path / "sealed-500-state-annotations.json"
    payload = _write_sealed_annotation_payload(
        payload_path,
        fixture_path=fixture_path,
        cell_ids=observed_cell_ids,
        labels=observed_labels,
        source_sha256=source_sha256,
    )
    expected_payload_sha256 = sha256_file(payload_path)
    if drift == "payload_bytes":
        payload["undeclared"] = True
        payload_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    elif drift == "fixture_bytes":
        fixture_path.write_bytes(b"changed fixture")
    elif drift == "label":
        payload["records"][0]["label"] = "tampered"
        payload_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        expected_payload_sha256 = sha256_file(payload_path)

    with pytest.raises(ValueError):
        convergence_runner._load_sealed_500_state_annotations(
            payload_path,
            fixture_path=fixture_path,
            expected_cell_ids=tuple(expected_cell_ids),
            expected_payload_sha256=expected_payload_sha256,
            expected_fixture_sha256=hashlib.sha256(
                b"exact sealed fixture"
            ).hexdigest(),
            expected_source_sha256=source_sha256,
            expected_state_digest=sha256_lines(expected_labels),
        )


def test_sealed_annotation_payload_preserves_frozen_fixture_digests():
    fixture = convergence_runner._sealed_500_fixture(Path("."))

    assert fixture.source_data_digest == (
        "b63a7df6b57d4db5bf0ce9e091ca36db9d19ad7c6ea798c7224a52f3a7d51dff"
    )
    assert fixture.state_annotation_digest == (
        "420db70e69a1007a9c4b406da298dc07960856f6276951b22672ce7c5dcb7c3c"
    )
    assert fixture.data_digest == (
        "2445302fd60a8fd8d6e025ccda28294d600dbd967eabddf16e7d63dab156feb1"
    )
    assert fixture.split_digest == (
        "d0ff980030476546bb802fe811b38b61377c6f6c118acf75736b28f4028c5413"
    )


def test_execution_request_distinguishes_exact_grid_from_non_authoritative_probe():
    full = validate_diagnosis_request(
        fixtures=(
            "mixed",
            "unequal_cells",
            "sealed_500",
            "canonical_human_if_available",
        ),
        rows=("B1", "B2", "B3", "D0"),
        seeds=(0, 1, 2),
    )
    assert full.authoritative_full_grid is True
    assert full.n_fits == 48

    probe = validate_diagnosis_request(
        fixtures=("mixed",),
        rows=("B1",),
        seeds=(0,),
    )
    assert probe.authoritative_full_grid is False
    assert probe.n_fits == 1

    with pytest.raises(ValueError, match="declared order"):
        validate_diagnosis_request(
            fixtures=("mixed",),
            rows=("B2", "B1"),
            seeds=(0,),
        )
    with pytest.raises(ValueError, match="Unknown"):
        validate_diagnosis_request(
            fixtures=("invented",),
            rows=("B1",),
            seeds=(0,),
        )


def test_persisted_metrics_are_reordered_to_the_frozen_dictionary():
    payload = metric_payload_template(
        contract_adapter=historical_redesign_contract_adapter(),
    )
    reversed_payload = dict(reversed(tuple(payload.items())))

    ordered = _ordered_metric_payload(reversed_payload)

    assert tuple(ordered) == tuple(payload)


def test_sealed_v3_metric_dictionary_preserves_contract_order(tmp_path):
    adapter = prospective_redesign_contract_adapter()
    payload = load_metric_dictionary(contract_adapter=adapter)
    path = tmp_path / "metric-dictionary.json"

    _write_metric_dictionary(path, payload)
    restored = load_metric_dictionary(
        path,
        contract_adapter=adapter,
    )

    assert tuple(restored["metrics"]) == adapter.metric_ids
    assert validate_metric_dictionary(
        restored,
        contract_adapter=adapter,
    ) == restored


def test_nested_publish_fallback_refuses_overwrite_and_moves_manifest_last(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source"
    nested = source / "checkpoints" / "fit"
    nested.mkdir(parents=True)
    (nested / "model.pt").write_text("state\n", encoding="utf-8")
    (source / "run-manifest.json").write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "destination"

    def unsupported(*_args, **_kwargs):
        raise RuntimeError(
            "No-replace fallback supports only non-empty regular-file runs."
        )

    monkeypatch.setattr(
        "benchmarks.mrtotalvi.run_convergence_diagnosis._rename_no_replace",
        unsupported,
    )
    _publish_run_no_replace(source, destination)

    assert not source.exists()
    assert (destination / "checkpoints" / "fit" / "model.pt").is_file()
    assert (destination / "run-manifest.json").is_file()

    second_source = tmp_path / "second-source"
    second_source.mkdir()
    (second_source / "run-manifest.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError):
        _publish_run_no_replace(second_source, destination)
    assert second_source.is_dir()


def test_sealed_manifest_identities_are_recomputed_not_trusted():
    code_payload = {
        "git_head": "a" * 40,
        "files": {"source.py": "b" * 64},
    }
    code_manifest = {
        **code_payload,
        "code_digest": _payload_digest(code_payload),
    }
    configuration = {
        "schema_version": "mrtotalvi-convergence-execution-v2",
        "fixture_manifests": [{"fixture_id": "mixed", "data_digest": "c" * 64}],
    }
    manifest = SimpleNamespace(
        code_digest=code_manifest["code_digest"],
        config_digest=_payload_digest(configuration),
        data_digest=_payload_digest(configuration["fixture_manifests"]),
    )

    _validate_manifest_identity_bindings(
        manifest,
        code_manifest=code_manifest,
        configuration=configuration,
    )

    for field in ("code_digest", "config_digest", "data_digest"):
        changed = SimpleNamespace(**vars(manifest))
        setattr(changed, field, "d" * 64)
        with pytest.raises(ValueError, match="identity"):
            _validate_manifest_identity_bindings(
                changed,
                code_manifest=code_manifest,
                configuration=configuration,
            )


def test_sealed_fit_recomputes_checkpoint_convergence_and_collapse():
    history = [
        {"epoch": 5 * (index + 1), "value": 100.0 - index}
        for index in range(31)
    ]
    checkpoint = {
        "monitor": "elbo_validation",
        "mode": "min",
        "epoch": 155,
        "value": 70.0,
        "state_digest": "a" * 64,
        "artifact_name": "best",
    }
    metrics = {
        "validation_objective_history": history,
        "best_checkpoint_identity": checkpoint,
        "factual_z_effective_rank": 3.0,
        "factual_z_latent_variance": 0.2,
        "factual_z_posterior_scale": 0.3,
        "latent_all_finite": 1.0,
    }
    collapse = {
        "failed": False,
        "failed_metrics": [],
        "required_representations": ["factual_z"],
        "latent_dimension": 4,
        "observed": {
            "factual_z_effective_rank": 3.0,
            "factual_z_latent_variance": 0.2,
            "factual_z_posterior_scale": 0.3,
            "latent_all_finite": 1.0,
        },
    }
    result = {
        "schema_version": "mrtotalvi-convergence-fit-v2",
        "candidate_id": "B1",
        "n_cells": 20,
        "latent_dimension": 4,
        "learning_rate": 1e-3,
        "training_history": {"elbo_validation": history},
        "best_checkpoint_identity": checkpoint,
        "convergence": {
            "status": "converged",
            "stable_validation_plateau": True,
            "stopped_early": True,
            "trainer_epochs": 155,
            "validation_checks": 31,
            "reached_minimum_epochs": True,
            "reached_maximum_epochs": False,
        },
        "collapse": collapse,
        "optimization_identity": frozen_optimization_identity(
            n_obs=20,
            learning_rate=1e-3,
        ),
        "metrics": metrics,
    }

    _validate_recomputed_fit_identity(
        result,
        artifact_state_digest="a" * 64,
    )

    changed = deepcopy(result)
    changed["convergence"]["status"] = "non_converged_at_maximum"
    with pytest.raises(ValueError, match="convergence"):
        _validate_recomputed_fit_identity(
            changed,
            artifact_state_digest="a" * 64,
        )

    changed = deepcopy(result)
    changed["collapse"]["failed"] = True
    with pytest.raises(ValueError, match="collapse"):
        _validate_recomputed_fit_identity(
            changed,
            artifact_state_digest="a" * 64,
        )

    with pytest.raises(ValueError, match="checkpoint state"):
        _validate_recomputed_fit_identity(
            result,
            artifact_state_digest="b" * 64,
        )

    changed = deepcopy(result)
    changed["optimization_identity"]["declared"]["eps"] = 1e-8
    with pytest.raises(ValueError, match="optimizer"):
        _validate_recomputed_fit_identity(
            changed,
            artifact_state_digest="a" * 64,
        )


def test_exact_source_snapshot_is_copied_and_reverified(tmp_path):
    relatives = (
        "benchmarks/__init__.py",
        "benchmarks/mrtotalvi/convergence.py",
        "benchmarks/mrtotalvi/metric_dictionary.json",
    )
    records = {
        relative: {
            "sha256": sha256_file(Path(relative)),
            "bytes": Path(relative).stat().st_size,
            "snapshot_path": f"code-snapshot/{relative}",
        }
        for relative in relatives
    }
    code_manifest = {
        "schema_version": CODE_MANIFEST_SCHEMA,
        "files": records,
    }

    _write_code_snapshot(tmp_path, code_manifest)
    _verify_code_snapshot(tmp_path, code_manifest)
    _verify_live_sources_against_manifest(code_manifest)

    snapshot = tmp_path / records[relatives[1]]["snapshot_path"]
    snapshot.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot drifted"):
        _verify_code_snapshot(tmp_path, code_manifest)


def test_prospective_manifest_closes_top_level_package_import(monkeypatch):
    """The package initializer executed by parent and worker is snapshotted."""
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.run_convergence_diagnosis._git",
        lambda *args: "clean" if args == ("status", "--short") else "identity",
    )
    adapter = prospective_redesign_contract_adapter()

    manifest = _convergence_code_manifest(contract_adapter=adapter)

    assert "benchmarks/__init__.py" in manifest["files"]
    assert manifest["files"]["benchmarks/__init__.py"][
        "snapshot_path"
    ] == "code-snapshot/benchmarks/__init__.py"
    assert manifest["runtime_imports"]["schema_version"] == (
        "mrtotalvi-runtime-import-identity-v2"
    )
    assert manifest["runtime_imports"]["entrypoints"] == {
        "benchmarks": "benchmarks/__init__.py",
        "scvi": "src/scvi/__init__.py",
    }
    assert manifest["runtime_imports"]["sys_path"] == [
        str(Path(value).resolve()) for value in sys.path if value
    ]
    assert "benchmarks.mrtotalvi.run_convergence_diagnosis" in (
        manifest["runtime_imports"]["loaded_modules"]
    )


@pytest.mark.parametrize("namespace", ["benchmarks.escape", "scvi.escape"])
def test_runtime_import_identity_rejects_any_loaded_namespace_escape(
    tmp_path,
    monkeypatch,
    namespace,
):
    escaped = ModuleType(namespace)
    escaped.__file__ = str(tmp_path / "outside.py")
    monkeypatch.setitem(sys.modules, namespace, escaped)

    with pytest.raises(ValueError, match="outside the declared source tree"):
        convergence_diagnosis._runtime_import_identity()


def test_worker_runtime_identity_binds_exact_parent_sys_path():
    identity = convergence_diagnosis._runtime_import_identity()
    changed = deepcopy(identity)
    changed["sys_path"].append("/writable/import/escape")

    with pytest.raises(ValueError, match="Runtime import identity drifted"):
        convergence_diagnosis._validate_runtime_import_identity(changed)


def test_fit_worker_uses_safe_path_no_user_site_and_exact_pythonpath(tmp_path):
    command = _fit_worker_command(
        fixture_id="mixed",
        candidate_id="B1",
        seed=0,
        output_dir=tmp_path / "worker",
        code_manifest_path=tmp_path / "code-manifest.json",
        configuration_path=tmp_path / "configuration.json",
        run_contract_path=tmp_path / "redesign-run-contract.json",
        latent_integrity_policy_path=tmp_path / "latent-integrity-policy.json",
        metric_dictionary_path=tmp_path / "metric-dictionary.json",
    )
    environment = _fit_worker_environment()

    assert command[1:3] == ("-P", "-m")
    assert environment["PYTHONSAFEPATH"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONPATH"] == str(REPO_ROOT / "src") + (
        f":{REPO_ROOT}"
    )


def test_legacy_snapshot_replay_does_not_require_new_package_record(tmp_path):
    """Prospective closure does not reinterpret sealed v1 source manifests."""
    relative = "benchmarks/mrtotalvi/convergence.py"
    code_manifest = {
        "schema_version": LEGACY_CODE_MANIFEST_SCHEMA,
        "files": {
            relative: {
                "sha256": sha256_file(Path(relative)),
                "bytes": Path(relative).stat().st_size,
                "snapshot_path": f"code-snapshot/{relative}",
            }
        },
    }

    _write_code_snapshot(tmp_path, code_manifest)
    _verify_code_snapshot(tmp_path, code_manifest)
    _verify_live_sources_against_manifest(code_manifest)


def test_isolated_worker_uses_writable_run_workspace_and_preserves_snapshot(
    tmp_path,
    monkeypatch,
):
    """Relative worker writes stay in-run and cannot mutate source snapshots."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    relative = "benchmarks/__init__.py"
    code_manifest = {
        "schema_version": CODE_MANIFEST_SCHEMA,
        "files": {
            relative: {
                "sha256": sha256_file(Path(relative)),
                "bytes": Path(relative).stat().st_size,
                "snapshot_path": f"code-snapshot/{relative}",
            }
        },
        "code_digest": "a" * 64,
    }
    _write_json(run_dir / "code-manifest.json", code_manifest)
    for name in (
        "configuration.json",
        "redesign-run-contract.json",
        "latent-integrity-policy.json",
        "metric-dictionary.json",
    ):
        _write_json(run_dir / name, {})
    _write_code_snapshot(run_dir, code_manifest)
    snapshot = run_dir / "code-snapshot" / relative
    snapshot_before = snapshot.read_bytes()
    source_before = (REPO_ROOT / relative).read_bytes()
    observed = {}

    def fake_run(command, *, cwd, env, stdout, stderr, check):
        workspace = Path(cwd)
        observed.update(command=command, cwd=workspace, env=env)
        assert workspace.is_dir()
        assert workspace.is_relative_to(run_dir)
        assert workspace != REPO_ROOT
        (workspace / "relative-worker-write.txt").write_text(
            "contained\n",
            encoding="utf-8",
        )
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir()
        (output / "worker-manifest.json").write_text("{}\n", encoding="utf-8")
        (output / "result.json").write_text("{}\n", encoding="utf-8")
        (output / "representation.npz").write_bytes(b"mock")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        "benchmarks.mrtotalvi.run_convergence_diagnosis.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.run_convergence_diagnosis."
        "_validate_fit_worker_manifest",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "benchmarks.mrtotalvi.run_convergence_diagnosis._read_representations",
        lambda *_args, **_kwargs: {},
    )

    _, _, worker_dir, _ = _run_isolated_diagnosis_fit(
        fixture_id="mixed",
        candidate_id="B1",
        seed=0,
        run_dir=run_dir,
        contract_adapter=prospective_redesign_contract_adapter(),
    )

    assert worker_dir.is_relative_to(run_dir)
    assert observed["cwd"].joinpath("relative-worker-write.txt").is_file()
    assert observed["env"]["PYTHONPATH"] == (
        f"{(REPO_ROOT / 'src').resolve()}:{REPO_ROOT.resolve()}"
    )
    assert observed["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert snapshot.read_bytes() == snapshot_before
    assert (REPO_ROOT / relative).read_bytes() == source_before


def test_fresh_fit_process_contract_is_independent_of_grid_order(tmp_path):
    fits = [
        ("mixed", "B1", 0),
        ("mixed", "B2", 0),
        ("unequal_cells", "B1", 1),
    ]

    def commands(ordered):
        shared = {
            "code_manifest_path": tmp_path / "code-manifest.json",
            "configuration_path": tmp_path / "configuration.json",
            "run_contract_path": tmp_path / "redesign-run-contract.json",
            "latent_integrity_policy_path": (
                tmp_path / "latent-integrity-policy.json"
            ),
            "metric_dictionary_path": tmp_path / "metric-dictionary.json",
        }
        return {
            key: _fit_worker_command(
                fixture_id=key[0],
                candidate_id=key[1],
                seed=key[2],
                output_dir=tmp_path
                / f"{key[0]}--{key[1]}--seed{key[2]}",
                **shared,
            )
            for key in ordered
        }

    assert commands(fits) == commands(reversed(fits))
    output_paths = {
        command[command.index("--output-dir") + 1]
        for command in commands(fits).values()
    }
    assert len(output_paths) == 3

    adapter = prospective_redesign_contract_adapter()
    payload = {
        "schema_version": FIT_WORKER_SCHEMA,
        **version_binding_fields(adapter),
        "fixture_id": "mixed",
        "candidate_id": "B1",
        "training_seed": 0,
        "process_id": 101,
        "parent_process_id": 100,
        "python_executable": str(Path(sys.executable).resolve()),
        "memory_scope": FIT_MEMORY_SCOPE,
        "code_digest": "a" * 64,
    }
    _validate_fit_worker_manifest(
        payload,
        fixture_id="mixed",
        candidate_id="B1",
        seed=0,
        code_digest="a" * 64,
        contract_adapter=adapter,
    )
    payload["parent_process_id"] = payload["process_id"]
    with pytest.raises(ValueError, match="not isolated"):
        _validate_fit_worker_manifest(
            payload,
            fixture_id="mixed",
            candidate_id="B1",
            seed=0,
            code_digest="a" * 64,
            contract_adapter=adapter,
        )
