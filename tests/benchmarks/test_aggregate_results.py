"""Tests for benchmark result aggregation."""

from __future__ import annotations

import json
import sys

import pytest


def test_aggregate_results_input_directory_mode(tmp_path, monkeypatch):
    from benchmarks.common import aggregate_results

    results = tmp_path / "results"
    results.mkdir()
    (results / "roider_b3_s0.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "dataset": "roider",
                "b3": {
                    "task": "b3_panel_divergent",
                    "p1_holdout": {"macro_f1": 0.91},
                    "p2_concordance_vs_knn": {"agreement": 0.86},
                },
            }
        )
    )
    (results / "roider_b5_sweep_s0.json").write_text(
        json.dumps(
            {
                "seed": 0,
                "dataset": "roider",
                "b5": {
                    "task": "b5_holdout_sweep",
                    "best_auroc": 0.9,
                    "mean_auroc": 0.49,
                },
            }
        )
    )
    output = results / "final_summary.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results",
            "--input",
            str(results),
            "--output",
            str(output),
        ],
    )
    aggregate_results.main()

    payload = json.loads(output.read_text())
    assert payload["tasks"]["roider_b3_s0.json"]["b3"]["p1_holdout_macro_f1"] == 0.91
    # Old-key backward-compat: fixture uses p2_concordance_vs_knn, output column renamed.
    assert payload["tasks"]["roider_b3_s0.json"]["b3"]["p2_inter_method_agreement"] == 0.86
    assert payload["tasks"]["roider_b5_sweep_s0.json"]["b5"]["best_auroc"] == 0.9
    assert "b9" in payload["missing_optional"]
    assert str(output) not in payload["sources"]


def test_publication_manifest_rejects_unknown_nested_json(tmp_path, monkeypatch):
    from benchmarks.common import aggregate_results

    results = tmp_path / "results"
    nested = results / "stale"
    nested.mkdir(parents=True)
    expected = results / "b8.json"
    expected.write_text(
        json.dumps(
            {
                "dataset": "nunez",
                "seed": 0,
                "b8": {
                    "task": "b8_hce_label_transfer",
                    "flat_ce": {"macro_f1": 0.5},
                    "hce_hierarchical_predict": {"macro_f1": 0.7},
                    "delta_hierarchical_vs_flat_macro_f1": 0.2,
                },
            }
        )
    )
    (nested / "old.json").write_text(json.dumps({"dataset": "roider", "b3": {}}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "phase": "P5",
                        "task": "b8",
                        "path": str(expected),
                        "dataset": "nunez",
                        "seeds": [0],
                        "job_id": "25102620",
                        "status": "complete",
                        "required": True,
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results",
            "--manifest",
            str(manifest),
            "--input",
            str(results),
            "--output",
            str(tmp_path / "summary.json"),
        ],
    )
    with pytest.raises(SystemExit):
        aggregate_results.main()


def test_publication_manifest_rejects_positional_inputs(tmp_path, monkeypatch):
    from benchmarks.common import aggregate_results

    expected = tmp_path / "b8.json"
    expected.write_text(
        json.dumps(
            {
                "dataset": "nunez",
                "seed": 0,
                "b8": {
                    "task": "b8_hce_label_transfer",
                    "flat_ce": {"macro_f1": 0.5},
                    "hce_hierarchical_predict": {"macro_f1": 0.7},
                    "delta_hierarchical_vs_flat_macro_f1": 0.2,
                },
            }
        )
    )
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps({"dataset": "roider", "b3": {}}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "phase": "P5",
                        "task": "b8",
                        "path": str(expected),
                        "dataset": "nunez",
                        "seeds": [0],
                        "job_id": "25102620",
                        "status": "complete",
                        "required": True,
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results",
            str(stale),
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "summary.json"),
        ],
    )
    with pytest.raises(SystemExit):
        aggregate_results.main()


def test_publication_manifest_requires_boolean_required_field(tmp_path, monkeypatch):
    from benchmarks.common import aggregate_results

    expected = tmp_path / "b8.json"
    expected.write_text(
        json.dumps(
            {
                "dataset": "nunez",
                "seed": 0,
                "b8": {
                    "task": "b8_hce_label_transfer",
                    "flat_ce": {"macro_f1": 0.5},
                    "hce_hierarchical_predict": {"macro_f1": 0.7},
                    "delta_hierarchical_vs_flat_macro_f1": 0.2,
                },
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "phase": "P5",
                        "task": "b8",
                        "path": str(expected),
                        "dataset": "nunez",
                        "seeds": [0],
                        "job_id": "25102620",
                        "status": "complete",
                        "required": "false",
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "summary.json"),
        ],
    )
    with pytest.raises(ValueError, match="required.*boolean"):
        aggregate_results.main()


def test_publication_manifest_without_input_ignores_unknown_json(tmp_path, monkeypatch):
    from benchmarks.common import aggregate_results

    results = tmp_path / "results"
    nested = results / "stale"
    nested.mkdir(parents=True)
    expected = results / "b8.json"
    expected.write_text(
        json.dumps(
            {
                "dataset": "nunez",
                "seed": 0,
                "b8": {
                    "task": "b8_hce_label_transfer",
                    "flat_ce": {"macro_f1": 0.5},
                    "hce_hierarchical_predict": {"macro_f1": 0.7},
                    "delta_hierarchical_vs_flat_macro_f1": 0.2,
                },
            }
        )
    )
    (nested / "old.json").write_text(json.dumps({"dataset": "roider", "b3": {}}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "phase": "P5",
                        "task": "b8",
                        "path": str(expected),
                        "dataset": "nunez",
                        "seeds": [0],
                        "job_id": "25102620",
                        "status": "complete",
                        "required": True,
                    }
                ]
            }
        )
    )
    output = tmp_path / "summary.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )
    aggregate_results.main()

    payload = json.loads(output.read_text())
    assert list(payload["tasks"]) == ["b8.json"]
    assert str(expected) in payload["sources"]
    assert str(nested / "old.json") not in payload["sources"]


def test_publication_manifest_accepts_multiseed_per_seed_task_payload(
    tmp_path, monkeypatch
):
    from benchmarks.common import aggregate_results

    result = tmp_path / "b1_multiseed.json"
    result.write_text(
        json.dumps(
            {
                "dataset": "nunez",
                "seeds": [0, 1],
                "per_seed": {
                    "0": {
                        "seed": 0,
                        "b1": {
                            "task": "b1_label_transfer",
                            "cytoanvi": {"macro_f1": 0.9},
                            "cytovi_knn": {"macro_f1": 0.8},
                        },
                    },
                    "1": {
                        "seed": 1,
                        "b1": {
                            "task": "b1_label_transfer",
                            "cytoanvi": {"macro_f1": 0.92},
                            "cytovi_knn": {"macro_f1": 0.81},
                        },
                    },
                },
                "summary": {
                    "b1.cytoanvi.macro_f1": {"mean": 0.91, "std": 0.01},
                    "b1.cytovi_knn.macro_f1": {"mean": 0.805, "std": 0.005},
                },
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "phase": "P2",
                        "task": "b1",
                        "path": "b1_multiseed.json",
                        "dataset": "nunez",
                        "seeds": [0, 1],
                        "job_id": "local",
                        "status": "complete",
                        "required": True,
                    }
                ]
            }
        )
    )
    output = tmp_path / "summary.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )
    aggregate_results.main()

    payload = json.loads(output.read_text())
    summary = payload["tasks"]["b1_multiseed.json"]
    assert summary["b1_delta_macro_f1"]["mean"] == pytest.approx(0.105)


def test_publication_manifest_requires_required_artifacts(tmp_path, monkeypatch):
    from benchmarks.common import aggregate_results

    missing = tmp_path / "missing.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "phase": "P2",
                        "task": "b1",
                        "path": str(missing),
                        "dataset": "nunez",
                        "seeds": [0, 1, 2],
                        "job_id": "25102544",
                        "status": "pending",
                        "required": True,
                    }
                ]
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results",
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "summary.json"),
        ],
    )
    with pytest.raises(FileNotFoundError, match="required"):
        aggregate_results.main()


def test_publication_manifest_summarizes_optional_blocked_b9_and_b8_delta(
    tmp_path, monkeypatch
):
    from benchmarks.common import aggregate_results

    b8 = tmp_path / "b8.json"
    b8.write_text(
        json.dumps(
            {
                "dataset": "nunez",
                "seed": 0,
                "b8": {
                    "task": "b8_hce_label_transfer",
                    "flat_ce": {"macro_f1": 0.5},
                    "hce_hierarchical_predict": {"macro_f1": 0.7},
                    "delta_hierarchical_vs_flat_macro_f1": 0.2,
                },
            }
        )
    )
    b9 = tmp_path / "nunez_b9_s0.json"
    b9.write_text(
        json.dumps(
            {
                "dataset": "nunez",
                "seed": 0,
                "b9": {
                    "task": "b9_mapqc",
                    "status": "blocked",
                    "blocked_reason": "mapqc unavailable",
                },
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "phase": "P5",
                        "task": "b8",
                        "path": str(b8),
                        "dataset": "nunez",
                        "seeds": [0],
                        "job_id": "25102620",
                        "status": "complete",
                        "required": True,
                    },
                    {
                        "phase": "P6",
                        "task": "b9",
                        "path": str(b9),
                        "dataset": "nunez",
                        "seeds": [0],
                        "job_id": None,
                        "status": "blocked",
                        "required": False,
                    },
                ]
            }
        )
    )
    output = tmp_path / "summary.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_results",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
    )
    aggregate_results.main()

    payload = json.loads(output.read_text())
    assert payload["aggregation_mode"] == "publication_manifest"
    assert payload["tasks"]["b8.json"]["b8"]["delta_hierarchical_vs_flat_macro_f1"] == 0.2
    assert payload["tasks"]["nunez_b9_s0.json"]["b9"]["status"] == "blocked"
    assert payload["manifest_artifacts"][1]["required"] is False
