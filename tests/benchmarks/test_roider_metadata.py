"""Tests for Roider full-cohort metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.common.roider_metadata import entity_map_from_vignette, load_patient_entity_map


@pytest.fixture
def vignette_h5ad():
    path = Path("data/Roider_et_al_BNHL_panel1.h5ad")
    if not path.exists():
        pytest.skip("vignette panel1 h5ad not on disk")
    return path


def test_entity_map_from_vignette(vignette_h5ad):
    mapping = entity_map_from_vignette(vignette_h5ad)
    assert len(mapping) >= 30
    assert set(mapping.values()) <= {"DLBCL", "FL", "MCL", "MZL", "rLN"}
    assert mapping["LN0024"] == "FL"
    assert mapping["LN0076"] == "rLN"


def test_entity_map_from_supplementary():
    from benchmarks.common.roider_metadata import SUPPLEMENTARY_XLSX, entity_map_from_supplementary

    if not SUPPLEMENTARY_XLSX.exists():
        pytest.skip("supplementary workbook not on disk")
    mapping = entity_map_from_supplementary()
    assert len(mapping) == 101
    assert mapping["LN0040"] == "MCL"
    assert mapping["LN0002"] == "DLBCL"
    assert set(mapping.values()) <= {"DLBCL", "FL", "MCL", "MZL", "rLN"}


def test_load_patient_entity_full_cohort_coverage(tmp_path, monkeypatch):
    from benchmarks.common.roider_metadata import SUPPLEMENTARY_XLSX, load_patient_entity_map

    if not SUPPLEMENTARY_XLSX.exists():
        pytest.skip("supplementary workbook not on disk")
    out = tmp_path / "patient_entity.json"
    monkeypatch.setattr("benchmarks.common.roider_metadata.ENTITY_JSON", out)
    mapping = load_patient_entity_map(refresh=True)
    assert len(mapping) >= 63
    assert mapping["LN0040"] == "MCL"
    assert "Unknown" not in mapping.values()


def test_load_patient_entity_json_roundtrip(vignette_h5ad, tmp_path, monkeypatch):
    out = tmp_path / "patient_entity.json"
    monkeypatch.setattr("benchmarks.common.roider_metadata.ENTITY_JSON", out)
    mapping = load_patient_entity_map(vignette_h5ad, refresh=True)
    assert out.exists()
    assert json.loads(out.read_text()) == mapping
