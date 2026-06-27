"""Unit test for B7 task (no full training)."""

from __future__ import annotations

from scvi.external.cytovi.paired_cytoanvi import prepare_paired_cytoanvi

from benchmarks.cytoanvi.paired_rna_cytof import make_synthetic_paired_rna_cytof


def test_make_synthetic_paired():
    rna, cy, markers = make_synthetic_paired_rna_cytof(seed=0)
    assert rna.n_obs == 200
    assert cy.n_obs == 300
    assert len(markers) >= 4
    assert all(m in cy.var_names for m in markers)
    assert "sample_id" in rna.obs.columns
    assert "sample_id" in cy.obs.columns
    shared = set(rna.obs["sample_id"]) & set(cy.obs["sample_id"])
    assert len(shared) >= 1


def test_prepare_paired_integration():
    rna, cy, markers = make_synthetic_paired_rna_cytof(seed=1)
    merged, out_markers = prepare_paired_cytoanvi(
        rna, cy, markers=markers, nn_count=5, npcs=3, unlabeled_category="Unknown"
    )
    assert out_markers == markers
    assert merged.n_obs == rna.n_obs + cy.n_obs
    assert "scaled" in merged.layers
    assert set(merged.obs["modality"]) == {"RNA", "CyTOF"}
    assert "sample_id" in merged.obs.columns
