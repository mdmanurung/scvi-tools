# Algorithm Manifest

Tracks models and algorithms implemented in `src/`.

## CytoANVI (`src/cytoanvi/`) — RELEASE CANDIDATE

Semi-supervised, annotation-aware extension of CytoVI for antibody-based cytometry (flow, mass cytometry, CITE-seq protein).

| Component | File | Description |
|-----------|------|-------------|
| Model | `_model.py` | `CytoANVI` — main user-facing API |
| Module | `_module.py` | `CytoANVAE` — VAE with M1+M2 hierarchy and classifier |
| HCE | `_hce.py` | Hierarchical cross-entropy for cell type hierarchy |
| Continual | `_continual.py` | `ContinualUpdate` — EWC + replay for case-control updates |
| Uncertainty | `_uncertainty.py` | TTA-based uncertainty estimation |
| Hierarchy | `hierarchy.py` | scHPL / treeArches helpers |
| Mapping QC | `mapping_qc.py` | `score_query_mapping` / mapping QC scoring |

**Branch**: `feat/cytoanvi`
**Status**: implementation complete; public API is release-candidate until the required
Roider full-cohort B3/B5 artifacts pass strict publication-manifest aggregation.

## CytoVI (`src/scvi/external/cytovi/`) — STABLE

CytoVI: variational inference for mass/flow cytometry data. Baseline for CytoANVI benchmarks.

## Other external models

See `src/scvi/external/` for cellassign, contrastivevi, decipher, diagvi, gimvi, methylvi, mrvi, poissonvi, resolvi, scar, scbasset, scviva, solo, stereoscope, sysvi.
