# 01 — Acquire full Nuñez flow PBMC data + manual labels

Status: ready-for-human
Blocked-by: —

## Task

Download and preprocess the **full** Nuñez et al. flow cytometry PBMC data (not vignette subsample).

- FCS: Figshare `55982654` (batch 1), `55982657` (batch 2)
- Preprocessing: arcsinh cofactor **2000**, min-max [0, 1] on proteins + scatter
- Labels: reproduce paper manual annotation (Leiden on NN graph → leukocyte subsets) or obtain
  author-provided labels
- Exclude ambiguous cells for A2 (per paper)

Deliver `benchmarks/cytovi/data/nunez/` (or shared `benchmarks/common/data/`) with:
- `nunez_batch1.h5ad`, `nunez_batch2.h5ad`, `nunez_merged.h5ad`
- `metadata.json`: cell counts, obs column names, label version

## Acceptance

- Merged object has **≥ 50k cells** (paper uses 100k downsample; full QC'd cohort acceptable)
- `labels`, `batch` columns documented
- Loader in `benchmarks/cytovi/data.py` + shared preprocessing

## Comments

### 2026-06-17 — vignette labels (Track B unblocked for tutorial subsample)

- Vignette FCS in `data/Nunez_PBMCs_batch{1,2}.fcs` (200k merged cells).
- Tutorial 11-type labels: `data/nunez_annotated.h5ad` via
  `python -m benchmarks.cytoanvi.annotate_nunez` (CytoVI + Leiden r=0.4 + manual map).
- **Full cohort** (≥50k beyond vignette / paper manual gating) still open — see YosefLab
  cytovi-reproducibility notebook for paper-aligned workflow.

### 2026-06-17

Validate vignette assets: `python -m benchmarks.common.fetch_data --validate-only`
Fetch from networked host: `python -m benchmarks.common.fetch_data --fetch`
Full-cohort notes: `python -m benchmarks.common.fetch_data --list-full-cohort`
