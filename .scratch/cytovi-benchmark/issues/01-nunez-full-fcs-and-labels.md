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
