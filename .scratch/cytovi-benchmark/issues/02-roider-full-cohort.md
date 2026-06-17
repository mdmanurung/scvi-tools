# 02 — Ingest full Roider BNHL cohort (63 patients)

Status: ready-for-human
Blocked-by: —

## Task

Replace vignette subsampled `.h5ad` with the **full** Roider flow cytometry cohort.

- Source: Figshare [24915633](https://doi.org/10.6084/m9.figshare.24915633) (raw `.fcs` per paper)
- Gating: viable singlet T cells (FlowJo-equivalent filters in Python or pre-gated exports)
- Preprocessing: arcsinh cofactor **500**, min-max [0, 1]
- Subsample: **10,000 T cells per patient** (paper Methods)
- Panels: 2 antibody panels, 8 shared backbone markers; merge with `cytovi.merge_batches`
- Metadata: `PatientID`, `panel_batch`, disease entity (`rLN`, `DLBCL`, `MCL`, `MZL`, `FL`, …)

Keep vignette `.h5ad` (56891468/71) as `--smoke` only.

## Acceptance

- ~63 patients × 10k cells in merged object
- Disease entity column present (enables Track B B4/B6 case-control)
- Loader returns `(merged, panel1, panel2)` compatible with existing B3 API

## Comments
