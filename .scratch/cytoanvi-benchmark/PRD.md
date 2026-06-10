# PRD: CytoANVI real-data benchmarking

Status: ready-for-human
Owner: mdmanurung
Branch: feat/cytoanvi
Created: 2026-06-10

## Problem

CytoANVI (semi-supervised CytoVI + paper-faithful cscanvi continual update) passes 20/20 synthetic
tests but has **no real-data validation** — synthetic accuracy is meaningless (independent
ref/query). We need to show, on real cytometry, that:

1. the semi-supervised classifier transfers labels better than CytoVI's k-NN, while preserving a
   well-integrated latent; and
2. the panel-aware query mapping (`prepare_query_anndata` + scArches surgery) and novelty
   uncertainty (`get_uncertainty`) work on a genuine multi-panel dataset.

## Approach

Benchmark against CytoVI's **own** workflow on the datasets the CytoVI vignettes use (no
substituted data — primary-source rule).

- **D1 — Roider BNHL** (CytoVI *advanced* tutorial): 2 antibody panels, 10 shared backbone markers,
  ~9,966 cells, 4 batches, 33 donors, labels in panel 1 only. Figshare files `56891468`/`56891471`
  (preprocessed `.h5ad`). Primary — exactly CytoANVI's target scenario.
- **D2 — Nuñez PBMC** (CytoVI *batch* tutorial): single panel, 35 markers, 2 batches, 11 labelled
  populations. Figshare `55982654`/`55982657` (`.fcs`). Secondary.

Baseline throughout: **CytoVI latent + k-NN label transfer** (the vignette's method).

## Tasks (→ CytoANVI feature)

| # | Measures | API | Baseline | Issue |
|---|----------|-----|----------|-------|
| B1 | label-transfer accuracy / macro-F1 on held-out labels | `predict` | CytoVI + kNN | 02 |
| B2 | batch mixing vs bio conservation of the latent | `get_latent_representation` | CytoVI latent | 02 |
| B3 | panel-1 → panel-2 mapping | `prepare_query_anndata` + `load_query_data` + `predict` | CytoVI kNN (concordance) | 03 |
| B5 | flags a held-out (novel) cell type | `get_uncertainty` | — | 04 |
| B4 | continual case-control update preserves reference while surfacing case signal | `load_query_data_with_replay` | — | 06 (deferred) |
| B6 | tune `ewc_importance` (λ) for CytoVI's intensity likelihood | (output: default) | — | 06 (deferred) |

## Success criteria (pre-registered)

- **B1:** CytoANVI macro-F1 ≥ CytoVI k-NN on the holdout (target +0.03 absolute).
- **B2:** CytoANVI within ±0.02 cLISI/silhouette of CytoVI at equal-or-better batch mixing.
- **B3:** panel-1 holdout accuracy > chance and ≥ kNN; panel-2 concordance ≥ 0.7.
- **B5:** held-out-type AUROC > 0.7.
- **B4/B6:** a λ exists where reference-drift stays low and the case population is recovered; record
  it as the CytoVI-specific default (replaces the paper's 100).

## Artifacts

- Harness: `benchmarks/cytoanvi/` (committed `2f7cf944`). Smoke-tested end-to-end on synthetic data.
- Full plan + provenance: `notes/2026-06-10-cytoanvi-benchmark-plan.md`.

## Status / blockers

- Data download is **blocked in the dev environment** (Figshare returns HTTP 202 / egress blocked).
  Needs a human to fetch the files (issue 01).
- D2 needs an FCS reader (`readfcs`/`flowio`) not in the `scvi-test` env (issue 05).
- B4/B6 need a case/control axis the Roider tumor data lacks (issue 06).
- B1–B3, B5 are ready to run the moment D1 data lands.
