---
date: 2026-06-17
topic: cytovi-cytoanvi-benchmark
branch: feat/cytoanvi
status: plan
summary: >-
  Unified benchmark plan for Track A (scvi-tools CYTOVI vs paper/reference) and Track B
  (CytoANVI vs CytoVI). Full datasets, max_epochs=1000, scib-metrics throughout.
decisions:
  tracks: both
  data: full cohorts (not vignette subsamples)
  max_epochs: 1000
  integration_metrics: scib-metrics
---

# CytoVI / CytoANVI benchmark plan (unified)

**Paper:** Ingelfinger et al., *CytoVI: Deep generative modeling of antibody-based single cell
technologies* (bioRxiv [10.1101/2025.09.07.674699](https://doi.org/10.1101/2025.09.07.674699)).

**Reference code:** [YosefLab/cytovi-reproducibility](https://github.com/YosefLab/cytovi-reproducibility),
[YosefLab/cytovi-reference-implementation](https://github.com/YosefLab/cytovi-reference-implementation).

**Decisions (2026-06-17):**

| # | Decision |
|---|----------|
| 1 | Implement **both** Track A (CYTOVI fidelity) and Track B (CytoANVI vs CytoVI) |
| 2 | Use **full** paper cohorts — not vignette subsamples |
| 3 | Training: **`max_epochs=1000`** (paper default) |
| 4 | Integration: **`scib-metrics`** (`Benchmarker`, batch + bio aggregates) — no lightweight proxies |

Supersedes the metric/training defaults in `notes/2026-06-10-cytoanvi-benchmark-plan.md` for integration
and epoch count. Task definitions (B1–B6, A1–A6) are retained and extended below.

---

## Architecture

```
benchmarks/
  common/
    preprocessing.py   # arcsinh cofactors, min-max, harmonized obs keys
    scib.py            # scib-metrics Benchmarker wrapper (shared by both tracks)
    training.py        # paper-faithful CYTOVI / CytoANVI train helpers (epochs=1000)
    seeds.py           # multiseed runner + JSON aggregation
  cytovi/              # Track A — paper reproduction
    data.py
    tasks_ppc.py       # A1
    tasks_integration.py  # A2
    tasks_imputation.py   # A3
    tasks_multipanel.py   # A4
    tasks_roider_da.py    # A5
    baselines.py       # FA, Harmony, cyCombinePy
    run.py
    README.md
  cytoanvi/            # Track B — CytoANVI vs CytoVI (extend existing harness)
    data.py            # extend loaders for full cohorts
    tasks.py           # B1–B5, B7–B9
    baselines.py
    metrics.py         # label-transfer + novelty only; integration → common/scib.py
    run.py
    README.md
```

**Environment:** `scib-gpu` or `scvi-test` with `scib-metrics`, `python-igraph`, `leidenalg`,
`readfcs`/`flowio`, and optional [cyCombinePy](https://github.com/mdmanurung/cyCombinePy) for A2 baselines (batch correction only).

**Invocation (all runs):**

```bash
ENV=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scib-gpu  # or scvi-test
PYTHONPATH=src:. LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python \
  -m benchmarks.<track>.run --seed 0 --max-epochs 1000 ...
```

**Multiseed:** ≥3 seeds (0, 1, 2); report mean ± SD; pre-registered targets evaluated on mean.

---

## Shared training & preprocessing defaults

Aligned with paper Methods (pp. 21–23, 27–31).

| Parameter | Value |
|-----------|-------|
| `max_epochs` | **1000** |
| `n_latent` | `get_n_latent_heuristic(n_shared_markers)` unless experiment specifies fixed d |
| Prior | MoG, K = n_latent (isotropic only where paper says so, e.g. B-cell atlas d=4) |
| Label-informed prior | λ = 10 when labels available |
| Likelihood | Gaussian on arcsinh + min-max scaled data |
| Label transfer k | 20 (CytoVI k-NN baseline) |
| Imputation posterior samples | 50 (A3); 10 for visualization tasks |

### Preprocessing cofactors (arcsinh) + scaling

| Dataset | arcsinh cofactor | Scale | Notes |
|---------|------------------|-------|-------|
| Nuñez flow PBMC | 2000 | [0, 1] min-max | exclude FSC/SSC from protein-only steps where paper does |
| Roider BNHL T cells | 500 | [0, 1] | viable singlet T cells after FlowJo gating |
| Kreutmair flow PBMC | 2000 | [0, 1] | exclude CD1c (unspecific) |
| Ingelfinger mass cyt PBMC | 10 | [0, 1] | exclude CD138 |
| Hao CITE-seq protein | 5 | [0, 1] | 1 donor baseline for cross-tech |
| Glass B-cell mass cyt | 5 | [0, 1] | 48k B cells, equal per sample |

### scib-metrics protocol (all integration benchmarks)

Use `scib_metrics.benchmark.Benchmarker` on the latent embedding `obsm["X_latent"]`:

- **Batch key:** experimental batch / panel / study ID per experiment
- **Label key:** harmonized cell-type annotations; exclude ambiguous cells (paper A2)
- **Subsampling for scib compute:** paper uses 10k cells/batch for A2; for full cohorts use
  **10k per batch stratum** (fixed seed) unless total n < 20k (then use all cells)
- **Metrics:** default `BatchCorrection()` + `BioConservation()` bundles
- **Report:** per-metric scores + `batch_correction` aggregate + `bio_conservation` aggregate +
  overall score (same as paper Figure 2E table)
- **Embedding:** latent mean from `get_latent_representation()`; build neighbors on latent before
  scib if required by metric

---

## Track A — scvi-tools CYTOVI vs paper / reference

**Goal:** Quantitatively confirm scvi-tools `CYTOVI` reproduces the paper's benchmarks and matches
the reference implementation where outputs are available.

### Datasets (full cohorts)

| ID | Paper fig | Source | Full cohort spec | Access |
|----|-----------|--------|------------------|--------|
| **A-D1** | S2, 2 | Nuñez et al. 2023 flow PBMC | 100k cells, 35 markers + FSC/SSC; manual labels | Authors / Figshare FCS (`55982654`, `55982657`) — use **all cells** after QC, not vignette subsample |
| **A-D1-batch** | 2E | Nuñez technical replicate | Same donor, 2 batches, labelled | Same FCS files (batch 1 + batch 2) |
| **A-D2** | S2 | Ingelfinger mass cyt PBMC | 100k cells, 2 healthy donors, 36 markers | Authors |
| **A-D3** | S2 | Hao CITE-seq protein | 1 donor baseline, 228 ADT markers | GEO GSE164378 |
| **A-D4** | 3 | Nuñez + Kreutmair flow | Nuñez 100k + Kreutmair 100k; 15 shared backbone | Kreutmair: [10.17632/ffkvft27ds.2](https://doi.org/10.17632/ffkvft27ds.2) |
| **A-D5** | 3E | Glass B-cell mass cyt | 12 panels × 2 donors, union 350 markers; 48k B cells | FlowRepository FR-FCM-Z2MA |
| **A-D6** | 4 | Roider BNHL lymph node | **63 patients**, 2 panels, **10k T cells/patient**, disease entities incl. rLN | Figshare [24915633](https://doi.org/10.6084/m9.figshare.24915633) — **not** the tutorial `.h5ad` subsample |
| **A-D7** | 5 | Clinical CLL | 95 patients (45 positive + 50 controls) | On request (BASEC 2024-00006) — **Phase 4** |

Vignette `.h5ad` files (`56891468`, `56891471`) are smoke-test / dev only once full A-D6 is wired.

### Tasks

#### A1 — Posterior predictive checks (Fig S2)

| Item | Spec |
|------|------|
| Data | A-D1, A-D2, A-D3 |
| Models | CytoVI × 3 likelihood/preproc configs; sklearn Factor Analysis baseline |
| Latent d | 10 (mass cyt), 20 (flow, CITE) |
| Metric | MAE, Pearson, Spearman on CV (cell axis + protein axis) across 10 PPC samples |
| Pass | scvi-tools CYTOVI within ±0.05 of reference repo MAE/Pearson on A-D1 |

#### A2 — Batch integration (Fig 2E) — **priority**

| Item | Spec |
|------|------|
| Data | A-D1-batch (Nuñez replicate), full cells, ambiguous labels excluded |
| Methods | CytoVI, Harmony, cyCombinePy |
| Preproc sweep | min-max, z-score, rank (9 conditions × 3 methods) |
| Metric | **scib-metrics** full table + aggregates |
| Subsample | 10k cells/batch for scib (paper) |
| Pass | CytoVI aggregate ≥ best non-CytoVI method on ≥2/3 preproc schemes; within ±0.05 of
  reference repo on min-max (paper's recommended preproc) |

#### A3 — Semi-synthetic imputation (Fig S4)

| Item | Spec |
|------|------|
| Data | A-D1 single donor, 50k cells; drop CXCR3, PD-1 |
| Protocol | 2 pseudo-batches; mask one marker per iteration; 50 posterior samples |
| Baselines | KNN (k=10, data space) — cyCombine imputation omitted (cyCombinePy is batch-correction only) |
| Metric | Per-marker Pearson/Spearman; binary GMM positivity; uncertainty–error correlation |
| Pass | Mean Pearson across markers within ±0.05 of paper Table S4 / reference repo |

#### A4 — Multi-panel integration (Fig 3)

| Item | Spec |
|------|------|
| Data | A-D4 (Nuñez + Kreutmair full) |
| Setup | Label-informed MoG (λ=10); batch = study |
| Metric | scib on latent; optional per-marker imputation Pearson on held-out panel markers |
| Deliverable | Quantitative table + UMAP artifacts |

#### A5 — Roider differential abundance (Fig 4E–N)

| Item | Spec |
|------|------|
| Data | A-D6 full cohort |
| Pipeline | Train CytoVI (batch covariate) → DA scores per disease entity → k-means on
  [latent ∥ DA scores] |
| Metrics | ICC of cluster frequencies between panels (paper: 0.99); confusion vs manual T-cell
  labels; Mann-Whitney cluster freq by entity vs rLN |
| Pass | ICC ≥ 0.95; ≥1 DA cluster per major entity with p < 0.05 vs rLN |

#### A6 — CLL clinical transfer (Fig 5) — deferred

Requires A-D7 data access. Reference 10+10 patients, query 75, scArches + kNN, 5-fold CV AUC.

### Track A phases

| Phase | Tasks | Depends on |
|-------|-------|------------|
| **A-P0** | Harness skeleton, `common/scib.py`, `common/training.py` | scib-metrics env |
| **A-P1** | A2 (integration) | A-D1-batch FCS + labels |
| **A-P2** | A3 (imputation) | A-D1 |
| **A-P3** | A1 (PPC) | A-D1, A-D2, A-D3 |
| **A-P4** | A4, A5 | Kreutmair, full Roider |
| **A-P5** | `compare_ref.py` vs cytovi-reproducibility | A-P1–A-P3 outputs |
| **A-P6** | A6 | data access |

---

## Track B — CytoANVI vs original CytoVI

**Goal:** On the same full cohorts, show CytoANVI's semi-supervised classifier, query mapping, and
uncertainty improve on (or match) CytoVI's k-NN workflow without sacrificing integration quality.

**Baseline (all tasks):** `CYTOVI` latent + k-NN (k=20) ≡ `impute_categories_from_reference`.

### Datasets

| ID | Track A link | CytoANVI-specific use |
|----|--------------|------------------------|
| **B-D1** | A-D6 (full Roider) | Panel 1 labelled → panel 2 query; DA / case-control (rLN vs lymphoma) |
| **B-D2** | A-D1-batch (full Nuñez replicate) | B1 holdout label transfer; B2 integration (cleanest fully-labelled setting) |
| **B-D3** | A-D4 | Optional B7 imputation comparison |

### Tasks

| # | Question | Data | CytoANVI API | Baseline | Metrics |
|---|----------|------|--------------|----------|---------|
| **B1** | Classifier beats k-NN? | B-D2 | `setup_anndata` → `train` → `predict` | CytoVI k-NN | accuracy, macro-F1, per-class recall |
| **B2** | Latent integrates without losing biology? | B-D2 | `get_latent_representation` | CytoVI latent | **scib-metrics** (full aggregates) |
| **B3** | Panel-divergent mapping | B-D1 | `prepare_query_anndata` + `load_query_data` + `predict` | CytoVI k-NN | panel-1 holdout F1; panel-2 concordance |
| **B4** | Continual update preserves reference | B-D1 (rLN ref + healthy replay + lymphoma query) | `load_query_data_with_replay` | CytoVI static + k-NN | reference-latent drift (replay L2); DA recovery |
| **B5** | Uncertainty flags novel type | B-D1 or B-D2 | `get_uncertainty` | — | AUROC; **sweep all holdout types** |
| **B6** | Tune λ (`ewc_importance`) | B-D4 | `train(plan_kwargs={ewc_importance})` | — | drift vs case-signal curve; pick knee |
| **B7** | Imputation parity | B-D3 | shared decoder | CytoVI alone | per-marker Pearson (A3 protocol) |
| **B8** | DA cluster quality | B-D1 | same latent pipeline | CytoVI | ICC, entity enrichment |
| **B9** | Query latent fidelity | B-D1 | `load_query_data` | full retrain | mean L2(query latent) vs joint train |

### Pre-registered success criteria (mean over ≥3 seeds)

| Task | Criterion |
|------|-----------|
| B1 | CytoANVI macro-F1 ≥ CytoVI k-NN + **0.03** |
| B2 | CytoANVI scib `bio_conservation` within **±0.02** of CytoVI; `batch_correction` ≥ CytoVI |
| B3 | Panel-1 holdout macro-F1 ≥ k-NN; panel-2 concordance ≥ **0.70** |
| B4 | ∃ λ with replay drift < 0.1 × baseline drift **and** known entity DA cluster recovered |
| B5 | best holdout-type AUROC > **0.70** |
| B6 | documented CytoVI-specific λ default (replaces RNA default 100) |

### Track B phases

| Phase | Tasks | Notes |
|-------|-------|-------|
| **B-P0** | Migrate `metrics.py` integration → `common/scib.py`; default `max_epochs=1000` | breaks current lightweight B2 |
| **B-P1** | B1, B2 on B-D2 full Nuñez | needs `readfcs` |
| **B-P2** | B3, B5 sweep on B-D1 full Roider | replaces vignette `.h5ad` |
| **B-P3** | B4, B6 using rLN vs lymphoma on B-D1 | unblocks deferred continual |
| **B-P4** | B7–B9 | after A3/A4 infrastructure exists |

### Prior results (vignette subsample, epochs=100, lightweight B2 — **not final**)

`roider_seed0_summary.json`: B1/B3 pass; B5 fails on "Naive CD4 T" (AUROC 0.59). Re-run all on
full A-D6 with scib and epochs=1000 before PR claims.

---

## Data acquisition plan (full cohorts)

| Priority | Dataset | Action |
|----------|---------|--------|
| P0 | Nuñez FCS (both batches) | Download Figshare; verify cell count ≫ vignette; build manual labels per paper |
| P0 | Roider raw | Figshare 24915633; reimplement gating + 10k/patient subsample in `data.py` |
| P1 | Kreutmair | Mendeley 10.17632/ffkvft27ds.2 |
| P2 | Mass cyt PBMC | Request from Ingelfinger authors or locate public deposit |
| P2 | CITE-seq GSE164378 | Standard GEO pull |
| P3 | Glass B cells | FlowRepository FR-FCM-Z2MA |
| P4 | CLL clinical | Ethics-gated request |

**Vignette `.h5ad` (56891468/71):** keep for `--smoke` and CI; mark `subsampled=True` in loader metadata.

---

## Issue tracker

| Track | PRD | Issues dir |
|-------|-----|------------|
| Shared infra | `.scratch/cytovi-benchmark/PRD.md` | `.scratch/cytovi-benchmark/issues/` |
| Track B | `.scratch/cytoanvi-benchmark/PRD.md` (updated) | `.scratch/cytoanvi-benchmark/issues/` |

---

## Execution order (recommended)

1. **Shared P0** — `benchmarks/common/{scib,training,preprocessing}.py`; wire scib into cytoanvi B2
2. **A-P1** — A2 Nuñez integration (validates CYTOVI = paper; highest leverage)
3. **B-P1** — B1/B2 on same Nuñez data (head-to-head with scib)
4. **A-P2 + data** — Full Roider ingest; A5 + B-P2 in parallel
5. **A-P2** — A3 imputation
6. **A-P3–A-P5** — remaining Track A
7. **B-P3** — B4/B6 (rLN case-control now available on full Roider)
8. **A-P6 / B-P4** — clinical + extensions

---

## Provenance

- Paper: `cytovi.pdf` (bioRxiv 10.1101/2025.09.07.674699)
- Prior plan: `notes/2026-06-10-cytoanvi-benchmark-plan.md`
- CytoVI vignettes: [batch](https://docs.scvi-tools.org/en/1.4.0/tutorials/notebooks/cytometry/CytoVI_batch_correction_tutorial.html),
  [advanced](https://docs.scvi-tools.org/en/1.4.1/tutorials/notebooks/cytometry/CytoVI_advanced_tutorial.html)
- Repro repo: https://github.com/YosefLab/cytovi-reproducibility
