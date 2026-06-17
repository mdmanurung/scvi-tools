# PRD: CytoANVI real-data benchmarking (Track B)

Status: ready-for-human
Owner: mdmanurung
Branch: feat/cytoanvi
Created: 2026-06-10
Updated: 2026-06-17

## Problem

CytoANVI passes synthetic tests but needs real-data validation against **original CytoVI** (k-NN
label transfer in the CytoVI latent). Prior smoke results used vignette subsamples, `max_epochs=100`,
and lightweight integration metrics — not publication-grade.

## Decisions (2026-06-17)

| Decision | Value |
|----------|-------|
| Scope | Track B of unified plan (Track A in parallel) |
| Data | **Full** Nuñez + full Roider cohorts (see Track A data issues) |
| Training | **`max_epochs=1000`** |
| Integration metrics | **`scib-metrics`** via `benchmarks/common/scib.py` |

Master plan: `notes/2026-06-17-cytovi-cytoanvi-benchmark-plan.md`.

## Approach

Extend `benchmarks/cytoanvi/` against CytoVI's k-NN baseline on full cohorts.

- **B-D1** — Full Roider (63 patients, 10k T cells/patient): B3, B4, B5, B8
- **B-D2** — Full Nuñez batch replicate: B1, B2
- **B-D3** — Nuñez + Kreutmair: B7 (optional)

## Tasks

| # | Measures | API | Baseline | Issue |
|---|----------|-----|----------|-------|
| B1 | label-transfer macro-F1 | `predict` | CytoVI kNN | 08 |
| B2 | scib integration aggregates | `get_latent_representation` | CytoVI latent | 08 |
| B3 | panel-1 → panel-2 mapping | `prepare_query_anndata` + `load_query_data` + `predict` | CytoVI kNN | 09 |
| B4 | continual case-control update | `load_query_data_with_replay` | static CytoVI | 10 |
| B5 | novelty AUROC (holdout sweep) | `get_uncertainty` | — | 09 |
| B6 | λ (`ewc_importance`) sweep | continual plan kwargs | — | 10 |
| B7–B9 | imputation / DA / query fidelity | various | CytoVI | 11 (later) |

## Success criteria (mean over ≥3 seeds)

- **B1:** CytoANVI macro-F1 ≥ CytoVI k-NN + **0.03**
- **B2:** CytoANVI scib `bio_conservation` within **±0.02** of CytoVI; `batch_correction` ≥ CytoVI
- **B3:** panel-1 holdout macro-F1 ≥ k-NN; panel-2 concordance ≥ **0.70**
- **B5:** best holdout-type AUROC > **0.70**
- **B4/B6:** λ knee documented as CytoVI-specific default

## Prior results (INVALID for PR — vignette / epochs=100 / lightweight B2)

`roider_seed0_summary.json`: B1/B3 pass; B5 fails on single holdout. **Re-run on full B-D1 with
scib + epochs=1000.**

## Vignette smoke results (2026-06-17) — harness validation only

**Data:** CytoVI tutorial `.h5ad` subsamples (`data/Roider_et_al_BNHL_panel{1,2}.h5ad`), not full
cohort. **`max_epochs=100`**, lightweight B2 metrics (not full scib aggregates).

**Summary:** `.scratch/cytoanvi-benchmark/results/roider_multiseed_summary.json`

| Task | Result | PRD target (smoke) |
|------|--------|-------------------|
| **B1** (3 seeds) | CytoANVI **0.925 ± 0.009** vs k-NN **0.810 ± 0.025** (Δ **+0.115**) | ≥ +0.03 — **pass** |
| **B2** (3 seeds, scib) | Bio conservation **+0.099** vs CytoVI; batch correction **−0.040** | bio ±0.02 / batch ≥ baseline — **fail** (better bio, worse batch) |
| **B3** (seed 0) | p1 holdout F1 **0.913**; p2 concordance **0.863** | ≥ 0.7 — **pass** |
| **B5** (13 holdouts) | **Tfh 0.875**, Treg CD69+ **0.779**, Ttox EM3 **0.742** (>0.7) | ≥ 1 type — **pass** |

Per-seed JSON: `results/roider_seed{0,1,2}_b{1,2}.json`, `roider_seed0_b{3,5}.json`,
`results/b5_sweep/*.json`.

**Harness fixes:** `baselines.py` skip k-NN when zero unlabeled cells; `run.py` `--holdout-type`,
`--holdout-sweep`, B5 kwargs filter; `holdout_safe_name()` for sweep filenames (`+`/`-` safe).

**Next:** issues 08–10 on **full** cohorts (`max_epochs=1000`, scib B2) — blocked on
`cytovi-benchmark/01–02` (data ingest). Issue 03 (scib infra) done; issue 05 (readfcs) unblocks
Nuñez loader once FCS files land.

## Artifacts

- Harness: `benchmarks/cytoanvi/` (existing)
- Shared: `benchmarks/common/` (issue cytovi-benchmark/03)
- Results: `.scratch/cytoanvi-benchmark/results/`

## Blockers

- Full data: cytovi-benchmark issues 01, 02
- scib infra: cytovi-benchmark issue 03
- FCS reader for Nuñez: issue 05 (unchanged)
