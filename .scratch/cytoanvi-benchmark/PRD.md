# PRD: CytoANVI real-data benchmarking (Track B)

Status: ready-for-human
Owner: mdmanurung
Branch: feat/cytoanvi
Created: 2026-06-10
Updated: 2026-06-29

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
| B8 | flat CE vs HCE holdout macro-F1 | `set_hierarchy`, `predict`, `predict_hierarchical` | flat CE | 12 |
| B9 | mapQC on query controls after surgery | `score_query_mapping` / `mapping_qc` | low control `mapqc_score > 2` | 13 |
| B7–B11 | imputation / DA / query fidelity | various | CytoVI | 11 (later) |

## Success criteria (mean over ≥3 seeds)

- **B1:** CytoANVI macro-F1 ≥ CytoVI k-NN + **0.03**
- **B2:** CytoANVI scib `bio_conservation` within **±0.02** of CytoVI; `batch_correction` ≥ CytoVI
- **B3:** panel-1 holdout macro-F1 ≥ k-NN; panel-2 concordance ≥ **0.70**
- **B5:** best holdout-type AUROC > **0.70**
- **B4/B6:** deferred until real rLN reference/replay plus FL/MCL query metrics exist
- **B8:** document Δ macro-F1 (HCE vs flat CE) when coarse types are observed model labels
- **B9:** query control `mapqc_score > 2` rate documented on real case/control cohort (synthetic = plumbing only)

## Known failure points (mapQC / B9)

Synthetic `--task b9` validates joint latent construction only (`run_mapqc=False`). Full mapQC
scoring requires a real case/control atlas (≥3 reference samples, matched query controls). See issue 13.

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
`cytovi-benchmark/01–02` (data ingest). Validate/fetch vignette assets:
`python -m benchmarks.common.fetch_data --validate-only`. Issue 03 done; readfcs installed (05).

## Vignette e1000 results (2026-06-17) — publication training, still vignette data

**Data:** CytoVI tutorial subsamples (Roider `.h5ad`; Nuñez via `data/nunez_annotated.h5ad`).
**`max_epochs=1000`**, scib B2 aggregates.

**Roider summary:** `.scratch/cytoanvi-benchmark/results/e1000/roider_e1000_partial_summary.json`

| Task | Result (3 seeds) | PRD target |
|------|------------------|------------|
| **B1** | CytoANVI **0.908 ± 0.008** vs k-NN **0.787 ± 0.039** (Δ **+0.121**) | ≥ +0.03 — **pass** |
| **B2** | Bio CytoANVI **0.737** vs CytoVI **0.628**; batch **0.792** vs **0.798** | bio ±0.02 / batch ≥ baseline — **fail** (better bio, batch slightly worse) |
| **B3** | p1 holdout F1 **0.917 ± 0.018**; p2 concordance **0.877 ± 0.012** | F1 ≥ k-NN; concordance ≥ 0.70 — **pass** |

**Nuñez e1000:** B1/B2 in flight from `run_e1000.sh` (uses tutorial 11-type labels). Smoke @
epochs=100 with Leiden r=0.05 superseded by `nunez_annotated.h5ad`.

**Nuñez labels:** `python -m benchmarks.cytoanvi.annotate_nunez` → `data/nunez_annotated.h5ad`;
`load_nunez()` auto-loads when present.

## Artifacts

- Harness: `benchmarks/cytoanvi/` (existing)
- Shared: `benchmarks/common/` (issue cytovi-benchmark/03)
- Results: `.scratch/cytoanvi-benchmark/results/`

## Execution status (2026-06-27)

- Phase 0 validation: green under the queue environment. Compile passed; touched-file ruff passed;
  focused benchmark/external tests passed; full targeted pytest passed (`54 passed`).
- Phase 2 Nuñez B1/B2: job `25102544` still RUNNING on `res-hpc-gpu15` at last check
  (`sacct`, elapsed `02:20:16`).
- Phase 3 Roider B3/B5: job `25102555` COMPLETED (`0:0`). B3 p1 holdout macro-F1
  `0.941 +/- 0.012`; p2 concordance `0.862 +/- 0.009`. B5 seed-0 best AUROC `0.909`;
  mean AUROC `0.490`.
- Phase 4 Roider B4/B6: old job `25102606` FAILED because the benchmark scored query controls
  through the reference model with unseen batch categories. Fixed to report `replay_latent_drift`;
  resubmitted as job `25102622`.
- Phase 5 B8 HCE: resubmitted as job `25102620`, dependency `afterok:25102544`.
- Phase 6 B9 mapQC: skipped/blocked because `mapqc` is not installed in the benchmark env.
  `phase6_b9_mapqc.slurm` now preflights `import mapqc` and exits instead of installing packages.
- Phase 7 aggregation: submitted as job `25102623`, dependency
  `afterok:25102544:25102620:25102622`. Current aggregation smoke wrote
  `.scratch/cytoanvi-benchmark/results/final_summary.json` and marks B4/B6/B9 missing until queued
  jobs complete.

## Blockers

- Nuñez Phase 2 still running; Phase 5 and final aggregation are waiting on it.
- B9 requires preinstalled `mapqc`; no dependency installation is allowed inside the SLURM queue.

## Second-pass publication gate (2026-06-28)

- Publication aggregation now uses `.scratch/cytoanvi-benchmark/publication_manifest.json` via
  `benchmarks.common.aggregate_results --manifest`. Recursive `--input` aggregation remains
  exploratory only.
- Manifest mode rejects any JSON under the input directory that is not explicitly listed, so older
  Roider vignette/full-cache outputs are not silently mixed into publication summaries.
- Existing Roider artifacts from jobs that used `--dataset roider` are smoke/provenance only, not
  publication evidence. Phase 3 now targets `--dataset roider-full` and writes `roider_full_*`
  outputs. Phase 4 remains a manual plumbing/smoke script only; do not use B4/B6 as publication
  evidence until real rLN reference/replay plus FL/MCL query implementation and metrics exist.
- Current scheduler state from `sacct` on 2026-06-28: Nuñez P2 job `25102544` is RUNNING
  (elapsed `06:01:36`, started `2026-06-27T19:41:20`); B8 P5 job `25102620` is PENDING; old P7
  job `25102623` is PENDING but should be ignored because it was submitted before manifest-mode
  aggregation.
- B9 is optional/blocked until `mapqc` is provisioned. Phase 6 now writes the stable manifest path
  `results/nunez_b9_s0.json` for both blocked and successful outcomes instead of installing
  dependencies inside the queue.

## Recovery execution status (2026-06-28 → 2026-06-29)

- Stale jobs **25102620** and **25102623** were canceled after `squeue` showed
  `DependencyNeverSatisfied`.
- Nuñez P2 job **25102544** failed with exit code `11:0` after producing B1 seeds 0/1/2 and B2
  seeds 0/1. Missing B2 seed 2 recovered by one-off SLURM job **25104249** (`recover_nunez_b2_s2.slurm`);
  exited `0:0`; `nunez_b2_s2.json` validated (cytoanvi total=0.7739, cytovi total=0.7747).
- Roider Phase 3 job **25104250** **FAILED** (exit `11:0`, elapsed 06:56:12). Root cause: two
  compounding problems — (1) default `batch_size=128` on 1.24 M cells → ~52 h/seed, infeasible; (2)
  NaN divergence at epoch 94 in the CytoVI encoder (`baselines.py::cytovi_latent_and_knn`).
  Fix: `batch_size=8192` threaded through all four benchmark files (training.py, baselines.py,
  tasks.py, run.py) + `--batch-size` CLI arg added. Pytest re-run: benchmark 20 passed, CytoANVI
  72 passed 1 skipped. Phase 3 scripts restructured: `phase3_b3b5_roider.slurm` → B3-only 14 h;
  `smoke_b3_roider.slurm` (NEW) → 20-epoch timing + Leiden cluster count;
  `phase3b_b5sweep_roider.slurm` (NEW) → B5 sweep 48 h placeholder (size from smoke test).
  **Pending user action:** (a) `scancel 25102546 25102547 25102610` (stale DependencyNeverSatisfied);
  (b) submit smoke test; (c) resubmit Phase 3a/3b after smoke confirms epoch time + cluster count.
- B8 Phase 5 job **25104252** RUNNING since 2026-06-28 22:38. Sequential seeds at ~8 h each:
  `nunez_b8_s0.json` done (06:50 Jun 29); s1 expected ~15:02; s2 expected ~23:14. 24 h wall ends
  22:38 → s2 will likely be killed ~36 min short. Per-seed recovery job needed after wall hit.
- Optional B9 Phase 6 job **25104251** wrote `nunez_b9_s0.json` with `b9.status == "blocked"`
  (`mapqc` not installed). Stable terminal state.
- `publication_manifest.json` marks existing complete artifacts as `complete`; Roider B3/B5 remain
  `pending` (blocked on Phase 3 resubmission). Manifest-mode aggregation remains blocked.
