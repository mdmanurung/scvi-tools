# Last Session Summary — 2026-07-06 (session 16, B1 Roider complete)

## 1. What was accomplished

**B1 Roider full-cohort COMPLETE**

Jobs 25149326/27/28 finished and produced results:
| Seed | CytoANVI | CytoVI-kNN | Δ |
|------|----------|------------|---|
| 0 | 0.9314 | 0.8906 | +0.0409 |
| 1 | 0.9295 | 0.8913 | +0.0383 |
| 2 | 0.9340 | 0.8967 | +0.0373 |
| **agg** | **0.9317±0.0022** | **0.8928±0.0034** | **+0.0388±0.0018 ✅** |
XGBoost 0.9516. Gate (Δ≥+0.03) PASSES. Aggregated → `roider_full_b1_multiseed.json`.

**B9 mapQC BLOCKED AGAIN (job 25149329, FAILED)**

mapqc 0.1.1 is installed but crashes with `IndexError: single positional indexer is out-of-bounds` in `_get_per_cell_filtering_info` → `mode().iloc[0]` on empty result. Library bug triggered by Nuñez dataset. Not reportable.

**All plan tasks complete (T1/T2/T3)**

- T1 (CytoANVI-kNN latent OOD code): already in `tasks.py` — running in B5 diagnostic jobs
- T2 (propagate honest numbers): CHANGELOG, ANALYSIS_MANIFEST, FINDINGS_REGISTRY, TODO_REGISTRY all updated; stale grep passes
- T3 (version + push): `pyproject.toml` already at `0.1.0`; pushed to GitHub

**Still waiting: B5 diagnostic (25149032/33/34)**

At session start jobs were at 11h34m. Seed 0 at epoch 539/1000 for one holdout type (~21min left for that run). Seed 2 at epoch 461/1000. All 3 still RUNNING.

## 2. Key numbers (publication-grade)

| Task | Dataset | Metric | Value | Status |
|------|---------|--------|-------|--------|
| B1 | Roider full | CytoANVI macro-F1 | **0.9317±0.0022** | ✅ |
| B1 | Roider full | CytoVI-kNN macro-F1 | 0.8928±0.0034 | baseline |
| B1 | Roider full | Δ macro-F1 | **+0.0388±0.0018** | ✅ gate |
| B1 | Nuñez full | CytoANVI macro-F1 | **0.9751±0.0003** | ✅ ceiling |
| B3 | Roider full | p1 macro-F1 | **0.828±0.015** | ✅ |
| B3 | Roider full | p2 concordance | 0.671±0.008 | ❌ gate (≥0.80) |
| B5 | Roider full | TTA mean_auroc | 0.484±0.019 | ❌ NEGATIVE |
| B5 | Roider full | CytoVI kNN-OOD | 0.775±0.002 | baseline |
| B8 | Nuñez full | Δ_hier_vs_flat | +0.0862±0.0027 | ✅ |

## 3. Open items

| Item | Status | Job(s) |
|------|--------|--------|
| B5 diagnostic (CytoANVI-kNN in own latent) | RUNNING | 25149032/33/34 |
| B9 mapQC | BLOCKED — mapqc library bug | — |
| B3 p2 ground-truth labels | Blocked — data acquisition | — |
| B5 better-than-chance novelty | Requires new formulation or external OOD | — |
| B4/B6 real case/control | Blocked — real data | — |

## 4. Next session

1. **B5 diagnostic re-aggregation** (after 25149032/33/34 complete):
   ```bash
   python .scratch/cytoanvi-benchmark/aggregate_b5_multiseed.py \
     --out results/roider_full_b5_sweep_multiseed_diag.json \
     results/roider_full_b5_sweep_s0.json \
     results/roider_full_b5_sweep_s1.json \
     results/roider_full_b5_sweep_s2.json
   ```
   - Check `cytoanvi_knn_mean_auroc_mean` in the output:
     - ≈0.77: TTA is the weak link → latent is fine; recommend latent-kNN as the novelty scorer
     - ≈0.48: CytoANVI latent is itself weaker OOD space → negative stands, strengthened
   - Update F-013 in FINDINGS_REGISTRY.md with verdict
   - Update `publication_summary.json` via `aggregate_results.py --manifest`
   - Also update publication_manifest.json: mark B5 diagnostic artifacts as `complete`

2. **CRITICAL NOTE: The diagnostic jobs overwrite `roider_full_b5_sweep_s{0,1,2}.json`** (same output path). They run ALL ~47 Leiden clusters (not the original 11 most-populous used in `phase3b_b5redesign_roider_s*.slurm` via `--b5-max-holdout-types 11`). Original 11-type seed files are preserved as `roider_full_b5_sweep_s{0,1,2}_11type_orig.json`. Publication result is `roider_full_b5_sweep_multiseed.json` (11-type, status=complete in manifest) — DO NOT overwrite it. Aggregate the diagnostic to a DIFFERENT file: `roider_full_b5_sweep_diag_multiseed.json`.

3. **B9**: Low priority until mapqc releases a fix for the IndexError in `_get_per_cell_filtering_info`.
