# Last Session Summary — 2026-07-06 (session 15, compute jobs + registry cleanup)

## 1. What was accomplished

This session continued the "Do all" instruction from session 14. Five actions taken:

**B1 Roider full-cohort submitted (3 seeds)**

Created `phase_b1_roider_s{0,1,2}.slurm` mirroring the B3 Roider pattern:
- `--dataset roider-full --task b1 --batch-size 16384 --cytoanvi-recipe publication --max-epochs 1000`
- 3 parallel single-seed jobs: 25149326 (s0), 25149327 (s1), 25149328 (s2)
- Output: `roider_full_b1_s{0,1,2}.json`
- All PENDING as of submission; added to `publication_manifest.json` as `in_progress`

**B9 mapQC resubmitted**

`sbatch phase6_b9_mapqc.slurm` → job 25149329 (PENDING). mapqc 0.1.1 is installed in scvi-test
env, so this will execute the real B9 run (not a blocked JSON). Updated manifest: `in_progress`.

**TODO registry cleaned up**

Stale statuses corrected:
- "Push feat/cytoanvi branch" → done
- "Reset dist version" → done
- "BLOCKER: rerun B5 3 seeds" → done
- "B5 REDESIGN" → done (results 0.484±0.019 NEGATIVE; diagnostic running)
- "B3 RERUN" → done (results 0.828±0.015 p1, 0.671±0.008 p2)
- "manifest-mode aggregation" → done
- Added new in_progress rows: B1 Roider (25149326/27/28), B9 (25149329), B5 diagnostic (25149032/33/34)

**publication_manifest.json updated**

Added B1 Roider entries (3 seeds, in_progress) and updated B9 from `blocked` → `in_progress` (job 25149329).

**B5 diagnostic still running**

Jobs 25149032 (seed 0) and 25149033 (seed 1) were RUNNING at ~31min; 25149034 (seed 2) PENDING.
These add `cytoanvi_knn_mean_auroc` to the B5 result to resolve the TTA-vs-latent question.

## 2. Decisions this session

No new architectural decisions. All work was compute/registry maintenance.

## 3. Key numbers (publication-grade, unchanged from session 14)

| Task | Metric | Value |
|------|--------|-------|
| B1 (Nuñez full) | CytoANVI macro-F1 | **0.9751 ± 0.0003** ✅ |
| B3 | p1 holdout macro-F1 | **0.828 ± 0.015** ✅ defensible headline |
| B3 | p2 concordance vs CytoVI-kNN | **0.671 ± 0.008** ❌ gate NOT met (≥0.80 required) |
| B5 | TTA mean_auroc | **0.484 ± 0.019** ❌ NEGATIVE (below chance) |
| B5 | CytoVI kNN-OOD baseline | **0.775 ± 0.002** (for comparison) |
| B8 | Δ_hierarchical_vs_flat | **+0.0862 ± 0.0027** ✅ |

## 4. Open blockers (compute/data)

| Blocker | Status | Job(s) |
|---------|--------|--------|
| B1 Roider full-cohort (3 seeds) | RUNNING/PENDING | 25149326/27/28 |
| B5 diagnostic (CytoANVI-kNN) | RUNNING/PENDING | 25149032/33/34 |
| B9 mapQC full run | PENDING | 25149329 |
| B3 p2 ground-truth labels (Roider panel-2 gating) | Data acquisition required |  |
| B5 better-than-chance novelty detection | Requires new formulation or external OOD dataset | |
| B4/B6 real case/control | Pseudo-batch = plumbing only; real validation blocked | |
| Real DOIs | Swap "in preparation" once preprints post | |

## 5. Next session

1. **B5 diagnostic re-aggregation** (after 25149032/33/34 complete):
   - `python .scratch/cytoanvi-benchmark/aggregate_b5_multiseed.py`
   - `python -m benchmarks.common.aggregate_results --manifest .scratch/cytoanvi-benchmark/publication_manifest.json`
   - Update F-013 with verdict: if `cytoanvi_knn_mean_auroc ≈ 0.77` → TTA is the weak link (ship latent-kNN novelty scorer); if `≈ 0.48` → latent itself is weak (negative stands, strengthened).
2. **B1 Roider results** (after 25149326/27/28 complete):
   - Write `aggregate_b1_roider_multiseed.py` analogous to `aggregate_b5_multiseed.py`
   - Update manifest + ANALYSIS_MANIFEST.md with final B1 Roider numbers
3. **B9 result verification** (after 25149329 complete):
   - Check `nunez_b9_s0.json` is not a blocked JSON
   - If successful: update manifest `status: complete`, update ANALYSIS_MANIFEST.md, consider adding F-014
