# Last Session Summary — 2026-07-05 (session 14, honest-numbers + B5 diagnostic + branch push)

## 1. What was accomplished

Three tasks from the approved plan were executed:

**Task 1 — B5 CytoANVI-latent kNN-distance OOD diagnostic (complete, diagnostic pending)**

Added `knn_distance_novelty(z_ref, z_eval, n_neighbors)` helper to `benchmarks/cytoanvi/baselines.py`, refactored `cytovi_novelty_score` to call it, and wired up a `cytoanvi_knn_baseline` result in the inductive branch of `task_b5_novelty` (extracted from live model BEFORE `del model`). Surfaced `cytoanvi_knn_mean_auroc` from `task_b5_holdout_sweep` and propagated to `aggregate_b5_multiseed.py` and `aggregate_results.py`. Added `--b5-cytovi-baseline` flag to all three B5 SLURM seed scripts. Submitted diagnostic re-run jobs 25149032 (seed 0), 25149033 (seed 1), 25149034 (seed 2). These were PENDING at session end. After completion, re-run `aggregate_b5_multiseed.py` and `aggregate_results.py --manifest` to regenerate `publication_summary.json` with both `cytovi_knn_mean_auroc` and `cytoanvi_knn_mean_auroc`. Committed as `66a1b806`.

**Task 2 — Propagate honest full-cohort numbers to all prose surfaces (complete)**

Updated all human-readable surfaces to show honest full-cohort Roider results:
- `benchmarks/ANALYSIS_MANIFEST.md`: B3 row shows p1 0.828±0.015 / p2 0.671±0.008 (gate NOT met); B5 row shows 0.484±0.019 NEGATIVE vs CytoVI kNN 0.775; prior smoke table renamed SUPERSEDED.
- `.living/findings/FINDINGS_REGISTRY.md`: F-006 → SUPERSEDED by F-012; F-007 → SUPERSEDED by F-013; added F-012 (B3 full-cohort) and F-013 (B5 full-cohort NEGATIVE).
- `.living/findings/cross-panel-mapping.md`: annotated F-006/F-007/F-010 as SUPERSEDED; appended full F-012 and F-013 sections.
- `CHANGELOG.md`: replaced "not yet final-publication ready" with all final full-cohort numbers.
- `docs/plans/2026-06-27-cytoanvi-execution-queue.md`: Phase 3 metrics annotated as "(⚠️ SUPERSEDED — roider-e1000 subset)" with inline full-cohort corrections.
- `notes/2026-06-18-cytoanvi-strengthening.md`: B3 and B5 rows annotated with full-cohort corrections.
- `analysis/ideas/2026-06-30-publication-readiness/02_statistical-methods-critic.md`: inline annotation at the 0.877 figure.
- `.living/decisions.md`: added D-012 (honest numbers) and D-013 (version + push).

**Task 3 — Package & push (complete)**

`pyproject.toml` version bumped from `1.5.0rc1` → `0.1.0`. Branch `feat/cytoanvi` pushed to `origin` (github.com/mdmanurung/scvi-tools) — first push ever.

## 2. Decisions this session

- **D-012**: Honest full-cohort numbers must replace stale e1000 numbers in all prose. B3 p2 concordance gate NOT met. B5 is a robust negative result.
- **D-013**: Version `0.1.0` for standalone CytoANVI. Branch pushed; no PyPI upload yet.

## 3. Key numbers (publication-grade, full-cohort Roider, 3 seeds)

| Task | Metric | Value |
|------|--------|-------|
| B3 | p1 holdout macro-F1 | **0.828 ± 0.015** ✅ defensible headline |
| B3 | p2 concordance vs CytoVI-kNN | **0.671 ± 0.008** ❌ gate NOT met (≥0.80 required) |
| B5 | TTA mean_auroc | **0.484 ± 0.019** ❌ NEGATIVE (below chance) |
| B5 | CytoVI kNN-OOD baseline | **0.775 ± 0.002** (for comparison) |

## 4. Open blockers (compute/data)

| Blocker | Status |
|---------|--------|
| B5 diagnostic re-run (CytoANVI-kNN) | Jobs 25149032/33/34 PENDING — re-aggregate after completion |
| B3 p2 ground-truth labels (Roider panel-2 gating) | Data acquisition required — see Idea 4 in analysis/ideas/ |
| B5 better-than-chance novelty detection | Requires new formulation or external OOD dataset |
| B4/B6 real case/control | Pseudo-batch = plumbing only; real validation blocked |
| Real DOIs | Swap "in preparation" once preprints post |

## 5. Next session

1. Check job status for 25149032/33/34 (`squeue -j 25149032,25149033,25149034`). If complete, run `python .scratch/cytoanvi-benchmark/aggregate_b5_multiseed.py` then `python -m benchmarks.common.aggregate_results --manifest ...` to update `publication_summary.json` with diagnostic results.
2. Interpret `cytoanvi_knn_mean_auroc` vs `cytovi_mean_auroc` to resolve the TTA-vs-latent diagnostic question (see F-013 pending note).
3. Consider whether to open a GitHub issue/discussion for the B3 ground-truth labels (Roider contact or FCS audit).
