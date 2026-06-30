# Last Session Summary — 2026-06-30 (continued)

## 1. What was accomplished

**Inline review complete**: Prior session's sub-agents hit spend-limit; review was completed inline across 6 checklist domains. Report written to `.living/outputs/reviews/2026-06-30-branch-vs-main.md` (1 major + 8 minor findings, F1–F9).

**Three fixes applied this session**:
- **F1 (MAJOR)**: B5 AUROC SE formula — added missing `(n1+n2+1)` numerator in `benchmarks/cytoanvi/tasks.py:365`. Without fix, z-scores inflated by up to 74×; all B5 FDR calls were effectively meaningless. Updated docstring comment too.
- **F2**: Seeded `np.random.choice` → `np.random.default_rng(42).choice` in `tests/cytoanvi/test_hierarchy_schpl_mock.py:53`. Missed by P1 round.
- **F3**: Removed local `NAN_LAYER = "_nan_mask"` in `benchmarks/cytovi/tasks_imputation.py:16`; added `NAN_LAYER` to import from `benchmarks.common.training`.

**Post-action protocol completed**: Added L-019 to `.living/learnings.md`; added 4 rows to `todo/TODO_REGISTRY.md`.

## 2. Review findings summary (inline, 2026-06-30)

**1 Major finding**:
- **F1**: B5 AUROC standard error formula — missing `(n1+n2+1)` numerator (Wilcoxon AUC SE). Inflated z-scores by `sqrt(n1+n2+1)` ≈ 74× for typical group sizes. **Fixed**.

**8 Minor findings** (not yet fixed):
- F2: Unseeded `np.random.choice` in `test_hierarchy_schpl_mock.py:53`. **Fixed**.
- F3: Local `NAN_LAYER` redefinition in `tasks_imputation.py:16`. **Fixed**.
- F4: B9 pseudo-sample assignment round-robin (seed not used for sample labels) — document or switch to `rng.integers`.
- F5: `DEFAULT_PROTEIN_GENE_MAP` has two keys mapping to `HLA-DRA`; inverse dict silently drops `HLADR` alias.
- F6: B5 transductive calibration consequence understated — add `calibration_note` to output dict.
- F7: `task_b6_lambda_sweep` doesn't expose `_fallback_split` — add to B6 output dict.
- F8: B5 FDR comment described the wrong SE formula (fixed by F1 fix — comment also updated).
- F9: `_pseudobulk_expression` O(N) Python loop — worth profiling before A3 runs at 50k cells.

## 3. Publication risks (combined from prior sessions + this review)

### Critical (fix before publication runs)
1. **F3/B5 (prior session)**: `best_auroc` cherry-pick — replace with `mean_auroc` + BH FDR (F1 fix now makes BH FDR correct)
2. **F2/B3 (prior session)**: Circular concordance metric — needs p2 expert labels or rename as `inter-method agreement`
3. **F1 (prior session)**: Nuñez annotation transductive leakage — regenerate `nunez_annotated_v2.h5ad`
4. **B5 result JSONs must be regenerated** after F1 SE fix (all existing `n_fdr_significant` values are wrong)
5. **F4/F7/F5/F18 (prior session)**: B4/B6 drift direction + control sampling + key-name mismatch — awaiting commit
6. **F12 (prior session)**: scennep z-score scale mismatch — save `orig_expr` before `sc.pp.scale`
7. **F22/F23/F24 (prior session)**: Fix CHANGELOG imports, C-002, ADR 0003 doc divergences
8. Archive `nunez_annotated.h5ad` to Figshare + wire auto-download

### High (major revision risk)
- B1: Add FlowSOM, Phenograph, XGBoost baselines
- B4: Real rLN vs FL/MCL biological split OR demote to supplement
- F14 (prior): Revert `accelerator='gpu'` → `'auto'` in TrainRunner
- F17 (prior): Narrow B1 harmony `except Exception` to `ImportError | ValueError`

## 4. Publication gate (current)

| Task | Dataset | Result | Gate | Status |
|------|---------|--------|------|--------|
| B1 Δ macro-F1 | Roider (≈5k, 3 seeds) | +0.121±0.040 | ≥+0.03 | ✅ PASS |
| B1 Δ macro-F1 | Nuñez r0.05 (3 seeds) | −0.013±0.028 | ≥+0.03 | ❌ FAIL (annotation leakage?) |
| B2 batch Δ | Roider (3 seeds) | −0.006 | ≤0.05 | ✅ PASS |
| B3 p2 concordance | Roider (3 seeds) | 0.877±0.012 | ≥0.80 | ✅ PASS (circular metric) |
| B5 FDR significant | — | INVALID (SE bug fixed — must re-run) | ≥5 types | ❌ RERUN |
| B8 HCE vs flat | — | PENDING (job 25108052) | HCE≥flat | — |

## 5. Active SLURM jobs

| Job ID | Name | Status |
|--------|------|--------|
| 25108052 | cytoanvi_p5_b8_hce | Was RUNNING at prior session — check `squeue -j 25108052` |
| 25102610 | cytoanvi_p7_aggregat | DependencyNeverSatisfied — cancel |
| 25102547 | cytoanvi_p6_b9_mapqc | DependencyNeverSatisfied — cancel |

## 6. Open todos (prioritized)

| Priority | Item | Source |
|----------|------|--------|
| critical | Regenerate B5 result JSONs (AUROC SE fix invalidates all FDR fields) | F1 (this session) |
| critical | Commit locally-fixed issues: B4/B6 drift, B4 rng.choice, B8 leaf_held | F4/F5/F7 (prior) |
| critical | B4/B6 key-name fix: align tasks.py ↔ aggregate_results.py | F18 (prior) |
| critical | B5: promote mean_auroc headline + BH FDR | F3 (prior) |
| critical | B3: acquire p2 ground-truth labels OR rename metric | F2 (prior) |
| critical | scennep z-score fix: save orig_expr before sc.pp.scale | F12 (prior) |
| critical | Fix CHANGELOG, C-002, ADR 0003 doc-code divergences | F22/F23/F24 (prior) |
| critical | Archive nunez_annotated.h5ad to Figshare + auto-download | idea 10 (prior) |
| critical | Nuñez annotation leakage: retrain on train-only CytoVI + kNN transfer | F1 (prior session) |
| high | Monitor B8 job 25108052; scancel stale jobs 25102610, 25102547 | user action |
| high | B1: add FlowSOM + Phenograph + XGBoost to baselines.py | prior |
| high | TrainRunner: revert accelerator='gpu' → 'auto' | F14 (prior) |
| high | B4 real biological split (rLN vs FL/MCL) OR demote to supplement | prior |
| high | Narrow B1 harmony except Exception to ImportError | F17 (prior) |
| medium | B9 round-robin sample labels — document or randomize with rng | F4 (this session) |
| medium | B6 output: expose _fallback_split field | F7 (this session) |
| medium | DEFAULT_PROTEIN_GENE_MAP: remove duplicate HLA-DRA alias | F5 (this session) |
| medium | Novelty detection threshold API: get_uncertainty_threshold(specificity=0.95) | prior |
