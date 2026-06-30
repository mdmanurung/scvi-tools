# Last Session Summary — 2026-06-30 (autonomous continuation)

## 1. What was accomplished

**Living-repo committed**: All `.living/`, `CLAUDE.md`, `todo/`, `.claude/last-session.md` staged
and committed as `d5308022` (previously untracked despite being populated).

**Three additional code fixes applied** (commit `d22c7283`):
- **F14**: `accelerator='gpu'` → `'auto'` in both `train_cytovi` and `train_cytoanvi`
  (`benchmarks/common/training.py`). Hardcoded `'gpu'` crashed CPU-only envs.
- **F4**: Added comment clarifying B9 round-robin sample assignment is deterministic;
  seeded `rng` used only for subsequent random status labels (`tasks.py`).
- **F6**: Added `calibration_note` field to `task_b5_holdout_sweep` return dict, flagging
  transductive uncertainty calibration for downstream consumers.

**Verified already clean** (no code changes needed):
- F18 (B4/B6 key-name mismatch): `replay_latent_drift` is consistent in both files
- F17 (broad except): `tasks.py:110` already uses `except (ImportError, ValueError, KeyError)`
- F12 (scennep z-score): `scennep.py:208-211` already saves `orig_expr` before `sc.pp.scale`
- F22/F24 (CHANGELOG/ADR 0003): CHANGELOG already uses `cytoanvi.CytoANVI`; ADR 0003 "unmapped
  leaves silently ignored" is CORRECT — it refers to scHPL leaves not matching any model label
  (zero-match case in `_infer_leaf_to_model_mapping`), not model labels without a tree node

## 2. Cumulative fixes across all sessions (commits)

| Commit | Fix |
|--------|-----|
| `6c99afe5` | B5 AUROC SE formula (F1/MAJOR), test seed (F2), NAN_LAYER DRY (F3), HLA-DR alias (F5), B6 fallback flag (F7) |
| `d5308022` | Committed .living/ scaffold, CLAUDE.md, todo/, last-session.md |
| `d22c7283` | accelerator='auto' (F14), B9 comment (F4), B5 calibration_note (F6) |

## 3. Publication risks — current state

### Critical (fix before publication runs)
1. **B5 result JSONs must be regenerated** after AUROC SE fix (all `n_fdr_significant` wrong)
2. **Nuñez annotation transductive leakage** — retrain on train-only CytoVI + kNN transfer (F1/prior)
3. **B3 circular concordance metric** — acquire p2 expert labels OR rename to `inter-method agreement`
4. Archive `nunez_annotated.h5ad` to Figshare + wire auto-download

### High (major revision risk)
- B1: Add FlowSOM, Phenograph, XGBoost baselines
- B4: Real rLN vs FL/MCL biological split OR demote to supplement
- B9: Install `mapqc` in conda env (currently blocked, DependencyNeverSatisfied on SLURM)

## 4. Publication gate (current)

| Task | Dataset | Result | Gate | Status |
|------|---------|--------|------|--------|
| B1 Δ macro-F1 | Roider (≈5k, 3 seeds) | +0.121±0.040 | ≥+0.03 | ✅ PASS |
| B1 Δ macro-F1 | Nuñez r0.05 (3 seeds) | −0.013±0.028 | ≥+0.03 | ❌ FAIL (leakage suspected) |
| B2 batch Δ | Roider (3 seeds) | −0.006 | ≤0.05 | ✅ PASS |
| B3 p2 concordance | Roider (3 seeds) | 0.877±0.012 | ≥0.80 | ✅ PASS (circular metric) |
| B5 FDR significant | — | INVALID — must re-run after SE fix | ≥5 types | ❌ RERUN |
| B8 HCE vs flat | — | PENDING (job 25108052 — check squeue) | HCE≥flat | — |

## 5. Active SLURM jobs

| Job ID | Name | Status |
|--------|------|--------|
| 25108052 | cytoanvi_p5_b8_hce | Was RUNNING — check `squeue -j 25108052` |
| 25102610 | cytoanvi_p7_aggregat | DependencyNeverSatisfied — USER ACTION: scancel |
| 25102547 | cytoanvi_p6_b9_mapqc | DependencyNeverSatisfied — USER ACTION: scancel |

## 6. Open todos (prioritized)

| Priority | Item | Source |
|----------|------|--------|
| critical | Regenerate B5 result JSONs (AUROC SE fix invalidates all FDR fields) | F1 (prior) |
| critical | Nuñez annotation leakage: retrain on train-only CytoVI + kNN transfer | F1 (prior) |
| critical | B3: acquire p2 ground-truth labels OR rename metric | F2 (prior) |
| critical | B5: promote mean_auroc headline + BH FDR; cherry-pick `best_auroc` removed | F3 (prior) |
| critical | Archive nunez_annotated.h5ad to Figshare + auto-download wiring | idea 10 |
| high | Monitor B8 job 25108052; scancel stale jobs 25102610, 25102547 | USER ACTION |
| high | B1: add FlowSOM + Phenograph + XGBoost to baselines.py | prior |
| high | B4 real biological split (rLN vs FL/MCL) OR demote to supplement | prior |
| high | Install `mapqc` in conda env before next SLURM submission (unblocks B9) | prior |
| medium | Novelty detection threshold API: get_uncertainty_threshold(specificity=0.95) | prior |
| medium | Profile `_pseudobulk_expression` before vectorizing (F9) | this session |
