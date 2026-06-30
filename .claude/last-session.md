# Last Session Summary — 2026-06-30 (autonomous continuation, session 4)

## 1. What was accomplished

**Two new commits** — B5 and B8 multiseed results both finalized:

| Commit | Content |
|--------|---------|
| `8abb784a` | B5 3-seed e1000 multiseed: ran `aggregate_b5_multiseed.py`, updated F-007 + F-010 in FINDINGS_REGISTRY and cross-panel-mapping.md, marked B5 ✓ in ANALYSIS_MANIFEST |
| `16015632` | B8 3-seed e1000 final: ran `aggregate_b8_multiseed.py`, updated F-011 in FINDINGS_REGISTRY and continual-update.md, marked B8 ✅ pub-gate in ANALYSIS_MANIFEST |

**Key results unlocked:**

B5 — Novelty detection (Roider 13-type T-cell holdout sweep, 3 seeds):
- `best_auroc`: 0.833 ± 0.122; `mean_auroc`: 0.462 ± 0.075; `n_fdr_sig`: 5.0
- 2/13 types pass ≥0.70 mean AUROC: **Ttox EM3** (0.776 ± 0.071, low variance), **Tfh** (0.724 ± 0.258, high variance)
- Tpr near-threshold: 0.693 ± 0.027 (very consistent)
- Bimodal confirmed: 5 types consistently near-chance (Treg CD69- worst, 0.120 ± 0.010)
- Publication framing: "detects immunologically distinct effector subtypes; fails for phenotypically overlapping naive/regulatory clusters"

B8 — HCE vs flat CE (Nuñez full 200k, 3 seeds):
- `delta_hierarchical_vs_flat_macro_f1` = **+0.0862 ± 0.0027** ✅ pub-gate passed
- Per-seed: +0.0887, +0.0866, +0.0833 (monotonically consistent, very low variance)
- `flat_ce macro_F1` = 0.9783 ± 0.0011 (stable baseline)
- Direct HCE prediction = −0.0984 ± 0.0851 (expected negative — benefit is post-hoc hierarchical decoding)
- B8 pub-gate passed for Nuñez cohort; Roider B8 still needed for generalizability

## 2. Cumulative benchmark status

| Task | Status | Result |
|------|--------|--------|
| B1 Roider | ✓ pub-grade | Δ+0.121±0.040 |
| B1 Nuñez inductive | RUNNING (PID 1520357, seed-0 epoch ~700/1000 at 18:49, ETA all-3-seeds ~06:00 Jul 1) | leakage fix; reversal expected to narrow |
| B2 | ✓ | bio +0.108, batch Δ−0.006 |
| B3 | ✓ | concordance 0.877±0.012 |
| B4 | Blocked (pseudo split) | plumbing only |
| B5 | ✓ multiseed | best 0.833±0.122, mean 0.462±0.075 |
| B6 | Blocked | plumbing only |
| B8 | ✅ pub-gate | Δ_hier +0.0862±0.0027 |
| B9 | Blocked (mapqc) | — |

## 3. Cumulative fixes across all sessions (commits)

| Commit | Fix |
|--------|-----|
| `6c99afe5` | B5 AUROC SE formula (F1/MAJOR), test seed (F2), NAN_LAYER DRY (F3), HLA-DR alias (F5), B6 fallback flag (F7) |
| `d5308022` | Committed .living/ scaffold, CLAUDE.md, todo/, last-session.md |
| `d22c7283` | accelerator='auto' (F14), B9 comment (F4), B5 calibration_note (F6) |
| `7657cd03` | B3 aggregator backward-compat + column rename |
| `5069d6bc` | XGBoost, Phenograph, FlowSOM B1 baselines |
| `3eb95baa` | F13 cofactor docs, F20 TTA RNG, F30 MockTreeNode dedup |
| `8abb784a` | B5 3-seed multiseed findings (F-007, F-010) |
| `16015632` | B8 3-seed final findings (F-011) — pub-gate ✅ |

## 4. Publication risks — current state

### Critical (fix before publication runs)
1. **B5 result JSONs must be regenerated** after AUROC SE fix (all `n_fdr_significant` counts use corrected Wilcoxon SE — already in the e1000 runs; old smoke results may differ)
2. **Nuñez annotation transductive leakage** — in progress via PID 1520357 (inductive kNN rerun)
3. **B3 circular concordance metric** — inter-method agreement (metric renamed in aggregator, underlying data gap remains)
4. Archive `nunez_annotated.h5ad` to Figshare + wire auto-download

### High (major revision risk)
- B4: Real rLN vs FL/MCL biological split OR demote to supplement
- B9: Install `mapqc` in conda env (currently blocked)

## 5. Open items for next session

**Immediate (B1 inductive, ETA ~06:00 Jul 1):**
- Read `results/e1000/nunez_inductive_b1_multiseed.json` when it appears
- Determine if reversal (CytoANVI < kNN) persists or narrows after leakage fix
- Update F-003 in FINDINGS_REGISTRY and label-transfer-accuracy.md
- Update ANALYSIS_MANIFEST B1 row

**Blocked on user:**
- `scancel 25089685 25102610 25102547` (3 stale DependencyNeverSatisfied SLURM jobs)
- Install `mapqc` in conda env (unblocks B9)
- B4: real rLN vs FL/MCL biological split OR demote to supplement

**High (release):**
- Push `feat/cytoanvi` branch + upstream PR
