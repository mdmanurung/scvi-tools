# TODO Registry

Tracks future work items, ideas, and planned improvements.

| Title | Priority | Status | Category | Date | Author | File |
|-------|----------|--------|----------|------|--------|------|
| Full Nuñez/Roider benchmarks at max_epochs=1000 | critical | open | analysis | 2026-06-10 | mdmanurung | [b-track-full-benchmarks.md](b-track-full-benchmarks.md) |
| Real case/control axis for B4 biological validation | high | open | analysis | 2026-06-18 | mdmanurung | — |
| Tune and document CytoVI-specific default λ after B6 sweep | medium | open | analysis | 2026-06-18 | mdmanurung | — |
| Push feat/cytoanvi branch + upstream PR | high | open | infrastructure | 2026-06-18 | mdmanurung | — |
| B9 mapQC on query controls after surgery | medium | blocked | validation | 2026-06-29 | mdmanurung | — |
| Roider Phase-3 smoke test then resubmit (B3/B5 full-cohort) | critical | in_progress | analysis | 2026-06-29 | mdmanurung | — |
| Nuñez B8 seed-2 recovery (job 25108052 running ~3.5h as of 2026-06-29) | high | done | analysis | 2026-06-29 | mdmanurung | — |
| Run manifest-mode aggregation after all Phase-3/5 artifacts land | high | open | infrastructure | 2026-06-29 | mdmanurung | — |
| Fix ruff lint findings on cytoanvi package | medium | done | infrastructure | 2026-06-29 | mdmanurung | — |
| Fix B5 AUROC SE formula (Wilcoxon numerator missing — FDR all-significant) | critical | done | analysis | 2026-06-30 | mdmanurung | — |
| Fix unseeded np.random.choice in test_hierarchy_schpl_mock.py | low | done | testing | 2026-06-30 | mdmanurung | — |
| Fix NAN_LAYER local redefinition in tasks_imputation.py (DRY violation) | low | done | infrastructure | 2026-06-30 | mdmanurung | — |
| Regenerate B5 result JSONs after AUROC SE fix (existing values are wrong) | critical | done | analysis | 2026-06-30 | mdmanurung | — |
| Install `mapqc` in conda env before next SLURM submission (unblocks B9) | high | open | infrastructure | 2026-06-29 | mdmanurung | — |
| USER ACTION: scancel stale jobs 25102610 25102547 (25102546 already gone) | high | open | infrastructure | 2026-06-29 | mdmanurung | — |
| Monitor B1 Nuñez reversal on full-cohort run (F-003: Δ−0.013 on vignette) | high | open | validation | 2026-06-29 | mdmanurung | — |
| Re-run Nuñez B1 e1000 seeds 0/1/2 after inductive kNN annotation fix (PID 2539851) | critical | in_progress | analysis | 2026-06-30 | mdmanurung | — |
| Fix Nuñez transductive leakage in annotate_nunez.py (L-022) | critical | done | analysis | 2026-06-30 | mdmanurung | — |
| Fix accelerator='gpu' → 'auto' in training.py (F14) | medium | done | infrastructure | 2026-06-30 | mdmanurung | — |
| Add B9 round-robin sample comment + B5 calibration_note (F4/F6) | low | done | analysis | 2026-06-30 | mdmanurung | — |
| B3 aggregator backward-compat for key rename (F2) | medium | done | infrastructure | 2026-06-30 | mdmanurung | — |
| XGBoost, Phenograph, FlowSOM baselines for B1 (F15) | high | done | analysis | 2026-06-30 | mdmanurung | — |
| Fix mask_augment nan_mask branch global RNG (F20) | low | done | infrastructure | 2026-06-30 | mdmanurung | — |
| Correct ARCSINH_COFACTORS dict (F13) | low | done | documentation | 2026-06-30 | mdmanurung | — |
| Consolidate MockTreeNode via conftest import (F30) | low | done | testing | 2026-06-30 | mdmanurung | — |
