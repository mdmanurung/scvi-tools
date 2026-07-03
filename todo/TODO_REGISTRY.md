# TODO Registry

Tracks future work items, ideas, and planned improvements.

| Title | Priority | Status | Category | Date | Author | File |
|-------|----------|--------|----------|------|--------|------|
| Full Nuñez/Roider benchmarks at max_epochs=1000 | critical | open | analysis | 2026-06-10 | mdmanurung | [b-track-full-benchmarks.md](b-track-full-benchmarks.md) |
| Real case/control axis for B4 biological validation | high | open | analysis | 2026-06-18 | mdmanurung | — |
| Tune and document CytoVI-specific default λ after B6 sweep | medium | open | analysis | 2026-06-18 | mdmanurung | — |
| Push feat/cytoanvi branch + upstream PR | high | open | infrastructure | 2026-06-18 | mdmanurung | — |
| B9 mapQC on query controls after surgery | medium | blocked | validation | 2026-06-29 | mdmanurung | — |
| Roider Phase-3 B3 full-cohort (job 25140597 RUNNING ~11h, seed 0; supersedes 25132400) | critical | in_progress | analysis | 2026-07-03 | mdmanurung | — |
| Roider Phase-3 B5 holdout sweep (job 25140598 RUNNING ~11h, seed 0; supersedes 25132895/25132401) | critical | in_progress | analysis | 2026-07-03 | mdmanurung | — |
| Nuñez B8 seed-2 recovery (job 25108052 running ~3.5h as of 2026-06-29) | high | done | analysis | 2026-06-29 | mdmanurung | — |
| Run manifest-mode aggregation after all Phase-3/5 artifacts land | high | open | infrastructure | 2026-06-29 | mdmanurung | — |
| Fix ruff lint findings on cytoanvi package | medium | done | infrastructure | 2026-06-29 | mdmanurung | — |
| Fix B5 AUROC SE formula (Wilcoxon numerator missing — FDR all-significant) | critical | done | analysis | 2026-06-30 | mdmanurung | — |
| Fix unseeded np.random.choice in test_hierarchy_schpl_mock.py | low | done | testing | 2026-06-30 | mdmanurung | — |
| Fix NAN_LAYER local redefinition in tasks_imputation.py (DRY violation) | low | done | infrastructure | 2026-06-30 | mdmanurung | — |
| Regenerate B5 result JSONs after AUROC SE fix (existing values are wrong) | critical | done | analysis | 2026-06-30 | mdmanurung | — |
| Install `mapqc` in conda env before next SLURM submission (unblocks B9) | high | done | infrastructure | 2026-07-03 | mdmanurung | — |
| USER ACTION: scancel stale jobs 25102610 25102547 (25102546 already gone) | high | done | infrastructure | 2026-07-03 | mdmanurung | — |
| Monitor B1 Nuñez reversal on full-cohort run (F-003: resolved — inductive Δ+0.017) | high | done | validation | 2026-06-29 | mdmanurung | — |
| Re-run Nuñez B1 e1000 seeds 0/1/2 after inductive kNN annotation fix (PID 3186154 v3) | critical | done | analysis | 2026-06-30 | mdmanurung | — |
| Fix Nuñez transductive leakage in annotate_nunez.py (L-022) | critical | done | analysis | 2026-06-30 | mdmanurung | — |
| Fix accelerator='gpu' → 'auto' in training.py (F14) | medium | done | infrastructure | 2026-06-30 | mdmanurung | — |
| Add B9 round-robin sample comment + B5 calibration_note (F4/F6) | low | done | analysis | 2026-06-30 | mdmanurung | — |
| B3 aggregator backward-compat for key rename (F2) | medium | done | infrastructure | 2026-06-30 | mdmanurung | — |
| XGBoost, Phenograph, FlowSOM baselines for B1 (F15) | high | done | analysis | 2026-06-30 | mdmanurung | — |
| Fix mask_augment nan_mask branch global RNG (F20) | low | done | infrastructure | 2026-06-30 | mdmanurung | — |
| Correct ARCSINH_COFACTORS dict (F13) | low | done | documentation | 2026-06-30 | mdmanurung | — |
| Consolidate MockTreeNode via conftest import (F30) | low | done | testing | 2026-06-30 | mdmanurung | — |
| Publication-readiness review + fixes (6 parallel agents) — code/docs/tests/packaging/benchmark-reporting | critical | done | infrastructure | 2026-07-03 | mdmanurung | — |
| Fix AnnData.concatenate → anndata.concat in hierarchy.py update path (L-032) | high | done | infrastructure | 2026-07-03 | mdmanurung | — |
| Document EWC Fisher = (E[grad])² approximation; λ not portable (L-033) | medium | done | documentation | 2026-07-03 | mdmanurung | — |
| Add direct ELBO-component unit tests (L-034); Fisher sanity; training-descends; slow-marker CI guard | high | done | testing | 2026-07-03 | mdmanurung | — |
| Standalone packaging: name=cytoanvi, dual-BSD LICENSE, sdist excludes, .gitattributes (D-008) | high | done | infrastructure | 2026-07-03 | mdmanurung | — |
| B5 mean_auroc primary; B3 concordance-not-accuracy relabel; B4/B6/B9/B8 demoted in manifests (D-009/D-010) | high | done | analysis | 2026-07-03 | mdmanurung | — |
| BLOCKER (needs data): independent manually-gated panel-2 labels for a real B3 accuracy claim | critical | open | analysis | 2026-07-03 | mdmanurung | — |
| BLOCKER (needs compute): rerun B5 roider-full at 3 seeds (currently seeds=[0] only) | critical | open | analysis | 2026-07-03 | mdmanurung | — |
| BLOCKER (needs data): external novel-cell-type dataset to make B5 mean AUROC meaningful (>0.7) | high | open | analysis | 2026-07-03 | mdmanurung | — |
| Fill real CytoANVI/CytoVI preprint DOIs in docs once posted (currently "in preparation") | medium | open | documentation | 2026-07-03 | mdmanurung | — |
| Deferred code items DONE: EWC per-sample Fisher (exact E[grad²], default); GPU-sync opt-out (CYTOANVI_DISABLE_FINITE_CHECKS); reconst_loss→reconstruction_term rename | low | done | infrastructure | 2026-07-03 | mdmanurung | — |
| Session-13 actions DONE: scancel 3 stale jobs; mapqc installed (B9 import unblocked); B5 seeds 1-2 submitted (25144240/1) | high | done | infrastructure | 2026-07-03 | mdmanurung | — |
| B9 mapQC: import unblocked (mapqc 0.1.1 in scvi-test env); full validation still needs a real query-mapping run | medium | open | validation | 2026-07-03 | mdmanurung | — |
| Engineering maturity DONE: fixed 15 CI workflows (scvi-tools→cytoanvi install target), py.typed, __version__, ruff lint+format clean, build verified (L-036) | high | done | infrastructure | 2026-07-03 | mdmanurung | — |
| scvi/ namespace collision (L-035): DEFERRED per D-011 — internal/clean-env use only, not on PyPI. Revisit Replace vs Coexist vs upstream-PR before any public upload | high | deferred | infrastructure | 2026-07-03 | mdmanurung | — |
| Reset dist version from inherited scvi-tools 1.5.0rc1 for standalone cytoanvi release (release-strategy) | medium | open | infrastructure | 2026-07-03 | mdmanurung | — |
