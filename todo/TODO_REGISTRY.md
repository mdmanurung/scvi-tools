# TODO Registry

Tracks future work items, ideas, and planned improvements.

| Title | Priority | Status | Category | Date | Author | File |
|-------|----------|--------|----------|------|--------|------|
| Full Nuñez/Roider benchmarks at max_epochs=1000 | critical | open | analysis | 2026-06-10 | mdmanurung | [b-track-full-benchmarks.md](b-track-full-benchmarks.md) |
| Real case/control axis for B4 biological validation | high | open | analysis | 2026-06-18 | mdmanurung | — |
| Tune and document CytoVI-specific default λ after B6 sweep | medium | open | analysis | 2026-06-18 | mdmanurung | — |
| Push feat/cytoanvi branch + upstream PR | high | done | infrastructure | 2026-06-18 | mdmanurung | — |
| B9 mapQC on query controls after surgery | medium | blocked | validation | 2026-06-29 | mdmanurung | — |
| Roider Phase-3 B3 full-cohort (job 25140597 RUNNING ~11h, seed 0; supersedes 25132400) | critical | in_progress | analysis | 2026-07-03 | mdmanurung | — |
| Roider Phase-3 B5 holdout sweep (job 25140598 RUNNING ~11h, seed 0; supersedes 25132895/25132401) | critical | in_progress | analysis | 2026-07-03 | mdmanurung | — |
| Nuñez B8 seed-2 recovery (job 25108052 running ~3.5h as of 2026-06-29) | high | done | analysis | 2026-06-29 | mdmanurung | — |
| Run manifest-mode aggregation after all Phase-3/5 artifacts land | high | done | infrastructure | 2026-06-29 | mdmanurung | — |
| B5 REDESIGN COMPLETE: jobs 25145052/53/54 (seeds 0/1/2, 11 types, CytoVI baseline, batch 16384) + merge 25145055. RESULT: CytoANVI mean_auroc 0.484±0.019 (NEGATIVE) vs CytoVI kNN 0.775±0.002. Diagnostic jobs 25149032/33/34 running (CytoANVI-kNN in own latent) | high | done | analysis | 2026-07-04 | mdmanurung | — |
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
| BLOCKER (needs compute): rerun B5 roider-full at 3 seeds (currently seeds=[0] only) | critical | done | analysis | 2026-07-03 | mdmanurung | — |
| BLOCKER (needs data): external novel-cell-type dataset to make B5 mean AUROC meaningful (>0.7) | high | open | analysis | 2026-07-03 | mdmanurung | — |
| Fill real CytoANVI/CytoVI preprint DOIs in docs once posted (currently "in preparation") | medium | open | documentation | 2026-07-03 | mdmanurung | — |
| Deferred code items DONE: EWC per-sample Fisher (exact E[grad²], default); GPU-sync opt-out (CYTOANVI_DISABLE_FINITE_CHECKS); reconst_loss→reconstruction_term rename | low | done | infrastructure | 2026-07-03 | mdmanurung | — |
| Session-13 actions DONE: scancel 3 stale jobs; mapqc installed (B9 import unblocked); B5 seeds 1-2 submitted (25144240/1) | high | done | infrastructure | 2026-07-03 | mdmanurung | — |
| B9 mapQC FAILED AGAIN 2026-07-06: job 25149329 ExitCode 1:0. mapqc 0.1.1 installed but IndexError in _get_per_cell_filtering_info (mode().iloc[0] on empty result). Library bug — blocked until mapqc patched or dataset workaround. manifest updated status=failed | medium | blocked | validation | 2026-07-06 | mdmanurung | — |
| B1 Roider full-cohort COMPLETE 2026-07-06: jobs 25149326/27/28. CytoANVI 0.9317±0.0022 vs kNN 0.8928±0.0034, Δ+0.0388±0.0018 ✅ gate. XGBoost 0.9516. Aggregated → roider_full_b1_multiseed.json. publication_manifest.json updated to complete | high | done | analysis | 2026-07-06 | mdmanurung | — |
| Engineering maturity DONE: fixed 15 CI workflows (scvi-tools→cytoanvi install target), py.typed, __version__, ruff lint+format clean, build verified (L-036) | high | done | infrastructure | 2026-07-03 | mdmanurung | — |
| scvi/ namespace collision (L-035): DEFERRED per D-011 — internal/clean-env use only, not on PyPI. Revisit Replace vs Coexist vs upstream-PR before any public upload | high | deferred | infrastructure | 2026-07-03 | mdmanurung | — |
| Reset dist version from inherited scvi-tools 1.5.0rc1 for standalone cytoanvi release (release-strategy) | medium | done | infrastructure | 2026-07-03 | mdmanurung | — |
| Leiden recompute FIXED (L-040): scanpy 1.12/igraph 1.0.0 seed incompat — unblocks coarser resolutions (faster training + more interpretable labels) | medium | done | infrastructure | 2026-07-04 | mdmanurung | — |
| B5 DIAGNOSTIC RUNNING: kNN-distance OOD in CytoANVI's OWN latent (jobs 25149032/33/34). Will add cytoanvi_knn_mean_auroc to publication_summary.json. Re-aggregate after completion to resolve TTA-vs-latent question (F-013 pending) | high | in_progress | analysis | 2026-07-05 | mdmanurung | — |
| COARSE-Leiden B3/B5 PREPARED (L-041 speed lever): 6 scripts phase3{a,b}_{b3,b5}coarse_*.slurm w/ __RES__ placeholder; calibration job 25145242 picks resolution ~12 clusters (~3-4x faster training + more interpretable). NOT submitted — run after current jobs finish | medium | open | analysis | 2026-07-04 | mdmanurung | — |
| B3 RERUN COMPLETE: jobs 25145151/52/53 (seeds 0/1/2). RESULT: p1 macro-F1 0.828±0.015 ✅, p2 concordance 0.671±0.008 ❌ (gate ≥0.80 NOT met — concordance not accuracy). Panel-2 ground-truth labels still unavailable (see BLOCKER row) | high | done | analysis | 2026-07-04 | mdmanurung | — |
| LFC B1: implement `compute_h_from_x_eps` in MrTotalVAE + MrMultiVAE modules | high | done | infrastructure | 2026-07-11 | mdmanurung | — |
| LFC B2: remove NotImplementedError from `_stats.py`, extend `_construct_design_matrix`, port MRVI LFC block | high | done | infrastructure | 2026-07-11 | mdmanurung | — |
| LFC B3: add LFC kwargs + xarray assembly to `mrtotalvi/_model.py` and `mrmultivi/_model.py` wrappers | high | done | infrastructure | 2026-07-11 | mdmanurung | — |
| LFC B4: add store_lfc tests (shape/coord/finiteness, pde∈[0,1], D2 determinism, backward-compat) | high | done | testing | 2026-07-11 | mdmanurung | — |
| LFC B5: update ADR-0005/0006 + mr_multimodal.md to remove "not implemented" + record D-020–D-023 | medium | done | documentation | 2026-07-11 | mdmanurung | — |
