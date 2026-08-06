# CytoANVI Scientific Review — Fix Plan

Source: `.living/outputs/reviews/2026-06-30-feat-cytoanvi-main.md`
Review type: mycelium:review (stats-causal, data-pipeline, bioinformatics, llm-failure-modes, doc-schema, code-quality)
Created: 2026-06-30

**Status legend:** `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` won't fix

---

## Phase 0 — Commit locally-fixed issues (≈30 min, unblocks all benchmark reruns)

- [x] **P0-A (F4)**: `_replay_latent_drift` in tasks.py:466,648,652 — already uses ref/replay cells ✓
- [x] **P0-B (F7)**: `rng.choice(query_adata.n_obs, n_ctrl, replace=False)` at tasks.py:528 ✓
- [x] **P0-C (F5)**: `leaf_held = held & ~np.isin(true, list(internal_labels))` at tasks.py:732 ✓
- [x] **P0-D (F18)**: Both tasks.py and aggregate_results.py use `replay_latent_drift`; `recommendation_status` confirmed present (2026-06-30)

---

## Phase 1 — Documentation fixes (≈2h, parallel, no benchmark impact)

### 1a. Import path / convention correctness (F22/F23/F24/F25)
- [x] **P1-A (F22)**: CHANGELOG already uses `cytoanvi.CytoANVI` (done in src/cytoanvi refactor)
- [x] **P1-B (F25)**: CHANGELOG already lists `latent_to_anndata` in hierarchy helpers (done in refactor)
- [x] **P1-C (F23)**: `.living/conventions.md` C-002 already says ValueError at construction (done)
- [x] **P1-D (F24)**: `docs/adr/0003-cytoanvi-hce-schpl-hierarchy.md:38` — added ambiguous-assignment ValueError note (done 2026-06-30)

### 1b. Shared constants (F26/F27)
- [x] **P1-E (F26)**: No duplication — `data.py` imports `VIGNETTE` from `fetch_data.py` and derives `FIGSHARE` from it (single source)
- [x] **P1-F (F27)**: `data.py:26` already imports `NAN_LAYER, SCALED_LAYER` from `benchmarks.common.training` (done)

### 1c. TrainRunner accelerator regression (F14)
- [x] **P1-G (F14)**: `src/scvi/train/_trainrunner.py` — fixed docstring to say "Defaults to 'auto'" (done 2026-06-30); parameter was already `'auto'`, only docstring was stale

### 1d. Seed aggregation (F8)
- [x] **P1-H (F8)**: `benchmarks/common/seeds.py:40` already uses `ddof=1` (done in prior session)

### 1e. API footguns (F19/F20/F21)
- [x] **P1-I (F19)**: `src/cytoanvi/_module.py:loss_with_replay` — already raises `ValueError` when `ewc_importance is None` and continual is active (done in refactor)
- [x] **P1-J (F20)**: `src/cytoanvi/_uncertainty.py:mask_augment` — already threads `generator` to `torch.randperm` (done in refactor)
- [x] **P1-K (F21)**: `benchmarks/cytoanvi/baselines.py:109` — already uses `random_state=seed` (done in refactor)

### 1f. Test reproducibility (F28/F29/F30)
- [-] **P1-L (F28)**: Skip — tasks.py creates `rng = np.random.default_rng(seed)` internally; API refactor not justified
- [x] **P1-M (F29)**: `tests/cytoanvi/test_cytoanvi.py:35` already uses `np.random.default_rng(42).choice(...)` (done in test move)
- [x] **P1-N (F30)**: `tests/cytoanvi/conftest.py` has `MockTreeNode`; test files import from it (done in test move)

---

## Phase 2 — Benchmark logic fixes (≈3–5h, requires benchmark rerun to verify)

### 2a. Baseline error handling (F17)
- [x] **P2-A (F17)**: `benchmarks/cytoanvi/tasks.py:117` — `except (ImportError, ValueError, KeyError) as e:` already narrowed ✓

### 2b. B3 metric rename (F2)
- [x] **P2-B (F2)**: `benchmarks/cytoanvi/tasks.py:302` — renamed to `"p2_inter_method_agreement_vs_knn"` ✓; aggregate_results.py:47-49 backward-compat reads both keys ✓

### 2c. B3 RNG order fix (F10)
- [x] **P2-C (F10)**: `benchmarks/cytoanvi/tasks.py:259-263` — `p1_idx = np.where(~is_p2)[0]` holdout on p1 cells only ✓

### 2d. B4/B6 fallback warning (F11)
- [x] **P2-D (F11)**: `benchmarks/cytoanvi/tasks.py:449-450` — `warnings.warn(...)` before random-split fallback; `"_fallback_split"` in output ✓

### 2e. B5 FDR correction (F3) — CRITICAL
- [x] **P2-E (F3)**: `benchmarks/cytoanvi/tasks.py:391-427` — `multipletests(..., method="fdr_bh")`; `n_fdr_significant`, `mean_auroc_fdr_sig` in output ✓

### 2f. EWC docstring (F16)
- [x] **P2-F (F16)**: `src/cytoanvi/_continual.py:58-60` — semi-supervised ELBO + Fisher importances docstring confirmed ✓

---

## Phase 3 — Data pipeline fixes (≈half day)

### 3a. scennep z-score scale mismatch (F12) — CRITICAL
- [x] **P3-A (F12)**: `src/scvi/external/cytovi/scennep.py:208-213` — `orig_expr` saved before `sc.pp.scale`; pseudobulk uses `orig_expr` ✓

### 3b. A3 KNN imputation — inductive fix (F6)
- [x] **P3-B (F6)**: `benchmarks/cytovi/tasks_imputation.py:119` — `KNNImputer(...).fit(x[~holdout]).transform(x[holdout_idx])[:, midx]` inductive fix ✓

### 3c. B5 evaluation mode flag (F9)
- [x] **P3-C (F9)**: `benchmarks/cytoanvi/tasks.py:351` — `"b5_evaluation_mode": "calibration_transductive"` in output ✓

---

## Phase 4 — Data regeneration (requires GPU, days)

### 4a. Nuñez annotation v2 — train-only CytoVI + k-NN transfer (F1) — CRITICAL
- [x] **P4-A (F1)**: `benchmarks/cytoanvi/annotate_nunez.py` has `annotate_inductive_knn()` + `--inductive` flag (verified); B1 inductive 3-seed re-run is finalized (the "in progress via PID 1520357 (ETA ~07:00 Jul 1)" note is stale). Result: B1 Nuñez full inductive e1000 3-seed, CytoANVI 0.9751±0.0003 vs CytoVI kNN 0.9581±0.0007 (Δ+0.0170) — see `.living/findings/FINDINGS_REGISTRY.md` F-003 (`nunez-full-inductive-e1000-3seed`) and `benchmarks/ANALYSIS_MANIFEST.md` B1 row (`nunez-full-inductive-e1000 ✓`). Marked done during the 2026-08-06 pass.
- [-] **P4-B (F1)**: `annotation_version` param in `load_nunez()` — deferred until v2 `.h5ad` exists (blocked on P4-A GPU run completing and file validation)
- [-] **P4-C (idea 10)**: Archive to Figshare — blocked on user credentials; mark as user action

### 4b. Arcsinh cofactor fix (F13/idea 7)
- [x] **P4-D (F13)**: `benchmarks/common/preprocessing.py:ARCSINH_COFACTORS = {"nunez": 5, "roider": 500, ...}` — CyTOF cofactor=5 already correct ✓
- [x] **P4-E (idea 7)**: `src/scvi/external/cytovi/_preprocessing.py` — added `technology` param + UserWarning when `technology != 'cytof' and global_scaling_factor == 5` (done 2026-06-30)

---

## Phase 5 — Extended features (design decisions required, user-triggered)

- [x] **P5-A (idea 1/6)**: `benchmarks/cytoanvi/baselines.py:162,215` — `flowsom_knn`, `phenograph_knn` present; XGBoost baseline exists ✓
- [x] **P5-B (idea 8)**: `benchmarks/cytoanvi/data.py` — `BNHL_CONTINUAL_SPLIT = {"ref": ["FL", "DLBCL"], "query": ["MCL", "rLN"]}` + `_split_by_entity()` added (entity_key="Entity", done 2026-06-30)
- [x] **P5-C (idea 9)**: `src/cytoanvi/_uncertainty.py` — `get_uncertainty_threshold()` added + exported from `cytoanvi.__init__`; `benchmarks/cytoanvi/metrics.py` — `precision_at_specificity()` added (done 2026-06-30)
- [x] **P5-D (F15)**: `src/scvi/external/cytovi/marker_harmonization.py:collapse_cd45_markers()` — `.. warning::` docstring about naive/memory T cell distinction loss already present ✓
- [-] **P5-E (idea 11)**: conda lock + REPRODUCE.md + Singularity.def — deferred (requires user decision on env spec; mark as user action)

---

## Sequencing

```
P0 (commit) ──┬──► P2 (benchmark logic) ──► P3 (pipeline) ──► P4 (GPU reruns)
              │
P1 (docs)   ──┘ (parallel, no deps)

P5 after P2-P4 validated, before PR
```

## Check-off order for next session

1. `git diff HEAD benchmarks/cytoanvi/tasks.py | grep -E 'replay_latent_drift|leaf_held|rng.choice'` — confirm locals
2. P0-A → P0-D (commit + push) — 30 min
3. P1-A, P1-C, P1-D (CHANGELOG + convention docs) — 30 min
4. P1-G (TrainRunner revert) — 5 min
5. P2-E (B5 FDR) — 2h
6. P3-A (scennep z-score) — 30 min + unit test
7. P2-A (harmony except narrowing) — 15 min
8. P0-D rerun aggregation with fixed keys — 10 min
9. Resubmit roider-full B1/B2/B3/B5 after above
