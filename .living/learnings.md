# Learnings Log

Record unexpected findings, gotchas, and edge cases. Entries feed the crystallize cycle and convention generation.

## Format

```
### L-NNN — [YYYY-MM-DD] Title
**Category**: bug | gotcha | performance | design | data | infra
**Tags**: [comma-separated]
**mitigation_type**: structural | convention | ambient-awareness
**structural_mitigation_candidate**: (name a test or invariant if structural; else leave blank)
**Body**: What happened, what was surprising, what to watch for.
```

---

### L-001 — [2026-06-18] `nan_layer` masking must be applied per-cell during TTA
**Category**: bug
**Tags**: nan-layer, tta-uncertainty, panel-aware, masking
**mitigation_type**: structural
**structural_mitigation_candidate**: test_tta_respects_nan_layer_per_cell
**Body**: TTA uncertainty was averaging over full marker set including unobserved (NaN) backbone positions. The fix requires masking at the per-cell level using `nan_layer`, not applying a global panel mask. Failing to do so inflates uncertainty estimates for multi-panel datasets.

### L-002 — [2026-06-18] `n_labels == 0` guard needed in classifier forward pass
**Category**: bug
**Tags**: classifier, edge-case, n_labels
**mitigation_type**: structural
**structural_mitigation_candidate**: test_cytoanvi_no_labels_smoke
**Body**: When all cells are unlabeled (e.g., during query-only inference), `n_labels` resolves to 0 and the classifier softmax has shape (N, 0) — causing a silent NaN. Added a guard that returns uniform priors in this case.

### L-003 — [2026-06-18] Fisher importances must subsample for large atlases
**Category**: performance
**Tags**: fisher, ewc, continual-update, memory
**mitigation_type**: convention
**structural_mitigation_candidate**: 
**Body**: Computing Fisher importances on full Roider (630k+ cells) OOMed on a 40GB GPU. Subsampling to 10k cells with a log-progress callback is now standard. The approximation is sufficient for EWC regularization since Fisher is used as a weighting mask, not a precision matrix.

### L-004 — [2026-06-18] Surgery from saved model path requires persisted encoder mask
**Category**: bug
**Tags**: surgery, save-load, encoder-mask, query
**mitigation_type**: structural
**structural_mitigation_candidate**: test_surgery_from_saved_path
**Body**: `prepare_query_anndata` reconstructs the panel-aware encoding on the fly. When called from a saved model path (no original training AnnData), the encoder marker mask was missing. Persisting `encoder_marker_mask_` alongside model state dicts fixes this.

### L-005 — [2026-06-10] scib batch correction metrics need >1 batch per sample
**Category**: data
**Tags**: scib, batch-correction, nunez, B2
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: 
**Body**: scib's `graph_connectivity` and `batch_ASW` metrics return NaN or error when a dataset has only one batch. Nuñez has two batches by design; Roider needs batch column correctly set. Always assert `n_batches > 1` before running B2.

### L-006 — [2026-06-29] Mycelium initialized from scratch (no scripts)
**Category**: infra
**Tags**: mycelium, init, living-repo
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: 
**Body**: Mycelium scripts (init_repo.py, install_convention.py, etc.) were not cloned. Living repo scaffolded manually following the core skill protocol. Future runs should clone the mycelium repo to get automated validation and convention install tooling.

### L-007 — [2026-06-01] Unfit module stays in train mode; `get_latent_representation` must force eval
**Category**: bug
**Tags**: eval-mode, train-mode, latent, inference
**mitigation_type**: structural
**structural_mitigation_candidate**: test_get_latent_in_eval_mode
**Body**: `from_cytovi` latent reproduction failed because an unfit module stays in train mode (dropout/BN active). `get_latent_representation` and `get_uncertainty` do not auto-force eval. Fix: both methods now call `module.eval()` before inference and restore training state after. Source: `notes/2026-06-01-cytoanvi.md`.

### L-008 — [2026-06-01] EWC `old_params` must snapshot the reference module; control importances from the batch-extended query model
**Category**: bug
**Tags**: ewc, continual-update, old-params, fisher, reference-module
**mitigation_type**: structural
**structural_mitigation_candidate**: test_ewc_snapshot_from_reference_not_query
**Body**: Two related EWC construction bugs: (1) `old_params` must be snapshotted from the **reference** module, not the query (query has a resized batch embedding — shape mismatch at penalty time); (2) control importances must be computed on the **batch-extended query model**, not the reference. Both were fixed in commits `3eeb1fd8` and `bf34dcba`. Source: `notes/2026-06-01-cytoanvi.md`.

### L-009 — [2026-06-01] EWC state (`importances` / `old_params` / replay batches) is not in `state_dict`
**Category**: bug
**Tags**: ewc, continual-update, save-load, state-dict, replay
**mitigation_type**: structural
**structural_mitigation_candidate**: test_continual_state_survives_round_trip
**Body**: The EWC anchor (`old_params`), Fisher importances, combine rule, and replay buffer are held as Python attributes — not registered in `state_dict`. A save/reload cycle silently drops them. After loading a continual model, `ContinualUpdate` is absent; exact replay-resume requires re-calling `load_query_data_with_replay(..., replay_adata=...)`. A `train()` warning was added when continual update is active at save but missing after load. Source: `notes/2026-06-01-cytoanvi.md`, `docs/user_guide/models/cytoanvi.md`.

### L-010 — [2026-06-29] Roider Phase-3 root cause: batch_size=128 → 52h/seed + NaN divergence at epoch 94
**Category**: performance
**Tags**: batch-size, nan-divergence, roider, training, slurm, baselines
**mitigation_type**: convention
**structural_mitigation_candidate**: 
**Body**: Full Roider (1.24M cells after panel concat) with `batch_size=128` produces ~9,688 steps/epoch → ~52h/seed (infeasible within 48h wall). Additionally, `baselines.py::cytovi_latent_and_knn` diverged to NaN at epoch 94 under this regime. Fix: thread `--batch-size 8192` through the full call graph (`run.py`, `tasks.py`, `training.py`, `baselines.py`) — reduces to ~64× fewer steps, ~50 min/seed. Source: `.scratch/cytoanvi-benchmark/issues/09-roider-b3-b5-full.md`, `docs/review-clear-execute-tasks.md`.

### L-011 — [2026-06-28] Recursive `--input` aggregation silently mixes smoke / synthetic / stale results
**Category**: infra
**Tags**: aggregation, publication, manifest, results
**mitigation_type**: convention
**structural_mitigation_candidate**: 
**Body**: `aggregate_results.py --input .scratch/cytoanvi-benchmark/results/` gathered all JSONs including smoke (epochs=100), synthetic (B8 tiny-data), and stale `roider_*` files alongside valid e1000 results — mixing them into a single summary. Always use manifest-mode (`--manifest publication_manifest.json`) for publication aggregation. Recursive mode is exploratory only. Source: `.scratch/cytoanvi-benchmark/PRD.md` (2026-06-28 decisions).

### L-012 — [2026-06-29] B8 `leaf_held` bias: internal-node cells must be excluded from HCE delta evaluation
**Category**: bug
**Tags**: B8, hce, leaf-held, hierarchical, evaluation-bias
**mitigation_type**: structural
**structural_mitigation_candidate**: test_b8_delta_excludes_internal_node_labels
**Body**: `delta_hierarchical_vs_flat_macro_f1` was computed over all held cells, including cells whose true labels are internal nodes. `predict_hierarchical(leaf_only=True)` can never correctly predict an internal-node label, systematically biasing delta against HCE. Fix: `leaf_held = held & ~isin(true_labels, internal_labels)`. Job 25107490 was cancelled after 2h because of this; see D-XXX. Source: `.scratch/cytoanvi-benchmark/issues/12-b8-hce-label-transfer.md`.

### L-013 — [2026-06-29] B8 HCE and flat-CE arms must both route through `train_cytoanvi`
**Category**: bug
**Tags**: B8, hce, routing, train-cytoanvi, benchmark-parity
**mitigation_type**: convention
**structural_mitigation_candidate**: 
**Body**: The original B8 harness set up HCE manually (hand-rolled hierarchy init) while the flat-CE arm used `train_cytoanvi`. Config drift between the two arms (e.g., different `classification_ratio`, different `plan_kwargs`) made the comparison unfair. Fix: both arms route through `train_cytoanvi(hierarchy_edges=...)` / `train_cytoanvi()` with identical kwargs. Job 25107490 cancelled; resubmitted as 25108052. See D-XXX and L-012. Source: `.scratch/cytoanvi-benchmark/issues/12-b8-hce-label-transfer.md`.

### L-015 — [2026-06-29] B5 novelty AUROC is bimodal: phenotypically distinct types detectable, similar subtypes at chance
**Category**: finding
**Tags**: B5, novelty, auroc, uncertainty, bimodal, cytometry
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: 
**Body**: B5 (novelty detection via uncertainty) on Roider e1000: 4/13 T cell subtypes achieve AUROC ≥0.70 (Tfh 0.744, Treg CD69+ 0.741, Tpr 0.710, Ttox EM3 0.702); 9/13 are below chance (mean 0.467), including Treg CD69- (0.131), Ttox EM2 (0.097), Tdp (0.269). Pattern: phenotypically distant types (Tfh, Treg CD69+) are detectable as novel; highly similar subtypes (CD69+/CD69- Treg pair, EM1/EM2/EM3 Ttox series) are not. This is scientifically interpretable — similarity-based uncertainty cannot separate phenotypically overlapping subtypes — but the mean AUROC (0.467) is a caveat for the publication. Need ≥3 seeds to confirm and frameable as "per-type novelty recall depends on phenotypic distance." Source: `results/e1000/roider_e1000_b5_sweep.json`.

### L-016 — [2026-06-30] Transductive evaluation is a recurring footgun across B-tasks: train/test contamination
**Category**: bug
**Tags**: transductive, leakage, B5, B3, annotation, knn, evaluation
**mitigation_type**: structural
**structural_mitigation_candidate**: test_evaluate_only_on_inductive_holdout
**Body**: Three independent code paths repeat the same transductive evaluation error: (1) Nuñez annotation (`annotate_nunez.py:185`) trains CytoVI on all cells including future test cells, so labels are partly determined by those cells' own latent positions; (2) A3 KNN imputation (`tasks_imputation.py:118`) runs `KNNImputer.fit_transform(x)` on the full matrix including holdout, while CytoVI is strictly inductive; (3) B5 novelty detection trains the model on novel cells' marker profiles (labels blanked), so "novelty detection" is really uncertainty calibration not OOD detection. All three inflate held-out apparent performance by leaking test information into training-set geometry. Pattern: any loop that includes test/query data inside a `fit` or `train` call before the held-out evaluation step is transductive, regardless of whether labels are withheld. Tripwire: assert that no held index appears in the model's training set (not just label list). Source: `/mycelium:review` 2026-06-30, findings F1/F6/F9.

### L-017 — [2026-06-30] Doc/convention drift: 3 cases where .living/ or ADR docs describe wrong runtime behavior
**Category**: bug
**Tags**: convention, adr, documentation, drift, C-002, ADR-0003, changelog
**mitigation_type**: structural
**structural_mitigation_candidate**: test_convention_describes_actual_behavior
**Body**: Review found three doc/code divergences: (1) C-002 says "return uniform priors when n_labels==0" — code raises `ValueError` at construction; (2) ADR 0003 says `set_hierarchy_from_schpl` raises `ValueError` for unmapped leaves — code silently ignores them (only raises for ambiguous, not extra leaves); (3) CHANGELOG lists `scvi.external.CytoANVI` imports — package was promoted to top-level `cytoanvi.CytoANVI`. All three were written accurately at one point but code was refactored without updating the docs. The C-002 case is dangerous because a developer following the convention would implement dead code that masks hard failures. Fix class: when refactoring a behavior, search `.living/conventions.md`, `docs/adr/`, and `CHANGELOG.md` for descriptions of the old behavior and update them in the same commit. Source: `/mycelium:review` 2026-06-30, findings F22/F23/F24.

### L-018 — [2026-06-30] Broad `except Exception` in baseline runners silently converts crashes to missing data
**Category**: bug
**Tags**: exception-handling, baseline, harmony, silent-failure, B1
**mitigation_type**: structural
**structural_mitigation_candidate**: test_baseline_raises_on_code_errors
**Body**: B1 harmony baseline wraps the entire `harmony_latent_and_knn(...)` call in `except Exception as e: result = {"error": str(e)}`. When `aggregate_results.py` later reads `_get(payload, "harmony_knn", "macro_f1")`, it returns None for an error dict — silently treating any code bug as a missing but otherwise-complete benchmark run. This makes B1 reports look complete when the third baseline crashed. Two expected failure modes exist (ImportError for missing harmonypy; ValueError/KeyError for config mismatch) — only these should be caught. Every other exception is a code bug and should propagate. This pattern may exist in other baseline wrappers; audit `baselines.py` for all bare `except Exception` clauses. Source: `/mycelium:review` 2026-06-30, finding F17.

### L-019 — [2026-06-30] AUROC Wilcoxon SE formula omitted `(n1+n2+1)` numerator — inflating z-scores by up to 74×
**Category**: bug
**Tags**: B5, auroc, wilcoxon, standard-error, fdr, statistics, normalization
**mitigation_type**: structural
**structural_mitigation_candidate**: test_b5_auroc_z_score_magnitude_sanity
**Body**: `task_b5_novelty` computed AUROC standard error as `sqrt(1 / (12*n1*n2))` (denominator only), omitting the `(n1+n2+1)` numerator from the Wilcoxon AUC variance formula `Var(AUROC) = (n1+n2+1) / (12*n1*n2)`. For typical holdout sizes (n_novel=500, n_ref=5000), z-scores were inflated by `sqrt(5501) ≈ 74×`, making essentially every AUROC above 0.5 trivially FDR-significant. The docstring described the wrong formula accurately (rare: docs-match-code but both-wrong pattern). Fixed in commit 2026-06-30. Sanity check: for a random classifier (AUROC≈0.5), the z-score should be ~0; if it's ≥3 before FDR correction the SE is almost certainly wrong. Source: `.living/outputs/reviews/2026-06-30-branch-vs-main.md`, F1.

### L-020 — [2026-06-30] λ (ewc_importance) is train-time-only; default from RNA experiments needs retuning for CytoVI
**Category**: gotcha
**Tags**: lambda, ewc, continual-update, cytovi, fisher, hyperparameter
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: 
**Body**: `ewc_importance` (λ) is passed at `train()` time, not at model construction — it does NOT persist on save/load, and is NOT part of the model config. The paper's default λ=100 was derived for scANVI on RNA count data; CytoVI uses arcsinh-transformed intensities, so Fisher importances have a different scale. B6 always sweeps λ before choosing a default for a given dataset. Additionally, user-facing docs must warn: replay batches (`replay_buffer`) are not serialized to disk — a loaded continual model cannot resume training with exact replay without re-supplying `replay_adata`. Source: `docs/user_guide/models/cytoanvi.md` and D-003.

### L-021 — [2026-06-30] Classifier reads backbone-latent only; panel-specific markers don't influence cell type call
**Category**: gotcha
**Tags**: classifier, backbone, panel-specific, label-transfer, B3, multi-panel
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: 
**Body**: The scANVI-style classifier in `CytoANVAE` operates on z1, which is derived from the shared backbone markers only. Panel-specific markers (present in p2 but not p1, or absent from the backbone) are encoded in the per-panel head but do not influence the label-transfer classifier. Consequence: cell types defined exclusively by panel-specific markers (e.g., a CD45RA/CD45RO Treg distinction in p2) will under-resolve relative to types also present in backbone space. This is a fundamental design constraint from M1+M2 sharing z1 across panels. Users running B3 with a high-resolution p2 panel should not expect CytoANVI to resolve panel-specific subtypes. Source: `docs/user_guide/models/cytoanvi.md`, D-001.

### L-022 — [2026-06-30] Nuñez joint-Leiden proxy labels are transductively leaky for B1 evaluation
**Category**: data
**Tags**: nunez, annotation, leakage, leiden, knn, proxy-labels, b1, transductive
**mitigation_type**: fix-applied
**structural_mitigation_candidate**: annotate_nunez.py --inductive flag
**Body**: The standard CytoVI-tutorial annotation runs Leiden on the joint latent (batch 0 + batch 1 together), then maps clusters to PBMC types. This is transductively leaky for B1 (label transfer from batch 0 → batch 1): batch 1 cells participate in the clustering that defines their own proxy labels. Fix: `annotate_inductive_knn()` keeps batch 0 labels from joint Leiden unchanged, fits kNN on batch 0 latent, predicts batch 1 labels from batch 0 structure alone. Result: 2,135/100k batch 1 cells (2.1%) received updated labels; batch 0 unchanged. The original joint-Leiden cluster ID→cell type dict (`NUNEZ_TUTORIAL_ANNOTATION`) must NOT be re-applied on a batch-0-only re-run of Leiden — cluster integer IDs are not stable across different cell subsets. Use existing labels from joint-Leiden on batch 0, then kNN-transfer. Committed e4170054; `data/nunez_annotated.h5ad` replaced; leaky backup at `data/nunez_annotated_leaky_v1.h5ad`.

### L-023 — [2026-06-30] `transform_arcsinh` lives in `cytovi/_preprocessing.py`, not `benchmarks/common/preprocessing.py`
**Category**: gotcha
**Tags**: preprocessing, arcsinh, cofactor, cytovi, file-location
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: 
**Body**: The `transform_arcsinh()` function (cofactor-aware arcsinh normalization) is defined in `src/scvi/external/cytovi/_preprocessing.py`. The review checklist (P4-E) incorrectly pointed to `benchmarks/common/preprocessing.py`, which has an unrelated `ARCSINH_COFACTORS` constant dict but not the function itself. The distinction: `benchmarks/common/preprocessing.py` stores dataset-level cofactor configs; `cytovi/_preprocessing.py` implements the transformation and now carries the `technology` guard param. Source: grep-based discovery, session 2026-06-30.

### L-024 — [2026-06-30] `run.py --max-cells` defaults to 100k; Nuñez "full dataset" is capped by default
**Category**: gotcha
**Tags**: nunez, max-cells, benchmark, default-value, run.py
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: 
**Body**: `benchmarks/cytoanvi/run.py` and `benchmarks/cytoanvi/data.load_nunez_data()` both default to `max_cells=100_000`. Any B1/B2 Nuñez run launched WITHOUT an explicit `--max-cells` flag uses the same 100k subsample as one launched with `--max-cells 100000`. This meant two seemingly different B1 processes (PID 1520357 without explicit flag, PID 2539861 with `--max-cells 100000` explicit) were in fact running identical workloads and the "race condition" on the shared output JSON was benign. Implication: the Nuñez full-dataset cell count at 100k may miss rare populations; if publication gate requires the genuine full cohort, set `--max-cells None` (or increase the cap in `data.py`).

### L-025 — [2026-06-30] harmonypy ≥0.2.0 returns `Z_corr` as (n_cells, n_components), not (n_components, n_cells)
**Category**: bug
**Tags**: baselines, harmonypy, b1-inductive, transposition, environment
**mitigation_type**: fix
**structural_mitigation_candidate**: yes — add shape convention check whenever wrapping harmonypy
**Body**: `harmony_latent_and_knn` in `baselines.py` applied `.T` to `ho.Z_corr` assuming old convention (n_components, n_cells). harmonypy 0.2.0 (both scvi and scvi-test envs) returns (n_cells, n_components) already — `.T` produces (n_components, n_cells) = (21, 100000), and boolean indexing with a (100000,) mask raises `IndexError`. Also, `IndexError` was not caught in the `tasks.py` try/except, killing the entire multiseed run. Fix: use `ho.Z_corr.T if ho.Z_corr.shape[0] == n_comp else ho.Z_corr`. Commit: e8a6d5f9.

### L-026 — [2026-07-01] B5 holdout sweep: nan_layer auto-detection unreliable across 47 successive model instantiations
**Category**: bug
**Tags**: nan-layer, b5, encoder-mask, setup-anndata, multi-model-loop, backbone-detection
**mitigation_type**: structural
**structural_mitigation_candidate**: always pass nan_layer explicitly; never rely on auto-detection in loops
**Body**: B5 holdout sweep (job 25132401) crashed at epoch 1 step 0 of the SECOND Leiden cluster with all-NaN z_encoder output (shape (8192,10)). Root cause: in `setup_anndata`, auto-detection (`if nan_layer is None and "_nan_mask" in adata.layers`) worked for the first model but not subsequent ones, leaving `encoder_marker_mask=None` and allowing NaN panel-2-specific marker columns to enter the encoder. Fix: pass `nan_layer=NAN_LAYER` explicitly in `benchmarks/cytoanvi/run.py` kw dict so PROTEIN_NAN_MASK is always registered regardless of auto-detection state. Also added `del model + gc.collect() + cuda.empty_cache()` in `task_b5_novelty` after scoring each holdout cluster to prevent GPU memory accumulation across 47 iterations. Commit: 3575b392.

### L-027 — [2026-07-01] B6 and B9 in run.py bypass the shared `kw` dict, missing `nan_layer`
**Category**: bug
**Tags**: nan-layer, run.py, b6, b9, kwarg-bypass, backbone-detection
**mitigation_type**: structural
**structural_mitigation_candidate**: test_run_b6_b9_nan_layer_threaded
**Body**: `run.py` builds a shared `kw` dict (lines 90–99) that includes `nan_layer=NAN_LAYER`, but B6 (`task_b6_lambda_sweep`) and B9 (`task_b9_mapqc`) both use explicit per-kwarg call sites that did not include `nan_layer`. B1/B2/B4/B5/B8 all correctly use `**kw` or `**b5_kw`/`**b8_kw`. The divergence happened because B6 and B9 needed non-standard extra kwargs (lambdas, mapqc params) and were written with a full explicit kwargs list that omitted `nan_layer`. Fix: added `nan_layer=NAN_LAYER` to both explicit call sites. Pattern: any future task call that does NOT use `**kw` must be manually audited for missing `nan_layer`.

### L-028 — [2026-07-01] aggregate_results.py resolves manifest artifact paths from CWD, not manifest location
**Category**: bug
**Tags**: aggregate-results, manifest, path-resolution, publication
**mitigation_type**: structural
**structural_mitigation_candidate**: test_manifest_path_resolution_non_cwd
**Body**: `_manifest_inputs` called `Path(artifact["path"])` which resolves relative to the current working directory, not relative to the manifest file. When the manifest contains relative paths like `.scratch/cytoanvi-benchmark/results/...`, calling the aggregator from any directory other than the repo root silently fails to find files. Fix: added `_resolve_artifact_path(raw, manifest_dir)` helper and `manifest_dir: Path | None` param to `_manifest_inputs`; call site passes `args.manifest.parent`. Rule: any function that reads paths from a config/manifest file must resolve them relative to the config file, not CWD.

### L-029 — [2026-07-01] Leiden clusters unseeded across all call sites; cache masks first-run non-reproducibility
**Category**: bug
**Tags**: leiden, seed, reproducibility, data.py, roider, nunez
**mitigation_type**: structural
**structural_mitigation_candidate**: test_leiden_labels_deterministic_across_envs
**Body**: `_leiden_labels` in `data.py` called `sc.tl.leiden` without `seed=`, making first-run Leiden labels environment-dependent. The roider Leiden cache partially mitigates (subsequent seeds load cached labels), but the initial cache creation and the Nuñez path (no cache) are non-reproducible. Fix: added `seed: int = 0` to `_leiden_labels` and threaded it through `load_nunez`, `apply_leiden_cell_types` in roider_metadata.py, and `annotate_roider_obs`. Note: do NOT include `seed` in the Leiden cache key — changing seed would invalidate existing caches.

### L-030 — [2026-07-01] EWC `combine_type="product"` silently underflows to 0 in float32
**Category**: bug
**Tags**: ewc, fisher, continual-update, combine-type, underflow, float32
**mitigation_type**: structural
**structural_mitigation_candidate**: test_ewc_product_combines_nonzero
**Body**: When `combine_type="product"`, the ContinualUpdateState computes `w = w * c` (Hadamard product of two Fisher importance vectors). For parameters with small-but-nonzero Fisher importances in both reference and control, the product underflows to float32 0.0, silently disabling the EWC penalty for those parameters. This does NOT raise an error; the penalty just becomes 0 for those params, effectively ignoring them. Fix: `w = torch.clamp(w * c, min=1e-10)`. Prefer `combine_type="sum"` for production; "product" is still useful for high-confidence penalization but requires the clamp.

### L-014 — [2026-06-01] Figshare egress returns HTTP 202 / 0 bytes in dev env; no pip in SLURM queue
**Category**: infra
**Tags**: figshare, data-download, slurm, mapqc, environment
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: 
**Body**: Two env constraints that block otherwise-straightforward steps: (1) Figshare download returns HTTP 202 (async processing) / 0-byte file in the dev environment — data must be acquired via user action outside the agent. (2) `pip install` is not allowed inside the SLURM job queue — `mapqc` (needed for B9) must be installed in the conda environment before job submission. B9 is currently blocked by (2). Source: `.scratch/cytoanvi-benchmark/issues/01` and `issues/13`.

### L-031 — [2026-07-01] locals().get() antipattern for snapshots taken inside try blocks
**Category**: code-quality
**Tags**: try-finally, was_training, snapshot, clarity
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: 
**Body**: `locals().get("was_training")` was used to guard the finally block against an OOM scenario where `was_training` might not yet be assigned. The fix: move the snapshot before the try block. The snapshot is read-only (no side effects), so it can safely execute before the try. The `locals().get()` pattern is always a red flag — if a variable might not be assigned, the real fix is to assign it before the try, not to look it up from the locals dict in finally.

### L-032 — [2026-07-03] AnnData.concatenate is removed in anndata ≥0.10 — use anndata.concat
**Category**: dependency-drift
**Tags**: anndata, concatenate, runtime-error, hierarchy, scHPL-update
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: grep for `.concatenate(` across the codebase in CI
**Body**: `src/cytoanvi/hierarchy.py` called `reference_adata.concatenate(query_adata)` in the scHPL `mode='update'` path. `AnnData.concatenate` was deprecated in anndata 0.8 and removed in 0.10+, so any user on a current anndata (the scvi-tools default) hit an `AttributeError` at runtime — but only when exercising that specific branch, so no test caught it. Fix: `anndata.concat([reference_adata, query_adata], join="outer", index_unique="-")` (the `index_unique="-"` preserves the batch-suffix obs-name dedup that the old API did implicitly). Lesson: removed-API calls hidden in rarely-exercised branches are invisible to smoke tests — grep for them.

### L-033 — [2026-07-03] EWC Fisher importance here is (E[grad])², not E[grad²] — lambda is not portable
**Category**: numerical-semantics
**Tags**: EWC, fisher, continual-update, ewc_importance, batch-mean-gradient
**mitigation_type**: documentation
**structural_mitigation_candidate**: per-sample gradients via torch.func.vmap+grad (deferred — too invasive)
**Body**: `fisher_importances` in `_continual.py` calls `loss.backward()` on a batch-MEAN loss, so the accumulated squared gradient is `(E[grad])²`, the square of the mean gradient — NOT the true diagonal Fisher `E[grad²]`. The two differ by a factor that scales with batch size. Training isn't broken (the bias is absorbed into the `ewc_importance` λ hyperparameter), but λ values are NOT portable across batch sizes or other EWC implementations, and absolute importances are not interpretable. Decision this session: document the approximation in the docstring rather than rewrite to per-sample gradients. Anyone copying an EWC λ from the RNA domain (e.g. 100) will get meaningless strength — λ must be tuned for this codebase.

### L-034 — [2026-07-03] The semi-supervised ELBO had zero direct unit tests — smoke tests can't see a dropped nan-mask
**Category**: test-coverage
**Tags**: elbo, unit-test, nan-mask, classification_ratio, N_EPOCHS-2
**mitigation_type**: structural
**structural_mitigation_candidate**: DONE — added tests/cytoanvi/test_cytoanvi_elbo_components.py
**Body**: Every CytoANVI test went through `model.train()` for `N_EPOCHS=2` then checked outputs. None exercised `CytoANVAE.loss()` directly, so a dropped `nan_mask` from the reconstruction term, a wrong KL coefficient, a collapsed M1+M2 hierarchy, or `classification_ratio` silently pinned to 0 would all still "train" and pass. Added a direct-loss test that asserts: finite loss/reconstruction/KL; masked markers do NOT change reconstruction (with `torch.manual_seed` reset to neutralize the stochastic z2 sample); `loss(ratio=1) == loss(ratio=0) + ce_loss`; and `n_labels==0` raises. Key gotcha discovered: `LossOutput.reconstruction_loss`/`kl_local` are dicts (`lo.reconstruction_loss["reconstruction_loss"]`), and the loss path samples z2 stochastically so deterministic comparisons need a seed reset before each call.

### L-033-UPDATE — [2026-07-03] Fisher approximation FIXED with per-sample estimator
**Category**: numerical-semantics
**Tags**: EWC, fisher, per-sample, resolved
**Body**: The L-033 batch-mean `(E[grad])²` bias is now fixed. `fisher_importances(..., per_sample=True)` (the new default) forces loader `batch_size=1` and accumulates per-cell `grad²`, returning `mean_i grad_i²` = the exact diagonal empirical Fisher `E[grad²]`, invariant to batching. Cost: one backward per cell, bounded by `max_cells` (10k) and computed once at surgery — the intended trade-off. `per_sample=False` keeps the old fast biased proxy for cheap estimates. `ContinualUpdate.configure` uses the exact default. Note: absolute scale still depends on ELBO magnitude / max_cells, so λ (`ewc_importance`) still needs codebase tuning — but it is now a genuine Fisher, not a squared-mean-gradient. Verified: test_cytoanvi_continual (non-negative/finite) passes on the per-sample path.

### L-035 — [2026-07-03] Standalone rename ships TWO import packages under one dist → scvi/ namespace collision
**Category**: packaging
**Tags**: pep-namespace, distribution, scvi-tools-fork, D-008, blocker
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: conflict guard on import, or vendor CytoVI under cytoanvi/
**Body**: The wheel for dist `cytoanvi` ships BOTH `scvi/` and `cytoanvi/` import packages (packages=['src/scvi','src/cytoanvi']) — the fork needs its *modified* scvi because CytoVI lives at `scvi.external.cytovi`. Consequence: `pip install cytoanvi` drops a top-level `scvi/` into site-packages OWNED by dist `cytoanvi`. If the real `scvi-tools` is also installed, two distributions claim the `scvi/` import name → file overwrites, and `pip uninstall scvi-tools` can rip out files cytoanvi needs. This is the classic "two dists, one import package" antipattern and the single biggest packaging-maturity defect. Does NOT block local CI/lint/build; DOES block a clean public PyPI upload. Resolution is a fork-in-the-road: (a) REPLACE model (never co-install with scvi-tools; document + add an import-time conflict guard), or (b) COEXIST (vendor CytoVI under cytoanvi/, or revert to upstream-PR). Also: dist version 1.5.0rc1 is inherited from scvi-tools and should be reset for a standalone release (user's call).

### L-036 — [2026-07-03] Standalone package rename silently broke all 15 CI workflows
**Category**: ci
**Tags**: rename, github-actions, install-target, D-008
**Body**: Renaming the dist to `cytoanvi` (D-008) left 15 `.github/workflows/*.yml` installing `"scvi-tools[...] @ ."`, which no longer resolves — CI would fail at the install step on every push/PR. The Dockerfile and .readthedocs.yaml were safe because they install by path (`.`, `.[extra]`), which is name-agnostic. Lesson: a dist rename must sweep every *name-based* install target (workflows, tox, requirements pins) — grep `scvi-tools\[|scvi-tools @|pip install scvi-tools` repo-wide; path-based installs are unaffected. Fixed in c818dcbd.

### L-037 — [2026-07-03] B3 full-cohort per-seed cost = ~4 sequential 1000-epoch trainings; 14h wall-time was far too short
**Category**: benchmark-ops
**Tags**: B3, slurm, wall-time, roider-full, timeout
**Body**: B3 job 25140597 (`phase3_b3b5_roider.slurm`, `for SEED in 0 1 2`) was given `--time=14:00:00` on a "~1h train/seed" assumption. Reality from the log: each seed runs ~4 full 1000-epoch trainings on the merged ~1.24M-cell panel-1+panel-2 cohort (`train_cytoanvi` on merged data + `cytovi_latent_and_knn` for the kNN baseline), ~1–1.7h each (6s/it early → 3.8s/it warm), plus `merge_batches` data prep. Seed 0 alone did not finish in 13.5h. A serial 3-seed B3 needs ~40–50h. The job timed out at 14h having written NO results (the only roider_full_b3_s0.json on disk is a stale Jul-1 artifact). Users cannot `scontrol update timelimit` on this cluster (Access denied), though gpu-long allows up to 7 days — so the fix is at submission time. Right-size the rerun: either split into 3 parallel single-seed jobs (~13–15h wall each, matching the B5 pattern) or one serial job with `--time=3-00:00:00`. HELD per maintainer decision 2026-07-03.

### L-038 — [2026-07-04] B5 47-cluster sweep is infeasible in 48h; redesigned to 11 types + CytoVI baseline
**Category**: benchmark-ops
**Tags**: B5, novelty, wall-time, cytovi-baseline, TTA, checkpoint
**Body**: The roider-full B5 holdout sweep holds out EVERY cell type in turn (`for ht in types`), and roider-full labels are ~47 Leiden clusters → 47 sequential runs/seed. Measured throughput from job 25140598: 7 clusters in 25h = ~3.5h/cluster (dominated NOT by the ~40min training but by ~2.9h of post-training evaluation — two `get_uncertainty` TTA passes at `tta_rep=50` + `precision_at_specificity` + full-AnnData copies). 47 × 3.5h ≈ 165h/seed ≫ 48h; and the sweep writes its JSON only at the end, so a timeout yields ZERO output (why no s0.json existed at 25h). No parameter tweak closes a 3.5× gap. Redesign (commit): (1) `--b5-max-holdout-types 11` sweeps only the 11 most populous clusters — feasible AND more interpretable; (2) `--b5-cytovi-baseline` adds an unsupervised CytoVI kNN-distance OOD AUROC per type so B5 is a fair COMPARISON, not a baseline-free number (the earlier review flagged the missing baseline); (3) `--b5-no-logit` drops the 2nd TTA pass; (4) `--b5-checkpoint` writes per-type incrementally so a timeout still yields partial results. Verified end-to-end on synthetic (2-type limit, both AUROCs, checkpoint, logit=null). The 3 timed-out-bound 47-cluster jobs (25140598/25144240/25144241) + merge job 25144367 were cancelled 2026-07-04. New scripts phase3b_b5redesign_roider_s{0,1,2}.slurm prepared; NOT yet submitted (pending approval).

### L-039 — [2026-07-04] B5 per-cluster cost is ~90% CytoANVI TRAINING; TTA uncertainty is free
**Category**: benchmark-ops
**Tags**: B5, profiling, TTA, training, batch-size, annbatch
**Body**: Direct micro-benchmark (job 25145028, .scratch/cytoanvi-benchmark/profile_tta.py, L40S, 104k train / 26k eval, batch 16384): CytoANVI train **596.8s (~88%)**, CytoVI baseline train+score **79.4s (~6%)**, latent TTA (tta_rep=50) **0.3s**, logit TTA **0.2s**, data load 13s (one-time). So the earlier hypothesis that the ~2.9h/cluster "overhead" was the TTA `get_uncertainty` passes was WRONG — TTA is essentially free (well vectorized), and `--b5-no-logit` saves ~0.2s (removed from the production scripts; logit metric kept for free). The real driver is **CytoANVI training**, which is ~7.5× slower than CytoVI for the same cells — largely the label-marginalized ELBO broadcasting over ~46 Leiden labels every step (`broadcast_labels` n_labels×). Consequences: (1) the old 3.5h/cluster was slow-GPU training (gpu11 ≈ ~4× slower than L40S), not TTA; (2) the right speed levers are **bigger batch** (fewer steps/epoch — user's instinct was correct; scripts now use 16384), **fewer/larger label classes** (coarser Leiden would cut the n_labels broadcast, but leiden recompute at a new resolution is broken in this env — igraph `community_leiden` TypeError), and **faster GPUs**; (3) **annbatch does NOT help** — data loading is 13s one-time, not the bottleneck; annbatch is for out-of-core scale (millions of cells), irrelevant to 620k×30 in-memory. The CytoVI baseline adds only ~6% — cheap, keep it.

### L-040 — [2026-07-04] Leiden recompute FIXED: scanpy 1.12 forwards seed= into igraph 1.0.0 which rejects it
**Category**: dependency-drift
**Tags**: leiden, scanpy, igraph, clustering, resolution, resolved
**Body**: `_leiden_labels` (benchmarks/cytoanvi/data.py) called `sc.tl.leiden(..., flavor="igraph", seed=seed)`. With scanpy 1.12.1 + igraph 1.0.0, scanpy forwards `seed=` into igraph's `community_leiden(**clustering_args)`, and igraph 1.0.0 raises `TypeError: unexpected keyword argument` (it never accepted seed). This broke any Leiden RECOMPUTE (new resolution / refresh); the cached r=1.0 parquet path was unaffected, which is why real runs worked but --leiden-resolution 0.3 failed (jobs 25145023/25145025). Fix: seed igraph's own RNG (`igraph.set_random_number_generator(random.Random(seed))` — it wants a random.Random, NOT a numpy Generator) and drop the seed kwarg; `n_iterations=2` matches scanpy's igraph-flavor default. Verified deterministic across same-seed runs. This UNBLOCKS coarser Leiden resolutions, which is the real CytoANVI training-speed lever (L-039: cost is the label-marginalized ELBO broadcasting over n_labels — fewer classes = faster) AND gives more interpretable label sets for B3/B5. Note: an explicit refresh at r=1.0 would now produce labels that differ from the cached parquet (different RNG path); default cache use is unaffected.

### L-041 — [2026-07-04] CytoANVI training is launch/overhead-bound; TF32 & torch.compile do NOT help
**Category**: performance
**Tags**: speed, tf32, torch.compile, launch-bound, n_labels, marginalization
**Body**: A/B speed profile (.scratch/cytoanvi-benchmark/profile_speed.py, job 25145234, A100-MIG, 150 fixed epochs / 100k cells / batch 16384, n_labels=47): baseline 82.9s, +TF32 85.8s (NO help — within noise), +TF32+no-finite-checks 80.2s (~3%), +TF32+torch.compile 109.7s (+32%, WORSE). Conclusion: CytoANVI training is **launch/overhead-bound** — the label-marginalized ELBO broadcasts z1 over ~n_labels (47) and runs the z2 encoder/decoder on n_labels×batch rows, spawning many small kernels. TF32 accelerates large matmuls, which aren't the bottleneck → no gain. torch.compile makes it WORSE (compile overhead + graph breaks on the dynamic scvi module returning a LossOutput dataclass). Removing the per-step finite-check CUDA syncs (CYTOANVI_DISABLE_FINITE_CHECKS=1) gives only ~3% and costs the NaN diagnostic, so not worth it in production. **There is no free code-level speedup.** The only real lever is structural: fewer label classes (coarser Leiden resolution, now unblocked by L-040) linearly shrinks the n_labels marginalization — but that changes label granularity, a SCIENCE decision, not a free optimization. NOTE (advisor): max_holdout_types (B5 redesign) reduced the NUMBER of trainings, not per-training cost — each training still marginalizes over all ~46 labels. This repeats the TTA lesson: measure the mechanism before implementing the assumed lever.

### L-043 — [2026-07-05] aggregate_b5_multiseed.py dropped dataset; summarize_multiseed missed B5 headline
**Category**: bug
**Tags**: aggregation, publication-summary, B5, dataset-field
**Body**: Two bugs blocked final publication aggregation. (1) `aggregate_b5_multiseed.py` built the merged payload without copying `dataset` from the per-seed inputs → `_validate_manifest_file` raised `ValueError: dataset=None does not match 'roider-full'`. Fix: collect `dataset` from the first seed file and include it in the payload. (2) `summarize_multiseed()` in `aggregate_results.py` only extracted B1/B2/B3 metrics from the `summary` dict; B5 headline metrics live in the `headline` key and were silently dropped, leaving the B5 entry in `publication_summary.json` with only `seeds`. Fix: add explicit extraction of `mean_auroc_mean/std`, `cytovi_mean_auroc_*`, `best_auroc_*`, `n_fdr_significant_mean` from `headline`. Result: `publication_summary.json` now contains all required B1/B2/B3/B5/B8 entries (commit f4f4f3d2).

### L-042 — [2026-07-04] B5 (robust, 3-seed): CytoVI OOD baseline decisively beats CytoANVI TTA-uncertainty
**Category**: finding
**Tags**: B5, novelty, negative-result, cytovi-baseline, TTA-uncertainty
**Body**: The redesigned 3-seed B5 (roider-full, 11 most-populous Leiden holdouts, jobs 25145052/53/54): CytoANVI TTA-uncertainty **mean_auroc = 0.484 ± 0.019 (below chance)** vs an unsupervised CytoVI kNN-distance OOD baseline **0.775 ± 0.002**. CytoANVI loses on 0/11 held-out types (seed 0: e.g. type 9 0.209 vs 0.722), often anti-correlated with novelty (AUROC < 0.5 = MORE confident on novel cells). Tight variance across seeds → not noise. This is a robust NEGATIVE result: **B5 as formulated does not support CytoANVI for novelty detection**; a trivial latent-distance detector is far better. Adding the CytoVI baseline (the redesign) is what turned a bare near-chance number into an interpretable, damning comparison — exactly why a novelty benchmark needs a baseline. Open follow-up (tracked): kNN-distance OOD in CytoANVI's OWN latent — if it also beats TTA-uncertainty, the fix is "ship latent-distance OOD, drop TTA"; if it also loses to CytoVI, CytoANVI's latent is the weaker OOD space. For the writeup: either reframe B5 around latent-distance OOD, or report the negative result honestly and drop the TTA-novelty claim.

### L-044 — [2026-07-06] CytoANVI training is COMPUTE-bound (corrects L-041); bf16 autocast = ~27% free speedup
**Category**: finding
**Tags**: speed, profiling, bf16, autocast, compute-bound, n_labels, marginalization
**Body**: Kernel-level torch.profiler trace (.scratch/cytoanvi-benchmark/profile_kernels.py, RTX PRO 6000 Blackwell MIG 1g.24gb, n_labels=45, batch=16384 → 737k marginalized rows) **corrects L-041's "launch/overhead-bound" conclusion**. The batch-scaling probe is decisive: per-step wall scales LINEARLY with batch (24.5/48.6/98.5 ms at 4k/8k/16k) and ms-per-1k-rows is flat (~6.0) — the textbook signature of **compute-bound**, not launch-bound (which would give flat ms/step across batch). CUDA-time category split: **75% elementwise** (batch_norm fwd+bwd ~14%, plus the label-marginalized distribution ops — Normal.log_prob, kl, add/mul/div/cat on the n_labels×batch tensors), **25% small SIMT matmuls** (`cutlass_80_simt`, `magma_sgemmEx` — NOT tensor-core kernels because n_hidden=128 is small). This explains L-041's TF32 null result mechanistically: TF32 only accelerates tensor-core matmuls, which are 25% of time AND run as SIMT kernels here → TF32 can't engage. **NEW lever L-041 missed (it only tested TF32 = matmul-only): bf16 autocast gives ~27% speedup** (97.5 → 70.7 ms/step) because it halves memory traffic for the dominant memory-bound elementwise ops AND makes matmuls tensor-core-eligible. Numerically sound over 40 steps: loss finite, tracks fp32 within 0.33% (same convergence). **VALIDATED end-to-end** (.scratch/cytoanvi-benchmark/ab_bf16_b1.py, real Nuñez B1, 80k cells, 30% blanked holdout, 200 fixed epochs, same seed/split): fp32 56.7s / macro-F1 0.9530 vs bf16-mixed 30.7s / macro-F1 0.9534 → **+45.8% faster wall, F1 delta +0.0004** (negligible, within seed noise; end-to-end gain exceeds the per-step 27% because bf16 also speeds the full-epoch/validation passes). **Confirmed at high n_labels** (.scratch/cytoanvi-benchmark/ab_bf16_roider47.py, real Roider panel-1, 180k cells, **47 Leiden labels**, batch 16384, 150 fixed epochs, same seed/split): fp32 152.5s / macro-F1 0.8273 vs bf16-mixed 107.7s / 0.8481 → **+29.4% faster, F1 delta +0.0208** (bf16 not degraded — slightly higher, single-seed so read as "no harm"). So the speedup is config-dependent (~29–46%, always positive) and F1-neutral across 11- and 47-label regimes. Wired into `benchmarks/common/training.py::train_cytoanvi` via `precision=` arg / `CYTOANVI_TRAIN_PRECISION` env var (fp32 remains the default). Enable via Lightning `precision="bf16-mixed"`, not manual autocast. Structural lever (fewer labels) still holds and is linear — but bf16 is the free code-level win L-041 wrongly concluded didn't exist.

### L-045 — [2026-07-06] CytoANVI assumes the unlabeled category sorts LAST (silent mislabel bug)
**Category**: bug
**Tags**: labels, unlabeled-category, hierarchy, HCE, setup_anndata, hao
**Body**: `CytoANVAE._observed_label_names()` returns `self._label_mapping[:self.n_labels]` — it takes the first n_labels entries of the (alphabetically sorted) category list and ASSUMES the unlabeled category is the last entry. This holds only when `unlabeled_category` sorts last among all label names. It breaks for any dataset with label names that sort AFTER the unlabeled string. Concretely on the Hao CITE-seq PBMC ref (celltype.l2 has lowercase-leading names pDC/cDC1/cDC2/dnT/gdT): with `unlabeled_category='Unknown'`, sorted mapping is [...,'Unknown'(idx31),...,'gdT','pDC'(idx36)], n_labels=36, so `_observed_label_names()=mapping[:36]` INCLUDES 'Unknown' and EXCLUDES 'pDC' — i.e. the model treats a real leaf ('pDC') as the unlabeled slot and 'Unknown' as a real class. Surfaces as `set_hierarchy` raising "hierarchy references labels not in the model's observed categories: ['pDC']", but the mislabeling would silently corrupt training/prediction even without a hierarchy. Nuñez never hit this because its cell types are all uppercase-leading and sort before 'Unknown'. **ROOT CAUSE (found + FIXED, commit pending):** the invariant "unlabeled is the final code" is established by scvi's `LabelsWithUnlabeledObsField._remap_unlabeled_to_final_category` (src/scvi/data/fields/_scanvi.py), which only moved the unlabeled category to last when it had CELLS (`if unlabeled in labels`). A **0-cell declared unlabeled category** (e.g. Hao with all cells labelled at l2 and 'Unknown' declared for future query use) hit neither branch → left in sorted position → the whole downstream chain (n_labels = registry-1, observed-label slicing, y_prior, class weights, classifier codes) mislabels the actual last category as the unlabeled slot. So it's NOT fundamentally about sort order — it's the 0-cell case; sort order only determines WHICH real label gets clobbered. Fix: change the guard to `if unlabeled in mapping` (move to last whenever present, incl. 0 cells). Regression test `test_zero_cell_unlabeled_category_moved_to_last`; SOLO (7) + HCE/hierarchy (21) suites still green (the one scanvi failure is a pre-existing missing-captum dep). The 'zzz_unlabeled' workaround is now unnecessary.

### L-046 — [2026-07-07] HCE is inert for leaf-only labels; helps only under mixed-granularity annotation
**Category**: finding
**Tags**: B8, hce, hierarchy, mixed-granularity, partial-annotation, hao, negative-then-positive
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: test_hce_equals_flat_ce_for_leaf_only_targets
**Body**: A first B8 run on the Hao CITE-seq ADT reference (real 8-internal-node l1->l2 tree, all cells labelled at fine l2 leaves) showed HCE ~0.03 WORSE than flat CE — the opposite of Nuñez's +0.086. Validation overturned this as an artifact: (1) `hierarchical_cross_entropy_loss` computes `-log((softmax @ R.T)[target] + eps)`; for a LEAF target the subtree mass `(softmax @ R.T)[leaf] = softmax[leaf]`, so HCE == flat CE EXACTLY (loss and gradients match to 1e-7 — unit-proven). The reachability internal rows are never indexed when every label is a leaf, so the hierarchy does literally nothing. (2) The spurious -0.03 was a bf16 numerical bug: `eps = finfo(bf16).eps ~ 7.8e-3` distorted the log (fixed by computing the loss in fp32, commit fcfff064) PLUS uncontrolled RNG/inference-method differences between the two separate trainings. A controlled check (leaf-only, same scvi.settings.seed, same predict()) gives delta -0.0012 ~ 0, confirming HCE==flat for fine labels. The CORRECT test coarsens a fraction of labelled cells to their l1 lineage so internal nodes become real targets (mixed-granularity, the realistic reference-atlas setting). Result (fp32, seed-controlled, leaf-masked inference; .scratch/cytoanvi-benchmark/b8_hao_mixed_granularity.py): coarse_frac 0.0/0.4/0.7 -> HCE-minus-flat leaf macro-F1 +0.001/+0.011/+0.012 and cross-lineage error delta -0.0001/-0.0005/-0.0013. Monotonic in both: HCE routes a coarse cell's mass to its whole subtree (learns fine leaves from coarse labels; flat CE wastes them) AND improves lineage coherence. **B8 must be framed around partial/mixed-granularity annotation, not all-fine labels — HCE is provably a no-op on the latter.** Modest effect, single seed; multi-seed to confirm.
