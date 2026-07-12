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
**Body**: A first B8 run on the Hao CITE-seq ADT reference (real 8-internal-node l1->l2 tree, all cells labelled at fine l2 leaves) showed HCE ~0.03 WORSE than flat CE — the opposite of Nuñez's +0.086. Validation overturned this as an artifact: (1) `hierarchical_cross_entropy_loss` computes `-log((softmax @ R.T)[target] + eps)`; for a LEAF target the subtree mass `(softmax @ R.T)[leaf] = softmax[leaf]`, so HCE == flat CE EXACTLY (loss and gradients match to 1e-7 — unit-proven). The reachability internal rows are never indexed when every label is a leaf, so the hierarchy does literally nothing. (2) The spurious -0.03 was a bf16 numerical bug: `eps = finfo(bf16).eps ~ 7.8e-3` distorted the log (fixed by computing the loss in fp32, commit fcfff064) PLUS uncontrolled RNG/inference-method differences between the two separate trainings. A controlled check (leaf-only, same scvi.settings.seed, same predict()) gives delta -0.0012 ~ 0, confirming HCE==flat for fine labels. The CORRECT test coarsens a fraction of labelled cells to their l1 lineage so internal nodes become real targets (mixed-granularity, the realistic reference-atlas setting). Result (fp32, seed-controlled, leaf-masked inference; .scratch/cytoanvi-benchmark/b8_hao_mixed_multiseed.py, 3 seeds), HCE-minus-flat at coarse_frac 0.0/0.4/0.7: **leaf macro-F1 +0.0002±0.0008 / +0.0094±0.0085 / +0.0059±0.0110** (NOISY — seed 1 negative, ~1σ, NOT a robust fine-accuracy win); **cross-lineage error −0.0001±0.0000 / −0.0008±0.0004 / −0.0011±0.0004** (CONSISTENT — all 3 seeds negative, monotonic, ~2.75σ at cf=0.7). **Corrected verdict: HCE's value is LINEAGE COHERENCE, not fine accuracy** — it reliably makes fewer gross wrong-lineage errors as coarse annotation grows, while fine-leaf accuracy is a wash. Magnitude is small (~9% relative cross-lineage-error reduction) because the 225-ADT panel already puts lineage acc ~99% (ceiling). **For the paper: report HCE with a lineage-level / hierarchical metric, not leaf macro-F1; frame around partial/mixed-granularity annotation (HCE is provably a no-op on all-fine labels).**

### L-047 — [2026-07-08] MULTIVAE KL interception uses qz_m/qz_v tensors, not a qz Distribution
**Category**: gotcha
**Tags**: multivae, mrmultivi, kl-divergence, qz_m, qz_v, inference-output
**mitigation_type**: structural
**structural_mitigation_candidate**: test_mrmultivi_qu_encoder_gradients_flow
**Body**: TOTALVAE and MULTIVAE differ in how the KL on the main latent posterior is communicated to `loss()`. TOTALVAE stores `out["qz"]` as a `torch.distributions.Normal` object; `loss()` computes `kld(qz, Normal(0,1))`. MULTIVAE has NO `qz` key — it stores the mean and variance as separate tensors `out["qz_m"]` and `out["qz_v"]`, and `loss()` computes `kld(Normal(qz_m, sqrt(qz_v)), Normal(0,1))` inline. To inject a sample-conditioned u-distribution (e.g. `EncoderXU_MultiVI`) into MULTIVAE's existing KL computation, override `outputs["qz_m"] = qu.loc` and `outputs["qz_v"] = qu.scale**2` — do NOT attempt `outputs["qz"] = qu`, which silently has no effect. The TotalVI pattern (`out["qz"] = qu`) does NOT port to MULTIVAE.

### L-048 — [2026-07-08] Gradient-flow test pattern: manual forward-backward on untrained model
**Category**: design
**Tags**: testing, gradient-flow, qu-encoder, conditional-normalization, forward-backward
**mitigation_type**: structural
**structural_mitigation_candidate**: test_qu_encoder_gradients_flow
**Body**: To verify that a specific submodule (e.g. `qu.cond_norm1.gamma_embedding`) is connected to the loss, the correct test shape is a single manual forward-backward pass on an UNTRAINED model (no training loop, no Lightning). Pattern: `module.train() → inf_inputs = module._get_inference_input(tensors) → inf_out = module.inference(**inf_inputs) → gen_inputs = module._get_generative_input(tensors, inf_out) → gen_out = module.generative(**gen_inputs) → loss_out = module.loss(tensors, inf_out, gen_out) → loss_out.loss.backward()`. Then assert `param.grad is not None` and `param.grad.abs().max() > 0`. Benefits: (1) runs in <1s (no training overhead), (2) directly tests gradient connectivity (the ground truth), (3) works on an untrained model so no initialization assumptions. Contrast with checking parameter drift after training, which is indirect and can give false negatives if LR is too small or training too short.

### L-049 — [2026-07-08] Per-cell ELBO reconstruction when parent loss() already called mean()
**Category**: design
**Tags**: mrtotalvi, mrmultivi, scale_observations, elbo, loss
**mitigation_type**: convention
**structural_mitigation_candidate**: test_scale_observations
**Body**: When a parent `loss()` returns a scalar (already called `torch.mean()`), reweighting per-cell ELBO requires reconstructing the per-cell total from the individual component tensors stored in `loss_out.kl_local` and `loss_out.reconstruction_loss`. Apply the exact same scaling factors as the parent formula (e.g. `kl_weight * pro_recons_weight` for protein in TOTALVAE), divide by `prefactors = n_obs_per_sample[sample_index]`, then take `.mean()`. Never try to undo a `mean()` — always work from the per-cell tensors before aggregation.

### L-050 — [2026-07-08] value_counts().sort_index() ordering for n_obs_per_sample
**Category**: gotcha
**Tags**: mrtotalvi, mrmultivi, scale_observations, n_obs_per_sample, indexing
**mitigation_type**: convention
**structural_mitigation_candidate**: (none)
**Body**: Computing `n_obs_per_sample` from `adata.obs["_scvi_sample"]` must use `value_counts().sort_index()` not `value_counts()` (which sorts by count, descending). The buffer must be indexed by integer sample index matching the embedding table row, so the sort must be by index (0, 1, 2, …), not by cell count. Wrong sort → wrong prefactor applied to each donor's cells → silently biased ELBO.

### L-051 — [2026-07-08] MrMultiVI model_kwargs injection causes duplicate kwargs at load()
**Category**: bug
**Tags**: mrmultivi, model-save-load, model_kwargs, init-params, duplicate-kwargs
**mitigation_type**: structural
**structural_mitigation_candidate**: test_mrmultivi_save_load_roundtrip
**Body**: `MrMultiVI.__init__` both declares `n_latent_sample`, `z_u_prior_scale`, `learn_z_u_prior_scale`, `use_map`, `scale_observations` as named params AND injects them into `model_kwargs` before calling `super().__init__(mdata, **model_kwargs)`. `_get_init_params(locals())` captures the named params as non-kwargs AND the `model_kwargs` dict containing the same keys. At load time, `cls(adata, **non_kwargs, **expanded_model_kwargs)` gets duplicate keyword arguments → `TypeError`. Fix: remove all injections into `model_kwargs`; they are already named params that flow through to `_setup_hierarchy` directly after `super().__init__()`. `MrTotalVI` does NOT have this bug (it passes params directly). This bug makes any saved `MrMultiVI` model unloadable.

### L-052 — [2026-07-08] Non-persistent buffer silently None after save/load when scale_observations=True  **[CLOSED 2026-07-11]**
**Category**: bug
**Tags**: mrtotalvi, mrmultivi, scale_observations, n_obs_per_sample, save-load, persistent-buffer
**mitigation_type**: structural
**structural_mitigation_candidate**: test_n_obs_per_sample_in_state_dict
**Body**: `n_obs_per_sample` was registered with `register_buffer(..., persistent=False)` so it was excluded from `state_dict` and not restored on direct `load_state_dict`. The standard `Model.load()` path re-runs `__init__` and recomputes the buffer, so it was unaffected. The gap was direct `load_state_dict` usage. Fix applied 2026-07-11: flipped both modules (`mrtotalvi/_module.py:241`, `mrmultivi/_module.py:256`) to `persistent=True`. The `UserWarning` guard for `None` buffer in `loss()` is retained as belt-and-suspenders for any code that creates the module without calling `_setup_hierarchy`. New test `test_n_obs_per_sample_in_state_dict` verifies `state_dict()` contains the buffer and `load_state_dict` round-trip preserves it.

### L-054 — [2026-07-10] pz_scale degeneracy direction: σ→0 (min), not σ→∞ (max)
**Category**: bug
**Tags**: mrtotalvi, mrmultivi, pz_scale, kl_z, degeneracy, prior-collapse
**mitigation_type**: structural
**structural_mitigation_candidate**: test_learnable_prior_scale_clamp
**Body**: Early analysis of the `pz_scale` degeneracy (session publication-readiness review) incorrectly stated that the collapse direction was `pz_scale→+∞` (σ→∞). The correct direction is `pz_scale→−∞` (σ→0). Under `use_map=True`, `kl_z = -log Normal(0, σ).log_prob(eps)`. Expanding: `-log_prob = 0.5*log(2π) + log(σ) + eps²/(2σ²)`. As σ→∞, `log(σ)→+∞` so loss INCREASES — the optimizer has no incentive to push σ large. As σ→0 and eps→0 jointly: `log(σ)→−∞` dominates (goes to −∞ faster than eps²/σ² diverges) → `kl_z→−∞` → ELBO unbounded above. The fix is therefore `clamp(min=...)`, NOT `clamp(max=...)`. Any future degeneracy analysis on a `log(σ)` term must check both limits before concluding direction.

### L-055 — [2026-07-10] `torch.allclose` across CUDA/CPU device raises RuntimeError in save-load tests
**Category**: bug
**Tags**: testing, save-load, cuda, device-mismatch, torch.allclose
**mitigation_type**: structural
**structural_mitigation_candidate**: test_mrtotalvi_save_load_preserves_latent_hierarchy, test_mrmultivi_save_load_preserves_latent_hierarchy
**Body**: `test_mrtotalvi_save_load_preserves_latent_hierarchy` and the mrmultivi equivalent were asserting `torch.allclose(loaded.module.u_prior_means, model.module.u_prior_means)`. After save-load, buffers are restored to CPU even if the original model was on GPU (`cuda:0`). The `torch.allclose` call raised `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`. Fix: `.cpu()` both sides before comparison. Pattern to remember: after `Model.load(path)`, all buffers and parameters land on CPU regardless of training device. Always call `.cpu()` on both sides of any post-load comparison.

### L-053 — [2026-07-08] value_counts() omits absent samples → wrong-length n_obs_per_sample
**Category**: bug
**Tags**: mrtotalvi, mrmultivi, scale_observations, n_obs_per_sample, value_counts, reindex
**mitigation_type**: structural
**structural_mitigation_candidate**: test_scale_observations_missing_sample
**Body**: `adata.obs["_scvi_sample"].value_counts().sort_index()` omits any sample index that has zero cells in the current `adata`. If the model was registered with `n_sample` donors but the current adata is a subset, the resulting tensor has length < `n_sample`. Indexing with a missing sample index raises `IndexError` at training time. Fix: use `.reindex(range(n_sample), fill_value=0).sort_index()` (after `value_counts()`) to ensure the tensor always has length `n_sample`. Also need a downstream guard: a `fill_value=0` entry causes division by zero in the `per_cell / prefactors` computation — raise a `ValueError` for zero-count donors before training begins.

### L-056 — [2026-07-11] `_differential_expression`: sqrtm loop (~128 cells × n_cov^3) is fast enough to not need vmap
**Category**: performance
**Tags**: mrtotalvi, mrmultivi, differential_expression, scipy, sqrtm
**Body**: `scipy.linalg.sqrtm` is called per cell in each data batch. For batch_size=128 and n_cov≤10 this is ~128 calls each O(n_cov^3) ≈ negligible (microseconds per call). Total overhead across 1000 cells is ~5 ms. No vectorized sqrtm for PyTorch tensors exists without autograd; the CPU loop via scipy is correct and fast enough.

### L-057 — [2026-07-11] `torch.vmap(torch.linalg.pinv)` works inside `torch.inference_mode()` — no error
**Category**: gotcha
**Tags**: mrtotalvi, mrmultivi, torch.vmap, inference_mode
**Body**: `torch.vmap(torch.linalg.pinv)` called inside `with torch.inference_mode()` works without error on PyTorch ≥2.0. `inference_mode` disables autograd tracking but does not prevent functorch transforms. Verified on CUDA (L40S) + PyTorch 2.x.

### L-058 — [2026-07-11] `false_discovery_control` requires `scipy>=1.10`; already available in codex env
**Category**: infra
**Tags**: mrtotalvi, mrmultivi, differential_expression, scipy, BH
**Body**: `scipy.stats.false_discovery_control` was introduced in scipy 1.10.0 (2023). The codex conda environment has scipy ≥ 1.10 already. If porting to a different environment, verify scipy version before using this function; fallback is `statsmodels.stats.multitest.multipletests(method="fdr_bh")`.

---

### L-059 — [2026-07-11] `py_["scale"]` in DecoderTOTALVI is stochastic via `rsample()`
**Category**: gotcha
**Tags**: mrtotalvi, mrmultivi, differential_expression, protein_lfc, stochastic_background
**Body**: `DecoderTOTALVI.forward` computes `rate_back = exp(Normal(back_alpha, back_beta).rsample())`
(confirmed `_base_components.py:~942`). This propagates through `rate_fore = rate_back * fore_scale`
to `py_["scale"] = normalize((1 - protein_mixing) * rate_fore, p=1, dim=-1)`. Two separate
forward calls (x_1, x_0 in the LFC contrast) draw **independent** background noise — the
difference does NOT simplify to a biological signal. The `rsample()` is inside the inherited
class and cannot be disabled via a kwarg. Fix: reconstruct `py_["scale"]` from `exp(back_alpha)`
on the contrast path in `compute_h_from_x_eps` (D-021). Verified from source; no MRVI
reference (MRVI is RNA-only). Test: null-covariate protein `lfc_std ≈ 0` confirms determinism.

---

### L-060 — [2026-07-11] MrTotalVI/MrMultiVI `qz` returns 3-tuple, not MRVI's 2-tuple
**Category**: gotcha
**Tags**: mrtotalvi, mrmultivi, differential_expression, qz, inference
**Body**: MRVI's `EncoderUZ.forward` returns `(z_base, eps)` 2-tuple. Both mr* derivatives
return `(z_base, eps, eps_dist)` 3-tuple — the extra `eps_dist` carries the Normal
distribution for analytic KL when `use_map=False` (D-015). All callers must unpack all
three values. Note: `mrmultivi/_module.py:281` has a 2-tuple unpack **inside a docstring
code-block** (not real code); the actual line `:330` is the 3-tuple assignment. Callers
who only need `z_base` and `eps` can use `z_base, eps, _ = self.qz(...)`.

---

### L-061 — [2026-07-11] `torch.vmap` breaks on `nn.BatchNorm` — use explicit loop for MrTotalVI LFC
**Category**: gotcha
**Tags**: mrtotalvi, vmap, batch_norm, differential_expression, lfc
**mitigation_type**: structural
**structural_mitigation_candidate**: `use_vmap=False` default in `differential_expression`
**Body**: `torch.vmap` (functorch) cannot be applied over models with `nn.BatchNorm` layers
because BatchNorm's running statistics are updated as a side effect and the layer tracks
a scalar mean/var over the *batch* dimension, which vmap maps to the *function* dimension.
Calling `vmap` over a BatchNorm-containing module raises a runtime error (or silently corrupts
statistics depending on PyTorch version). MrTotalVI's TotalVI decoder defaults
`use_batch_norm="both"` (encoder + decoder; see `_totalvae.py:146`). Mitigation: default
`use_vmap=False` in both models; the existing per-donor explicit loop is correct and fast
enough at the scales tested (D-023). MrMultiVI uses LayerNorm in MULTIVAE decoder, so
`use_vmap=True` is architecturally safe there but remains off by default for consistency.

---

### L-062 — [2026-07-11] donor_key==sample_key creates duplicate column → tuple MultiIndex in design matrix
**Category**: bug
**Tags**: mrtotalvi, mrmultivi, differential_expression, design_matrix, donor_key
**mitigation_type**: code_fix
**Body**: In `_stats.py._differential_expression`, the design matrix is built by selecting
`adata.obs[[model.sample_key] + cov_cols]`. When `donor_key == model.sample_key`, the same
column appears twice in the select list, producing a DataFrame with a tuple MultiIndex on
columns instead of a flat string Index. Downstream `.set_index()` and `.loc[]` calls then
raise `KeyError` or silently select wrong columns. Fix: filter `cov_cols_for_select = [c
for c in cov_cols if c != model.sample_key]` before the select. Same fix needed in
`_construct_design_matrix` when resolving the donor covariate: guard with
`if donor_key == sample_info.index.name: cov = sample_info.index.to_series()`.

### L-063 — [2026-07-11] VampPrior + encode_covariates=True: must pass batch_idx=zeros to _append_covariates
**Category**: bug
**Tags**: vamprior, encode-covariates, mrtotalvi, batch-index
**mitigation_type**: structural
**structural_mitigation_candidate**: test_vamprior_trains_finite_elbo (guards forward pass through qu)
**Body**: `MrTotalVI` defaults `encode_covariates=True`. `EncoderXU_TotalVI._append_covariates` raises `ValueError: batch_index is required when encode_covariates=True.` if `batch_index=None` is passed. `_vamp_component_dist` initially called `self.qu(x_p, y_p, sample_idx)` without `batch_index`, triggering this error during training. Fix: pass `batch_index=torch.zeros(K, 1, ...)` when `self.encode_covariates` is True. MrMultiVI is unaffected (defaults `encode_covariates=False`).

### L-065 — [2026-07-11] MrTotalVI/MrMultiVI `encode_covariates` default was True in `MrTotalVAE` (bug)
**Category**: bug
**Tags**: encode-covariates, mrtotalvi, checkpoint-load, model-load, size-mismatch
**mitigation_type**: structural
**structural_mitigation_candidate**: test_mrtotalvi_load_pretrained_checkpoint
**Body**: `MrTotalVAE.__init__` had `encode_covariates = kwargs.get("encode_covariates", True)` as its default. The reference MRVI design and `EncoderXU_TotalVI.__init__` both default to `False` (u-encoder should be batch-uninformed). The pre-trained checkpoint was saved with `False` — `qu.fc1.weight = [128, 10130]`. Loading with the buggy `True` default caused `n_input = 10130 + n_batch(6) = 10136` in the current model, producing a `RuntimeError: size mismatch for qu.fc1.weight` on load. Fix: change the default to `False` in `_module.py:110`. Same fix needed in MrMultiVI; check was also `False` there, confirming consistency. Any checkpoint trained with `encode_covariates=True` would need an explicit kwarg on load.

### L-066 — [2026-07-11] SLURM `gpu` partition may allocate TITAN Xp (sm_61), incompatible with PyTorch sm_75+
**Category**: infra
**Tags**: slurm, gpu, cuda, pytorch, titan-xp
**mitigation_type**: convention
**Body**: The `gpu` partition on the cluster includes TITAN Xp nodes (sm_61, Pascal). The scvi-test conda env's PyTorch build requires sm_75+. Allocating from the unspecified `gpu` partition randomly assigned a TITAN Xp → `cudaErrorNoKernelImageForDevice` at first CUDA call. Fix: use `--partition=gpu-long --gres=gpu:L40S:1` for any MrTotalVI/MrMultiVI GPU inference. L40S (sm_89, Ada) is available on `res-hpc-gpu[11-12]`. RTX6000/RTX8000 (sm_75) are also compatible but on the `short` partition.

### L-064 — [2026-07-11] MrMultiVI protein_in_encoder: VampPrior pseudoinputs must split at n_latent
**Category**: design
**Tags**: protein-in-encoder, vamprior, mrmultivi, pseudoinputs
**Body**: When both `protein_in_encoder=True` and `u_prior="vamp"` are active, `u_vamp_pseudo` has shape `(K, n_latent + n_input_proteins)`. In `_vamp_component_dist`, the tensor must be split: `u0_pseudo = pseudo[:, :n_latent]` (unconstrained, continuous latent space) and `y_pseudo = F.softplus(pseudo[:, n_latent:])` (positive, fed as log1p-transformed pseudo-counts). Using Softplus on the whole pseudo-tensor would incorrectly constrain u0 to be positive. The split index is `self.n_latent`, not `module.qu.n_input_proteins`.

### L-067 — [2026-07-11] Architectural prior for MrTotalVI/MrMultiVI protein resolution is only half-predictive
**Category**: data
**Tags**: mrmultivi, mrtotalvi, cell-state, protein, architectural-prior, benchmark
**Body**: Expected: MrTotalVI_u (raw protein in u-encoder) better on protein-defined states; MrMultiVI_u (protein second-hand via MULTIVAE) better on RNA-dominant states. Observed (schisto CITE-seq, 49k cells, kNN-F1): architectural prior correctly predicts Plasma (−0.071) and DC2 (−0.038) to be better in MrTotalVI_u. But CD16- NK cells (protein-defined by CD16/CD56) are *better* in MrMultiVI_u (+0.105). MAIT cells (primarily RNA/TCR-defined) are better in MrMultiVI_u (+0.044) — this one is consistent. The split is ~50/50 between architecturally predicted vs surprising. Likely explanation: MULTIVAE joint encoding is more effective than TotalVI's additive decomposition for some intermediate populations (like CD16- NK), even though those populations are protein-defined.

### L-070 — [2026-07-11] `torch.arange(n_batch)` defaults to CPU — passes CPU index into vmap where embedding weight is on CUDA
**Category**: bug
**Tags**: mrvi, vmap, device, batch_index, torch-arange, embedding
**mitigation_type**: structural
**structural_mitigation_candidate**: always pass device= when creating index tensors for vmap
**Body**: In `mrvi_torch/_model.py` line 1399, `batch_index_ = torch.arange(self.summary_stats.n_batch)[:, None]` creates a CPU tensor. This is passed as the second argument to the triple-nested vmap (`in_dims=0` in the outermost vmap). Inside vmap, each vmapped slice is still on CPU. When it flows through `compute_h_from_x_eps → generative → DecoderZXAttention.forward → batch_embedding(batch_covariate)`, the embedding weight is on CUDA but the index is on CPU → `RuntimeError: Expected all tensors to be on the same device`. All other tensors in the LFC block either explicitly use `device=eps_mean_.device` or are moved via `.to(self.device)` OUTSIDE the vmap — but `batch_index_` enters the vmap without a device move. Calling `.to(device)` INSIDE vmap is not vmap-traceable and causes a different crash. The fix is to add `device=self.device` to the `torch.arange` call before the vmap entry. This was a 1-character source change to `mrvi_torch/_model.py`. General rule: any tensor created with `torch.arange`, `torch.ones`, `torch.zeros`, or `torch.full` that will enter a vmap must have `device=` explicitly set.

### L-069 — [2026-07-11] `@auto_move_data` is vmap-incompatible in eval mode — requires instance-level monkey-patch
**Category**: gotcha
**Tags**: mrvi, vmap, auto_move_data, torch-func, monkey-patch, attention
**mitigation_type**: structural
**structural_mitigation_candidate**: make_mrvi_vmap_safe
**Body**: MRVI's `store_lfc=True` path uses triple-nested `torch.func.vmap`. In eval mode, `@auto_move_data` wraps every submodule's `forward`/`inference`/`generative` and calls `.to(device, non_blocking=True)` on every input tensor. Under `torch.vmap`, vmapped tensors are BatchedTensors that do not support `.to()` with non_blocking; the tensor becomes `None` downstream, crashing at `F.scaled_dot_product_attention` with "Expected proper Tensor but got None for argument #0 'self'". The wrapper is a no-op in intent (all tensors are already on the correct device post-load), but fatal under vmap. Fix: after `MRVI.load(...)`, iterate over `model.module.modules()`, find any class-level attribute that has `__wrapped__` set (confirming `@auto_move_data` / `@wraps` was used), and install an instance-level bound method from the original function via `setattr(submod, attr_name, types.MethodType(cls_attr.__wrapped__, submod))`. This overrides class-attribute lookup for the specific model instance without modifying any source files. Affected methods: `ResnetBlock.forward`, `MLP.forward`, `NormalDistOutputNN.forward`, `ConditionalNormalization.forward`, `AttentionBlock.forward`, `DecoderZXAttention.forward`, `EncoderUZ.forward`, `EncoderXU.forward`, `TorchMRVAE.inference`, `TorchMRVAE.generative` (10 methods).

### L-071 — [2026-07-11] MultiVI VampPrior pseudo-inputs are in continuous latent space — data-driven init not feasible at construction time
**Category**: design
**Tags**: mrmultivi, vamprior, pseudoinputs, latent-space, init
**mitigation_type**: ambient-awareness
**Body**: `EncoderXU_MultiVI` takes MULTIVAE's mixed latent `u0` (continuous, can be negative) as input — not raw counts. So MrMultiVI VampPrior pseudo-inputs (shape `(K, n_latent)`) live in `u0` space. Unlike MrTotalVI (raw count space, Softplus-constrained), you can't compute data centroids and apply softplus-inverse. The only way to seed them from data at init time would be to run a forward pass through the random (untrained) MULTIVAE encoder, which produces meaningless latents. `init_prior_from_data=True` is accepted syntactically by MrMultiVI but silently deferred. If data-driven MoG/VampPrior init is wanted for MultiVI, it must be done post-training via `model.module.u_prior_means.data = compute_latent_centroids(model)`.

### L-072 — [2026-07-12] `de["pvalue"]` in MrTotalVI DE is per-cell, not per-gene
**Category**: gotcha
**Tags**: mrtotalvi, de, pvalue, calibration, permutation-null, per-cell
**mitigation_type**: ambient-awareness
**Body**: `_stats.py` Wald chi² computes `ts = (betas_norm**2).sum(-1).mean(0)` where `betas_norm` has shape `(mc, cells, n_cov, latent_dim)`. After squaring + summing over latent + averaging over MC, `ts` has shape `(cells, n_cov)`. So `de["pvalue"]` = chi² p-value per CELL, not per gene. The LFC path (`store_lfc=True`) computes a separate gene-level contrast via `compute_h_from_x_eps`. The permutation null (`run_permutation_null.py`) tests whether per-cell p-values are enriched below 0.05 — with n=10 donors, each permutation creates a coherent alternative biology (not a true null), producing frac_below_0.05 in [0, 1] with sd≈0.47. Real data sits at 40th–55th percentile of null → test is uninformative for calibration at n=10. To calibrate gene-level DE, would need per-gene chi² statistics (not currently exposed), or a different null strategy (shuffle cells within timepoints, not donors between timepoints).

### L-073 — [2026-07-12] `model.eval()` not valid on scvi Model objects — use `model.module.eval()`
**Category**: bug
**Tags**: mrtotalvi, mrmultivi, eval-mode, pytorch, scvi-model
**mitigation_type**: fix
**Body**: scvi Model classes (MrTotalVI, MrTotalVI, etc.) do not inherit from `torch.nn.Module` and have no `.eval()` method. Calling `model.eval()` raises `AttributeError: 'MrTotalVI' object has no attribute 'eval'`. The underlying PyTorch module is at `model.module` — so `model.module.eval()` is correct. In practice, scvi Models are already in eval mode post-load for inference; the `.eval()` call is typically a defensive no-op. But in scripts that explicitly require eval mode, use `model.module.eval()`.

### L-074 — [2026-07-12] `_differential_expression` with `donor_key` returns 3D LFC DataArray
**Category**: gotcha
**Tags**: mrtotalvi, mrmultivi, de, lfc, xarray, donor_key
**mitigation_type**: fix
**Body**: When `donor_key` is set, `_differential_expression` returns `lfc` as an xarray DataArray with dims `[cell_name, covariate, feature]` and shape `(n_cells, n_fixed, n_features)`. Scripts that try to slice `de_result["lfc"].values[:n_genes]` are slicing the cell axis, not the feature axis — yielding garbage. Correct extraction: (1) inspect `de_result["lfc"].coords["covariate"].values` to find the relevant covariate label (e.g., `"timepoint_W22"` from `pd.get_dummies`); (2) select with `.sel(covariate="timepoint_W22")`; (3) average over cells with `.mean("cell_name")`; result is `(n_features,)`. This applies even when `n_fixed=1`. The covariate name comes from `_construct_design_matrix`: categorical `sample_cov_keys` → `pd.get_dummies(drop_first=True)` → columns named `"{key}_{category}"`.

### L-075 — [2026-07-12] `donor_key='sex'` has zero effect in a balanced paired design (orthogonal to timepoint)
**Category**: gotcha
**Tags**: mrtotalvi, de, donor_key, sex-confound, wls, instability
**mitigation_type**: ambient-awareness
**Body**: `donor_key='sex'` adds a sex dummy to the WLS design matrix for eps regression. In the schisto CITE-seq dataset (10 donors, 4M+6F, **all with both W00 and W22 timepoints**), sex is perfectly balanced across timepoints — sex is orthogonal to timepoint in the design matrix. Consequence: including sex in WLS does not change the timepoint beta at all; the sex-adjusted and naive multi-seed means are algebraically identical (rho=1.000). The seed-0 Y-chr collapse (F-027, job 25211187) was a stochastic artifact of one model seed's eps encoding; it did NOT replicate in seeds 1+2 (DDX3Y: s0=+0.113, s1=+0.471, s2=+0.740). **Correction**: prior documentation erroneously stated n=5 donors; the dataset has n=10 donors (4M: RC458, RC660, RC681, RC966; 6F: RC122, RC232, RC307, RC365, RC393, RC441). **Rule**: when timepoint and sex are orthogonal in the study design (balanced paired design), WLS sex adjustment cannot remove sex confounds in `eps` — the betas are independent by construction. The Y-chr confound should be reported as a limitation; it arises because the model's eps encodes cross-donor variation that includes sex-linked biology, which paired pseudobulk (F-029) correctly cancels via within-donor differencing.

### L-076 — [2026-07-12] Model eps-space timepoint contrast can reverse IFN sign vs pseudobulk ground truth — root cause: design matrix misspecification
**Category**: design
**Tags**: mrtotalvi, de, eps, ifn, calibration, pseudobulk, artifact, sample_key
**mitigation_type**: ambient-awareness
**Body**: In MrTotalVI DE (`sample_cov_keys=["timepoint"]`, model trained with `sample_key="donor"`), the WLS produces IFN genes as negative at W22, while donor pseudobulk (ground truth) shows IFN consistently UP in every cell type. The root cause is a **design matrix misspecification**, not just eps signal partitioning:

**Primary mechanism** (design matrix is wrong): `eps = EncoderUZ.forward(u, donor_id)` — the sample-specific residual depends only on the donor identity (embedding table has one entry per donor), NOT per timepoint. When `_differential_expression` builds `sample_info`, it calls `adata.obs[["donor","timepoint"]].drop_duplicates(subset="donor")` over the full 4-timepoint adata. This assigns each donor its first-seen timepoint in the data file, not its actual timepoint at DE time. For schisto CITE-seq the resulting "W22" flag = {RC441, RC232, RC365} vs others — a donor-group comparison with no temporal meaning. The WLS "timepoint_W22 beta" estimates the eps deviation of those 3 donors from the rest, not a within-donor W22 vs W00 effect.

**Secondary mechanism** (eps has no within-donor temporal information): even if the design matrix were constructed correctly, eps = attention(u, embed(donor_id)) produces the SAME residual for W00 and W22 cells from the same donor. There is no mechanism by which eps encodes within-donor temporal change when sample_key=donor.

**Correct setup for temporal DE**: retrain with `sample_key="donor_timepoint"` (20 samples) so each (donor, timepoint) pair has its own embedding, and eps captures temporal variation. Or use calibrated pseudobulk (F-029) as ground truth.

**Rule**: MrVI-style eps-space DE is only valid when `model.sample_key` granularity matches the condition variable in `sample_cov_keys`. When sample_key=donor and sample_cov_keys=["timepoint"], the analysis is invalid for within-donor temporal DE (see L-077).

### L-077 — [2026-07-12] MrVI-style DE: sample_key granularity must match condition granularity
**Category**: design
**Tags**: mrtotalvi, mrmultivi, de, sample_key, design-matrix, temporal, paired
**mitigation_type**: structural
**structural_mitigation_candidate**: IMPLEMENTED — `_stats.py` `_differential_expression` now raises `ValueError` when any sample maps to >1 value of any sample_cov_key. Tests: `test_mrtotalvi_de_raises_on_nonunique_cov`, `test_mrmultivi_de_raises_on_nonunique_cov`.
**Body**: The `_differential_expression` function builds its design matrix by `drop_duplicates(subset=model.sample_key)` over `adata.obs`. This assumes each sample (= one row in model's embedding table) maps to exactly ONE value of each `sample_cov_key`. If a donor spans multiple timepoints (paired design), the drop_duplicates silently picks the first-seen timepoint per donor — a data-order artifact, not a biological property.

**When it breaks**: `sample_key="donor"` + `sample_cov_keys=["timepoint"]` with paired (longitudinal) data. Each donor appears at multiple timepoints; drop_duplicates picks arbitrarily.

**When it works correctly**: `sample_key="donor_timepoint"` (each sample = one unique condition). Then drop_duplicates is a no-op (each sample has exactly one condition level), and the design matrix is correct.

**Structural mitigation (IMPLEMENTED 2026-07-12)**: `_stats.py` `_differential_expression` now validates that each sample maps to exactly one value of every `sample_cov_key` (and `donor_key` if provided). Raises `ValueError("... not constant within sample_key ...")` at call time if the invariant is violated. Tests added to both `test_mrtotalvi.py` and `test_mrmultivi.py`.

### L-079 — [2026-07-12] eps-space DE cannot detect treatment-induced cell-state changes — u absorbs the signal
**Category**: design
**Tags**: mrtotalvi, mrmultivi, de, eps, cell-state, treatment-effect, ifn, artifact, fundamental-limitation
**mitigation_type**: ambient-awareness
**Body**: Even after fixing the design matrix with DTP retraining (`sample_key="donor_timepoint"`), MrTotalVI/MrMultiVI eps-space DE still reports IFN genes as DOWN at W22 vs PyDESeq2 gold standard IFN UP (Spearman rho=−0.240, all 12 IFN genes wrong direction — see F-033). The reason is architectural: `u = EncoderXU(x)` is a function of the cell's gene expression `x`. IFN activation is a cell state that manifests as changes in `x` (higher STAT1, IFIT3, GBP1, etc.). The u encoder captures this cell state because it is trained on cell-level expression. So u at W22 is already "high-IFN" for IFN-activated cells. Then `eps = z − u` is the residual after removing the cell-intrinsic state — eps contains very little IFN signal because IFN information is already encoded in u. When the WLS regresses eps on the design matrix, the IFN direction is driven by noise/second-order effects, not by the primary biological signal.

**Contrast with donor effects**: eps WAS designed to capture donor-level variation (genetic background, technical effects) that affects all cells from a donor identically regardless of their individual cell state. For such effects, u (cell-state) is similar between donors and the donor-specific deviation lands in eps — making eps-space DE appropriate. For treatment effects that change cell states (e.g., IFN activation from infection treatment), the signal routes through u, not eps.

**Quantitative confirmation**: old donor model rho=−0.095 → DTP model rho=−0.240. Fixing the design matrix made things WORSE in concordance terms because DTP correctly identifies IFN as the primary temporal signal (it appears cleanly in the DTP top-DOWN genes) but still assigns it the wrong direction. The signal is real but inverted.

**Practical rule**: eps-space DE in MrVI-family models is NOT appropriate for detecting treatment-induced cell-state changes. Use calibrated pseudobulk (PyDESeq2, F-031) for those. eps-space DE is appropriate for identifying donor/batch covariate effects that persist across cell types (e.g., a covariate that shifts eps direction independently of cell state — a true sample-level additive effect in latent space). The u vs eps partition is fundamentally a "cell state" vs "sample effect" partition.

---

### L-080 — [2026-07-12] DTP MrTotalVI DA seed variance is catastrophic (std ≈ 8× mean)
**Category**: data
**Tags**: mrtotalvi, da, dtp, seed-instability, mog-prior, temporal
**mitigation_type**: ambient-awareness
**Body**: DTP MrTotalVI DA per-seed W22 enrichment: s0=+2.74, s1=−9.04, s2=+9.67; mean=+1.12, std=9.46. The seed with negative enrichment (s1=−9.04) reports cells as more likely at W00 than W22, the opposite of both the positive seeds and biological expectation. With 20 samples (10 donors × 2 timepoints), the sample-level MoG or VampPrior u-prior is harder to learn reliably than with the original 10-donor setup — the prior collapses differently per seed. The DA result is meaningless at 3 seeds. Contrast: DA with original `sample_key="donor"` on 10 samples was blocked (wrong granularity, L-078), so no baseline comparison is possible. **Rule**: DA requires sufficient samples (>> 20) and stable training across seeds; 3 seeds at 20 samples is insufficient for MrVI-style DA. The `differential_abundance` result is inherently less stable than `differential_expression` because it is computed entirely in latent u-space, making it more sensitive to the prior.

---

### L-068 — [2026-07-11] Subsample/h5ad version mismatch causes UMAP overlay misalignment
**Category**: infra
**Tags**: umap, subsample, h5ad, mismatch, benchmark
**Body**: Pre-computed UMAP coords (`umap_coords_MrMultiVI_u.tsv.gz`) generated Jul 11 11:35; the h5ad was subsequently modified (Jul 11 19:43). The subsample (SEED=0, same MAX_CELLS=50k, same `celltype_not_null` filter) produces a different set of obs_names when reindex is done on the new h5ad, yielding only 19861/49752 aligned cells for the Jaccard overlay. Analysis results (F1, purity, ARI, NMI, Leiden cross-recoverability) are fully valid on the current subsample. Fix: always regenerate UMAP coords in the same script that generates the analysis subsample, or save subsample obs_names as a sidecar file.

### L-081 — [2026-07-12] MrMultiVI.load() requires MuData, not AnnData — `_validate_var_names` calls `.get()` on numpy array
**Category**: bug
**Tags**: mrmultivi, load, save_load, mudata, anndata, validate_var_names
**mitigation_type**: code_fix
**Body**: `Model.load(str(model_dir), adata=adata)` calls `_validate_var_names` in `_save_load.py:231`. For MrMultiVI, `load_var_names` is the `var_names` attribute of the input data, which is a `numpy.ndarray` when an `AnnData` is passed. But `_validate_var_names` assumes a dict-like structure and calls `.get(mod_key)` on it — raising `AttributeError: 'numpy.ndarray' object has no attribute 'get'`. The fix is to pass a `MuData` (with `{"rna": rna_adata, "protein": protein_adata}` modalities) instead of a flat `AnnData`. The `make_mrmultivi_mdata(adata)` helper in `run_multiseed_de.py` implements this correctly; `run_dtp_da.py` was missing the conversion. Fix applied: check `if model_cls is MrMultiVI` in `run_dtp_da.py:run_da_seed` and call `make_mrmultivi_mdata(adata)` before the `.load()` call. General rule: any script loading MrMultiVI must pass MuData, not AnnData.

### L-082 — [2026-07-12] mr* `lfc_std` anti-conservatism: Jensen bias from qu.mean — FIXED with CRN
**Category**: design
**Tags**: mrtotalvi, mrmultivi, lfc, lfc_std, pde, qu_mean, crn, estimand, anti-conservative
**mitigation_type**: structural
**structural_mitigation_candidate**: `test_mrtotalvi_crn_identity`, `test_mrmultivi_crn_identity` (CRN identity: same u+eps → LFC == 0)
**Body**: `compute_h_from_x_eps` originally used `qu.mean` as the latent anchor, shifting the estimand from the posterior-marginalized LFC to a point-estimate LFC at u_mean. Two consequences: (1) Jensen bias — `dec(E[u]) ≠ E[dec(u)]` for nonlinear decoders; (2) Anti-conservative intervals — x_0 was computed once outside the MC loop, so `lfc_std`/`pde` captured only regression uncertainty in β_k.

**Fix implemented (2026-07-12)**: Common Random Numbers (CRN). Both modules now accept `u_anchor: torch.Tensor | None`. `_stats.py` stores `u_samples` per MC draw and passes `u_anchor=u_samples[mc_idx]` to both x_0 and x_1 decode calls. x_0 is now computed inside the MC loop. This gives the correct posterior-marginalized LFC estimator with proper variance. The `u_anchor=None` legacy path (uses qu.mean) is kept for standalone callers (e.g. test_d021_*) — see D-034.

**Residual awareness**: with default `use_map=True` in `EncoderUZ`, `qz(u_anchor, cf)` is deterministic given u_anchor — so the only randomness in the LFC MC loop is from `qu.rsample()`. `library_gene`/`libsize_expr` scale the rate but not `px_scale` (softmax) or `py_scale_det` (deterministic D-021), so they do not contaminate the contrast.

### L-078 — [2026-07-12] sample_key='donor' makes DA invalid in repeated-measures (paired) designs
**Category**: design
**Tags**: mrtotalvi, mrmultivi, differential_abundance, sample_key, design-matrix, temporal, paired, da
**mitigation_type**: structural
**structural_mitigation_candidate**: validate in _stats.differential_abundance that each sample_key maps to exactly one value of each sample_cov_key
**Body**: `differential_abundance(sample_cov_keys=["timepoint"])` aggregates per-sample log-probs in `u` space and regresses them on the covariate. This requires that each sample (`model.sample_key` level) maps to exactly one covariate value. With `sample_key="donor"` in a longitudinal design (each donor at 4 timepoints), `sample_info = adata.obs[["donor","timepoint"]].drop_duplicates(subset="donor")` silently picks the first-seen timepoint per donor — a data-order artifact. The covariate assignment is wrong, so DA results are meaningless.

**Fix**: retrain with `sample_key="donor_timepoint"` (38 samples, one per donor×timepoint). Each sample then has an unambiguous timepoint. MRVI (trained with `sample_key="final_label"` ≈ donor×timepoint) was already using the correct granularity; MrTotalVI/MrMultiVI were not.

**General rule**: for any MrVI-style model, the granularity of `sample_key` must be ≥ the granularity of every `sample_cov_key` used in `differential_expression` or `differential_abundance`. In a repeated-measures design with timepoint as condition, `sample_key` must be at donor×timepoint level.

**Structural fix pending**: add a pre-flight check in `_stats.differential_abundance` (and `_differential_expression`): for each `sample_cov_key`, assert that `adata.obs.groupby(model.sample_key)[key].nunique().max() == 1`.

### L-084 — [2026-07-12] model.history key is 'elbo_train' not 'train_loss_epoch' in MrTotalVI
**Category**: gotcha
**Tags**: mrtotalvi, training-history, pytorch-lightning, test-patterns
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: use candidate-scan pattern: `next((k for k in candidates if k in history), None)`
**Body**: `model.history` in scvi-tools (PyTorch Lightning back-end) exposes `'elbo_train'`, `'train_loss'`, `'reconstruction_loss_train'`, etc. — not the generic `'train_loss_epoch'` string that Trainer callbacks sometimes use. Hardcoding `history["train_loss_epoch"]` raises `KeyError`. Pattern: scan a candidate list in priority order (`["elbo_train", "train_loss", "train_loss_epoch"]`) and assert one is found, so the test remains valid across version changes.

### L-086 — [2026-07-12] Single-seed scIB bio sub-score estimate (+0.009) was noise; 3-seed analysis shows bio conservation flat (Δbio = −0.006±0.010)
**Category**: analysis-process
**Tags**: mrmultivi, scib, bio-conservation, multi-seed, adversarial-check
**mitigation_type**: structural
**structural_mitigation_candidate**: never report sub-score breakdown from a single seed; always replicate with ≥3 seeds before claiming directional component gains
**Body**: F-015 (single-seed scIB breakdown) showed MrMultiVI_u bio sub-score +0.009 over MultiVI. This was reported in the manifest as "bio conservation also marginally better." The 3-seed adversarial audit (F-038, skeptic agent session 50) showed that across all 3 seeds, Δbio = −0.006±0.010 — statistically indistinguishable from zero, with mean actually negative. The single-seed +0.009 was pure noise. The manifest claim and the `mr_multimodal.md` "bio Δ≈+0.009 marginal" wording were both corrected. Lesson: a single-seed sub-score estimate can misrepresent direction by ±0.015 — never report component-level breakdown without multi-seed confirmation. Also caught: the MultiVI baseline (0.593) is a single-seed point with no within-model error bar; published comparison should disclose asymmetric error bars.

### L-085 — [2026-07-12] User-facing docs did not disclose empirical eps-space DE anti-concordance — a publication-blocking gap
**Category**: process
**Tags**: mrtotalvi, mrmultivi, de, lfc, publication, documentation
**mitigation_type**: structural
**structural_mitigation_candidate**: always add empirical validation note to user-facing docs when a new API is shipped; distinguish "API works" (code tests pass) from "results are biologically valid" (concordance with gold standard)
**Body**: When `store_lfc=True` was implemented and the ADRs/mr_multimodal.md were updated, the docs described the feature as working and resolved. The empirical validation (F-036, F-037) was completed afterward and showed anti-concordance with PyDESeq2 gold standard across all 12 cell types. But `mr_multimodal.md` was never updated to reflect this. The v1 Limitations section had: BatchNorm/vmap note, ATAC rejection, ArchesMixin note, objective note — no mention that LFC results are anti-concordant with pseudobulk. Similarly, F-023's status was left as "preliminary (sex-adjusted pending)" after F-028 closed the question; F-021 pointed to retracted findings (F-020/F-017) as the evidence base; F-026 called the convergent IFN artifact "the most robust shared signal." These stale entries compound: a reader tracing the evidence chain encounters misleading status markers and conclusions. Fix: (1) add empirical eps-space DE disclaimer block to `mr_multimodal.md` v1 Limitations; (2) update stale FINDINGS_REGISTRY entries with retraction notes and status corrections; (3) add DA stability caveat for MrTotalVI; (4) add Publication Confidence Review section to ANALYSIS_MANIFEST (session 49).

### L-087 — [2026-07-12] MC marginalization in DA does not reduce cross-seed variance
**Category**: design
**Tags**: mrtotalvi, mrmultivi, differential_abundance, da-stability, vamp-prior, seed-variance
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: (none required — correct design choice made)
**Body**: Early framing suggested `n_mc_samples > 1` in `differential_abundance` would "convert cross-training-seed variance into within-model posterior uncertainty." This is incorrect. Drawing K samples from `q(u|x)` within a single trained model does NOT reduce cross-seed variance: different training seeds produce different `qu.loc` values, and all K draws from a given seed cluster around that seed's `qu.loc`. The sampling does NOT sample across the training-seed distribution. What MC marginalization actually does: corrects the Jensen gap between `log p(E_q[u]|ap_s)` and `E_q[log p(u|ap_s)]`. More importantly, switching from `give_mean=True` (deterministic given weights) to `give_mean=False` introduces inference-RNG non-determinism that does not currently exist — a regression against the stated stability goal. The correct fix is `freeze_prior_after_init` (D-038).

### L-083 — [2026-07-12] torch.Tensor.var() defaults correction=1 → NaN when n=1
**Category**: correctness
**Tags**: mrtotalvi, mrmultivi, lfc_std, mc_samples, pytorch, variance
**mitigation_type**: structural
**structural_mitigation_candidate**: `lfc_mc_cov.var(1, correction=0)` in `_stats.py`
**Body**: `torch.Tensor.var()` defaults to `correction=1` (Bessel's correction, divides by n−1). When `mc_samples=1`, the tensor along the MC axis has length 1, so `var(1)` computes `sum_sq / (1-1) = sum_sq / 0 = NaN`. This silently poisons `lfc_std` for any call with `mc_samples=1`. Fix: use `var(1, correction=0)` (population variance, divides by n). Population variance is the correct choice for an MC estimator where we care about the spread of the draws, not an unbiased estimate of a population parameter.

**Secondary**: `qu.loc` is a PyTorch `Normal` attribute that aliases `loc` (the mean parameter), but the attribute name is misleading when the intent is "use the posterior mean." Prefer `qu.mean` — semantically unambiguous (`.mean` is the distribution mean property, not a raw parameter alias). `qu.loc` and `qu.mean` are numerically identical for `Normal`.

### L-088 — [2026-07-12] B5 multiseed aggregate had wrong CytoVI AUROC (0.775 vs correct 0.855) due to stale per-seed files
**Category**: data-integrity
**Tags**: cytoanvi, b5, multiseed, aggregate, auroc, novelty-detection
**mitigation_type**: structural
**structural_mitigation_candidate**: always check per-seed file modification dates before trusting an existing aggregate; prefer regenerating from source files
**Body**: The existing `roider_full_b5_sweep_multiseed.json` reported `cytovi_mean_auroc_mean=0.775`. Investigation revealed this aggregate was built from `_11type_orig.json` files (July 6, covering only 11 Leiden types) rather than the July 7 reruns (jobs 25149032/33/34) which expanded to all 47 types AND added `cytoanvi_knn_baseline`. The per-seed files `s{0,1,2}.json` matched July 7 timestamps. Regenerating the aggregate from the correct per-seed files gave `cytovi_mean_auroc_mean=0.855±0.001` and `cytoanvi_knn_mean_auroc_mean=0.906±0.003`. The stale 0.775 figure had been copied into F-013 and potentially into earlier session notes.

**Secondary (path resolution)**: `aggregate_b5_multiseed.py` uses `HERE = Path(__file__).parent` for path resolution. Calling it with relative paths from the repo root causes doubling (`.scratch/cytoanvi-benchmark/.scratch/cytoanvi-benchmark/...`). Always pass absolute paths when calling aggregate scripts from outside their directory.

### L-089 — [2026-07-12] `init_prior_from_data` protein subsampling: integer row-index vs DataFrame column indexing
**Category**: bug
**Tags**: mrtotalvi, vamp-prior, init_prior_from_data, pandas, indexing
**mitigation_type**: structural
**structural_mitigation_candidate**: test that `init_prior_from_data=True` runs end-to-end without error (already required by Change D verification in the plan)
**Body**: In `mrtotalvi/_model.py:192`, `get_from_registry(PROTEIN_EXP_KEY)` returns a pandas DataFrame with protein-name string columns. Applying `[idx]` where `idx` is a numpy integer array of row positions was interpreted by pandas as column selection → `KeyError`. Fix: `.to_numpy()[idx]` extracts the underlying ndarray first so integer indexing selects rows. The same pattern at line 185 for `X_KEY` works because the gene expression registry returns a scipy sparse matrix (where integer array indexing is row selection). Root cause discovered from SLURM job failure logs: tasks 4-6 of array 25211780 failed at `__init__` within 55s.
