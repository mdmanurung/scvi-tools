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

### L-014 — [2026-06-01] Figshare egress returns HTTP 202 / 0 bytes in dev env; no pip in SLURM queue
**Category**: infra
**Tags**: figshare, data-download, slurm, mapqc, environment
**mitigation_type**: ambient-awareness
**structural_mitigation_candidate**: 
**Body**: Two env constraints that block otherwise-straightforward steps: (1) Figshare download returns HTTP 202 (async processing) / 0-byte file in the dev environment — data must be acquired via user action outside the agent. (2) `pip install` is not allowed inside the SLURM job queue — `mapqc` (needed for B9) must be installed in the conda environment before job submission. B9 is currently blocked by (2). Source: `.scratch/cytoanvi-benchmark/issues/01` and `issues/13`.
