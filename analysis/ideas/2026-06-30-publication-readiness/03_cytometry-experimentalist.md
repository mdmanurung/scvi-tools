# Cytometry Experimentalist — Publication Readiness Ideas (Refined)

**Persona**: Senior cytometry researcher (CyTOF + spectral flow, clinical and basic research)
**Session**: 2026-06-30 publication readiness ideation (refined 2026-06-30)
**Focus**: Biological / practical gaps that prevent lab adoption

---

## Idea 7 — Per-channel arcsinh cofactor support for non-CyTOF technologies

**Biological/practical gap**

`transform_arcsinh` in `src/scvi/external/cytovi/_preprocessing.py` defaults to `global_scaling_factor=5` (the CyTOF standard for metal ion counts). The function does accept a `scaling_dict` parameter for per-channel overrides, but no helper exists to estimate channel-appropriate cofactors, no warning fires when the default is used without a `scaling_dict`, and no QC path exists to surface the symptom (near-zero post-arcsinh values).

Conventional flow PMT values span 0–100k+; `arcsinh(x / 5)` saturates at ~9 for all values above ~75, making the entire dynamic range collapse to a single point. Spectral flow is identical in scale. A lab running a 30-color spectral flow panel on a Sony ID7000 will silently get near-zero transformed values — no warning, no diagnostic, no documentation that cofactor=5 is wrong for them.

**What to do**

*File: `src/scvi/external/cytovi/_preprocessing.py`*

1. Add `technology` parameter to `transform_arcsinh`. Change the signature to include `technology: str = "cytof"`. Add a `UserWarning` when `technology != "cytof"` and `scaling_dict is None`:
   ```
   "technology={!r} but no scaling_dict was provided and global_scaling_factor=5
   (CyTOF default). Conventional flow / spectral flow require per-channel cofactors
   of 150–1000. Pass scaling_dict={'ChannelA': 300, ...} or call
   cytovi.estimate_cofactor(adata) to compute channel-specific cofactors."
   ```

2. Add `estimate_cofactor(adata, raw_layer_key="raw", percentile=5.0, min_cofactor=5.0, max_cofactor=2000.0) -> dict[str, float]`. For each channel, take the 5th-percentile positive value (`x > 0`) as the cofactor (flowCore logicle heuristic approximation), clip to `[min_cofactor, max_cofactor]`. Returns `{marker: cofactor}`. Also writes result to `adata.var["cofactor"]` as a side effect — giving a persistent per-dataset record.

3. At the end of `transform_arcsinh`, write the effective per-channel cofactors back to `adata.var["cofactor"]` (as float64), so the transform is always reproducibly reconstructable from the AnnData.

4. Add `plot_arcsinh_qc(adata, transformed_layer_key="transformed", threshold=0.1, max_markers=20) -> matplotlib.figure.Figure`. For each channel, compute the fraction of cells with `|transformed| < threshold` (near-zero). Produce a horizontal bar chart sorted by that fraction. Channels exceeding 50% near-zero are labelled "DEGENERATE" in red. Return the figure without displaying it.

*File: `src/scvi/external/cytovi/__init__.py`*

5. Export `estimate_cofactor` and `plot_arcsinh_qc` from the cytovi public API alongside `transform_arcsinh`.

*File: `benchmarks/cytoanvi/data.py`*

6. In `load_roider_full()` (the `transform_arcsinh` call), add `technology="cytof"` to silence the new warning (Roider is CyTOF).

7. In `load_nunez()`, add `technology="cytof"` to the `transform_arcsinh` call for the same reason.

*File: `tests/external/cytoanvi/test_arcsinh_cofactor.py`* (new)

8. Write three tests:
   - `test_estimate_cofactor_writes_var`: call `estimate_cofactor(adata)`, assert `"cofactor"` in `adata.var.columns` and all values are finite.
   - `test_transform_arcsinh_persists_cofactor`: call `transform_arcsinh(adata, technology="cytof")`, assert `adata.var["cofactor"]` == 5.0 for all channels.
   - `test_transform_arcsinh_warns_non_cytof`: assert `UserWarning` is raised when calling `transform_arcsinh(adata, technology="flow")` with no `scaling_dict`.

**Done criteria**

1. `pytest tests/external/cytoanvi/test_arcsinh_cofactor.py -v` passes all three tests.
2. After `cytovi.transform_arcsinh(adata, technology="flow")` with default `global_scaling_factor=5` and no `scaling_dict`, `pytest.warns(UserWarning)` catches the warning and `adata.var["cofactor"]` is populated.

**Dependencies**: No dependency on Ideas 8 or 9. Enables safe use by flow/spectral-flow users before CytoANVI is ever instantiated.

**Effort**: Medium | **Priority**: Critical

---

## Idea 8 — B4/B6 disease-split validation and EWC `train()` hard error

**Biological/practical gap**

B4 (`task_b4_continual`) and B6 (`task_b6_lambda_sweep`) both use `_b4_setup` → `_split_reference_query`, which splits by `batch_key` values. For Roider this is a batch-index pseudo-split with no real biological shift. The EWC penalty acts on the same distribution in both directions, and `replay_latent_drift` is 0.0 regardless of `ewc_importance` (λ). The B6 code correctly emits `"recommendation_status": "no_recommendation"`.

The fix exists in the data: `adata.obs["Entity"]` (populated by `annotate_roider_obs()` in `benchmarks/common/roider_metadata.py`) has values `"rLN"` (reactive lymph node controls), `"FL"` (follicular lymphoma), `"MCL"` (mantle cell lymphoma). The rLN vs. FL/MCL split is the canonical case-control axis.

A secondary gap in the save/load path: `CytoANVI.train()` issues a `UserWarning` when `continual.replay_batches` is empty after a load/reload, but does not prevent the caller from silently training with EWC-only (no experience replay). Published cscanvi requires both components.

**What to do**

*File: `benchmarks/cytoanvi/tasks.py`*

1. Add `_split_roider_disease(adata, *, disease_key="Entity", ref_values=("rLN",), query_values=("FL", "MCL"), seed=0) -> tuple[AnnData, AnnData]`. This function:
   - Extracts `entity = np.asarray(adata.obs[disease_key].astype(str))`
   - Sets `is_ref = np.isin(entity, list(ref_values))` and `is_query = np.isin(entity, list(query_values))`
   - Raises `ValueError` with diagnostic message if either split has fewer than 64 cells
   - Returns `(adata[is_ref].copy(), adata[is_query].copy())`

2. Add `task_b4_disease(adata, labels_key="labels", unlabeled_category="Unknown", batch_key="batch", sample_key=None, nan_layer=None, seed=0, max_epochs=1000, n_latent=None, ewc_importance=1.0, control_frac=0.1, replay_frac=0.2, disease_key="Entity", ref_values=("rLN",), query_values=("FL", "MCL"), batch_size=None) -> dict`. This function:
   - Calls `_split_roider_disease(adata, ...)` → `ref_adata, query_adata`
   - Sets `query_adata.obs[labels_key] = unlabeled_category` (disease query cells are unlabeled during update)
   - Returns the same schema as `task_b4_continual` with `"split": "disease_rLN_vs_FL_MCL"` replacing `"note"`.

3. In `task_b6_lambda_sweep`, add `disease_split: bool = False, disease_key="Entity", ref_values=("rLN",), query_values=("FL", "MCL")` parameters. When `disease_split=True`, invoke `task_b4_disease` for each λ instead of `task_b4_continual`.

4. In `run.py`, add a `b4d` task entry that calls `task_b4_disease` when `--dataset roider-full`. Add `--disease-key`, `--ref-values`, `--query-values` CLI args (defaults: `Entity`, `rLN`, `FL,MCL`). Example:
   ```bash
   python -m benchmarks.cytoanvi.run --dataset roider-full --task b4d \
     --entity-key Entity --control-entities rLN --seed 0 --max-epochs 1000
   ```

5. Add a guard: if `disease_key not in adata.obs.columns`, raise `KeyError` with message `"Entity column missing — run load_roider_full() with annotate_metadata=True first"`.

*File: `src/cytoanvi/_model.py`*

6. In `CytoANVI.train()`, change the `UserWarning` for empty replay buffer to a `ValueError` with a two-part message: (a) that continuing without replay buffer breaks the cscanvi guarantee, and (b) recovery path: `CytoANVI.load_query_data_with_replay(..., replay_adata=ref_subset)`. Add `allow_empty_replay=False` keyword to `train()` so callers can explicitly opt into EWC-only mode.

*File: `tests/external/cytoanvi/test_continual.py`* (new or extend)

7. Add `test_train_errors_on_empty_replay`: load a model saved after `load_query_data_with_replay`, assert `ValueError` is raised when `model.train()` is called without re-supplying `replay_adata`.

8. Add `test_b4_disease_drift_nonzero`: on `make_synthetic_panels()` with forced entity labels in `obs["Entity"]`, verify that `task_b4_disease()` returns `continual_update.replay_latent_drift > 0` for at least one λ > 0.

**Done criteria**

1. `task_b4_disease(roider_full_merged, ...)` with `max_epochs=1000`, seeds 0/1/2 produces `continual_update.replay_latent_drift` values that vary monotonically with λ across at least one seed — i.e., B6 disease split returns `"recommendation_status": "recommended"` with λ > 0.
2. `pytest tests/external/cytoanvi/test_continual.py::test_train_errors_on_empty_replay -v` passes (raises `ValueError`, not `UserWarning`).

**Dependencies**: Requires `load_roider_full` + `annotate_roider_obs` working correctly to populate `adata.obs["Entity"]`. Does not depend on Ideas 7 or 9. B4 disease split is prerequisite for B6 λ recommendation.

**Effort**: High | **Priority**: Critical

---

## Idea 9 — Novelty detection needs actionable threshold and P@95S metric

**Biological/practical gap**

`CytoANVI.get_uncertainty()` returns a raw numpy array of Bregman Information floats with no calibration, no reference distribution, and no threshold. A lab receiving a BI score of 0.42 for a query cell has no way to decide whether to flag it. Current B5 reports AUROC only; clinical use requires a fixed operating point.

The standard choice for novelty detection is P@95S (precision at 95% specificity): given the reference distribution as the null, fix the threshold at the 95th percentile of reference-cell uncertainty scores, then report what fraction of cells above the threshold are truly novel. A z-score relative to the reference null also allows communicating "this cell is 3 sigma above the reference" without per-dataset calibration.

The existing `select_replay_by_uncertainty()` classmethod already calls `get_uncertainty()` on reference cells and uses the resulting distribution — those same reference scores are the natural null distribution for calibration.

**What to do**

*File: `benchmarks/cytoanvi/metrics.py`*

1. Add `precision_at_specificity(uncertainty, is_novel, specificity=0.95) -> dict`:
   ```python
   def precision_at_specificity(uncertainty, is_novel, specificity=0.95):
       is_novel = np.asarray(is_novel).astype(bool)
       threshold = np.percentile(uncertainty[~is_novel], 100.0 * specificity)
       flagged = uncertainty > threshold
       precision = float(flagged[is_novel].sum() / flagged.sum()) if flagged.sum() > 0 else float("nan")
       return {
           "threshold": float(threshold),
           "precision": float(precision),
           "n_flagged": int(flagged.sum()),
           "n_novel_flagged": int(flagged[is_novel].sum()),
           "specificity": specificity,
       }
   ```

2. Add `uncertainty_zscore(unc, ref_unc) -> np.ndarray`: computes `(unc - ref_unc.mean()) / (ref_unc.std() + 1e-8)`.

*File: `benchmarks/cytoanvi/tasks.py`*

3. In `task_b5_novelty()`: after computing `unc_latent` and `unc_logit`, compute reference-cell uncertainty `ref_latent = model.get_uncertainty(work[~is_novel], mode="latent")` (cache; do not call twice). Add to return dict:
   ```python
   "latent_p95s": metrics.precision_at_specificity(unc_latent, is_novel, specificity=0.95),
   "logit_p95s": metrics.precision_at_specificity(unc_logit, is_novel, specificity=0.95),
   "latent_zscore_stats": {
       "mean": float(metrics.uncertainty_zscore(unc_latent, ref_latent).mean()),
       "median": float(np.median(metrics.uncertainty_zscore(unc_latent, ref_latent))),
   },
   ```

4. In `task_b5_holdout_sweep()`: after collecting `per_type` results, add:
   ```python
   p95s_list = [v["latent_p95s"]["precision"] for v in per_type.values()
                if not np.isnan(v["latent_p95s"]["precision"])]
   ```
   Add `"mean_p95s_precision"` and `"best_p95s_precision"` to the return dict alongside existing AUROC aggregates.

*File: `src/cytoanvi/_model.py`*

5. Add `get_uncertainty_threshold(self, reference_adata, specificity=0.95, tta_rep=50, mode="latent", batch_size=None) -> float`:
   - Calls `ref_unc = self.get_uncertainty(reference_adata, tta_rep=tta_rep, mode=mode, batch_size=batch_size)`
   - Returns `float(np.percentile(ref_unc, 100.0 * specificity))`
   - Docstring: *"Returns the uncertainty score at the `specificity` quantile of reference-cell scores. Flag query cells whose uncertainty exceeds this threshold. At `specificity=0.95`, approximately 5% of reference cells are false positives."*

*File: `tests/external/cytoanvi/test_uncertainty.py`* (new or extend)

6. Add `test_precision_at_specificity_perfect`: construct `uncertainty = np.concatenate([np.zeros(100), np.ones(20)])` and `is_novel = np.array([False]*100 + [True]*20)`. Assert `precision_at_specificity(uncertainty, is_novel, 0.95)["precision"]` == 1.0.

7. Add `test_get_uncertainty_threshold_shape`: on `make_synthetic_panels()` data, train a minimal model (`max_epochs=3`), call `model.get_uncertainty_threshold(reference_adata=adata, specificity=0.95)`, assert result is a finite float.

8. Add `test_b5_novelty_has_p95s_key`: call `task_b5_novelty(adata, ...)` and assert `"latent_p95s"` in result and `"precision"` in result["latent_p95s"].

**Done criteria**

1. `pytest tests/external/cytoanvi/test_uncertainty.py -v` passes all three tests, including the perfect-precision fixture.
2. A full B5 holdout sweep on Nuñez data returns `"mean_p95s_precision" > 0` (threshold is not degenerate) and every per-type entry contains `"latent_p95s": {"threshold": <finite float>, "precision": <finite float>, ...}`.

**Dependencies**: Depends on nothing in Ideas 7 or 8. `get_uncertainty_threshold` directly enables Idea 7 users deploying query mapping to report a single actionable number. P@95S metric should be included in publication supplement alongside AUROC.

**Effort**: Medium | **Priority**: High
