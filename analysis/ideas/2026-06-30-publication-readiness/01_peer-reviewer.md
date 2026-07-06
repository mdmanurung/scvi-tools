# Peer Reviewer — Publication Readiness Ideas (Refined)

**Persona**: Independent peer reviewer, cytometry methods paper
**Session**: 2026-06-30 publication readiness ideation (refined 2026-06-30)
**Focus**: Scientific rigor, baseline comparisons, and evidence quality

---

## Idea 1 — Add FlowSOM/Phenograph baselines and stratify B1 by cohort type

**Risk**

Two compounding risks.

First, B1 currently benchmarks CytoANVI against CytoVI k-NN, raw-marker k-NN, and Harmony k-NN. None of these are the community-standard tools cytometry practitioners actually use. Any reviewer familiar with the field will immediately ask why FlowSOM and Phenograph are absent — both appear in virtually every cytometry methods paper.

Second, the Nuñez B1 delta is unstable and depends critically on label source. The Leiden-labeled `nunez_r005_e1000` run shows CytoANVI 0.954±0.032 vs CytoVI k-NN 0.967±0.004 → Δ **−0.013** (reversal). On `nunez_annotated.h5ad`-labeled runs, seeds 0/1/2 show Δ +0.022/+0.023/+0.016 — positive but all below the +0.03 publication gate. The claim is not safe until it is clear which batch drives the gap and whether the threshold is met at ≥3 seeds with expert labels.

**What to do**

1. Add `flowsom_and_knn()` to `benchmarks/cytoanvi/baselines.py`. Signature:
   ```python
   def flowsom_and_knn(adata, labels_key, unlabeled_category, n_clusters=15, seed=0, layer=SCALED_LAYER)
   ```
   Implementation: fit FlowSOM (`pip install flowsom`, the Saeyslab Python port) on labeled-cell scaled-marker matrix; build metacluster → majority-true-label map; assign unlabeled cells by nearest SOM node → metacluster → majority label. Return `(pred_for_unlabeled, None, unlabeled_mask)` — identical contract to `cytovi_latent_and_knn()`. Wrap in `try/except ImportError` so absence of the package degrades to `{"error": "flowsom not installed"}`.

2. Add `phenograph_and_knn()` to `benchmarks/cytoanvi/baselines.py`. Signature:
   ```python
   def phenograph_and_knn(adata, labels_key, unlabeled_category, k=30, seed=0, layer=SCALED_LAYER)
   ```
   Implementation: call `phenograph.cluster(X_all, k=k, seed=seed)` on all cells, build community → majority-label map from labeled cells only; assign unlabeled cells from the map. Return same contract. Wrap in `try/except ImportError`.

3. In `task_b1_label_transfer()` (`benchmarks/cytoanvi/tasks.py`), add two new try/except blocks after the `harmony_result` block. One for each new baseline. Add `"flowsom"` and `"phenograph"` to the return dict. Pattern is identical to `"harmony_knn"`: if the call raises any `Exception`, store `{"error": str(e)}` rather than aborting.

4. Add `stratify_key: str | None = None` parameter to `task_b1_label_transfer()`. When not `None`, compute `metrics.label_transfer_metrics()` per stratum (unique values of `work.obs[stratify_key]` among held cells) for each method. Add `"per_stratum": {"cytoanvi": {...}, "cytovi_knn": {...}, "flowsom": {...}, ...}` to the return dict.

5. Add `--stratify-key` argument to `argparse` in `benchmarks/cytoanvi/run.py` (default `None`). Thread it through to `task_b1_label_transfer(..., stratify_key=args.stratify_key)` in the B1 dispatch block. For Nuñez runs pass `--stratify-key batch`; for roider-full runs pass `--stratify-key Entity`.

6. Add `flowsom` and `phenograph` to `pyproject.toml` under a new optional extra: `[project.optional-dependencies] cytoanvi-baselines = ["flowsom", "phenograph"]`.

7. Run seed 0 Nuñez B1 with `--stratify-key batch --require-annotated-nunez` to confirm the per-batch breakdown. If the below-threshold delta (+0.02 vs +0.03 gate) is concentrated in one batch, document it and consider reporting per-batch results in the paper.

**Done criteria**

- B1 result JSON for any Nuñez seed contains top-level keys `"flowsom"` and `"phenograph"` (or `{"error": "..."}` if package absent) alongside `"cytoanvi"` and `"cytovi_knn"` — verified by `assert "flowsom" in result["b1"]` in `tests/benchmarks/test_cytoanvi_smoke.py`.
- At least one Nuñez run with `--stratify-key batch --require-annotated-nunez` produces a `"per_stratum"` dict with both `"batch_0"` and `"batch_1"` entries, confirming the weak overall delta is not driven by a pathological single batch.

**Dependencies**: Requires `data/nunez_annotated.h5ad` (already present). Does not depend on Ideas 2 or 3.

**Effort**: High | **Priority**: Critical

---

## Idea 2 — Reframe or fix B5 novelty detection (mean AUROC 0.467)

**Risk**

The B5 mean AUROC of 0.467 — below chance — is the most damaging number in the paper if reported without qualification. The 13-type Roider T-cell sweep shows a bimodal distribution: Tfh 0.744, Treg CD69+ 0.741, Tpr 0.710, Ttox EM3 0.702 (≥0.70); Treg CD69− 0.131, Ttox EM2 0.097, Tdp 0.269, Th CM1 0.206 (failing badly). The pattern is scientifically interpretable — phenotypically isolated types are detectable as novel; near-duplicate subtypes within the EM/Treg families are not. But this interpretation is unsubstantiated in code: there is no metric quantifying phenotypic similarity, no correlation analysis tying similarity to AUROC, and no alternative uncertainty mode tested.

**Diagnostic path (do first)**

1. Add `marker_isolation_score(adata, holdout_type, labels_key, layer=SCALED_LAYER)` to `benchmarks/cytoanvi/metrics.py`. Implementation: compute per-label centroids in `layer` space; return minimum L2 distance from the held-type centroid to any other label's centroid.

2. Update `task_b5_holdout_sweep()` in `benchmarks/cytoanvi/tasks.py`. For each `ht` in the sweep loop, call `marker_isolation_score(adata, ht, labels_key)` and store as `per_type[ht]["isolation_score"]`. After the loop, compute `spearmanr(isolation_scores, aurocs)` from `scipy.stats` and add `"spearman_auroc_vs_isolation": float` and `"spearman_p": float` to the return dict.

3. Add `mode="entropy"` and `mode="margin"` to `get_uncertainty()` in `src/scvi/external/cytoanvi/_model.py`. Entropy: `H = -sum(p * log(p + 1e-8), axis=-1)` over classifier softmax output. Margin: gap between top-2 softmax probabilities. Both derive from existing logit output — no new model parameters.

4. Update `task_b5_novelty()` to call `model.get_uncertainty(mode="entropy")` and `model.get_uncertainty(mode="margin")`, adding `"entropy"` and `"margin"` sub-dicts to the return dict. Update `task_b5_holdout_sweep()` to track `per_type[ht]["best_auroc"] = max(latent_auroc, logit_auroc, entropy_auroc, margin_auroc)`.

5. Interpret the Spearman r: if r ≥ 0.6 (p < 0.05), the isolation score explains the bimodal split and the reframing path is valid. Rewrite the B5 paper claim as: *"Novelty recall scales monotonically with phenotypic isolation (Spearman r = X.XX, p = Y.YY); CytoANVI correctly flags isolated novel populations but cannot distinguish near-duplicate subtypes."* Do not report 0.467 mean AUROC as a headline.

**Method-fix path (if Spearman r < 0.6)**

6. Add `mode="mc_dropout"` to `get_uncertainty()`. Implementation: call `module.train()` to activate dropout, run N=25 forward passes, compute per-cell softmax entropy across samples, call `module.eval()`. Update `task_b5_novelty()` to evaluate this mode's AUROC alongside the others.

**Done criteria**

- `task_b5_holdout_sweep()` return dict includes `"spearman_auroc_vs_isolation"` and `"isolation_score"` per type — verified by a new test in `tests/benchmarks/test_cytoanvi_smoke.py` asserting both keys are present on synthetic data.
- Either: `"spearman_auroc_vs_isolation" ≥ 0.60` on the roider seed-0 result (reframe path), OR: mean best-mode AUROC rises to ≥ 0.55 at ≥3 seeds after adding `mode="mc_dropout"` (method-fix path). In either case, 0.467 mean AUROC is not reported without conditioning.

**Dependencies**: Requires seeds 1 and 2 of B5 sweep (currently only seed 0 available). Spearman diagnostic can be computed from the single existing seed immediately to determine which path to take. No dependency on Ideas 1 or 3.

**Effort**: Med–High | **Priority**: Critical

---

## Idea 3 — Real case/control split for B4/B6, or demote to supplement

**Risk**

All current B4 and B6 results show `replay_latent_drift = 0.0` for both plain surgery and continual update at every λ ∈ {0, 1, 10, 100, 1000}. The B6 code correctly emits `"recommendation_status": "no_recommendation"`. The cause: these runs used `--dataset roider`, which splits by batch value — and the two Roider batches are patient replicates with identical biology. Without biological drift, EWC has nothing to constrain, λ selection is uninformative, and the "continual update" claim has zero empirical support.

The fix exists in the data: `adata.obs["Entity"]` in the full Roider cohort (populated by `benchmarks/common/roider_metadata.py:annotate_roider_obs()`) has values `rLN` (reactive lymph node controls), `FL` (follicular lymphoma), `MCL` (mantle cell lymphoma), `CLL`, `DLBCL`. Issue 10 specifies exactly this split.

**What to do**

1. Add `_split_by_entity(adata, entity_key="Entity", control_entities=("rLN",), seed=0) -> tuple[AnnData, AnnData, BooleanMask]` to `benchmarks/cytoanvi/tasks.py`, after the existing `_split_reference_query()` function. Raises `ValueError` with a diagnostic message if either split has fewer than 64 cells or if `entity_key` is not in `adata.obs`. No permuted fallback — fail loudly rather than silently degrading to a batch split.

2. Add `entity_key: str | None = None` and `control_entities: list[str] | None = None` parameters to `_b4_setup()`. When `entity_key` is not `None`, replace the `_split_reference_query()` call with `_split_by_entity(adata, entity_key=entity_key, control_entities=control_entities or ["rLN"])`. Propagate both parameters up through `task_b4_continual()` and `task_b6_lambda_sweep()`.

3. Update the `task_b4_continual()` return dict to include `"split_type": "entity"` or `"split_type": "batch_pseudo"`, plus `"reference_entities"` and `"query_entities"` string lists. This lets `aggregate_results.py` distinguish real vs. pseudo runs.

4. Add `--entity-key` (default `None`) and `--control-entities` (default `"rLN"`, comma-separated) arguments to `argparse` in `benchmarks/cytoanvi/run.py`. Thread through in the B4 and B6 dispatch blocks. Example: `python -m benchmarks.cytoanvi.run --dataset roider-full --task b4 --entity-key Entity --control-entities rLN --seed 0 --max-epochs 1000 --batch-size 8192`.

5. Add a guard to `_split_by_entity()`: if `entity_key not in adata.obs.columns`, raise `KeyError` with the message `"Entity column missing — run load_roider_full() with annotate_metadata=True first"`.

6. Add an assertion to `benchmarks/common/aggregate_results.py`: when aggregating B4 results with `"split_type": "entity"`, assert that `continual_update["replay_latent_drift"] <= plain_surgery["replay_latent_drift"]` at the recommended λ. This is the machine-checkable publication gate.

**Done criteria**

- Seed 0 of B4 entity-split on roider-full returns `plain_surgery["replay_latent_drift"] > 0.0` — proving biological gradient exists between rLN reference and FL/MCL query (drift=0.0 would mean this split has the same pathology as the pseudo-batch split and the task must be demoted).
- B6 entity-split at seed 0 emits `"recommendation_status": "recommended"` with a non-trivial `recommended_lambda` value (λ > 0), confirming the EWC λ-vs-drift curve has a real minimum.

**Dependencies**: Requires `roider-full` with `Entity` column populated. If metadata source (`41556_2024_1358_MOESM3_ESM.xlsx`) is unavailable, B4 entity-split cannot run and the claim must be demoted to supplement with explicit acknowledgment that Phase 2 is plumbing-only. Does not depend on Ideas 1 or 2.

**Effort**: Med–Low | **Priority**: High
