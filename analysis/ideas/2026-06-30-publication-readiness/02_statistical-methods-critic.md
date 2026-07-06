# Statistical Methods Critic — Publication Readiness Ideas (Refined)

**Persona**: Statistical methods reviewer, biostatistics orientation
**Session**: 2026-06-30 publication readiness ideation (refined 2026-06-30)
**Focus**: Statistical validity, multiple testing, fair comparisons

---

## Idea 4 — B3 concordance is a circular non-ground-truth metric

**Weakness**

`task_b3_panel_divergent` in `benchmarks/cytoanvi/tasks.py` computes `p2_concordance_vs_knn`: the fraction of panel-2 cells where CytoANVI prediction equals CytoVI-kNN prediction. Both models train on the same panel-1 reference, so their errors are correlated — systematic misclassification will appear as high concordance. The current `Roider_et_al_BNHL_panel2.h5ad` has `cell_type = 'unknown'` for all 4983 panel-2 cells, confirming no ground truth is currently ingested. Concordance ≥ 0.80 is the published gate; at seeds 0/1/2 concordance = 0.860 / 0.887 / 0.883 (mean 0.877). *(⚠️ SUPERSEDED — these are roider-e1000 subset numbers; full-cohort 3-seed result is **0.671±0.008** — gate NOT met; see F-012 in FINDINGS_REGISTRY.md)* This cannot distinguish agreement from joint error.

**What to do**

*Step 1 — Obtain panel-2 ground-truth labels (prerequisite, data acquisition)*

Audit `data/roider_raw/FlowCytometryData_Part{A,B,C,D}*/` for FCS files whose filenames or `$P*S` text annotations suggest panel-2 gating. Run:
```bash
find data/roider_raw -name "*.fcs" | xargs -I{} python -c "
import fcsparser; meta, _ = fcsparser.parse('{}', meta_data_only=True)
print('{}:', [k for k in meta if 'gate' in k.lower() or 'population' in k.lower()])
"
```
If gate metadata exists in FCS headers, write `benchmarks/cytoanvi/fetch_roider_p2_labels.py` to extract gate population assignments. If absent, file a data-access request to Roider et al. and document the blocker in `data/DATA_MANIFEST.md`:
```
| Roider panel-2 expert labels | Roider et al. (contact) | — | data/roider_p2_labels.csv | Per-cell gating for p2 | BLOCKED |
```

*Step 2 — Ingest labels into the h5ad*

When labels are available as CSV (`cell_barcode, cell_type`):
```python
import anndata, pandas as pd
adata = anndata.read_h5ad("data/Roider_et_al_BNHL_panel2.h5ad")
labels = pd.read_csv("data/roider_p2_labels.csv", index_col="cell_barcode")
adata.obs["cell_type"] = labels.reindex(adata.obs_names)["cell_type"].fillna("unknown")
adata.write_h5ad("data/Roider_et_al_BNHL_panel2.h5ad")
```
Verify: `adata.obs["cell_type"].nunique() > 1` must pass.

*Step 3 — Modify `task_b3_panel_divergent` to compute macro-F1 vs ground truth*

In `benchmarks/cytoanvi/tasks.py`, add `p2_true_labels_key: str | None = "cell_type"` to the signature. After the concordance computation, add:
```python
if p2_true_labels_key is not None:
    p2_true = np.asarray(p2.obs.get(p2_true_labels_key, "unknown").astype(str))
    known_mask = p2_true != "unknown"
    if known_mask.sum() > 0:
        p2_pred_at_known = pred[is_p2][known_mask]
        out["p2_macro_f1"] = metrics.label_transfer_metrics(
            p2_true[known_mask], p2_pred_at_known
        )
```

*Step 4 — Update `aggregate_results.py`*

In `_summarize_single_task` for `"b3"`, after `out["p2_concordance"]`, add:
```python
out["p2_macro_f1"] = _get(payload, "p2_macro_f1", "macro_f1")
```

In `summarize_multiseed`, after `out["b3_p2_concordance"]`, add:
```python
p2_f1 = _get(summary, "b3.p2_macro_f1.macro_f1")
if p2_f1:
    out["b3_p2_macro_f1"] = p2_f1
```

*Step 5 — Update publication gate in `benchmarks/ANALYSIS_MANIFEST.md`*

Replace: `Gate: concordance ≥ 0.80`
With: `Primary gate: macro-F1 vs expert labels ≥ 0.75. Secondary: concordance vs kNN reported for methods comparison only.`

*Step 6 — Re-run B3 at 3 seeds after label ingestion*

```bash
python -m benchmarks.cytoanvi.run --dataset roider --task b3 --seeds 0,1,2 \
  --max-epochs 1000 --labels-key cell_type \
  --out .scratch/cytoanvi-benchmark/results/e1000/roider_e1000_b3_multiseed_v2.json
```

**Done criteria**

1. `roider_e1000_b3_multiseed_v2.json` contains `b3.p2_macro_f1.macro_f1 > 0.0` for all 3 seeds; `b3.p2_concordance_vs_knn` still present as secondary.
2. `adata.obs["cell_type"].value_counts().iloc[0] < adata.n_obs` passes on `Roider_et_al_BNHL_panel2.h5ad` (i.e., not all-unknown).

**Dependencies**: Data acquisition (Step 1) is the binding constraint. Steps 3–6 can be written and tested with a synthetic p2 ground-truth stub. Does not block any other idea.

**Effort**: Medium (data acquisition unknown; code changes ~30 lines) | **Priority**: Critical

---

## Idea 5 — B5 reports max AUROC over 13 tests when mean is sub-chance

**Weakness**

`task_b5_holdout_sweep` in `benchmarks/cytoanvi/tasks.py` computes AUROC for each of 13 held-out types independently, then reports `best_auroc` (0.744, Tfh at seed 0) and `mean_auroc` (0.467, below 0.5 chance). The published gate "best holdout-type AUROC > 0.70" is the maximum over 13 independent tests with no multiple-testing correction. Under the null (AUROC ≡ 0.5 per type, independent), the expected maximum over 13 tests is well above 0.70 by chance — a 20-line simulation will confirm this. Per-type p-values and FDR correction are absent from the output.

**What to do**

*Step 1 — Add `auroc_pvalue` to `benchmarks/cytoanvi/metrics.py`*

After the `novelty_auroc` function, add:
```python
def auroc_pvalue(auroc: float, n_novel: int, n_ref: int) -> float:
    """One-sided p-value for AUROC > 0.5 via Mann-Whitney U normal approximation.
    Valid when n_novel, n_ref >= 10. Returns 1.0 if counts are too small."""
    import scipy.stats
    if n_novel < 10 or n_ref < 10:
        return 1.0
    U = auroc * n_novel * n_ref
    mu = n_novel * n_ref / 2.0
    sigma = np.sqrt(n_novel * n_ref * (n_novel + n_ref + 1) / 12.0)
    z = (U - mu) / sigma
    return float(scipy.stats.norm.sf(z))  # one-sided: P(AUROC > 0.5 | H0)
```
This derivation is exact: `roc_auc_score(is_novel, uncertainty) = U / (n_novel * n_ref)` where U is the Mann-Whitney U statistic, so the p-value is analytically recoverable from stored `auroc` and `n_novel` values without rerunning models.

*Step 2 — Modify `task_b5_novelty` to return `n_ref`*

In `tasks.py`, after defining `is_novel`, add:
```python
n_ref = int((~is_novel).sum())
```
Then add `"n_ref": n_ref` to the return dict.

*Step 3 — Add FDR correction and null simulation to `task_b5_holdout_sweep`*

After the per-type loop, before the return statement, add:
```python
from statsmodels.stats.multitest import multipletests

type_names = [ht for ht, v in per_type.items() if not np.isnan(v.get("auroc", float("nan")))]
pvals = [
    metrics.auroc_pvalue(per_type[ht]["auroc"], per_type[ht]["n_novel"],
                         per_type[ht].get("n_ref", adata.n_obs - per_type[ht]["n_novel"]))
    for ht in type_names
]
if pvals:
    reject, pvals_corrected, _, _ = multipletests(pvals, method="fdr_bh", alpha=0.05)
    for i, ht in enumerate(type_names):
        per_type[ht]["pval"] = float(pvals[i])
        per_type[ht]["bh_pval"] = float(pvals_corrected[i])
        per_type[ht]["bh_reject"] = bool(reject[i])
n_bh_significant = int(sum(per_type[ht].get("bh_reject", False) for ht in per_type))

# Null simulation: expected max AUROC over n_types tests under H0
rng_null = np.random.default_rng(42)
n_sim, n_types_valid = 10_000, len(aurocs)
null_maxes = np.array([
    np.max(0.5 + rng_null.normal(0, np.std(aurocs) if len(aurocs) > 1 else 0.18, size=n_types_valid))
    for _ in range(n_sim)
])
null_max_p95 = float(np.percentile(null_maxes, 95))
```

Add these keys to the return dict:
```python
"n_bh_significant": n_bh_significant,
"null_max_p95": null_max_p95,
"observed_best_exceeds_null_p95": bool(float(max(aurocs)) > null_max_p95) if aurocs else False,
```

*Step 4 — Update `aggregate_results.py` B5 block*

In `_summarize_single_task` for `"b5"`, add:
```python
out["n_bh_significant"] = payload.get("n_bh_significant")
out["null_max_p95"] = payload.get("null_max_p95")
out["observed_best_exceeds_null_p95"] = payload.get("observed_best_exceeds_null_p95")
```

*Step 5 — Update publication gate in `benchmarks/ANALYSIS_MANIFEST.md`*

Replace: `Gate: best holdout-type AUROC > 0.70`
With: `Gate: n_bh_significant ≥ 1 (FDR < 0.05 after BH correction over 13 types) AND mean_auroc > 0.5. Report best_auroc as exploratory secondary. Report null_max_p95 in supplementary to contextualise the uncorrected maximum.`

*Step 6 — Re-run B5 at seeds 0,1,2*

```bash
python -m benchmarks.cytoanvi.run --dataset roider --task b5 --holdout-sweep \
  --seeds 0,1,2 --max-epochs 1000 \
  --out .scratch/cytoanvi-benchmark/results/e1000/roider_e1000_b5_sweep_v2.json
```

**Done criteria**

1. `roider_e1000_b5_sweep_v2.json` contains `n_bh_significant`, `null_max_p95`, `observed_best_exceeds_null_p95`, and each per-type entry contains `pval`, `bh_pval`, `bh_reject`.
2. Unit test in `tests/benchmarks/test_cytoanvi_smoke.py`: `assert "n_bh_significant" in result["b5"]` and `assert "null_max_p95" in result["b5"]`.
3. The paper can now distinguish "Tfh is a genuinely detectable novel type (BH-corrected)" from "the 0.744 maximum is consistent with null-model chance."

**Dependencies**: No other ideas are prerequisites. Enables a defensible uncertainty-detection claim or a properly framed negative result. The p-value computation is a post-hoc closed-form formula from stored JSON — no model re-training until Step 6.

**Effort**: Low | **Priority**: Critical

---

## Idea 6 — B1 information asymmetry: add XGBoost supervised upper bound

**Weakness**

`task_b1_label_transfer` trains CytoANVI semi-supervised (labels enter ELBO + classifier loss) while CytoVI-kNN trains fully unsupervised (labels applied only post-hoc as kNN references). This is not a peer comparison. The Nuñez reversal (CytoANVI 0.954±0.032 vs kNN 0.967±0.004 across 3 seeds) is unexplained in the current framing. The raw B-cell recall collapse at seed 1 (CytoANVI: 0.633, kNN: 0.992) with 8× higher cross-seed std for CytoANVI (0.032 vs 0.004) identifies this as training instability, not a systematic defeat. Without a supervised upper bound (full label access, no VAE overhead), readers cannot calibrate whether CytoANVI's supervised advantage is realized anywhere in the result set.

**What to do**

*Step 1 — Add `xgboost_supervised` baseline to `benchmarks/cytoanvi/baselines.py`*

After `harmony_latent_and_knn`, add:
```python
def xgboost_supervised(
    adata, labels_key, unlabeled_category, layer=SCALED_LAYER,
    n_estimators=200, max_depth=6, random_state=0,
):
    """XGBoost trained on labeled cells only — supervised upper bound for B1."""
    import xgboost as xgb
    X = _get_dense(adata, layer)
    labels = np.asarray(adata.obs[labels_key].astype(str))
    labelled = labels != unlabeled_category
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=random_state, verbosity=0,
    )
    clf.fit(X[labelled], labels[labelled])
    unlabeled_mask = ~labelled
    pred_unlabeled = clf.predict(X[unlabeled_mask]) if unlabeled_mask.any() else np.array([], dtype=object)
    return pred_unlabeled, unlabeled_mask
```

*Step 2 — Add `xgboost_supervised` call to `task_b1_label_transfer` in `tasks.py`*

Import `xgboost_supervised` from `.baselines`. After the `harmony_result` block, add a parallel try/except block:
```python
try:
    xgb_pred_unlab, xgb_unlab_mask = xgboost_supervised(work, labels_key, unlabeled_category)
    xgb_full = masked.copy()
    xgb_full[xgb_unlab_mask] = xgb_pred_unlab
    xgb_result = metrics.label_transfer_metrics(true[held], xgb_full[held])
except Exception as e:  # noqa: BLE001
    xgb_result = {"error": str(e)}
```
Add `"xgboost_supervised": xgb_result` to the return dict.

*Step 3 — Add `xgboost` to `pyproject.toml` benchmark optional dependencies*

In the benchmark extras group (which contains `harmonypy`, `scib-metrics`), add `"xgboost>=1.7"`.

*Step 4 — Update `aggregate_results.py` model loop*

Change the model list in `_summarize_single_task` for `"b1"` from:
```python
for model in ("cytoanvi", "cytovi_knn", "raw_marker_knn", "harmony_knn"):
```
to:
```python
for model in ("cytoanvi", "cytovi_knn", "xgboost_supervised", "raw_marker_knn", "harmony_knn"):
```

*Step 5 — Write a diagnostic script for the Nuñez reversal*

Create `benchmarks/cytoanvi/diagnose_nunez_reversal.py`:
```python
"""Diagnose Nuñez B1 reversal: per-class recall stability across seeds."""
import json, numpy as np
from pathlib import Path
data = json.loads(Path(".scratch/cytoanvi-benchmark/results/e1000/nunez_r005_e1000_b1_multiseed.json").read_text())
for seed, per_seed in data["per_seed"].items():
    b1 = per_seed["b1"]
    b_anvi = b1["cytoanvi"]["per_class_recall"].get("B cells", float("nan"))
    b_knn  = b1["cytovi_knn"]["per_class_recall"].get("B cells", float("nan"))
    print(f"seed={seed}: CytoANVI macro_f1={b1['cytoanvi']['macro_f1']:.4f}, B-recall={b_anvi:.3f} | "
          f"kNN macro_f1={b1['cytovi_knn']['macro_f1']:.4f}, B-recall={b_knn:.3f}")
```
Document the B-cell recall collapse in `.living/learnings.md` as a new L-NNN entry: seed-1 classifier head instability, not a systematic defeat.

*Step 6 — Re-run B1 at seeds 0,1,2 with new baseline and restructure the B1 results table*

The table should have four columns in information-hierarchy order:

| Method | Supervision at train time | Nuñez macro-F1 | Roider macro-F1 |
|---|---|---|---|
| Raw-marker kNN | None | — | — |
| CytoVI + kNN | Unsupervised VAE, labels post-hoc | — | — |
| **CytoANVI** | Semi-supervised (labels in ELBO + classifier) | — | — |
| XGBoost | Fully supervised | — | — |

The Nuñez reversal is disclosed as a per-dataset anomaly with mechanistic explanation (B-cell recall collapse, classifier instability) rather than suppressed.

**Done criteria**

1. `task_b1_label_transfer` output contains `"xgboost_supervised"` key with `"macro_f1"` (not `"error"`) when `xgboost` is installed.
2. Unit test in `tests/benchmarks/test_cytoanvi_smoke.py`: `assert "macro_f1" in result["xgboost_supervised"]`.
3. Publication table reports all four methods for both datasets; Nuñez reversal disclosed with mechanistic note.

**Dependencies**: No dependency on Ideas 4 or 5. Roider B1 re-run should be coordinated with B3 re-run (Idea 4) to avoid redundant GPU time.

**Effort**: Low–Medium | **Priority**: High
