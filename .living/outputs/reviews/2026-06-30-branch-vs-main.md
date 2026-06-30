# Review — feat/cytoanvi vs main — 2026-06-30

**Scope**: Branch `feat/cytoanvi` vs `main` (full diff including 4 commits from the autonomous session: `e579011b`, `16f9347c`, `a032b313`, `c9c0d1b4`, plus upstream commits `45de72af`…`173cc842`)
**Files reviewed**: ~35 new/modified files across `src/cytoanvi/`, `benchmarks/cytoanvi/`, `benchmarks/common/`, `benchmarks/cytovi/`, `src/scvi/external/cytovi/`, `tests/cytoanvi/`
**Sub-agents run**: 0 (inline — spend-limit block prevented sub-agents; all six checklists applied in-line)

---

## Key decisions in this analysis

- **M1+M2 semi-supervised hierarchy over GMM** — CytoANVAE uses scANVI-style z1→classifier + z1→z2 hierarchy, with `prior_mixture=False` forced (see ADR-0001). This is the fundamental architecture choice.
- **EWC + replay for continual update** — `load_query_data_with_replay` wires EWC penalty into manual-optimization training plan; `ewc_importance` is a train-time argument, not a constructor arg (see ADR-0002). λ must be swept (B6) per dataset.
- **HCE gated by reachability matrix** — flat CE used when `reachability_matrix_ is None`; hierarchical CE only when `set_hierarchy` was called (see ADR-0003). Never silent fallback.
- **B3 holdout drawn on p1 cells only** — `_holdout()` now called on p1 cells before merging with p2, so the holdout draw is independent of p2 panel size. This was a fix from `16f9347c`.
- **B5 uncertainty mode is transductive** — the model is trained excluding the novel type, then uncertainty is scored on all cells including novel ones. Tagged `"b5_evaluation_mode": "calibration_transductive"` in output. No separate OOD validation set.
- **KNN imputation inductive** — `KNNImputer.fit(reference).transform(holdout)` pattern (fixed in `a032b313`). Prior code used `fit_transform(all)` (transductive leakage).
- **B5 per-type AUROC tested with BH FDR using normal approximation** — AUROC → z-score → one-sided p-value → BH correction. This is the right framework but the SE formula is incorrect (see F1 — MAJOR).

---

## Questions for the analyst

1. **B5 clinical bar**: What AUROC threshold constitutes "detectable novelty" for a cytometry atlas? The current code uses BH FDR q<0.05 but doesn't set a minimum AUROC floor (e.g., AUROC≥0.7). Should both FDR significance AND minimum effect size be required to count a type as detectable?

2. **B4/B6 biology split**: Is there a plan to replace the batch-split proxy with a genuine case-control split (e.g., Nuñez disease/healthy) before publication? The `note: "pseudo case/control via batch split — validates plumbing, not biology"` is honest, but reviewers will ask.

3. **scennep O(N) loop**: For 50k-cell imputation benchmarks, `_pseudobulk_expression`'s Python loop over cells is likely the bottleneck. Is runtime acceptable at max_epochs=1000, or should this be vectorized before the full benchmark runs?

4. **B9 mapQC dependency**: `mapqc` is currently an optional extra that triggers a graceful `blocked` return when absent. Is `mapqc` installable in the SLURM environment where full benchmarks run? If not, B9 will silently output `status: blocked` for all publication seeds.

5. **Figshare archive**: The `FIGSHARE` constant in `data.py` is constructed from `_VIGNETTE`. Are the Figshare IDs pointing to the correct, publicly accessible versions of the Nuñez and Roider datasets? (The dev environment returned HTTP 202 / 0-byte responses during earlier debugging.)

---

## Findings

### Statistics & causal inference

#### Major

##### F1. B5 AUROC standard error missing `(n1 + n2 + 1)` numerator

`benchmarks/cytoanvi/tasks.py:1667`

```python
se = np.sqrt(1.0 / np.maximum(12 * n_novel_arr * n_ref_arr, 1))
z = (np.array(aurocs) - 0.5) / se
p_vals = norm.sf(z)
```

**Why it matters here**: Under the Mann-Whitney U / Wilcoxon rank-sum formulation, `Var(AUROC) = (n1 + n2 + 1) / (12 * n1 * n2)`, so `SE(AUROC) = sqrt((n1+n2+1) / (12*n1*n2))`. The code omits the `(n1+n2+1)` numerator, making the SE `sqrt(n1+n2+1)` times too small and the z-score the same factor too large. For a typical holdout type with n_novel=500 and n_ref=5000, the z-score is inflated by ≈√5501 ≈ 74×, making essentially every AUROC above 0.5 trivially significant. The `n_fdr_significant` and `mean_auroc_fdr_sig` fields in the output will be wrong, and any publication figure derived from them will overclaim.

**Fix**:
```python
se = np.sqrt((n_novel_arr + n_ref_arr + 1) / np.maximum(12 * n_novel_arr * n_ref_arr, 1))
```

#### Minor

##### F2. `test_hierarchy_schpl_mock.py:53` — unseeded `np.random.choice`

`tests/cytoanvi/test_hierarchy_schpl_mock.py:53`

```python
adata.obs[SAMPLE_KEY] = np.random.choice(["group_a", "group_b"], size=adata.shape[0])
```

**Why it matters here**: The P1 round fixed this in `test_hce.py:33` (changed to `np.random.default_rng(42).choice(...)`), but `test_hierarchy_schpl_mock.py` was not updated. The two files have the same `_make_adata` helper pattern. While the `sample_key` assignment here doesn't affect test pass/fail (it's only registered, not sampled over in assertions), it's inconsistent and leaves a global RNG mutation that can affect subsequent tests in the same process.

**Fix**: Replace with `np.random.default_rng(42).choice(["group_a", "group_b"], size=adata.shape[0])`.

---

### Data pipeline & leakage

#### Minor

##### F3. `tasks_imputation.py` redefines `NAN_LAYER` locally

`benchmarks/cytovi/tasks_imputation.py:2277`

```python
from benchmarks.common.training import SCALED_LAYER, train_cytovi
...
NAN_LAYER = "_nan_mask"
MASK_BATCH = "pseudo_1"
```

**Why it matters here**: `SCALED_LAYER` is correctly imported from `benchmarks.common.training`, but `NAN_LAYER` is redefined as a local constant with the same string `"_nan_mask"`. If the canonical definition in `benchmarks.common.training` ever changes (e.g., a rename for multi-modal compatibility), this local copy will silently diverge and produce a key-not-found error at runtime rather than a clean import error.

**Fix**: `from benchmarks.common.training import NAN_LAYER, SCALED_LAYER, train_cytovi` and remove the local `NAN_LAYER = "_nan_mask"` line.

##### F4. B9 pseudo-sample assignment is round-robin, not seeded-random

`benchmarks/cytoanvi/tasks.py:2127–2130`

```python
rng = np.random.default_rng(seed)
ref_samples = [f"ref_s{i}" for i in range(4)]
...
sample_col[ref_idx] = np.take(ref_samples, np.arange(len(ref_idx)) % len(ref_samples))
```

**Why it matters here**: `rng` is instantiated with `seed` but the `sample_col` assignment uses `np.arange(...) % len(ref_samples)` — a deterministic round-robin based on cell order in `adata`, not on `rng`. The seed therefore has no effect on which cells get which sample labels. For a plumbing test this is harmless, but if B9 is ever promoted to a biological split the seed will not reproduce the intended randomization. Also: `rng` is currently only consumed for the status_choices, so the seed controls only case/control assignment, not sample-ID assignment.

**Fix** (if random is desired): `sample_col[ref_idx] = np.take(ref_samples, rng.integers(0, len(ref_samples), size=len(ref_idx)))`. Or document that round-robin assignment is intentional.

---

### Bioinformatics

#### Minor

##### F5. `DEFAULT_PROTEIN_GENE_MAP` has two keys mapping to `HLA-DRA`; inverse silently drops one

`src/scvi/external/cytovi/marker_harmonization.py:20–21`

```python
"HLADR": "HLA-DRA",
"HLA-DR": "HLA-DRA",
```

**Why it matters here**: `DEFAULT_GENE_TO_PROTEIN` is built by inverting `DEFAULT_PROTEIN_GENE_MAP`. Both `"HLADR"` and `"HLA-DR"` map to `"HLA-DRA"`, so the inverse dict has exactly one entry for `"HLA-DRA"` — whichever protein alias appears last in dict iteration order (`"HLA-DR"` in Python 3.7+ insertion order). If a cytometry panel uses `"HLADR"` as the column name (as some older CyTOF panels do), `rename_rna_to_protein_names` would correctly map `HLA-DRA` → `"HLA-DR"` (not `"HLADR"`), causing a downstream var_names mismatch against the cytometry column. This is a silent asymmetry.

**Fix**: Pick one canonical protein name (prefer `"HLA-DR"`, the standard), remove `"HLADR"`, and document the alias in a comment.

##### F6. B5 calibration is transductive — adequately flagged but publication consequence understated

`benchmarks/cytoanvi/tasks.py` (B5 task, `b5_evaluation_mode`)

**Why it matters here**: The AUROC for each holdout cell type is computed after training a model that saw the complement of that type. The uncertainty threshold (implicitly at AUROC > 0.5) is not calibrated on a held-out validation cohort. For a benchmark claiming to detect novel cell states, reviewers will ask for a calibration curve on truly OOD data. The `"b5_evaluation_mode": "calibration_transductive"` tag correctly flags this, but the implication — that the AUROC thresholds cannot be transferred to new datasets — should be stated in the paper's limitations.

**Fix**: Add a `note` to the return dict: `"calibration_note": "AUROC thresholds are not validated on external OOD data; treat as within-dataset diagnostic, not deployable threshold."` And ensure the user guide section mentions this.

---

### LLM coding antipatterns

#### Minor

##### F7. `task_b6_lambda_sweep` output does not expose `_fallback_split`

`benchmarks/cytoanvi/tasks.py:2072–2102`

```python
out = {
    "task": "b6_lambda_sweep",
    "seed": seed,
    "max_epochs": max_epochs,
    ...
    "per_lambda": per_lambda,
}
```

**Why it matters here**: When `_b4_setup` triggers the random 70/30 fallback split (emitting a `UserWarning`), the `task_b4_continual` output propagates `_fallback_split: True` in each per-lambda sub-dict under `"continual_update"`. But `task_b6_lambda_sweep` only stores `task_b4_continual(...)["continual_update"]` (which doesn't include `_fallback_split`). A downstream caller parsing B6 JSON can't determine if results came from a valid batch split or a random fallback without re-running `_b4_setup`. This matters because `aggregate_results.py` would silently treat a fallback-split B6 result as equally valid as a real one.

**Fix**: Add `"_fallback_split": setup["_fallback_split"]` to the B6 output dict so it's preserved and auditable.

---

### Documentation & schema fidelity

#### Minor

##### F8. B5 FDR formula in docstring matches code but code formula is wrong

`benchmarks/cytoanvi/tasks.py:1656–1658`

```python
# BH FDR over per-type AUROCs: normal approximation z = (AUROC - 0.5) / se,
# se = sqrt(1 / (12 * n_novel * n_ref)).
```

**Why it matters here**: The docstring accurately describes what the code does, but the formula documented is the erroneous one (see F1). When F1 is fixed, this comment must be updated too, otherwise future readers will apply the wrong formula thinking it's intentional.

**Fix**: After fixing F1, update to: `# se = sqrt((n_novel + n_ref + 1) / (12 * n_novel * n_ref))  # Wilcoxon AUC SE`.

---

### Code quality

#### Minor

##### F9. `_pseudobulk_expression` O(n_cells) Python loop is the bottleneck for 50k-cell benchmarks

`src/scvi/external/cytovi/scennep.py:2996–3018`

```python
for cell_id in range(n_cells):
    row = snn_graph.getrow(cell_id)
    ...
    out[cell_id] = (neighbor_expr * weights[:, None]).sum(axis=0) / weights.sum()
```

**Why it matters here**: In `task_a3_imputation`, `max_cells=50_000` and the loop runs for every marker × every cell. `snn_graph.getrow(cell_id)` is a Python-level sparse row extraction that generates a new sparse matrix object per call. For 50k cells × 30 markers = 1.5M iterations, this will dominate benchmark wall-time. The alternative is vectorized sparse matmul: `out = (snn_graph.toarray() * weights_matrix).dot(expr) / row_sums`.

**Fix**: Vectorize using `snn_graph @ expr` with normalized weights, or at minimum use `snn_graph.tocsr().indices` / `.data` arrays with `np.add.at` for batch aggregation.

---

## What was checked but is fine

- **Statistics & causal inference**: B3 holdout RNG independence fix is correct (draws on p1 cells only). B8 `leaf_held` metric is correctly restricted to leaf-held cells on both sides of the delta comparison. B6 lambda recommendation logic (`np.isclose` + NaN filtering) is correct.
- **Data pipeline & leakage**: A3 KNN imputation is now correctly inductive (`fit(reference).transform(holdout)`). B4 `query_adata.copy()` inside surgery calls correctly prevents shared-state mutation across lambda iterations. scennep saves `orig_expr` before z-scoring so pseudobulking uses unscaled values.
- **Bioinformatics**: `collapse_cd45_markers` docstring warning is adequate. `prior_mixture=False` is correctly forced in `CytoANVAE.__init__` with a clear inline comment pointing to the ADR. `ewc_importance` sentinel-None pattern is correct — raises if `continual is not None` but ewc_importance wasn't supplied.
- **LLM coding antipatterns**: `try/except ImportError` in B9 mapQC is correctly scoped (not bare `except`). `conftest.py` `MockTreeNode` is correct and `ancestor` pointer is set in `__init__`. `ScalarNameTreeNode` correctly inherits `get_leaves()` from `MockTreeNode`.
- **Documentation & schema fidelity**: `_split_reference_query` 3-tuple return with `fallback_used` is documented in both the function docstring and the downstream caller's comment. `collapse_cd45_markers` `.. warning::` is properly formed RST. B4 output `note: "Pseudo case/control via batch split"` is honest.
- **Code quality**: Harmony seed is correctly forwarded to both PCA (`random_state=seed`) and Harmony (`random_state=seed`). `TTA generator` param (`torch.Generator | None = None`) is correctly threaded through `mask_augment` and `compute_uncertainty_scores`. `NAN_LAYER` / `SCALED_LAYER` / `FIGSHARE` dedup in `benchmarks/cytoanvi/data.py` is correct.

---

## Notes

- **F1 (AUROC SE) is the only finding that would invalidate a publication figure.** All B5 FDR significance calls and `n_fdr_significant` values in existing result JSONs need to be regenerated after the fix. The `mean_auroc_fdr_sig` field is also affected.
- **F2 + F8** are cleanup tasks that pair together: fix the unseeded RNG and update the SE comment at the same time as the F1 fix.
- **F3 + F7** are output-schema hygiene items: add `NAN_LAYER` import and add `_fallback_split` to B6 output. Neither affects existing result correctness.
- **Commits `e579011b` and `16f9347c`** (the autonomous session's P1/P2 work) are correct except for F2 (unseeded test) which was missed. The `a032b313` data-pipeline fix (KNN inductive) is correct.
- The scennep implementation (`F9`) is functionally correct but may need vectorization before the imputation benchmark (A3) runs at scale. Worth profiling on a 50k-cell run first.
