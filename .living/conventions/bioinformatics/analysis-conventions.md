# Bioinformatics Analysis Conventions

Domain conventions for single-cell and cytometry pipelines. Installed 2026-06-29.

---

## Data handling

### BIO-01 — Validate batch structure before integration metrics
**Rule**: Assert `n_batches > 1` and all batches have ≥10 cells before running any scib integration metric.
**Why**: scib functions (batch_ASW, graph_connectivity) return NaN or error on single-batch inputs without warning. This silently invalidates results.
**How**: Add a pre-check function in benchmark harnesses:
```python
assert adata.obs[batch_key].nunique() > 1, "scib requires >1 batch"
assert adata.obs.groupby(batch_key).size().min() >= 10, "minimum 10 cells per batch required"
```

### BIO-02 — Use per-cell masking for variable-feature datasets
**Rule**: Apply feature masks at the per-cell level (e.g., via `nan_layer`) rather than global per-panel masks when computing per-cell statistics.
**Why**: Global masks inflate or deflate per-cell estimates for cells with partially observed feature sets (e.g., multi-panel cytometry, variable gene detection).
**How**: Index into the per-cell mask tensor for each cell's observed features, not a single shared mask for the full dataset.

### BIO-03 — Distinguish raw and processed data in manifests
**Rule**: Track all datasets in `data/DATA_MANIFEST.md` with an explicit `Status` column: `raw | processed | derived`.
**Why**: Prevents accidentally re-processing already-processed data or confusing raw FCS files with preprocessed h5ad files.
**How**: Update `DATA_MANIFEST.md` whenever a new dataset file is added or a processing step is completed.

---

## Model evaluation

### BIO-04 — Always report results at convergence, not early stopping
**Rule**: Publication benchmarks must run at the intended `max_epochs` on full-cohort data. Smoke/vignette results at reduced epochs are for development only and must be labeled as such.
**Why**: Under-trained models systematically underestimate integration quality and can reverse rankings between methods.
**How**: Use `max_epochs=1000` (or domain-appropriate convergence criterion) for publication. Store smoke results separately in `.scratch/`.

### BIO-05 — Seed everything and report mean ± SD over ≥3 seeds
**Rule**: All quantitative benchmark comparisons must include ≥3 random seeds and report mean ± SD (or IQR) in the results.
**Why**: Single-seed results are not reproducible and cannot be compared across methods.
**How**: Pass `--seed 0 1 2` (or equivalent) and aggregate results with `benchmarks/common/aggregate_results.py`.

### BIO-06 — Label baselines clearly and run them from the same data split
**Rule**: Baseline models must receive the same training/query splits as the comparison model. Document the baseline API method used.
**Why**: Asymmetric data splits or different preprocessing pipelines invalidate comparisons.
**How**: In the benchmark harness, build both model and baseline from the same `adata` and `adata_query` objects.

---

## Model development

### BIO-07 — Persist all inference-time state alongside weights
**Rule**: Any tensor or attribute needed for inference or surgery (that can't be reconstructed from config/weights alone) must be saved in the model checkpoint.
**Why**: Surgery from saved paths fails when inference state was only held in memory during training.
**How**: Register attributes in `save_params` or equivalent serialization hooks. Add a round-trip test.

### BIO-08 — Guard semi-supervised classifiers against all-unlabeled batches
**Rule**: All classifier forward passes must handle `n_labels == 0` (or all cells unlabeled) by returning uniform priors.
**Why**: Softmax over a zero-dimensional output is undefined and produces NaN, which propagates silently through the loss.
**How**: Add an early return before the softmax when `n_labels == 0`. Cover with a unit test.

### BIO-09 — Subsample for Fisher/importance computations on large atlases
**Rule**: Limit Fisher importance matrix computation to ≤10k cells. Log the subsample fraction.
**Why**: Full-atlas Fisher computation OOMs on GPU for datasets >100k cells. The 10k approximation is sufficient for EWC regularization.
**How**: Subsample in the `ContinualUpdate` constructor before Fisher accumulation. Report in logs: `"Fisher computed on {n_sample}/{n_total} cells"`.

---

## Continual learning

### BIO-10 — Document continual update hyperparameters and sweep them
**Rule**: EWC importance λ (`ewc_importance`) must be swept over at least one order of magnitude before setting a default.
**Why**: λ controls the trade-off between plasticity and stability; the right value is dataset-dependent.
**How**: Run B6-style sweep (see `benchmarks/cytoanvi/tasks.py`) and report F1 vs λ curve. Document the chosen default and rationale in CLAUDE.md.
