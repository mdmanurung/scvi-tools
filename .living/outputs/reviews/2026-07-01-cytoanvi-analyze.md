# Review — CytoANVI Implementation Correctness — 2026-07-01

**Scope**: Full feat/cytoanvi implementation review via 5 parallel sub-agents (mycelium:analyze)
**Files reviewed**: benchmarks/cytoanvi/run.py, benchmarks/common/aggregate_results.py,
  benchmarks/cytoanvi/data.py, benchmarks/common/roider_metadata.py,
  src/cytoanvi/_continual.py, src/cytoanvi/_hce.py,
  .scratch/cytoanvi-benchmark/publication_manifest.json
**Sub-agents run**: 5 (core model, continual/HCE, tasks/metrics, data/baselines, orchestration)
**Date**: 2026-07-01

---

## Key decisions validated (no action required)

- **M1+M2 latent with `prior_mixture=False`**: correct — z2 prior is per-label not a mixture, consistent with scANVI design and D-001.
- **encoder_marker_mask registration**: correct — PROTEIN_NAN_MASK gating is applied correctly when `nan_layer` is explicitly provided; auto-detection is unreliable (L-026).
- **scArches surgery**: correct — `load_query_data` → `_set_params_online` path verified clean.
- **B5 FDR computation (tasks.py)**: correct — `scipy.stats.mannwhitneyu` p-values + BH correction present; keys were computed but not surfaced by aggregator (F3 — half-fixed, now fully fixed).
- **B8 leaf_held filter and HCE routing**: correct — leaf_only filter present, `train_cytoanvi` correctly routed for HCE baseline (post session9 fixes).
- **EWC old_params snapshot**: correct — snapped from reference module, control importances computed on batch-extended query.
- **HCE reachability matrix**: correct — `R[i,j]=1` means j is descendant-of-i; `.T` in loss accumulates over descendants (mathematically correct). Docstring is self-consistent.
- **Accelerator**: correct — `accelerator='auto'` (commit d5308022).
- **B3 nan_layer threading**: correct — `kw` dict includes `nan_layer=NAN_LAYER` for B3/B4/B5/B8.

---

## Findings

### F1 — MAJOR: B6 missing `nan_layer` in run.py explicit call
`benchmarks/cytoanvi/run.py:145–156`
```python
results["b6"] = task_mod.task_b6_lambda_sweep(
    p1,
    labels_key=args.labels_key,
    ...
    # MISSING: nan_layer=NAN_LAYER
)
```
**Why it matters**: On roider-full (multi-panel), `encoder_marker_mask` remains None → NaN z_encoder for panel-2-specific markers on the second sweep λ. Silent correctness failure, same root cause as L-026.
**Fix applied**: Added `nan_layer=NAN_LAYER` to explicit kwargs (same session, run.py line 151).
**Status**: FIXED ✓

### F2 — MAJOR: B9 missing `nan_layer` in run.py explicit call
`benchmarks/cytoanvi/run.py:183–197`
```python
results["b9"] = task_mod.task_b9_mapqc(
    p1, ...,
    # MISSING: nan_layer=NAN_LAYER
)
```
**Why it matters**: Same NaN risk as F1 for mapqc query inference path.
**Fix applied**: Added `nan_layer=NAN_LAYER` to explicit kwargs (run.py line 189).
**Status**: FIXED ✓

### F3 — MAJOR: aggregate_results.py resolves manifest artifact paths from CWD
`benchmarks/common/aggregate_results.py:217,227`
```python
expected = {Path(a["path"]).resolve() for a in artifacts}
...
path = Path(artifact["path"])  # resolves from CWD, not manifest file location
```
**Why it matters**: `python benchmarks/common/aggregate_results.py --manifest .scratch/...` works from repo root; `python aggregate_results.py --manifest /abs/path/pub.json` from any other CWD silently fails to find files.
**Fix applied**: Added `_resolve_artifact_path(raw, manifest_dir)` helper; `_manifest_inputs` now accepts `manifest_dir: Path | None`; call site passes `args.manifest.parent`.
**Status**: FIXED ✓

### F4 — MAJOR: Leiden clustering unseeded in `_leiden_labels`
`benchmarks/cytoanvi/data.py:139`
```python
sc.tl.leiden(a, resolution=resolution, flavor="igraph", directed=False)
# Missing: seed=seed
```
**Why it matters**: First-run Leiden labels are non-reproducible across environments. All downstream label-dependent metrics (B1 macro-F1, B5 AUROC) are irreproducible without a cache.
**Fix applied**: Added `seed: int = 0` to `_leiden_labels`; passed to `sc.tl.leiden`; threaded through `load_nunez` and `apply_leiden_cell_types` → `annotate_roider_obs` → `load_roider_full`.
**Status**: FIXED ✓

### F5 — MAJOR: EWC Hadamard product (`combine_type="product"`) silently underflows to 0
`src/cytoanvi/_continual.py:243–244`
```python
if self.combine_type == "product":
    w = w * c   # float32 underflow to 0.0 for many parameters
```
**Why it matters**: Product of two Fisher importance vectors frequently underflows to 0 in float32 for small-importance parameters, silently disabling EWC penalty for those parameters. B6 λ-sweep results may appear robust to λ when they are not.
**Fix applied**: `w = torch.clamp(w * c, min=1e-10)` — prevents silently zeroed weights.
**Status**: FIXED ✓

### F6 — MAJOR: publication_manifest.json superseded B1 entries have `required: true`
`.scratch/cytoanvi-benchmark/publication_manifest.json`
```json
{"status": "superseded", "required": true, ...}
```
**Why it matters**: `_manifest_inputs` raises `ValueError` for `required=true, status!="complete"`. All three superseded Nuñez B1 entries would abort the aggregator.
**Fix applied**: Set all 3 superseded entries to `"required": false`; updated inductive B1 entry from `status: "running"` → `"complete"` (completed 2026-07-01 05:58); updated B3/B5 entries with job IDs 25132400/25132895.
**Status**: FIXED ✓

### F7 — MINOR: B5 FDR keys (`n_fdr_significant`, `mean_auroc_fdr_sig`) computed but not surfaced
`benchmarks/common/aggregate_results.py:91–97`
```python
elif task == "b5":
    out["best_auroc"] = payload.get("best_auroc")
    out["mean_auroc"] = payload.get("mean_auroc")
    # FDR keys present in tasks.py output but missing here
```
**Why it matters**: F3 (AUROC SE fix) was only half complete — the aggregator didn't expose `n_fdr_significant` and `mean_auroc_fdr_sig` from the holdout sweep result.
**Fix applied**: Added both keys to the B5 branch of `_summarize_single_task`.
**Status**: FIXED ✓

---

## What was checked and is fine

- **Core model (Agent 1)**: encoder masking, M1+M2 latent hierarchy, scArches surgery, `prior_mixture=False` — all correct. No major findings.
- **HCE matrix convention (Agent 2)**: R[i,j]=1 (j descendant of i) is consistent between docstring and code. `.T` in loss is intentional math, not a sign error.
- **B5 GPU cleanup (Agent 3)**: `del model + gc.collect() + cuda.empty_cache()` committed in 3575b392 — correct.
- **B3 aggregator backward-compat (Agent 5)**: `p2_inter_method_agreement_vs_knn` fallback to `p2_concordance_vs_knn` confirmed present.
- **B8 leaf_held bias and HCE routing (Agent 3)**: both previously fixed, verified clean.

---

## Fixes committed this session

All 6 code fixes + manifest update committed in one commit (see LOG_REGISTRY session11).

| Fix | File | Status |
|-----|------|--------|
| F1 B6 nan_layer | benchmarks/cytoanvi/run.py | ✓ |
| F2 B9 nan_layer | benchmarks/cytoanvi/run.py | ✓ |
| F3 manifest path | benchmarks/common/aggregate_results.py | ✓ |
| F4 Leiden seed | benchmarks/cytoanvi/data.py, benchmarks/common/roider_metadata.py | ✓ |
| F5 EWC underflow | src/cytoanvi/_continual.py | ✓ |
| F6 manifest required | .scratch/cytoanvi-benchmark/publication_manifest.json | ✓ |
| F7 B5 FDR keys | benchmarks/common/aggregate_results.py | ✓ |
