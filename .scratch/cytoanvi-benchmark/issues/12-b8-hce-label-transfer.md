# 12 — B8: flat CE vs HCE label transfer (hierarchy-aware training)

Status: smoke-done; real-data-pending
Blocked-by: none (synthetic smoke); full Nuñez/Roider blocked on annotated labels + hierarchy JSON

## Task

Compare CytoANVI **flat CE** vs **HCE** when a user ontology matches observed model labels
(coarse parent + fine children). Reports holdout macro-F1 for:

- `predict()` after flat-CE training
- `predict()` after HCE training
- `predict_hierarchical(leaf_only=True)` after HCE training

```bash
# Smoke (synthetic default hierarchy)
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset synthetic --task b8 --max-epochs 50 --seed 0 \
  --out .scratch/cytoanvi-benchmark/results/b8_synthetic_s0.json

# Real data: supply parent→children edges JSON (observed labels only)
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset nunez --task b8 --max-epochs 1000 --seed 0 \
  --labels-key <col> --batch-key batch \
  --hierarchy-edges benchmarks/cytoanvi/hierarchy_nunez_tutorial.json \
  --out .scratch/cytoanvi-benchmark/results/nunez_b8_s0.json
```

## Acceptance

- Smoke: task completes; JSON includes `flat_ce`, `hce_flat_predict`, `hce_hierarchical_predict`
- Real (≥3 seeds): document whether HCE improves held-out macro-F1 vs flat CE when coarse types
  are observed labels (not identity reachability)

## Notes

- Default synthetic edges mirror `tests/cytoanvi/test_hce.py` (`label_1` → `label_2/3/4`).
- scHPL-learned trees with sibling-only leaves often yield identity reachability → HCE ≈ flat CE;
  use explicit `set_hierarchy(edges=...)` for meaningful B8 comparisons.

### 2026-06-23 — synthetic smoke complete

`results/b8_synthetic_s0.json` — task completes, all keys present (`flat_ce`, `hce_flat_predict`,
`hce_hierarchical_predict`). Flat CE macro_f1=0.295, HCE=0.166 (Δ −0.129) — expected on tiny
synthetic data at 50 epochs (not converged; identity reachability noted). **Smoke passes.**

Next: real Nuñez B8 run requires `benchmarks/cytoanvi/hierarchy_nunez_tutorial.json` (coarse→fine
edges for the 11 tutorial PBMC types). Blocked until that file is authored.

### 2026-06-27 — hierarchy file confirmed; SLURM job submitted
`benchmarks/cytoanvi/hierarchy_nunez_tutorial.json` exists with Dendritic→Plasmacytoid edges.
Phase 5 SLURM script `.scratch/cytoanvi-benchmark/slurm/phase5_b8_hce.slurm` submitted as job
**25102524** (dependency: Nuñez P2 25102520). Seeds 0/1/2, max_epochs=1000. Status: `pending (after P2)`.

### 2026-06-27 — resubmitted after stale dependency cancellation
Phase 5 was resubmitted with the active Nuñez Phase 2 dependency:

- Job: **25102620**
- Dependency: `afterok:25102544`
- Status at last `sacct`: PENDING

No B8 result should be interpreted until Phase 2 finishes successfully and this job runs.

### 2026-06-28 — still pending on Nuñez P2
`sacct` reports job **25102620** as PENDING while Nuñez P2 job **25102544** remains RUNNING.
No B8 real-data JSON should be considered publication evidence until `nunez_b8_s{0,1,2}.json`
exist and pass manifest aggregation. The aggregation summary now reports
`delta_hierarchical_vs_flat_macro_f1`.

### 2026-06-28 — requeued behind B2 seed-2 recovery
Canceled stale dependency job **25102620** after Nuñez P2 failed. Requeued Phase 5 as SLURM job
**25104252** with dependency `afterok:25104249`, where **25104249** is the one-off Nuñez B2 seed-2
recovery job. Keep B8 real-data status pending until `nunez_b8_s0.json`, `nunez_b8_s1.json`, and
`nunez_b8_s2.json` exist and match the publication manifest.

### 2026-06-29 — cancelled 25107490; resubmitted as 25108052 after second review fixes
Second code review (findings #3 and #8) found two B8 correctness issues in 25107490's code:
- **#3 (leaf_held bias):** `delta_hierarchical_vs_flat_macro_f1` was evaluated over ALL held cells,
  including those with internal-node true labels. `predict_hierarchical(leaf_only=True)` can never
  emit internal labels → those cells always scored wrong, biasing delta against HCE.
  Fix: `leaf_held = held & ~np.isin(true, internal_labels)`; both arms evaluated on leaf-held only.
- **#8 (HCE routing):** HCE arm hand-rolled setup/train instead of routing through `train_cytoanvi`,
  risking silent drift from the flat arm's training config.
  Fix: both arms now go through `train_cytoanvi(hierarchy_edges=...)`.

Job **25107490** cancelled (2:03 elapsed, no result files written). Resubmitted as **25108052**
under fully corrected code. No B8 result from any prior job is publication-valid.
