# 12 — B8: flat CE vs HCE label transfer (hierarchy-aware training)

Status: ready-for-agent
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

- Default synthetic edges mirror `tests/external/cytoanvi/test_hce.py` (`label_1` → `label_2/3/4`).
- scHPL-learned trees with sibling-only leaves often yield identity reachability → HCE ≈ flat CE;
  use explicit `set_hierarchy(edges=...)` for meaningful B8 comparisons.
