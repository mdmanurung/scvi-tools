# 03 — Run B3 (panel-divergent mapping) on D1

Status: ready-for-agent
Blocked-by: 01

## Task

Map Roider panel 2 (unlabelled, different antibody panel) onto a panel-1 reference via the
panel-aware path, and transfer labels. This exercises `prepare_query_anndata` + `load_query_data`
+ `predict` on the real backbone/panel-specific split.

```bash
... python -m benchmarks.cytoanvi.run --dataset roider --task b3 \
  --labels-key <col> --batch-key <col> --unlabeled <value> \
  --out .scratch/cytoanvi-benchmark/results/roider_b3.json
```

## Acceptance

- `p1_holdout` accuracy > chance and ≥ the CytoVI k-NN baseline.
- `p2_concordance_vs_knn` ≥ 0.7 (CytoANVI agrees with CytoVI k-NN on the unlabelled panel-2 cells).
- Confirm the real 10-marker backbone is what the encoder uses (check the "Backbone markers" log
  line) and that `prepare_query_anndata` masks the panel-specific markers (no backbone-marker
  rejection).
- Results JSON committed; summary in PRD.

## Comments
