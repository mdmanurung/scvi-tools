# 05 — Run A3 semi-synthetic imputation benchmark (Fig S4)

Status: ready-for-agent
Blocked-by: 01, 03

## Task

Implement `benchmarks/cytovi/tasks_imputation.py`.

- 50k cells, 2 pseudo-batches, mask-one-marker loop
- 50 posterior samples per cell
- Compare CytoVI vs KNN (k=10) vs cyCombine
- Per-marker Pearson/Spearman + uncertainty–error correlation
- `max_epochs=1000`

## Acceptance

- Per-marker correlation CSV committed under `.scratch/cytovi-benchmark/results/`
- Mean Pearson documented vs paper/reference tolerance (±0.05)

## Comments
