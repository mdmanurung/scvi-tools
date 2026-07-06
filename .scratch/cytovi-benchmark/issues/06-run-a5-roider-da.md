# 06 — Run A5 Roider differential abundance (Fig 4)

Status: ready-for-agent
Blocked-by: 02, 03

## Task

Implement `benchmarks/cytovi/tasks_roider_da.py` on full A-D6.

- Train CytoVI (`max_epochs=1000`, batch covariate)
- DA scores per disease entity vs rest
- k-means on [latent ∥ DA scores]
- ICC between panels for cluster frequencies
- Mann-Whitney cluster freq vs rLN controls

## Acceptance

- ICC ≥ 0.95 (or documented gap vs paper 0.99 with reason)
- Results JSON + optional UMAP artifacts in `.scratch/cytovi-benchmark/results/`

## Comments
