# 08 — Run B1 + B2 on full Nuñez (scib, epochs=1000)

Status: ready-for-agent
Blocked-by: cytovi-benchmark/01

## Task

On **full** Nuñez batch replicate (B-D2):

- **B1:** 5-fold stratified holdout (20% labels → unlabeled); CytoANVI `predict` vs CytoVI k-NN
- **B2:** scib-metrics on both latents (`benchmarks/common/scib.py`)
- `max_epochs=1000`, seeds 0, 1, 2

```bash
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset nunez --task b1 --max-epochs 1000 --seed 0 \
  --labels-key <col> --batch-key batch --out .scratch/cytoanvi-benchmark/results/nunez_b1_s0.json
```

## Acceptance

- B1: macro-F1 mean ± SD over 3 seeds; pass if ≥ baseline +0.03
- B2: scib aggregates for CytoANVI and CytoVI; bio within ±0.02, batch ≥ baseline

### 2026-06-17

Vignette FCS (17 MB each) in ``data/Nunez_PBMCs_batch{1,2}.fcs``, symlinked into harness.
**Issue 08 B1/B2** running at ``max_epochs=1000`` (background log:
``results/nunez_full_b12.log``). ``readfcs`` installed.

### 2026-06-17

- cytovi-benchmark/03 (scib infra) and issue 05 (readfcs) are done
- Vignette Nuñez FCS still blocked on Figshare egress from HPC; use
  `python -m benchmarks.common.fetch_data --fetch` from a networked shell

Supersedes issue 02 (vignette Roider B1/B2) for primary integration/transfer validation — Nuñez is
the paper's clean fully-labelled batch-replicate setting.
