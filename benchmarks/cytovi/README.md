# Track A — CYTOVI paper benchmarks

Reproduces quantitative claims from the CytoVI paper against baselines.

Master plan: `notes/2026-06-17-cytovi-cytoanvi-benchmark-plan.md`.

## Optional dependencies

```bash
pip install scib-metrics harmonypy
pip install "git+https://github.com/mdmanurung/cyCombinePy.git"
```

| Package | Used for | Notes |
|---------|----------|-------|
| `scib-metrics` | All integration tasks (A2, B2) | Required |
| `harmonypy` | A2 Harmony baseline | Optional (`--no-harmony`) |
| [cyCombinePy](https://github.com/mdmanurung/cyCombinePy) | A2 cyCombine baseline | Optional (`--no-cycombinepy`). **Batch correction only** — A3 imputation uses CytoVI + KNN. |

## Smoke test (synthetic)

```bash
ENV=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test
PYTHONPATH=src:. LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python \
  -m benchmarks.cytovi.run --dataset synthetic --task a2 --max-epochs 3 \
  --subsample-per-batch 200 --labels-key labels --batch-key batch
```

## A2 — batch integration (Figure 2E)

```bash
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytovi.run \
  --dataset nunez --task a2 --max-epochs 1000 --seeds 0,1,2 \
  --labels-key <col> --batch-key batch \
  --out .scratch/cytovi-benchmark/results/a2_nunez.json
```

Runs CytoVI, Harmony, and cyCombinePy under min-max / z-score / rank preprocessing; reports
scib-metrics aggregates.

## A3 — marker imputation (Figure S4)

CytoVI vs KNN (k=10) only — cyCombinePy has no imputation API.

```bash
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytovi.run \
  --dataset roider --task a3 --max-epochs 1000 --max-cells 50000 \
  --batch-key batch --out .scratch/cytovi-benchmark/results/a3_roider.json
```
