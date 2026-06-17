# 04 — Run A2 batch integration benchmark (Fig 2E)

Status: ready-for-agent
Blocked-by: 01, 03

## Task

Implement `benchmarks/cytovi/tasks_integration.py` + `run.py --task a2`.

**Design (paper-faithful):**
- Data: Nuñez technical replicate (A-D1-batch), ambiguous labels excluded
- Methods: CytoVI, Harmony, **cyCombinePy** ([mdmanurung/cyCombinePy](https://github.com/mdmanurung/cyCombinePy))
- Preprocessing sweep: min-max, z-score, rank
- scib-metrics on latent (CytoVI) or corrected expression (Harmony/cyCombine)
- Subsample 10k cells/batch for scib; `max_epochs=1000` for CytoVI
- Seeds 0, 1, 2

```bash
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytovi.run \
  --task a2 --dataset nunez --max-epochs 1000 --seeds 0,1,2 \
  --out .scratch/cytovi-benchmark/results/a2_nunez.json
```

## Acceptance

- JSON table: 9 rows (3 preproc × CytoVI) + Harmony + cyCombine baselines
- scib `batch_correction`, `bio_conservation`, overall scores reported
- Comparison note vs cytovi-reproducibility if ref outputs checked out

## Comments
