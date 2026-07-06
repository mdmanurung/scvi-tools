# 03 — Shared benchmark infrastructure (scib + training defaults)

Status: ready-for-human
Blocked-by: —

## Task

Create `benchmarks/common/`:

1. **`scib.py`** — `run_scib_benchmark(adata, latent_key, batch_key, label_key, subsample_per_batch=10000, seed)`
   using `scib_metrics.benchmark.Benchmarker`; return dict with per-metric + aggregates.
2. **`training.py`** — `train_cytovi(adata, ..., max_epochs=1000)`, `train_cytoanvi(...)` with paper
   defaults (`n_latent=None` → heuristic, MoG prior, Gaussian likelihood).
3. **`preprocessing.py`** — `arcsinh_scale(adata, cofactor, layer_out="scaled")` per dataset table in
   master plan.
4. **`seeds.py`** — `run_multiseed(fn, seeds=[0,1,2])` → mean ± SD JSON summary.

Update `benchmarks/cytoanvi/metrics.py`: remove `batch_mixing` / `bio_conservation`; B2 calls
`common.scib`. Default `--max-epochs` in `run.py` → **1000**.

## Acceptance

- `python -m benchmarks.cytoanvi.run --dataset synthetic --task b2 --max-epochs 3` still passes (smoke)
- Unit test on tiny synthetic data: scib returns numeric aggregates
- README documents `scib-gpu` env requirement

## Comments

### 2026-06-17

Implemented in `benchmarks/common/` (`scib.py`, `training.py`, `seeds.py`, `preprocessing.py`).
CytoANVI B2 now calls `run_scib_benchmark`. Vignette results in `cytoanvi-benchmark/results/`.
Unit test still TODO.

### 2026-06-17 — implemented

- `benchmarks/common/{scib,training,seeds,preprocessing}.py` in place
- B2 uses `run_scib_benchmark` (aggregates: `batch_correction`, `bio_conservation`, `total`)
- Synthetic smoke: `python -m benchmarks.cytoanvi.run --dataset synthetic --task b2 --max-epochs 3`
- Unit test: `tests/benchmarks/test_common_scib.py` (4 passed)
- Default `--max-epochs` → 1000 in `run.py`
