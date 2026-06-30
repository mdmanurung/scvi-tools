# 09 — Run B3 + B5 on full Roider (scib, epochs=1000, holdout sweep)

Status: ready-for-agent
Blocked-by: —

## Task

On **full** Roider cohort (B-D1):

- **B3:** panel-1 reference → panel-2 query; holdout on panel-1 for hard F1; concordance vs k-NN on panel-2
- **B5:** `--holdout-sweep` over all cell types; report best/worst/mean AUROC
- `max_epochs=1000`, seeds 0, 1, 2

Panel-1 ``cell_type`` = **Leiden clusters** (``r=1.0``, cached under ``data/roider_full/``); not manual
gating names.

## Acceptance

- B3: concordance mean ≥ 0.70; holdout F1 ≥ k-NN
- B5: ∃ holdout type with AUROC > 0.70 (or document all types in sweep table)

## Comments

### 2026-06-17 — vignette e1000 B3 complete

`results/e1000/roider_e1000_b3_multiseed.json`: p1 holdout macro-F1 **0.917 ± 0.018**; p2 concordance
**0.877 ± 0.012** (3 seeds, epochs=1000). **Pass** PRD vignette targets. B5 sweep + full cohort still
pending (`cytovi-benchmark/02`).

### 2026-06-17 — vignette prototype done; full cohort pending

Vignette B3+B5 smoke on tutorial `.h5ad` validates harness (`--holdout-type`, B5 sweep, 3-seed B1/B2
with scib). All PRD smoke targets pass or documented (B2: better bio, worse batch). **Full-cohort
run blocked** on `cytovi-benchmark/02` (63-patient ingest). Re-run with `max_epochs=1000` when data
ready. Data helper: `python -m benchmarks.common.fetch_data --list-full-cohort`.

### 2026-06-27 — SLURM job submitted
Phase 3 SLURM script `.scratch/cytoanvi-benchmark/slurm/phase3_b3b5_roider.slurm` created and
submitted as job **25102521** (dependency: data-check 25102517 ✓). Seeds 0/1/2, max_epochs=1000,
B5 holdout sweep enabled. Pass criteria: B3 concordance ≥ 0.70; B5 AUROC > 0.70.
Status: `queued → running`.

### 2026-06-27 — running on gpu-long (job 25102555)
Previous submission failed with `KeyError: 'labels'`; Roider data also uses `cell_type` column.
Fixed with `--labels-key cell_type` in phase3 and phase4 scripts. Resubmitted as job **25102555**
(gpu-long partition, res-hpc-gpu15). Currently RUNNING.

### 2026-06-27 — completed
Phase 3 job **25102555** COMPLETED with exit code `0:0`.

Artifacts:
- `.scratch/cytoanvi-benchmark/results/roider_b3_s0.json`
- `.scratch/cytoanvi-benchmark/results/roider_b3_s1.json`
- `.scratch/cytoanvi-benchmark/results/roider_b3_s2.json`
- `.scratch/cytoanvi-benchmark/results/roider_b5_sweep_s0.json`

Metrics:
- B3 p1 holdout macro-F1: **0.941 +/- 0.012** across seeds 0/1/2
- B3 p2 concordance vs kNN: **0.862 +/- 0.009** across seeds 0/1/2
- B5 seed-0 holdout sweep: best AUROC **0.909**, mean AUROC **0.490**

Status: B3 passes the concordance target (`>= 0.70`). B5 has at least one holdout type above
AUROC `0.70`; detailed per-type interpretation remains in the result JSON.

### 2026-06-28 — relabeled as non-publication evidence; rerun script fixed
The artifacts above came from commands using `--dataset roider`, not `--dataset roider-full`.
Treat them as smoke/provenance only, not publication evidence.

`.scratch/cytoanvi-benchmark/slurm/phase3_b3b5_roider.slurm` now runs `--dataset roider-full` and
writes:

- `.scratch/cytoanvi-benchmark/results/roider_full_b3_s0.json`
- `.scratch/cytoanvi-benchmark/results/roider_full_b3_s1.json`
- `.scratch/cytoanvi-benchmark/results/roider_full_b3_s2.json`
- `.scratch/cytoanvi-benchmark/results/roider_full_b5_sweep_s0.json`

These paths are the required Roider entries in `publication_manifest.json`.

### 2026-06-28 — publication rerun submitted
Submitted `.scratch/cytoanvi-benchmark/slurm/phase3_b3b5_roider.slurm` as SLURM job **25104250**.
The job is expected to write the required `roider_full_*` manifest paths listed above. Until those
files exist and validate, previous `roider_b3_*` and `roider_b5_*` artifacts remain smoke/provenance
only.

### 2026-06-29 — job 25104250 failed; root cause identified and fixed

Job **25104250** exited `11:0`, elapsed **06:56:12**. Two compounding problems:

1. **Wall-time infeasible:** `batch_size` defaulted to scvi's 128. On 1.24 M cells that is ~9,688
   steps/epoch → ~189 s/epoch → ~52.6 h/seed. The script requested 48 h for three B3 seeds *plus*
   the B5 holdout sweep — it cannot complete even one seed.

2. **NaN divergence at epoch 94:** gradient clipping (`_GRAD_CLIP=1.0`) was already in place but
   batch=128 still diverged in the CytoVI encoder:
   `ValueError: Expected parameter loc (Tensor of shape (128,10)) Normal ... found invalid values: nan`.
   The divergence path is `benchmarks/cytoanvi/baselines.py::cytovi_latent_and_knn → train_cytovi`,
   not the `train_cytoanvi` helper.

**Fix implemented (2026-06-29):** plumbed `batch_size: int | None = None` through all four
benchmark files:
- `benchmarks/common/training.py` — `train_cytovi` + `train_cytoanvi` accept and forward.
- `benchmarks/cytoanvi/baselines.py` — `cytovi_latent_and_knn` (the actual NaN path).
- `benchmarks/cytoanvi/tasks.py` — b1, b2, b3, b5_novelty, b5_holdout_sweep, b8 inline.
- `benchmarks/cytoanvi/run.py` — `--batch-size` CLI arg; B3 explicit wiring.

`batch_size=8192` → ~152 steps/epoch (~64× fewer) → projected ~3 s/epoch → ~50 min/seed.
Per-dataset values: roider-full = 8192, nunez = 128 (default; Nuñez artifacts not re-run).
Comparison within dataset remains fair (same batch_size for CytoANVI and CytoVI baseline).

**SLURM scripts restructured:**
- `phase3_b3b5_roider.slurm` → B3-only, `--batch-size 8192`, `--time=14:00:00`.
- `smoke_b3_roider.slurm` → NEW: 20-epoch timing smoke + Leiden cluster count (gates Phase 3b sizing).
- `phase3b_b5sweep_roider.slurm` → NEW: B5 sweep only, 48 h placeholder; resize from smoke test.

**Status:** `smoke_b3_roider.slurm` submitted as SLURM job **25129287** (gpu-long, Priority queue,
2026-06-30 ~20:40 CEST). Awaiting start; once complete, review epoch timing + Leiden cluster
count, then submit `phase3_b3b5_roider.slurm` (B3 full-cohort 3 seeds) and size/submit
`phase3b_b5sweep_roider.slurm` (B5 holdout sweep).

### 2026-06-30 — smoke test RUNNING; Leiden count confirmed

Smoke test job **25129287** started at 20:41:58 CEST on `res-hpc-gpu14` (gpu-long, 8h budget).

Early output from log (cytoanvi_smoke_b3_roider_25129287.log):
- **Leiden clusters at res=1.0: 47**
- Panel shapes: `p1=(620000, 22)`, `p2=(620000, 22)` (1.24M cells total)
- B3 seed-0 training started at batch_size=8192; 20-epoch timing pending (log still updating)

**B5 sweep wall-time estimate** (updated for 47 clusters):
- B5 uses panel-1 only (620k cells); at batch_size=8192: ~76 steps/epoch
- Estimated ~29 min/cluster; 47 clusters × 29 min = ~23h + 1h data = **~24h total** → fits in 48h
- `phase3b_b5sweep_roider.slurm` comment updated with Leiden count and revised estimate

**Next steps once smoke test completes:**
1. Confirm no NaN divergence from smoke log
2. Read actual epoch timing (epochs 2-10) from smoke log to refine B5 estimate
3. Submit `phase3_b3b5_roider.slurm` for B3 full-cohort (3 seeds, ~14h)
4. Submit `phase3b_b5sweep_roider.slurm` for B5 sweep (48h, expect ~24h)
