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

### 2026-07-01 — Smoke gate PASSED; B3 + B5 full-cohort RUNNING

Smoke job **25129287** COMPLETED at 22:14:05 CEST Jun 30 (elapsed 1:32:07).

**Smoke timings (batch_size=8192, 1.24M cells):**
- CytoVI adversarial: 7.82 s epoch 1 → **6.55 s/epoch** steady state; 20 epochs done; loss 120→−11.8
- CytoANVI ELBO: 3.54 s epoch 4 → **3.49 s/epoch** steady state; 20 epochs done; loss −24.9→−29.3
- No NaN divergence; 20-epoch concordance 0.641 (at smoke scale; expect higher at 1000 epochs)

**Wall-time projections:**
- B3 (3 seeds, 1000 epochs): CytoANVI 3.49s × 1000 × 3 + CytoVI 6.55s × 1000 × 3 + data ≈ 10–11h → `--time=14:00:00` ✅
- B5 (47 clusters × ~29 min + 1.5h data, seed 0): ≈ 24h → `--time=48:00:00` ✅

**Phase 3 jobs submitted 2026-07-01:**
- Job **25132400** — B3 full-cohort (3 seeds, `phase3_b3b5_roider.slurm`); ETA ~13:00 CEST Jul 1
  - Outputs: `results/roider_full_b3_s{0,1,2}.json`
- Job **25132401** — B5 sweep seed 0 (`phase3b_b5sweep_roider.slurm`); ETA ~06:00 CEST Jul 2
  - Output: `results/roider_full_b5_sweep_s0.json`

**After jobs complete:**
- Check: no NaN, B3 concordance ≥ 0.70, B5 ∃ AUROC > 0.70
- Run manifest-mode aggregation: `python benchmarks/common/aggregate_results.py --manifest .scratch/cytoanvi-benchmark/publication_manifest.json`
- Update FINDINGS_REGISTRY (add F-012+ for roider-full-e1000 B3/B5)

### 2026-07-01 — B5 job 25132401 FAILED (NaN crash); fix applied; resubmitted as 25132895

Job **25132401** failed at 00:57:27 elapsed (exit code 1:0). The first Leiden cluster trained
successfully for 1000 epochs (~57 min). The **second cluster** crashed at epoch 1/1000 step 0 with:

```
ValueError: Expected parameter loc (Tensor of shape (8192, 10)) of distribution Normal ... to satisfy
the constraint Real(), but found invalid values: tensor([[nan, nan, nan, ...], ...])
```

**Root cause:** `encoder_marker_mask` was `None` for the second model, so the encoder received full
data including NaN panel-2-specific marker columns → all-NaN z_encoder output. `encoder_marker_mask`
is None when `PROTEIN_NAN_MASK` is not registered, which happens when `setup_anndata` is called with
`nan_layer=None` and auto-detection (`if nan_layer is None and "_nan_mask" in adata.layers`) fails
across successive model instantiations in the same Python process.

**Fix (commit 3575b392):**
1. Pass `nan_layer=NAN_LAYER` explicitly in `benchmarks/cytoanvi/run.py` `kw` dict — never rely on
   auto-detection for multi-model sweeps (47 sequential instantiations).
2. Add `del model + gc.collect() + cuda.empty_cache()` in `task_b5_novelty` after scoring each
   holdout cluster to prevent GPU memory accumulation across 47 iterations.

See L-026 in `.living/learnings.md` for full gotcha writeup.

**Resubmitted as job 25132895 (RUNNING, 2026-07-01):**
- First cluster training normally; epoch 108/1000, train_loss=−24.9, no NaN, no UserWarning
- GPU: NVIDIA L40S on res-hpc-gpu12
- ETA for all 47 clusters: ~24h from job start

### 2026-07-03 — B3/B5 publication reruns failed; manifest corrected; rerun required

Scheduler state checked on 2026-07-03:

- Job **25132400** (`phase3_b3b5_roider.slurm`) is `FAILED` (`1:0`, elapsed 05:20:06,
  node `res-hpc-gpu14`). It wrote `results/roider_full_b3_s0.json`, then seed 1 failed with
  `Trainer.__init__() got an unexpected keyword argument 'lr'`. The current code routes
  publication learning rate through `plan_kwargs["lr"]`; B3 needs a clean rerun.
- Job **25132895** (`phase3b_b5sweep_roider.slurm`) is `FAILED` (`1:0`, elapsed 00:55:57,
  node `res-hpc-gpu12`). It hit the old `Normal(..., sqrt(pz1_v))` scale failure in CytoANVI
  loss. The current module clamps tiny variances and raises targeted errors for non-finite or
  negative variance; B5 needs a clean rerun.

`publication_manifest.json` now marks the required roider-full B3/B5 artifacts as `failed`.
Manifest-mode aggregation should therefore stop until new complete artifacts replace these entries.
Rerun scripts should use `--cytoanvi-recipe publication` so empirical label priors, class weighting,
LR plateau scheduling, `lr=5e-4`, and `gradient_clip_val=1.0` are applied consistently.

### 2026-07-03 — Publication reruns resubmitted

After focused tests and a synthetic publication-recipe smoke passed, the rerun scripts were submitted:

- **B3** `phase3_b3b5_roider.slurm` → job **25140597**, `RUNNING` on `res-hpc-gpu11`,
  time limit 14:00:00, output
  `.scratch/cytoanvi-benchmark/slurm/out/cytoanvi_p3a_roider_b3_25140597.log`.
- **B5** `phase3b_b5sweep_roider.slurm` → job **25140598**, `RUNNING` on `res-hpc-gpu11`,
  time limit 48:00:00, output
  `.scratch/cytoanvi-benchmark/slurm/out/cytoanvi_p3b_roider_b5sweep_25140598.log`.

The manifest now records these jobs as `running`. Aggregation remains gated until both jobs finish
and the required artifacts are marked `complete`.
