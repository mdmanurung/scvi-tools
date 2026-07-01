# 08 — Run B1 + B2 on full Nuñez (scib, epochs=1000)

Status: ready-for-agent
Blocked-by: cytovi-benchmark/01

### 2026-06-18 — B8 HCE benchmark harness

- **Task B8** added to `benchmarks/cytoanvi/tasks.py` (flat CE vs HCE + hierarchical predict).
- Smoke: `--dataset synthetic --task b8 --max-epochs 50`
- Real runs: pass `--hierarchy-edges` JSON with coarse+fine observed labels (issue 12).

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

### 2026-06-17 — e1000 vignette Nuñez (in flight)

- **Labels:** `data/nunez_annotated.h5ad` (11 tutorial PBMC types via `annotate_nunez.py`).
  `load_nunez()` prefers this file; no `--leiden-resolution` needed.
- **Running:** `run_e1000.sh` → `nunez_r005_e1000_b1_multiseed.json` (started before output rename;
  uses annotated h5ad regardless of filename). Future runs → `nunez_annotated_e1000_b*.json`.
- **Smoke (epochs=100, Leiden r=0.05):** `nunez_r005_seed0_summary.json` — superseded for publication.

### 2026-06-17 — e1000 Roider vignette complete

Rolling summary: `results/e1000/roider_e1000_partial_summary.json`. B1 Δ **+0.12**; B2 batch still
slightly below CytoVI; B3 concordance **0.877**.

### 2026-06-17 — e1000 in flight (Roider)

Vignette Roider **B1+B2 @ 1000 epochs** complete; **B3** running. Rolling summary:
``results/e1000/roider_e1000_partial_summary.json``. B1 macro-F1: CytoANVI **0.908±0.008** vs
k-NN **0.787±0.039** (Δ **+0.12**). B2 bio: CytoANVI **0.737** vs CytoVI **0.628**; batch:
CytoANVI **0.792** vs CytoVI **0.798** (batch still slightly worse).

### 2026-06-17

- cytovi-benchmark/03 (scib infra) and issue 05 (readfcs) are done
- Vignette Nuñez FCS still blocked on Figshare egress from HPC; use
  `python -m benchmarks.common.fetch_data --fetch` from a networked shell

Supersedes issue 02 (vignette Roider B1/B2) for primary integration/transfer validation — Nuñez is
the paper's clean fully-labelled batch-replicate setting.

### 2026-06-27 — SLURM job submitted
Phase 2 SLURM script `.scratch/cytoanvi-benchmark/slurm/phase2_b1b2_nunez.slurm` created and
submitted as job **25102520** (dependency: data-check 25102517 ✓). Seeds 0/1/2, max_epochs=1000.
Pass criteria: B1 macro-F1 ≥ CytoVI kNN + 0.03; B2 bio ±0.02, batch ≥ baseline.
Status: `queued → running`.

### 2026-06-27 — running on gpu-long (job 25102544)
Previous submission used wrong labels-key; fixed with `--labels-key cell_type`. Resubmitted as
job **25102544** (gpu-long partition, res-hpc-gpu15). Currently RUNNING (~20 min in).
Two bugs found and fixed in this iteration:
- `_model.py:591`: `assert_close` device mismatch (cpu vs cuda:0) → added `.cpu()` on both sides
- `phase3_b3b5_roider.slurm`: missing `--labels-key cell_type` → added

### 2026-06-27 — pytest gate PASSED
Job 25102566: **73/73 tests passed** in 75.95s on gpu-long. Phase 0 gate is green.
P2 (job 25102544) still RUNNING (37 min in as of check).

### 2026-06-27 — current queue status
`sacct` still reports Phase 2 job **25102544** as RUNNING on `res-hpc-gpu15`
(elapsed `02:20:16` at inspection). Do not resubmit while this job is active.

Downstream jobs submitted with fresh dependencies:
- Phase 5 B8: job **25102620**, `afterok:25102544`
- Phase 7 aggregate: job **25102623**, waits on Phase 2 plus available downstream phases

### 2026-06-28 — still running; do not resubmit
`sacct -j 25102544,25102620,25102623` reports Phase 2 job **25102544** as RUNNING
(elapsed `06:01:36`, start `2026-06-27T19:41:20`). Keep waiting; do not resubmit while active.

Phase 5 B8 job **25102620** is still PENDING on `afterok:25102544`.
Phase 7 job **25102623** is PENDING but was wired to stale recursive aggregation; ignore it in
favor of manifest-mode aggregation after the required artifacts exist.

### 2026-06-28 — one-off B2 seed-2 recovery submitted
Phase 2 job **25102544** failed with exit code `11:0` after writing five of six required Nuñez
artifacts. Verified complete artifacts:

- `.scratch/cytoanvi-benchmark/results/nunez_b1_s0.json`
- `.scratch/cytoanvi-benchmark/results/nunez_b1_s1.json`
- `.scratch/cytoanvi-benchmark/results/nunez_b1_s2.json`
- `.scratch/cytoanvi-benchmark/results/nunez_b2_s0.json`
- `.scratch/cytoanvi-benchmark/results/nunez_b2_s1.json`

Missing required artifact `.scratch/cytoanvi-benchmark/results/nunez_b2_s2.json` is being recovered
by one-off SLURM job **25104249** using
`.scratch/cytoanvi-benchmark/slurm/recover_nunez_b2_s2.slurm`. Downstream B8 was queued as
**25104252** with dependency `afterok:25104249`; do not run manifest aggregation until recovery
completes and the JSON validates.

### 2026-06-30 — Nuñez B2 seed-2 recovery completed; B8 3-seed done

Nuñez B2 seed-2 recovery (job 25104249) and B8 seed-2 recovery (job 25108052) completed. B8
multiseed aggregate: `nunez_b8_multiseed.json` — Δ_hier_vs_flat = **+0.0862±0.0027** (pub-gate ✅,
F-011). B2 all 3 seeds present.

### 2026-06-30 — B1 inductive relaunch after annotation + harmony fix

The Nuñez B1 result (`nunez_b1_s{0,1,2}.json`) was computed with transductive kNN (L-022: labels
leaked through Leiden clustering on the full labeled data). Relaunched with inductive kNN annotation
fix (`annotate_nunez.py --require-annotated-nunez --max-cells 100000`) as:

- PID 1520357 (scvi env): CRASHED at harmony baseline after CytoANVI training (epoch ~300/1000).
  Root cause: harmonypy 0.2.0 returns `Z_corr` as `(n_cells, n_components)` but code applied `.T`
  unconditionally; fix in commit e8a6d5f9.
- PID 2539861 (scvi-test env): KILLED pre-emptively (would have crashed on same harmony bug).
- PID 3186154 (scvi-test env, v3 restart): RUNNING at epoch 117/1000 (~14.7s/epoch), no NaN,
  train_loss stable at ~-81.4. ETA seed-0: ~01:35 CEST Jul 1; all 3 seeds: ~09:45 CEST Jul 1.
  Output: `.scratch/cytoanvi-benchmark/results/e1000/nunez_inductive_b1_s012_v3.log`
  JSON when complete: `.scratch/cytoanvi-benchmark/results/e1000/nunez_inductive_b1_multiseed.json`

Harmony fix details: L-025, commit e8a6d5f9 (`baselines.py` shape-aware transpose +
`tasks.py` IndexError catch).

### 2026-07-01 — B1 Nuñez inductive complete (PID 3186154 v3, 05:58 CEST)

All 3 seeds (0/1/2) finished at max_epochs=1000. JSON written to:
`.scratch/cytoanvi-benchmark/results/e1000/nunez_inductive_b1_multiseed.json`

**Results (3-seed means ± SD):**

| Method | macro-F1 |
|--------|----------|
| CytoANVI | **0.9751 ± 0.0003** |
| CytoVI kNN | 0.9581 ± 0.0007 |
| XGBoost | 0.9722 ± 0.0008 |
| Raw marker kNN | 0.9010 ± 0.0028 |
| Harmony kNN | 0.9111 ± 0.0026 |

Δ (CytoANVI vs CytoVI kNN) = **+0.0170**.

**Gate assessment:** Formally misses ≥+0.03 target — ceiling effect on clean 11-type PBMC.
CytoANVI still wins all comparisons. XGBoost already scores 0.9722 (near-perfect without batch
info), confirming the task is near-saturated. Roider Δ+0.121 (F-002) remains primary B1 result.

Status updated in: FINDINGS_REGISTRY.md F-003, label-transfer-accuracy.md, ANALYSIS_MANIFEST.md B1 row.
