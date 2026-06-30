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
