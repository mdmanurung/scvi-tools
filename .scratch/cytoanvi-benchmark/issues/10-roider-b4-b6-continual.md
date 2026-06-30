# 10 — Run B4 + B6 continual update on full Roider (rLN case-control)

Status: needs-info
Blocked-by: cytovi-benchmark/02, cytovi-benchmark/03

## Task

Full Roider now has disease entities — use **rLN (resected control)** as reference/replay and
lymphoma entities as case query.

- Reference: rLN patients (panel 1 labelled)
- Replay buffer: rLN healthy-control cells
- Query: case lymphoma patients (panel 2 or mixed per design)
- Sweep `ewc_importance` ∈ {0, 1, 10, 100, 1000}
- Metrics: replay latent drift (L2 pre/post); DA cluster recovery for known entity-enriched population
- `max_epochs=1000` for reference; query epochs per continual plan defaults

## Acceptance

- Drift vs λ curve JSON
- Documented λ knee as CytoVI-specific default
- B4 pass: low drift + case signal recovered at some λ

## Comments

Unblocks former issue 06 (deferred for lack of case/control on vignette subsample).

### 2026-06-27 — SLURM job submitted
Phase 4 SLURM script `.scratch/cytoanvi-benchmark/slurm/phase4_b4b6_continual.slurm` created and
submitted as job **25102526** (dependency: Roider P3 25102521). ewc_importance sweep over
{0,1,10,100,1000}, seed 0, max_epochs=1000. Pass criteria: drift-vs-λ curve documents knee.
Status: `pending (after P3)`.

### 2026-06-27 — benchmark bug fixed and phase resubmitted
Old Phase 4 job **25102606** FAILED quickly (`1:0`) at B4 drift scoring. Root cause: the benchmark
computed control latent drift by passing query-control cells through the original reference model;
those controls can carry query batch categories absent from the reference registry.

Patch:
- `benchmarks/cytoanvi/tasks.py` now reports `replay_latent_drift` on replay/reference cells for
  both plain surgery and continual update.
- B6 now selects `recommended_lambda` by lowest `replay_latent_drift` and writes
  `recommended_replay_latent_drift`.
- Added regression coverage for unseen query batches in
  `tests/benchmarks/test_cytoanvi_smoke.py`.

Validation:
- Synthetic B4 smoke completed and wrote `/tmp/cytoanvi_b4_smoke.json`.
- Focused tests passed (`2 passed`); broader targeted pytest passed earlier (`54 passed`).

Resubmitted fixed Phase 4 as job **25102622**. Status: PENDING at last check.

### 2026-06-27 — fixed Phase 4 completed
Job **25102622** COMPLETED with exit code `0:0` in `00:14:22` on `res-hpc-gpu14`.

Artifacts:
- `.scratch/cytoanvi-benchmark/results/roider_b4_s0.json`
- `.scratch/cytoanvi-benchmark/results/roider_b6_sweep_s0.json`

B4 seed-0 metrics:
- Plain surgery replay latent drift: `0.0`
- Plain surgery query macro-F1: `0.859`
- Continual update replay latent drift: `0.0`
- Continual update query macro-F1: `0.862`

B6 seed-0 sweep:
- All replay latent drifts are `0.0`, so the current heuristic tie-selects λ=`0.0`.
- Best query macro-F1 in the sweep is λ=`1.0` with macro-F1 `0.888`.
- Recorded recommendation field: `recommended_lambda=0.0`,
  `recommended_replay_latent_drift=0.0`, `recommended_query_macro_f1=0.862`.

Interpretation note: because replay drift is exactly tied at zero across λ values in this pseudo
batch-split run, λ selection is not biologically informative yet. Use this result as plumbing
evidence; retune the λ decision rule on a real case/control split.

### 2026-06-28 — recommendation suppression + roider-full rerun target
The previous B4/B6 artifacts used `--dataset roider` and a pseudo batch split, so they are plumbing
evidence only. The benchmark code now suppresses `recommended_lambda` when replay drift is tied or
non-informative; B6 reports the full λ table plus `recommendation_status: no_recommendation`.

`.scratch/cytoanvi-benchmark/slurm/phase4_b4b6_continual.slurm` now targets `--dataset roider-full`
and writes `roider_full_b4_s0.json` / `roider_full_b6_sweep_s0.json`, but remains a manual
plumbing/smoke script. It is not submitted by `submit_all.sh` and is not publication evidence until
a real rLN reference/replay plus FL/MCL query implementation and metric design exist. These manifest
entries are optional/deferred and missing-tolerant.
