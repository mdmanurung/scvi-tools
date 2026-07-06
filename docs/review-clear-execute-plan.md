# Review-Clear-Execute Plan: CytoANVI Benchmark Recovery

## Objective

Make the CytoANVI publication-readiness benchmark state unambiguous, recover missing required
benchmark artifacts, and regenerate the final summary only through manifest-mode aggregation.

## Current State Reviewed

- `docs/handoffs/2026-06-27-cytoanvi-review-clear-execute.md`
- `docs/plans/2026-06-27-cytoanvi-execution-queue.md`
- `docs/plans/2026-06-28-cytoanvi-next-steps.md`
- `docs/review-clear-execute-plan.md`
- `docs/review-clear-execute-tasks.md`
- `.scratch/cytoanvi-benchmark/publication_manifest.json`
- `.scratch/cytoanvi-benchmark/slurm/phase2_b1b2_nunez.slurm`
- `.scratch/cytoanvi-benchmark/slurm/phase3_b3b5_roider.slurm`
- `.scratch/cytoanvi-benchmark/slurm/phase5_b8_hce.slurm`
- `.scratch/cytoanvi-benchmark/slurm/phase6_b9_mapqc.slurm`
- `.scratch/cytoanvi-benchmark/slurm/phase7_aggregate.slurm`

Live state on 2026-06-28:

- Completed: local code-review hygiene packet; Nuñez B1 seeds 0/1/2; Nuñez B2 seeds 0/1.
- Incomplete: Nuñez B2 seed 2; Roider `roider-full` Phase 3 outputs; Nuñez B8 outputs; optional
  B9 blocked marker; manifest-mode final aggregation.
- Stale: `25102620` and `25102623` are pending with `DependencyNeverSatisfied`; the existing
  `final_summary.json` is exploratory recursive aggregation and not publication evidence.
- Risk: `publication_manifest.json` currently marks missing required artifacts as `complete`; treat
  it as a target contract, not proof of completion.

## Frozen Execution Plan

1. Preflight the worktree, scheduler, and artifact state.
   - Run `git status --short` and stop if files needed by this plan changed unexpectedly.
   - Run `sacct` and `squeue` for `25102544`, `25102620`, and `25102623`.
   - Check every required path in `.scratch/cytoanvi-benchmark/publication_manifest.json`.

2. Clean up stale scheduler state without touching result artifacts.
   - If `25102620` and `25102623` are still pending because of `DependencyNeverSatisfied`, cancel
     those stale jobs.
   - Do not use `.scratch/cytoanvi-benchmark/slurm/submit_all.sh` for recovery.

3. Recover the missing Nuñez Phase 2 artifact.
   - Submit a one-off SLURM recovery job that sources `.scratch/cytoanvi-benchmark/slurm/_env.sh`.
   - Run only B2 seed 2 with `--dataset nunez --task b2 --labels-key cell_type --max-epochs 1000
     --seed 2 --out .scratch/cytoanvi-benchmark/results/nunez_b2_s2.json`.
   - Record the new job ID and require `COMPLETED 0:0` before downstream B8 or aggregation.

4. Regenerate Roider Phase 3 publication artifacts.
   - Submit `.scratch/cytoanvi-benchmark/slurm/phase3_b3b5_roider.slurm`.
   - Require `roider_full_b3_s0.json`, `roider_full_b3_s1.json`, `roider_full_b3_s2.json`, and
     `roider_full_b5_sweep_s0.json`.
   - Treat older `roider_*` outputs as smoke/provenance only.

5. Run Nuñez B8 only after Nuñez Phase 2 recovery succeeds.
   - Submit `.scratch/cytoanvi-benchmark/slurm/phase5_b8_hce.slurm`.
   - Require `nunez_b8_s0.json`, `nunez_b8_s1.json`, and `nunez_b8_s2.json`.

6. Record optional B9 state.
   - Run `.scratch/cytoanvi-benchmark/slurm/phase6_b9_mapqc.slurm` only to produce
     `nunez_b9_s0.json` with either `b9.status == "blocked"` or a completed mapQC result.
   - Do not install `mapqc` inside the queue.

7. Reconcile manifest, PRD, and issue files from evidence.
   - Update manifest statuses and job IDs only after matching files exist and validate.
   - Update `.scratch/cytoanvi-benchmark/PRD.md` and issues 08, 09, 12, and 13 with actual job IDs,
     exit codes, artifact paths, and blockers.

8. Run publication aggregation.
   - Run manifest-mode aggregation only after all required manifest paths exist.
   - Confirm the final summary has `"aggregation_mode": "publication_manifest"` and excludes old
     `roider_*`, synthetic, and recursive `e1000/*` sources.

## Validation Commands

- `git status --short`
- `sacct -j 25102544,25102620,25102623 --format=JobID,JobName%40,State,ExitCode,Reason,Elapsed,Start,End -P`
- `squeue -j 25102620,25102623 -o "%i %j %T %M %R"`
- `MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONPATH=src conda run -n scvi-test python -m ruff check src/cytoanvi benchmarks/cytoanvi tests/cytoanvi`
- `MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONPATH=src conda run -n scvi-test python -m pytest tests/cytoanvi/test_cytoanvi.py tests/cytoanvi/test_hce.py tests/cytoanvi/test_mapping_qc_mock.py -q`
- `MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONPATH=src conda run -n scvi-test python -m pytest tests/benchmarks -q`
- `MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONPATH=src conda run -n scvi-test python -m benchmarks.common.aggregate_results --manifest .scratch/cytoanvi-benchmark/publication_manifest.json --output .scratch/cytoanvi-benchmark/results/final_summary.json`

## Constraints and Non-Goals

- Do not use CodeRabbit or external review tooling.
- Do not fetch external data, use credentials, or install dependencies inside SLURM jobs.
- Do not use `submit_all.sh` for this recovery.
- Do not treat old `roider_*`, synthetic, e1000, or recursive aggregation outputs as publication
  evidence.
- Do not rewrite unrelated dirty worktree changes.
