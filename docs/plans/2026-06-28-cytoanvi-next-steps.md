# CytoANVI Next Steps

Date: 2026-06-28
Project: /exports/para-lipg-hpc/mdmanurung/scvi-tools
Status: reviewed; active recovery checklist lives in `docs/review-clear-execute-tasks.md`

## Objective

Recover the CytoANVI publication-readiness benchmark queue from stale dependencies and missing
publication artifacts, then run manifest-mode aggregation only after required evidence exists.

## Plan

1. Preflight live state before any queue changes: run `git status --short`, `sacct`/`squeue` for
   stale jobs `25102544`, `25102620`, and `25102623`, and verify every required
   `publication_manifest.json` path exists or is explicitly missing.
2. Conditionally cancel stale pending jobs `25102620` and `25102623` only if they are still pending
   because of `DependencyNeverSatisfied`.
3. Submit a narrow recovery job for only the missing Nuñez artifact, using the benchmark environment
   from `.scratch/cytoanvi-benchmark/slurm/_env.sh` and the required labels key:
   `--dataset nunez --task b2 --labels-key cell_type --seed 2 --max-epochs 1000 --out .scratch/cytoanvi-benchmark/results/nunez_b2_s2.json`.
4. Rerun fixed Phase 3 for `roider-full` so the required publication artifacts exist:
   `roider_full_b3_s0.json`, `roider_full_b3_s1.json`, `roider_full_b3_s2.json`, and
   `roider_full_b5_sweep_s0.json`.
5. After Nuñez recovery succeeds, rerun Phase 5 B8 to create `nunez_b8_s{0,1,2}.json`.
6. Run Phase 6 only to record the optional B9 blocked artifact unless `mapqc` is provisioned in the
   benchmark environment.
7. Update `.scratch/cytoanvi-benchmark/publication_manifest.json` statuses/job IDs only after the
   corresponding JSON files exist and validate against the manifest.
8. Run manifest-mode aggregation with `.scratch/cytoanvi-benchmark/publication_manifest.json`.
9. Update `.scratch/cytoanvi-benchmark/PRD.md` and issue statuses to reflect actual scheduler and
   artifact state.

## Validation

- `git status --short`
- `sacct -j <new_job_ids> --format=JobID,JobName%40,State,ExitCode,Elapsed,Start,End -P`
- Verify all required manifest paths exist before aggregation.
- Verify optional B9 is either `status: blocked` in `nunez_b9_s0.json` or complete with mapQC
  installed.
- `MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONPATH=src conda run -n scvi-test python -m ruff check src/cytoanvi benchmarks/cytoanvi tests/cytoanvi`
- `MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONPATH=src conda run -n scvi-test python -m pytest tests/cytoanvi/test_cytoanvi.py tests/cytoanvi/test_hce.py tests/cytoanvi/test_mapping_qc_mock.py -q`
- `MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONPATH=src conda run -n scvi-test python -m pytest tests/benchmarks -q`
- `MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONPATH=src conda run -n scvi-test python -m benchmarks.common.aggregate_results --manifest .scratch/cytoanvi-benchmark/publication_manifest.json --output .scratch/cytoanvi-benchmark/results/final_summary.json`
- Confirm `.scratch/cytoanvi-benchmark/results/final_summary.json` has
  `"aggregation_mode": "publication_manifest"` and excludes old `roider_*`, synthetic, and
  recursive `e1000/*` sources.

## Notes

- Keep review local; do not use CodeRabbit.
- Treat old `roider` outputs as smoke/provenance only, not publication evidence.
- Treat B4/B6 and B9 as optional or deferred, not publication evidence.
- Keep the accepted top-level package decision: `cytoanvi.CytoANVI`, not
  `scvi.external.CytoANVI`.
