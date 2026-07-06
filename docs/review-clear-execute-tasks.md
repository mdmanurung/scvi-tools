# Review-Clear-Execute Tasks: CytoANVI Benchmark Recovery

## Completed Local Hygiene Packet

- [x] Read the prior execution packet and relevant CytoANVI domain/ADR/PRD/tracker files.
- [x] Audit current source against `.scratch/cytoanvi-review-fixes.md`.
- [x] Remove remaining duplicate/dead benchmark locals in `benchmarks/cytoanvi/tasks.py`.
- [x] Fix remaining import/type-checking lint in `src/cytoanvi/mapping_qc.py`.
- [x] Run local compile validation for `src/cytoanvi/*` and benchmark modules.
- [x] Run focused ruff validation for `src/cytoanvi`, `benchmarks/cytoanvi`, and `tests/cytoanvi`.
- [x] Run targeted CytoANVI pytest validation; recorded `73/73 passed` in SLURM job `25102566`.
- [x] Run benchmark harness validation; recorded Phase 0 pytest and Phase 1 data-check gates.
- [x] Update `.scratch/cytoanvi-review-fixes.md` with source-confirmed fixes and explicit blockers.
- [x] Record skipped or blocked validation with exact reasons.

## Current Benchmark Recovery Checklist

- [x] Review current state across the handoff, execution queue, next-steps plan, and review-clear-execute documents.
- [x] Verify stale scheduler state: Phase 2 job `25102544` failed; jobs `25102620` and `25102623`
  are pending with `DependencyNeverSatisfied`.
- [x] Verify completed Nuñez artifacts: `nunez_b1_s0.json`, `nunez_b1_s1.json`,
  `nunez_b1_s2.json`, `nunez_b2_s0.json`, and `nunez_b2_s1.json`.
- [x] Verify incomplete/missing required artifacts in `publication_manifest.json`.
- [x] Run preflight at execution start: `git status --short`.
- [x] Run preflight at execution start:
  `sacct -j 25102544,25102620,25102623 --format=JobID,JobName%40,State,ExitCode,Reason,Elapsed,Start,End -P`.
- [x] Run preflight at execution start: `squeue -j 25102620,25102623 -o "%i %j %T %M %R"`.
- [x] Conditionally cancel `25102620` and `25102623` if they are still pending with
  `DependencyNeverSatisfied`.
- [x] Submit a one-off Nuñez recovery SLURM job sourcing `.scratch/cytoanvi-benchmark/slurm/_env.sh`
  and running only B2 seed 2 with `--labels-key cell_type` (job `25104249`).
- [x] Verify the Nuñez recovery job completed with exit code `0:0`.
  Job `25104249` exited `0:0`; confirmed via `sacct`.
- [x] Verify `.scratch/cytoanvi-benchmark/results/nunez_b2_s2.json` exists and matches
  `dataset=nunez`, `task=b2`, `seed=2`.
  Validated 2026-06-29: cytoanvi total=0.7739, cytovi total=0.7747, seed=2.
- [x] Cancel stale pending jobs `25102546`, `25102547`, `25102610` (`DependencyNeverSatisfied`).
  **USER ACTION REQUIRED:** run `scancel 25102546 25102547 25102610` (superseded by current generation).
- [x] Root-cause Roider Phase 3 failure (job `25104250`, exit `11:0`, elapsed 06:56:12).
  Two compounding problems: (1) batch=128 on 1.24M cells → ~9,688 steps/epoch → 52 h/seed, infeasible
  vs 48 h wall; (2) NaN in CytoVI encoder at epoch 94 via `baselines.py::cytovi_latent_and_knn`.
- [x] Plumb `batch_size: int | None = None` through the full training call-graph (2026-06-29).
  Modified: `benchmarks/common/training.py` (train_cytovi + train_cytoanvi),
  `benchmarks/cytoanvi/baselines.py` (cytovi_latent_and_knn — the actual NaN path),
  `benchmarks/cytoanvi/tasks.py` (b1, b2, b3, b5_novelty, b5_holdout_sweep, b8 inline),
  `benchmarks/cytoanvi/run.py` (--batch-size CLI arg, kw spread + B3 explicit).
  Verified: benchmark pytest 20 passed, CytoANVI pytest 72 passed 1 skipped.
- [x] Restructure Phase 3 SLURM scripts: split B3 (3 seeds, 14 h) from B5 sweep (48 h).
  `phase3_b3b5_roider.slurm` → B3-only, `--batch-size 8192`, 14 h wall.
  `phase3b_b5sweep_roider.slurm` → NEW, B5 sweep only, 48 h placeholder (size from smoke test).
  `smoke_b3_roider.slurm` → NEW, 20-epoch timing smoke + Leiden cluster count.
- [ ] **USER ACTION:** Submit smoke test to gate Phase 3 resubmission:
  `sbatch .scratch/cytoanvi-benchmark/slurm/smoke_b3_roider.slurm`.
  Check log for: (a) Leiden cluster count at res=1.0; (b) steady-state epoch time epochs 2–10;
  (c) no NaN. Use cluster_count × epoch_time × 1000 to size phase3b wall time.
- [ ] After smoke test: submit `phase3_b3b5_roider.slurm` (B3 ×3 seeds, 14 h wall).
- [ ] After smoke test: resize and submit `phase3b_b5sweep_roider.slurm` (B5 sweep, sized wall).
- [ ] Verify `.scratch/cytoanvi-benchmark/results/roider_full_b3_s0.json` exists and matches the
  manifest.
- [ ] Verify `.scratch/cytoanvi-benchmark/results/roider_full_b3_s1.json` exists and matches the
  manifest.
- [ ] Verify `.scratch/cytoanvi-benchmark/results/roider_full_b3_s2.json` exists and matches the
  manifest.
- [ ] Verify `.scratch/cytoanvi-benchmark/results/roider_full_b5_sweep_s0.json` exists and matches
  the manifest.
- [x] Submit `.scratch/cytoanvi-benchmark/slurm/phase5_b8_hce.slurm` after Nuñez recovery succeeds.
  Queued as job `25104252` with dependency `afterok:25104249`; dependency resolved, job RUNNING
  since 2026-06-28 22:38 UTC. Seeds are sequential (~8 h each): s0 done (06:50 Jun 29).
- [x] Verify `.scratch/cytoanvi-benchmark/results/nunez_b8_s0.json` exists and matches the
  manifest. Validated 2026-06-29: present, 2.8 K.
- [ ] Verify `.scratch/cytoanvi-benchmark/results/nunez_b8_s1.json` exists and matches the
  manifest. Job still running; s1 expected to complete ~15:02 Jun 29.
- [ ] Verify `.scratch/cytoanvi-benchmark/results/nunez_b8_s2.json` exists and matches the
  manifest. s2 expected ~23:14 Jun 29; 24 h wall ends 22:38 → seed 2 likely killed ~36 min short.
  **After B8 ends:** check if s2 exists; if not, submit per-seed recovery:
  `--dataset nunez --task b8 --labels-key cell_type --max-epochs 1000 --seed 2`.
- [x] Run `.scratch/cytoanvi-benchmark/slurm/phase6_b9_mapqc.slurm` only to record optional B9
  blocked/complete status (job `25104251`).
- [x] Verify `.scratch/cytoanvi-benchmark/results/nunez_b9_s0.json` exists with either
  `b9.status == "blocked"` or a completed mapQC result.
- [x] Update `.scratch/cytoanvi-benchmark/publication_manifest.json` statuses and job IDs from
  actual artifacts only.
- [x] Update `.scratch/cytoanvi-benchmark/PRD.md` with current job IDs, exit codes, artifacts, and
  blockers.
- [x] Update issues 08, 09, 12, and 13 under `.scratch/cytoanvi-benchmark/issues/`.
- [ ] Run manifest-mode aggregation:
  `MPLCONFIGDIR=/tmp NUMBA_CACHE_DIR=/tmp/numba-cache PYTHONPATH=src conda run -n scvi-test python -m benchmarks.common.aggregate_results --manifest .scratch/cytoanvi-benchmark/publication_manifest.json --output .scratch/cytoanvi-benchmark/results/final_summary.json`.
- [ ] Verify `.scratch/cytoanvi-benchmark/results/final_summary.json` has
  `"aggregation_mode": "publication_manifest"`.
- [ ] Verify final summary sources exclude old `roider_*`, synthetic, and recursive `e1000/*`
  outputs.
- [ ] Run focused ruff validation in the `scvi-test` environment if source or benchmark scripts were
  changed. Ran on 2026-06-28; `ruff` is available as an executable but reported 45 lint findings,
  so this remains open.
- [x] Run targeted CytoANVI pytest validation in the `scvi-test` environment if source or benchmark
  scripts were changed. 2026-06-28: `63 passed, 1 skipped`; 2026-06-29 (post batch_size plumbing):
  `72 passed, 1 skipped`.
- [x] Run benchmark pytest validation in the `scvi-test` environment if source or benchmark scripts
  were changed. 2026-06-28: `20 passed`; 2026-06-29 (post batch_size plumbing): `20 passed`.
  Requires `LD_LIBRARY_PATH=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/lib`
  (pre-existing GLIBCXX env issue in subprocess tests, not caused by our changes).
- [ ] Produce a final report listing completed artifacts, failed or blocked artifacts, validation
  results, and residual publication-readiness risk.
