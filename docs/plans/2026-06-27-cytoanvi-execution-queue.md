# CytoANVI Publication Execution Queue

Date: 2026-06-27
Project: /exports/para-lipg-hpc/mdmanurung/scvi-tools
Status: in-progress

## Current Status Reconciliation (2026-06-28)

This queue document contains useful history, but some 2026-06-27 queue lines are stale. The live
state checked on 2026-06-28 is:

- [x] Phase 0 local validation gate completed in the queue-like environment.
- [x] Phase 1 data-check gate completed; vignette/full-cohort prerequisites were available enough
  for the attempted benchmark queue.
- [ ] Phase 2 is incomplete. Job `25102544` is now `FAILED` (`11:0`); written artifacts are
  `nunez_b1_s0.json`, `nunez_b1_s1.json`, `nunez_b1_s2.json`, `nunez_b2_s0.json`, and
  `nunez_b2_s1.json`. Missing required artifact: `nunez_b2_s2.json`.
- [ ] Phase 3 publication evidence is incomplete. Job `25102555` completed, but it wrote old
  `roider_*` outputs using `--dataset roider`; these are smoke/provenance only. Required
  `roider-full` outputs are missing: `roider_full_b3_s0.json`, `roider_full_b3_s1.json`,
  `roider_full_b3_s2.json`, and `roider_full_b5_sweep_s0.json`.
- [ ] Phase 5 B8 is not completed. Job `25102620` is pending with `DependencyNeverSatisfied`;
  missing required artifacts are `nunez_b8_s0.json`, `nunez_b8_s1.json`, and `nunez_b8_s2.json`.
- [ ] Phase 6 B9 remains optional/blocked unless `mapqc` is available. The expected blocked marker
  `nunez_b9_s0.json` is not present yet.
- [x] Phase 4 B4/B6 plumbing smoke completed previously, but it is not publication evidence and is
  optional/deferred in the manifest.
- [ ] Phase 7 final aggregation is not completed. Job `25102623` is pending with
  `DependencyNeverSatisfied`; the existing `final_summary.json` is stale exploratory output, not a
  valid manifest-mode publication summary.

Use `docs/plans/2026-06-28-cytoanvi-next-steps.md` and `docs/review-clear-execute-tasks.md` for the
active recovery checklist.

## Objective
Execute the remaining CytoANVI publication-readiness work in a deterministic order, closing local validation blockers first and then finishing full-cohort benchmark tasks with documented pass or fail outcomes.

## Plan
1. Phase 0: unblock environment and run local validation gates (ruff and pytest suites listed in docs/review-clear-execute-tasks.md).
2. Phase 1: confirm data prerequisites for full Nunez and full Roider cohorts via shared benchmark fetch and validation utilities.
3. Phase 2: run Issue 08 (B1 and B2 on full Nunez) for seeds 0, 1, and 2 at max_epochs=1000.
4. Phase 3: run Issue 09 (B3 and B5 on full Roider), including B5 holdout sweep artifact generation.
5. Phase 4: run Issue 10 (B4 and B6 continual update on full Roider) with ewc_importance sweep over {0,1,10,100,1000}.
6. Phase 5: run Issue 12 real-data B8 (flat CE vs HCE) with benchmarks/cytoanvi/hierarchy_nunez_tutorial.json across 3 seeds.
7. Phase 6: run Issue 13 real-data B9 mapQC with scvi-tools[cytoanvi-mapping-qc] installed and record mapQC completion outputs.
8. Phase 7: aggregate results, update .scratch/cytoanvi-benchmark/PRD.md and issue statuses, and publish final go or no-go summary.

## Validation
- python3 -m ruff check src/cytoanvi benchmarks/cytoanvi tests/cytoanvi
- python3 -m pytest tests/cytoanvi/test_cytoanvi.py tests/cytoanvi/test_hce.py tests/cytoanvi/test_mapping_qc_mock.py -q
- python3 -m pytest tests/benchmarks -q
- python -m benchmarks.common.fetch_data --validate-only
- python -m benchmarks.common.fetch_data --list-full-cohort
- python -m benchmarks.common.aggregate_results --manifest .scratch/cytoanvi-benchmark/publication_manifest.json --output .scratch/cytoanvi-benchmark/results/final_summary.json

Do not use recursive `--input .scratch/cytoanvi-benchmark/results` aggregation for publication
claims; it mixes smoke, synthetic, e1000, and stale `roider` outputs.

## Notes
- Tracker references: issues 08, 09, 10, 12, and 13 under .scratch/cytoanvi-benchmark/issues/.
- Success criteria follow .scratch/cytoanvi-benchmark/PRD.md thresholds (B1, B2, B3, B5, B4, B6, B8, B9).
- Execute in this order for risk burn-down: Phase 0, 1, 2, 3, 5, 6, 4, 7.
- If a phase is blocked, record the exact blocker in the corresponding issue file before proceeding.

## Execution update (2026-06-27)

Validation completed in the queue-like environment:
- Compile: `benchmarks/cytoanvi/tasks.py`, `benchmarks/common/aggregate_results.py`,
  `benchmarks/cytoanvi/run.py` passed.
- Touched-file ruff: passed for `benchmarks/cytoanvi/tasks.py`,
  `benchmarks/common/aggregate_results.py`, `tests/benchmarks/test_cytoanvi_smoke.py`, and
  `tests/benchmarks/test_aggregate_results.py`.
- Broad ruff command runs but still reports pre-existing lint in unrelated benchmark files; see
  command output from the 2026-06-27 execution turn.
- Targeted pytest: `54 passed` for CytoANVI external tests, mapping-QC mocks, benchmark smoke, and
  aggregation tests.
- Synthetic B4 smoke: passed and wrote `/tmp/cytoanvi_b4_smoke.json`.
- Aggregation smoke: passed and wrote
  `.scratch/cytoanvi-benchmark/results/final_summary.json`.

Code fixes applied:
- B4/B6 continual metrics now use `replay_latent_drift` on replay/reference cells, avoiding the
  invalid reference-model scoring of query batches unseen by the reference registry.
- `benchmarks.common.aggregate_results` now supports both positional `inputs --out` and queue mode
  `--input DIR --output PATH`.
- Phase 6 mapQC script no longer installs packages; it preflights `import mapqc` and exits blocked
  if missing.

Queue state at last check:
- Phase 2 Nuñez B1/B2: job **25102544**, RUNNING on `res-hpc-gpu15`.
- Phase 3 Roider B3/B5: job **25102555**, COMPLETED (`0:0`).
- Phase 4 old B4/B6: job **25102606**, FAILED (`1:0`) before the replay-drift fix.
- Phase 4 fixed B4/B6: job **25102622**, COMPLETED (`0:0`) in `00:14:22`.
- Phase 5 B8: job **25102620**, PENDING with dependency `afterok:25102544`.
- Phase 6 B9: skipped/blocked because `mapqc` is not installed.
- Phase 7 aggregate: job **25102623**, PENDING with dependency
  `afterok:25102544:25102620:25102622`.

Completed Phase 3 metrics (⚠️ SUPERSEDED — roider-e1000 subset, not full cohort):
- B3 p1 holdout macro-F1: **0.941 +/- 0.012** across seeds 0/1/2.  *(full-cohort: 0.828±0.015)*
- B3 p2 concordance vs kNN: **0.862 +/- 0.009** across seeds 0/1/2.  *(full-cohort: 0.671±0.008)*
- B5 seed-0 holdout sweep: best AUROC **0.909**, mean AUROC **0.490**.  *(full-cohort 3-seed: mean_auroc 0.484±0.019 — NEGATIVE vs CytoVI kNN-OOD 0.775)*

Completed Phase 4 metrics:
- B4 plain surgery query macro-F1 **0.859**; continual update query macro-F1 **0.862**.
- B6 λ sweep replay latent drift tied at **0.0** for all tested λ values.
- B6 highest query macro-F1 was λ=`1.0` (**0.888**), but the recorded drift-first heuristic
  recommends λ=`0.0` because all drifts tie. Treat this as plumbing evidence, not a biological
  λ default.
