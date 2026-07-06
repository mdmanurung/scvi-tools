# Fresh Session Handoff: CytoANVI Publication Readiness

## Status Reconciliation 2026-06-28

This handoff covered the local code-review hygiene pass, not the current benchmark recovery queue.
Its task list in `docs/review-clear-execute-tasks.md` was completed: local compile/lint/test
validation was recorded, source-confirmed fixes were marked in `.scratch/cytoanvi-review-fixes.md`,
and skipped/data-dependent checks were reported.

Do not use this 2026-06-27 handoff as the active benchmark recovery plan. The active recovery state
is recorded in:

- `docs/plans/2026-06-28-cytoanvi-next-steps.md`
- `docs/review-clear-execute-plan.md`
- `docs/review-clear-execute-tasks.md`

Current benchmark queue status from 2026-06-28:

- Completed: local hygiene packet; Nuñez B1 seeds 0/1/2; Nuñez B2 seeds 0/1.
- Not completed: Nuñez B2 seed 2; Roider `roider-full` Phase 3 artifacts; Nuñez B8 artifacts;
  optional B9 blocked marker; manifest-mode final aggregation.
- Stale: pending jobs `25102620` and `25102623` are blocked by the failed Phase 2 dependency and
  should be conditionally canceled if still pending.

## Objective

Finish local, code-review-supported CytoANVI publication-readiness fixes feasible in this checkout,
then report validation and remaining data-dependent benchmark blockers.

## Repository

/exports/para-lipg-hpc/mdmanurung/scvi-tools

Revised Plan Path: docs/review-clear-execute-plan.md
Task List Path: docs/review-clear-execute-tasks.md

## Frozen Plan

1. Preserve the documented CytoANVI architecture decisions: M1+M2 with CytoVI GMM prior disabled,
   paper-faithful continual update with replay/product Fisher, and opt-in fail-fast hierarchy.
2. Audit current dirty-tree CytoANVI source against `.scratch/cytoanvi-review-fixes.md`.
3. Apply only remaining local hygiene fixes supported by current source and the tracker.
4. Update `.scratch/cytoanvi-review-fixes.md` for fixes confirmed in the current source.
5. Run feasible local compile, ruff, and targeted pytest validation.
6. Report skipped validation with exact blocker.

## Constraints and Non-Goals

- Parent review created this packet and did not intentionally implement source changes.
- Do not use CodeRabbit or external review tooling.
- Do not install dependencies, fetch external data, use credentials, or run external benchmark data
  downloads.
- Do not run full publication-grade Nuñez/Roider benchmarks unless required data are already local.
- Do not rewrite unrelated dirty worktree changes.
- Do not move package files with live importers in this pass.

## Validation Commands

- `python3 -m py_compile src/cytoanvi/_module.py src/cytoanvi/_continual.py src/cytoanvi/_uncertainty.py src/cytoanvi/_model.py src/cytoanvi/mapping_qc.py benchmarks/cytoanvi/tasks.py benchmarks/cytoanvi/data.py benchmarks/cytoanvi/metrics.py`
- `python3 -m ruff check src/cytoanvi benchmarks/cytoanvi tests/cytoanvi`
- `python3 -m pytest tests/cytoanvi/test_cytoanvi.py tests/cytoanvi/test_hce.py tests/cytoanvi/test_mapping_qc_mock.py -q`
- `python3 -m pytest tests/benchmarks -q`

## Stop Conditions

- Stop if repository state has drifted from assumptions that affect the plan.
- Stop if any step is ambiguous after inspecting the named files.
- Stop before destructive operations not explicitly listed in the frozen plan.
- Stop before accessing credentials, production systems, external review tools, or restricted
  network resources without explicit permission.
- Stop if required permissions, dependencies, or data are missing.

## Required Artifacts

- `docs/review-clear-execute-plan.md`
- `docs/review-clear-execute-tasks.md`
- This handoff file
- Updated `.scratch/cytoanvi-review-fixes.md` if source-confirmed fixes are marked complete
- Final summary of changes, validation, skipped checks, and remaining blockers
