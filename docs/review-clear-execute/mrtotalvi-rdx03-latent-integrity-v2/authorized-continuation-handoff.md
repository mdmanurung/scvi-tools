# Authorized Continuation Handoff: MrTotalVI RDX-03 Latent-Integrity v2

Frozen: 2026-07-31

## Objective

Continue the frozen MrTotalVI RDX-03 latent-integrity v2 plan from the first verified unfinished
task, using test-driven development, and execute every in-scope issue through the RDX-03 seal and
downstream handoff. Preserve all existing work and immutable evidence.

## Repository and controlling packet

Repository:
`/exports/para-lipg-hpc/mdmanurung/scvi-tools`

Read these files completely before acting:

1. `docs/review-clear-execute/mrtotalvi-rdx03-latent-integrity-v2/handoff.md`
2. `docs/review-clear-execute/mrtotalvi-rdx03-latent-integrity-v2/plan.md`
3. `docs/review-clear-execute/mrtotalvi-rdx03-latent-integrity-v2/tasks.md`

The original plan and task list remain controlling. This continuation handoff records the later
authorization and does not rewrite the frozen scientific decisions.

## Authorization ledger

The user subsequently instructed:

- `execute all issues please tdd`
- `i approve`

This authorizes:

- local read, edit, test, review, and immutable evidence work required by the controlling packet;
- the specified non-authoritative one-fit CPU probe after its prerequisites pass;
- exactly one authoritative RDX-03 CPU submission with:
  `sbatch --parsable .scratch/mrtotalvi-v2-redesign/relaunch-rdx03-grid.sbatch`;
- read-only monitoring of that one job through terminal scheduler and artifact state;
- Issue 11 seal/adjudication and Issue 12 downstream handoff or local TDD work that does not require
  another scheduler launch.

This does not authorize:

- a second submission, automatic retry, array repair, or splicing;
- any later Stage A/B, Milo, human-safety, or other scheduler launch;
- network access, dependency installation, GPU use, commit, push, publication, or external
  environment mutation;
- unlocking factual W22-versus-W00 differential abundance;
- overwriting an existing artifact, creating `latest`, cleaning/resetting the worktree, or
  modifying preserved historical evidence.

The single RDX-03 submission is conditional. Do not call `sbatch` until every Issue 01-09
pre-submit requirement and all high/medium independent-review findings are demonstrably closed.
If any precondition fails, preserve evidence, keep the submission unused, and continue only with
safe local remediation.

## Current verified state to re-inspect

- The worktree is intentionally dirty and shared. Preserve all unrelated and pre-existing edits.
- Scientific and compatibility second-pass reviews had no open high/medium findings before later
  launcher-containment edits.
- Operations review remains the controlling open gate until source snapshot, data closure,
  execution containment, Git provenance, environment closure, boundary-code binding, and their
  negative TDD cases pass an independent second pass.
- No scheduler submission has been made under this authorization.
- Existing partial, failed, historical, and evidence directories are immutable inputs. Never
  promote or splice them.
- Source changes after an earlier validation invalidate dependent validation/evidence claims.
  Re-run and freshly seal every affected gate after the source stabilizes.

Treat these statements as handoff context, not substitutes for repository inspection. Verify the
actual issue files, source, tests, evidence, scheduler state, and immutable identities before
claiming completion.

## Execution protocol

1. Read the controlling packet and project issue-tracker/domain instructions completely.
2. Inspect the actual worktree, issue files, implementation, evidence, and live agent edits.
3. Resume at the first verified unfinished task; do not redo already valid work merely for
   appearance.
4. For every behavioral or safety change, establish a failing test first, then the smallest
   implementation, then focused and regression validation.
5. Close all high/medium independent-review findings. Re-review after execution-relevant changes.
6. Generate fresh Issue 06 evidence with commands, exact environment, source/input identities,
   stdout/stderr, exit codes, checksums, and supersession of invalidated packets.
7. Run and verify the one-fit probe without allowing it to complete RDX-03 or unlock factual DA.
8. Freeze Issue 09 only after the exact launcher, pre-submit boundary, source/data/environment
   snapshot, authorization, and one-shot claim are end-to-end bound and independently pass.
9. Submit exactly once, record the returned job ID, and monitor without retry.
10. On terminal success, perform every Issue 11 internal, snapshot, live-source, scheduler,
    checksum, exact-grid, aggregate-recompute, and independent-review gate.
11. Complete the Issue 12 handoff or authorized local TDD slices. Stop before any further
    scheduler authority boundary.
12. Update the task checklist and issue records only from contemporaneous evidence.

## Stop conditions

Stop the affected path and preserve its evidence if:

- execution-relevant identity or immutable input drift cannot be reconciled;
- exact v1 replay, lineage, Python 3.14, environment, source/data snapshot, or import containment
  fails;
- an active conflicting RDX-03 job/process exists;
- the authorization/claim cannot prove that zero prior submissions consumed this authority;
- `sbatch` fails or the submitted job does not finish as an exact sealed valid run;
- completion would require any authority excluded above.

Never convert a blocked, running, submitted, partial, failed, or merely package-tested state into a
scientifically complete claim.

## Required final report

Report:

- issue-by-issue completion or exact blocker;
- tests and independent reviews with pass/fail counts;
- fresh evidence and immutable run paths;
- whether the single submission authority was unused or, if consumed, the exact job ID and
  terminal scheduler evidence;
- RDX-03 scientific decomposition without outcome-guided retuning;
- downstream handoff state and the next authority boundary.
