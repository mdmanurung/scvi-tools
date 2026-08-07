# No-parent-history executor handoff: CytoANVI and MrTotalVI usage readiness

## Objective

Execute every locally authorized engineering and governance step in the frozen CytoANVI/MrTotalVI
usage-readiness remediation packet. Produce fail-closed P1 contracts, trustworthy guidance, exact
artifact tooling and (only if clean dependency authority is locally available) one accepted
`cytoanvi 0.2.0` wheel, frozen-or-explicitly-blocked P2 protocols, and a non-promoting 19-row
capability matrix. Never substitute engineering tests for scientific acceptance or agent judgment
for human promotion.

## Repository

`/exports/para-lipg-hpc/mdmanurung/scvi-tools`

Revised Plan Path:
`/exports/para-lipg-hpc/mdmanurung/scvi-tools/docs/review-clear-execute/cytoanvi-mrtotalvi-usage-readiness/plan.md`

Task List Path:
`/exports/para-lipg-hpc/mdmanurung/scvi-tools/docs/review-clear-execute/cytoanvi-mrtotalvi-usage-readiness/tasks.md`

Baseline commit: `297769d3c62b9228244a05469dc8349a55e4174c`

## Frozen Plan

1. Preserve and fingerprint the dirty baseline; add ADR/migration/schema/quarantine and artifact
   harness contracts without building the candidate.
2. Establish one authoritative evidence-status surface and repair every contradictory guide,
   docstring, installation path, and both executable treeArches paths.
3. Implement CytoANVI fail-closed adversarial, replay, TTA, split-boundary, control-data, and mapQC
   contracts with defect-inverting tests.
4. Implement MrTotalVI prior/supervision migration, exhaustive counts, sample metadata/subsets,
   protein identity/access, streaming export, latent schema, Vamp split, vmap, and public DE refusal
   contracts with checkpoint and negative tests.
5. Run terminal source validation, review/stage only task-owned files, create a scoped local commit,
   reconstruct a clean tree from it, and build at most one `0.2.0` wheel.
6. Run isolated installed-wheel acceptance only if an exact local dependency authority exists;
   otherwise record `blocked_dependency_authority` and leave the candidate unaccepted.
7. Author machine/human P2 protocols, reusing only pre-existing immutable authority and marking
   missing scientific choices blocked; launch no jobs.
8. Materialize and validate all 19 capability rows without agent-issued scientific or promotion
   signatures, then report terminal engineering evidence and exact remaining gates.

The complete normative details, migration tables, capability IDs, validation commands, and stop
conditions are in the revised plan and must be read before editing.

## Authorization

Allowed:

- Read repository files and existing local evidence needed for this packet.
- Modify task-scoped source, tests, documentation, ADRs, workflows, local scripts, schemas, protocol
  contracts, capability tables, packet files, and ignored local candidate evidence.
- Run bounded local CPU/source tests, static checks, clean-tree/archive operations under the repo or
  `/tmp`, local wheel builds using already available dependencies, and isolated wheel acceptance
  using an already available exact local dependency authority.
- Create scoped local commits containing only task-owned changes because the requested artifact
  contract requires a recorded source commit.
- Run the smallest engineering-only MrMultiVI compatibility test only if a shared helper changed;
  this does not authorize MrMultiVI scientific review or remediation.

Approval required:

- Any network access, dependency installation/resolution, external-environment write, credential or
  restricted-data access.
- Any scheduler/GPU submission, cancellation, modification, or live scientific run.
- Any CI trigger, push, tag, GitHub/PyPI release, deployment, publication action, or external write.
- Independent protocol approval, result adjudication, or P3 capability promotion/signature.
- Any scope expansion or change to MrMultiVI behavior.

Forbidden or out of scope:

- Reset, clean, stash, discard, overwrite, or broad staging of user-owned worktree state.
- Moving, deleting, overwriting, or rebuilding the existing `0.1.0` wheel.
- MrMultiVI scientific inspection, remediation, testing beyond narrow compatibility, recommendation,
  or promotion.
- Publication claims, broad public release, CPU fallback for GPU-qualified scientific runs, and
  treating warnings/tests/synthetic fixtures as biological validation.
- Selecting cohorts, labels, thresholds, or acceptance margins after observing outcomes.
- Splicing the preserved 42/48 RDX-03 run or restoring effective rank as a terminal v2 gate.

## Constraints and Non-Goals

- Source presence, source tests, installed-wheel tests, scientific evidence, and promotion are
  separate states.
- New breaking version is `0.2.0`; build only after P1 is committed and build it at most once.
- Keep the existing stale wheel at its current path and SHA; quarantine through tracked authority.
- No new CytoANVI EWC-only public mode is introduced.
- Stable TTA/legacy DE paths fail closed; explicit experimental/private code is not a promoted
  capability.
- MrTotalVI new calls are unsupervised by default even when labels are registered.
- All 19 capability rows, including no-go and experimental surfaces, are mandatory.
- Missing P2 authority is a blocker, not permission to invent a protocol or launch a run.

## Validation Commands

- `env LD_LIBRARY_PATH=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/lib MPLCONFIGDIR=/tmp/cytoanvi-mpl NUMBA_CACHE_DIR=/tmp/cytoanvi-numba PYTHONPATH=src /exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest -q -p no:cacheprovider tests/cytoanvi`
- `env LD_LIBRARY_PATH=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/lib MPLCONFIGDIR=/tmp/cytoanvi-mpl NUMBA_CACHE_DIR=/tmp/cytoanvi-numba PYTHONPATH=src /exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest -q -p no:cacheprovider tests/external/mrtotalvi`
- `env LD_LIBRARY_PATH=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/lib MPLCONFIGDIR=/tmp/cytoanvi-mpl NUMBA_CACHE_DIR=/tmp/cytoanvi-numba PYTHONPATH=src /exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python -m pytest -q -p no:cacheprovider tests/benchmarks/mrtotalvi tests/benchmarks/test_cytoanvi_smoke.py tests/benchmarks/test_cytoanvi_baselines.py tests/benchmarks/test_aggregate_results.py`
- `git diff --check`
- `scripts/accept_usage_readiness_wheel --wheel dist/cytoanvi-0.2.0-py3-none-any.whl --manifest docs/artifacts/cytoanvi-0.2.0/manifest.json --dependency-authority <exact-local-lock-or-wheelhouse>`

## Stop Conditions

- Stop if repository target state drifted or an edit would overlap user-owned changes.
- Stop before reset/clean/stash, broad staging, destructive operations, or artifact overwrite.
- Stop before network, dependency installation, restricted data, credentials, external writes,
  scheduler/GPU, CI trigger, push, tag, release, deployment, or publication action.
- Stop and record `blocked_dependency_authority` if isolated installed-wheel acceptance needs new
  authority; do not call a contaminated smoke an acceptance pass.
- Stop rather than weaken raw-count exhaustiveness, namespace isolation, checkpoint migration,
  leakage boundaries, terminal evidence, or independent/human approval.
- Stop before changing MrMultiVI behavior or broadening its scope.
- Stop with P2 `draft_unfrozen`/blocked rather than making outcome-guided scientific choices.

## Required Artifacts

- Completed/annotated task checklist and frozen plan/handoff.
- ADR-0010, migration/changelog, authoritative evidence-status page, and executable tutorials.
- P1 source/tests for both models plus exact source-test receipts.
- Artifact/receipt/protocol/matrix schemas, build and acceptance harnesses, quarantine record, and CI
  definition.
- At most one candidate wheel with exact manifest/inventory/receipt, or a truthful pre-seal/clean-
  acceptance blocker.
- Human- and machine-readable P2 protocols with explicit freeze state.
- Strict 19-row capability matrix with pending independent/human fields where evidence is absent.
- Final report of changed files, scoped commits, exact test exits/counts/runtimes, artifact identity,
  blockers, residual risks, and the one next required approval/action.

## Execution evidence (2026-08-07)

- CytoANVI source/tutorial suite: 178 passed, 3 skipped, 672 warnings in 882.40s; exit 0.
- MrTotalVI: the original collect-only inventory of 195 nodes passed as non-overlapping terminal
  partitions 41 + 68 + 30 + 1 + 55. A final independent audit added one frozen-Vamp checkpoint
  save/load/retrain regression: it passed alone in 11.18s and the now-56-node affected partition
  passed in 223.80s, giving a current total of 196. The narrow shared-helper MrMultiVI
  labelled-Vamp regression also passed (1 test).
- Frozen benchmark selection: 224 passed, 2 skipped, 48 warnings in 341.12s; exit 0.
- Governance/harness selection: final rerun 68 passed in 6.69s; the repository validator accepted all 12
  schemas/state/protocol files. Scoped Ruff, Python compilation, and diff checks passed.
- The unchanged `dist/cytoanvi-0.1.0-py3-none-any.whl` still hashes to
  `340dfbd2d571e44cf5e8b6d1bc8a62798ce9753abc5df099654a026095f19c8d`.
- The uninitialized tutorial gitlink remains at `e36f55865a4d5f197045a62a6edd227e67ade843` with its
  two inspected files restored byte-for-byte (SHA-256
  `8ea180e2d944d0059fc6a126546e9f8e4533d32bcec229b3dffd03f9b36eeadc` and
  `242d08751f429c80cfc887c4d965763c5a08a9002e8c20dbc1a4f48eac0501af`). No external
  submodule write occurred.

Build state is explicitly `blocked_build_backend_authority`. The recorded source-test interpreter
contains `build` but not the frozen `hatchling` backend. No dependency installation or network
resolution was authorized, and the build harness therefore was not invoked: no global claim,
candidate directory, backend process, or `0.2.0` wheel exists. Installed acceptance remains
`blocked_dependency_authority` because there is also no exact local hash-locked dependency authority
and complete wheelhouse. The manifest and receipt remain non-passing; no stale pass was reused.
