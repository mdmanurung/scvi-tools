# Fresh Session Handoff: Implement MrTotalVI RDX-03 Latent-Integrity v2

## Objective

Implement and independently verify the prospective latent-integrity v2 policy while preserving
exact v1 replay, then stop at the scheduler-authorization checkpoint before launching the fresh
48-fit RDX-03 CPU grid.

## Repository

`/exports/para-lipg-hpc/mdmanurung/scvi-tools`

Revised Plan Path:
`docs/review-clear-execute/mrtotalvi-rdx03-latent-integrity-v2/plan.md`

Task List Path:
`docs/review-clear-execute/mrtotalvi-rdx03-latent-integrity-v2/tasks.md`

Prior Redesign Packet:
`docs/review-clear-execute/mrtotalvi-v2-redesign/`

Current base HEAD observed by the planning session:
`cb343ca1628b83f78b256b85e6cf218fb92dff3f`

## Frozen Plan

1. Freeze current source/artifact identities and correct the preserved partial-run inventory.
2. Add the prospective ADR and latent-integrity v2 policy without modifying v1 evidence.
3. Implement explicit v1/v2 contract and assessment seams through red-green vertical slices.
4. Bind all prospective v3 run payloads to complete sealed contract and policy payloads.
5. Run focused/full tests, Python 3.14 compatibility, Ruff, compilation, lineage, and artifact
   invariance checks.
6. Obtain independent review and verify one non-authoritative probe.
7. Stop for separate scheduler authorization.
8. After authorization, run exactly one fresh immutable 48-fit CPU grid, seal and independently
   verify it, and hand every valid RDX-03 result to D1-D5.

## Critical Decisions

- Effective rank remains recorded but is alert-only for v2. It does not decide terminal
  eligibility or suppress geometry.
- V2 terminal integrity failures are nonfinite representation/integrity inputs, exactly zero
  centered variation, any invalid posterior-scale element, and incomplete MrTotalVI
  residual-gradient coverage.
- Convergence remains a separate hard failure.
- Existing state-recovery, prediction, leakage, stability, and Milo gates remain frozen.
- V1 contracts, verifiers, artifacts, and verdicts remain exactly replayable.
- The partial directory contains 42 paired records plus one orphan result/representation; none may
  be promoted or spliced.
- A valid negative RDX-03 result continues to D1-D5. RDX-03 cannot issue `candidate` or terminal
  `stop`.

## Constraints and Non-Goals

- Preserve the complete dirty worktree and every immutable artifact.
- Do not reset, clean, overwrite, create a `latest` pointer, or splice runs.
- Do not change public APIs, defaults, existing modes, or checkpoint semantics.
- Do not derive a threshold or preprocessing rule from previous outcomes.
- Keep factual human W22-versus-W00 DA locked.
- Do not use the network, install dependencies, use a GPU, commit, push, publish, or write external
  environments without separate authority.
- Do not submit a scheduler job without a new explicit authorization from the user.

## Validation Commands

Use:

```bash
env PYTHONPATH=src:. \
  LD_LIBRARY_PATH=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/lib:${LD_LIBRARY_PATH:-} \
  CUDA_VISIBLE_DEVICES="" \
  NUMBA_CACHE_DIR=/tmp/rdx03-numba \
  MPLCONFIGDIR=/tmp/rdx03-mpl \
  /exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/python \
  -m pytest \
  tests/benchmarks/mrtotalvi/test_convergence.py \
  tests/benchmarks/mrtotalvi/test_convergence_runner_contract.py \
  tests/benchmarks/mrtotalvi/test_redesign_contract.py \
  tests/benchmarks/mrtotalvi/test_redesign_governance.py \
  -q -p no:cacheprovider
```

Then run:

- the complete `tests/benchmarks/mrtotalvi` suite;
- `tests/external/mrtotalvi`;
- `tests/external/mrmultivi`;
- scoped Ruff;
- compilation of every changed Python file;
- `git diff --check`;
- the pure contract suite under the exact previously used Python 3.14 runtime;
- the authoritative RDX-01 verifier;
- pre/post hashes for the preserved partial directory.

## Stop Conditions

- Stop if execution-relevant repository or artifact drift invalidates the frozen assumptions.
- Stop if v1 replay changes.
- Stop if lineage, dependency, environment, or Python 3.14 verification is unavailable.
- Stop before overwriting any existing artifact.
- Stop before factual-human outcome-guided tuning.
- Stop before `sbatch` until the user gives separate explicit scheduler authorization.
- On any run failure, preserve the evidence and do not retry or splice automatically.

## Required Artifacts

- ADR-0009 and the human-readable v2 policy amendment;
- canonical policy/contract payloads and digests;
- v1 golden replay plus v2 behavior/tamper tests;
- versioned execution/fit/aggregate/worker/metric schemas;
- focused/full validation and lineage evidence;
- unchanged preserved-partial hashes;
- independent review record;
- verified non-authoritative probe;
- after later authorization, one sealed exact 48-fit RDX-03 run and independent aggregate review.

