# Fresh Session Handoff: Execute MrTotalVI Stable Latent and DA Redesign

## Objective

Execute the frozen MrTotalVI redesign through the first genuine scientific, lineage, dependency,
permission, or compute stop condition. Produce exactly one eligible opt-in candidate or a
documented `stop`/`blocked` result while preserving all legacy and existing-v2 semantics.

## Repository

`/exports/para-lipg-hpc/mdmanurung/scvi-tools`

Revised Plan Path:
`docs/review-clear-execute/mrtotalvi-v2-redesign/plan.md`

Task List Path:
`docs/review-clear-execute/mrtotalvi-v2-redesign/tasks.md`

Source Tracker Path:
`docs/plans/2026-07-26-mrtotalvi-stable-latent-da-redesign.md`

## Frozen Plan

1. Verify the packet, base commit, dirty worktree, frozen evidence, and current regressions.
2. Record the redesign amendment, ADR/checkpoint boundary, candidates, metrics, and verdict schema.
3. Derive and seal the 46,817-cell human W00/W22 cohort from harmonized IDs plus final-parent QC,
   or record `blocked` without substituting another universe.
4. Extend the benchmark with matched scVI/TotalVI comparators and separate `u`/factual-`z`
   convergence, quality, leakage, stability, prediction, and resource metrics.
5. Diagnose whether current C4 was undertrained under identical convergence controls.
6. Implement D1-D5 test-first behind package-private construction hooks while leaving all old
   branches and `EncoderUZ.forward()` unchanged.
7. Run the preregistered adaptive known-truth screen and advance no more than two redesigns.
8. Build the exact named-representation Milo bridge and enforce DA FDR/power/localization gates.
9. Run the canonical human non-inferiority and null/semi-synthetic screen without opening factual
    W22-minus-W00 DA.
10. Apply the frozen selection rule and freeze configuration/code.
11. Expose only a frozen eligible `sample_blind_scaled` or `sample_blind_totalvi` mode, then run
    factual human DA only for a selected candidate and complete package/build/independent-review
    evidence. Add no mode for D0-only, `stop`, or `blocked` outcomes.

## Frozen Starting Evidence

- Base commit: `d8c8e997a67997a53f55923eb3ab14e6cf06f94c`
- Source tracker SHA-256:
  `b7ee438f41c7c2b8cf29b140bc226f2044c9c47d8040183aeaeb71f50d6f2dcc`
- Original MrTotalVI plan SHA-256:
  `4011a896fa6ceee834064ed59dfd52c25d531dcaf055a4ce0b078a93dcb79154`
- Original MrTotalVI tasks SHA-256:
  `4395026fb46700b08533b01550864acd2066a75ff880b8d0a8c5eb1f502fa973`
- Original MrTotalVI handoff SHA-256:
  `c56e3d44f1d35527cbcaba500e1b470ecd8bf48115e9fbfb6fc47bc25ee0b2a8`
- Validation report SHA-256:
  `f7a9690ea6f51b8b807c92854a6f0d9335b9d638411a0f46c621f2ddc10d4fa5`
- Lineage blocker SHA-256:
  `e89e9fcb33c4f494e1cc9a8677a21acfd9891ed97645809c54d26e2c91ab1afc`

## Constraints and Non-Goals

- Preserve the full pre-existing dirty worktree; do not reset, clean, or overwrite unrelated files.
- Preserve existing immutable run directories and never write a `latest` pointer.
- Preserve legacy, centered-v2, current sample-blind, checkpoint, and MrMultiVI behavior.
- Use only the frozen human lineage; do not choose 47,709 or 51,174 cells as a fallback.
- Never tune on factual human DA or inspect it before candidate freeze.
- Do not run or implement DE, LEMUR, miloDE, macaque validation, causal claims, publication jobs,
  default promotion, new-sample surgery, minified mode, vectorization, or non-MAP residuals.
- Do not commit, push, publish, install dependencies, access the network, submit scheduler jobs, or
  write external environments without separate authority.
- Use test-first vertical slices for encoder behavior; do not batch speculative source changes.

## Validation Commands

Adapt cache paths and the existing Python 3.13 `scvi-test` environment as recorded in the source
tracker. Start with:

- `python -m pytest tests/benchmarks/mrtotalvi -q -p no:cacheprovider`
- `python -m pytest tests/external/mrtotalvi -q -p no:cacheprovider`
- `python -m pytest tests/external/mrmultivi -q -p no:cacheprovider`
- `python -m ruff check src/scvi/external/mrtotalvi src/scvi/external/mrmultivi tests/external/mrtotalvi tests/external/mrmultivi benchmarks/mrtotalvi tests/benchmarks/mrtotalvi`
- `python -m py_compile benchmarks/mrtotalvi/*.py src/scvi/external/mrtotalvi/*.py`
- `git diff --check`

Run the pure contract subset under Python 3.14 and the R miloR toy/bridge checks after their
implementations exist. Use the exact commands recorded into the task evidence rather than
inventing an untracked environment.

## Stop Conditions

- Stop if HEAD or frozen hashes drift in a way that affects execution.
- Stop human work if the source hashes, exact 46,817-cell count, parent QC, counts, features, or
  covariates cannot be reproduced.
- Stop before overwriting existing or unrelated worktree files.
- Stop before any missing dependency, external write, network request, GPU/scheduler action, or
  permission boundary.
- Stop a representation on non-convergence, nonfinite values, latent collapse, failed/NA Milo
  primary fits, or any hard-gate failure.
- Stop the redesign with a valid negative report if no candidate passes; do not improvise another
  architecture or retune on human results.

## Required Artifacts

- updated progress tracker and issue evidence;
- redesign amendment and ADR/checkpoint contract;
- typed/tested D0-D5 benchmark schema;
- immutable human lineage/feature/split manifest or explicit blocker;
- immutable synthetic and permitted human runs with exact manifests;
- Milo bridge, diagnostics, and calibrated aggregate;
- any eligible opt-in package mode with compatibility/docs/build evidence;
- formal `candidate`, `stop`, or `blocked` report; and
- final summary of changes, validation, failures, and residual risks.
