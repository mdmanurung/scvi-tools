# Fresh Session Handoff: Execute MrTotalVI v2 Latent and DA Plan

## Objective

Implement the frozen, opt-in MrTotalVI v2 latent-decoding and differential-abundance plan through
the first genuine scientific, data, environment, permission, or compute gate, preserving legacy
checkpoints and unrelated worktree artifacts.

## Repository

`/exports/para-lipg-hpc/mdmanurung/scvi-tools`

Revised Plan Path:
`docs/review-clear-execute/mrtotalvi-v2/plan.md`

Task List Path:
`docs/review-clear-execute/mrtotalvi-v2/tasks.md`

Original Strategy Path:
`docs/plans/2026-07-25-mrtotalvi-v2-latent-da.md`

## Frozen Plan

1. Freeze governance, evidence lineage, canonical human/macaque data contracts, exact environments,
   and durable artifact policy.
2. Accept ADR-0007 with the raw-residual prior, exact centering, sample-blind `u`, decoder
   estimands, checkpoint, and claim contracts.
3. Implement test-first latent feasibility while preserving legacy and MrMultiVI numerics.
4. Implement bounded counterfactual latent/expression datasets and descriptive local enrichment.
5. Build the tracked benchmark, metric dictionary, exogenous-truth simulation harness, and
   immutable manifest aggregation.
6. Validate the exact named-representation Milo GLMM bridge and fail-closed diagnostics.
7. Run the three-seed human development screen and freeze or reject one candidate.
8. Run DE only as an optional gated track using LEMUR, miloDE, and within-state pseudobulk.
9. Obtain explicit approval before publication-scale simulation and scheduler submissions, then run
   precision-driven calibration and locked macaque validation.
10. Complete release hardening, independent reproduction, and claim-by-claim scientific review.

## Constraints and Non-Goals

- Do not rely on parent-session conversation, historical `.scratch` outputs, or the refuted D-041
  claim as evidence.
- Do not overwrite or clean unrelated dirty/untracked files.
- Do not modify the default behavior of shared `EncoderUZ.forward()` or MrMultiVI without frozen
  numerical regression evidence.
- Missing checkpoint mode metadata means `legacy`; never infer v2 from weights.
- V2 is registered-sample-only, MAP-residual-only, and opt-in.
- Only sample-blind C4 may support the primary Milo DA claim.
- Local enrichment is descriptive and is never an inferential competitor to Milo.
- Do not call the macaque dataset untouched; use the signed locked-analysis protocol.
- Do not launch publication-scale jobs without explicit compute/storage approval.
- Do not promote DE if its independent gate fails.
- Do not change the v2 default without a later ADR and full scientific/release sign-off.

## Validation Commands

- `conda run -n mrtotalvi-v2 python -c "import scvi, torch, anndata, scanpy, xarray"`
- `conda run -n mrtotalvi-v2 python -m ruff check src/scvi/external/mrtotalvi src/scvi/external/mrmultivi tests/external/mrtotalvi tests/external/mrmultivi benchmarks/mrtotalvi`
- `conda run -n mrtotalvi-v2 python -m pytest tests/external/mrtotalvi -q`
- `conda run -n mrtotalvi-v2 python -m pytest tests/external/mrmultivi -q`
- `conda run -n mrtotalvi-v2 python -m pytest tests/benchmarks/mrtotalvi -q`
- `conda run -n mrtotalvi-v2 python -m benchmarks.mrtotalvi.run_fixture`
- `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/Rscript benchmarks/mrtotalvi/tests/test_milo_fixture.R`

## Stop Conditions

- Stop if repository state has drifted from the packet in a way that changes the frozen contracts.
- Stop if canonical data lineage, raw integer counts, donor pairing, or protein identity cannot be
  established.
- Stop before destructive operations, credentials, external writes, or production systems not
  explicitly authorized.
- Stop and request approval if creating the missing project environment requires external
  filesystem/network access.
- Stop if exact centered-v2 semantics, legacy arrays, gradients, or MrMultiVI regression tests fail.
- Stop if C4 fails the latent/biology/calibrated-null gate; do not promote conditioned `u`.
- Stop if Milo uses the wrong embedding, has mismatched design rows, or has failed/NA primary fits.
- Stop before publication-scale scheduler submissions until compute/storage approval is explicit.
- Stop at an inconclusive simulation precision cap or a failed locked external-validation gate.

## Required Artifacts

- Updated feature PRD/issues with evidence and canonical statuses.
- ADR-0007 and tracked data/environment/metric/manifests.
- Backward-compatible source changes and focused tests.
- Counterfactual API and local-enrichment documentation.
- Reproducible benchmark/Milo/simulation entrypoints.
- Immutable candidate or stop/blocked report after the three-seed screen.
- Optional DE verdict kept separate from latent/DA verdicts.
- Publication-scale and release reports only after their explicit gates.

