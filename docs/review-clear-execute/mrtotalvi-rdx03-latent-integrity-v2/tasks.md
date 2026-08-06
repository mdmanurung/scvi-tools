# Execution Tasks: MrTotalVI RDX-03 Latent-Integrity v2

Repository: `/exports/para-lipg-hpc/mdmanurung/scvi-tools`

## Packet and evidence freeze

- [ ] Read `handoff.md`, `plan.md`, ADR-0008, the collapse-gate challenge, and the current RDX-03
  implementation/review records completely.
- [ ] Record HEAD and `git status --porcelain=v1 --untracked-files=all`.
- [ ] Hash every consumed source, configuration, fixture, policy, and lineage artifact.
- [ ] Hash the preserved partial directory without writing inside it.
- [ ] Record the 42 paired records and orphan canonical-human B3 seed-0 record.
- [ ] Reproduce the focused 31-test baseline before source edits.
- [ ] Stop and report any execution-relevant drift before editing.

## Amendment

- [ ] Add ADR-0009, prospectively superseding only ADR-0008's effective-rank hard-gate clause.
- [ ] Add `.scratch/mrtotalvi-v2-redesign/latent-integrity-policy-v2.md`.
- [ ] Freeze and hash the canonical latent-integrity v2 policy payload.
- [ ] Retain effective rank as an alert-only diagnostic.
- [ ] Define exact terminal integrity conditions with no result-derived threshold.
- [ ] Keep convergence separate and factual human DA locked.

## TDD compatibility slices

- [x] RED: add a golden v1 contract-payload/digest test.
- [x] GREEN: preserve exact v1 `redesign_run_contract()` behavior.
- [x] RED: add a golden v1 rank-failure replay test.
- [x] GREEN: preserve exact v1 `assess_latent_collapse()` behavior.
- [x] RED: add a low-rank, otherwise-valid v2 behavior test.
- [x] GREEN: add `assess_latent_integrity_v2()` with alert-only rank.
- [x] RED: add an invertible anisotropic-rescaling eligibility test.
- [x] GREEN: make v2 terminal eligibility independent of effective rank.
- [x] RED/GREEN: add representation-specific nonfinite-value failure behavior.
- [x] RED/GREEN: add representation-specific exact-zero-variation failure behavior.
- [x] RED/GREEN: add all-elements-positive posterior-scale behavior.
- [x] RED/GREEN: add exact MrTotalVI residual-gradient-coverage behavior.
- [x] RED/GREEN: prove a `u` failure does not disqualify valid factual `z`, and vice versa.
- [x] RED/GREEN: prove rank alerts do not suppress paired or cross-seed geometry.
- [x] RED/GREEN: prove genuine terminal failure and non-convergence create explicit no-calls.
- [x] RED/GREEN: add v2 contract, execution, fit, aggregate, partial, worker, assessment, and
  metric-dictionary schemas.
- [x] RED/GREEN: bind every v3 payload to contract and policy IDs/digests.
- [x] RED/GREEN: seal complete contract and policy payloads.
- [x] RED/GREEN: reject policy/schema/digest tampering and cross-version substitution.
- [x] RED/GREEN: register and replay both historical v1 governance payload variants.
- [x] RED/GREEN: require an explicit contract adapter in metric-schema validation.
- [x] RED/GREEN: keep exact-grid validation at 48 unique cells.
- [x] Refactor only after all focused tests are green.

## Validation

- [ ] Run the focused convergence/runner/contract/governance suite.
- [ ] Run all `tests/benchmarks/mrtotalvi`.
- [ ] Run all `tests/external/mrtotalvi`.
- [ ] Run all `tests/external/mrmultivi`.
- [ ] Run scoped Ruff.
- [ ] Compile every changed Python file.
- [ ] Run `git diff --check`.
- [ ] Recover the exact prior Python 3.14 interpreter and run the pure contract suite.
- [ ] Verify the authoritative RDX-01 lineage run.
- [ ] Rehash the preserved partial directory and prove byte identity.
- [ ] Verify v1 artifacts retain historical classifications.

## Independent review and probe

- [x] Obtain independent scientific, compatibility, and operations review.
- [x] Close all high- and medium-severity findings.
- [ ] Run the mixed/B1/seed-0 probe.
- [ ] Prove the subset is labeled `probe` and cannot complete RDX-03.
- [ ] Verify the probe with sealed-payload, code-snapshot, and live-repository verifiers.
- [ ] Confirm rank alerts do not cause terminal failure or geometry no-calls.
- [ ] Confirm factual human DA remains locked.

## Scheduler authorization checkpoint

- [ ] Stop and request separate explicit authorization before any `sbatch`.

## Fresh 48-fit grid after authorization

- [ ] Add the required `LD_LIBRARY_PATH` and preflight assertions to the launcher.
- [ ] Confirm no active RDX-03 job or process.
- [ ] Preserve every stale `.tmp`, failed, and partial run.
- [ ] Freeze HEAD, full status, source snapshot, environment, grid, and all relevant digests.
- [ ] Verify the exact 4-fixture by 4-row by 3-seed grid.
- [ ] Submit the launcher exactly once with `sbatch --parsable`.
- [ ] Record the returned job ID.
- [ ] Monitor `squeue`, logs, fit-complete count, and `sacct` without automatic resubmission.
- [ ] Preserve and adjudicate any failed run without splicing.

## Seal and RDX-03 decision

- [ ] Require scheduler `COMPLETED`, exit `0:0`, and final `sealed`.
- [ ] Require one new immutable non-`.tmp`, non-`-failed` directory with no `latest`.
- [ ] Verify exactly 48 unique fits and all v3 policy/contract bindings.
- [ ] Verify results, representations, histories, checkpoints, workers, code, configuration,
  fixtures, lineage, and checksums.
- [ ] Run internal-artifact, sealed-code-snapshot, and live-repository verification.
- [ ] Independently recompute and compare the aggregate.
- [ ] Obtain final independent read-only review.
- [ ] Mark RDX-03 complete only after every artifact gate passes.
- [ ] Record scientific fit failures as negative evidence, not execution blockage.
- [ ] Record `blocked` only for invalid execution, lineage, dependency, permission, or compute
  evidence.

## D1-D5 handoff

- [ ] Record D0 convergence and terminal-integrity decomposition.
- [ ] Hand every valid RDX-03 result to RDX-04.
- [ ] Execute D1 exact-transform behavior test and implementation.
- [ ] Execute D2 sample-blind TotalVI-trunk behavior test and implementation.
- [ ] Execute D3-D5 exact-axis behavior tests and implementations.
- [ ] Add invariance, technical-covariate, gradient, centering, and chunk-equivalence tests.
- [ ] Run frozen MrTotalVI and MrMultiVI legacy-oracle regressions.
- [ ] Keep D1-D5 package-private.
- [ ] Require separate authority for every later scheduler launch.
- [ ] Issue `candidate` or `stop` only after the complete valid downstream scientific screen.
