# Execution Tasks: MrTotalVI Stable Latent and DA Redesign

Repository: `/exports/para-lipg-hpc/mdmanurung/scvi-tools`

## Packet and Baseline

- [x] Read `docs/review-clear-execute/mrtotalvi-v2-redesign/handoff.md`.
- [x] Read `docs/review-clear-execute/mrtotalvi-v2-redesign/plan.md`.
- [x] Read `docs/plans/2026-07-26-mrtotalvi-stable-latent-da-redesign.md`.
- [x] Read ADR-0007, the validation report, lineage blocker, package verification, and issue files
  01-06.
- [x] Confirm HEAD is `d8c8e997a67997a53f55923eb3ab14e6cf06f94c` or assess drift before editing.
- [x] Snapshot `git status --porcelain=v1` and preserve unrelated changes.
- [x] Verify frozen packet and source-tracker hashes.
- [x] Run current narrow MrTotalVI/MrMultiVI and benchmark tests before source changes.
- [x] Record pre-change test, environment, and source evidence.

## RDX-00 Governance

- [x] Add the redesign package/scientific amendment without changing frozen documents.
- [x] Add an ADR amendment for topology-specific encoder/checkpoint semantics.
- [x] Add typed B0-B3/D0-D5 configurations.
- [x] Add tests that reject unknown candidates and undeclared axis changes.
- [x] Add formal `candidate`, `stop`, and `blocked` verdict validation.
- [x] Record source, configuration, metric-dictionary, and old-run hashes.
- [x] Mark RDX-00 complete only after its tests and hash checks pass.

## RDX-01 Human Lineage

- [x] Write a failing fixture test for harmonized-order/final-parent-QC intersection.
- [x] Implement the read-only, fail-closed cell-universe derivation.
- [x] Add source-hash and expected-count checks.
- [x] Verify exact shared RNA/protein counts and required metadata.
- [x] Verify 46,817 W00/W22 cells and 10 complete donors.
- [x] Add deterministic within-sample hash splitting and tests.
- [x] Add training-only 5,000 Pearson-residual-HVG selection and tests.
- [x] Freeze and hash the 130-protein non-isotype order.
- [x] Seal the derived object and exact artifact manifest atomically.
- [x] Write the project-specific human lineage amendment.
- [x] If any lineage check fails, record `blocked` and do not relabel existing objects.
- [x] Mark RDX-01 complete only after independent manifest re-read.

RDX-01 authority:
`.scratch/mrtotalvi-v2-redesign/human-lineage-runs/20260726T124847Z-d976773e-607f16dd-c67d109b`.
The source-backed verifier, all seven sealed checksums, 16 focused tests, 48 full benchmark tests,
and scoped Ruff/compilation passed. Independent re-review closed all shared-parent-metadata,
covariate-level, and split/HVG-verifier findings. Earlier complete and blocked attempts remain
preserved as superseded evidence. Factual human DA remains `locked_not_computed_or_inspected`.

## RDX-02 Benchmark Contract

- [x] Add stock scVI RNA-only comparator configuration and runner.
- [x] Add stock TotalVI matched multimodal comparator configuration and runner.
- [x] Export `u` and factual `z` as distinct named representations.
- [x] Record full training history and best-checkpoint identity.
- [x] Record RNA/protein reconstruction, KL, posterior-scale, residual, gradient, rank, parameter,
  runtime, and memory diagnostics.
- [x] Add tested CKA/Procrustes and cross-seed kNN metrics.
- [x] Add tested state, sample-leakage, and technical-batch metrics.
- [x] Add tested modality-specific held-out prediction/calibration metrics.
- [x] Reject multimodal ELBO ranking of scVI.
- [x] Update the metric dictionary and schema tests.
- [x] Mark RDX-02 complete only after the pure contract suite passes in Python 3.13 and 3.14.

## RDX-03 Convergence Diagnosis

- [ ] Add a frozen convergence configuration shared by B1-B3 and D0.
- [ ] Run mixed-fixture B1-B3/D0 at seeds 0-2.
- [ ] Run unequal-cell B1-B3/D0 at seeds 0-2.
- [ ] Run sealed 500-cell B1-B3/D0 at seeds 0-2.
- [ ] Run canonical-human diagnosis only if RDX-01 passed.
- [ ] Verify best-checkpoint restoration and record non-convergence/collapse.
- [ ] Decide mechanically whether D0 already passes all downstream gates.
- [ ] Seal the diagnostic aggregate and mark RDX-03 complete.

## RDX-04 Test-First Encoder Ablations

- [ ] RED: add one behavior test for exact TotalVI per-modality input transformation.
- [ ] GREEN: implement the reusable transformation without changing old branches.
- [ ] RED: add one behavior test for the TotalVI-`FCLayers` sample-blind posterior.
- [ ] GREEN: implement the new package-private encoder.
- [ ] Add sample-index invariance and technical-covariate sensitivity tests.
- [ ] Add core, conditioning, and all-residual-row gradient tests.
- [ ] Add D1-D5 exact-axis configuration tests.
- [ ] Add centered hierarchy and target/cell chunk equivalence tests.
- [ ] Run legacy MrTotalVI and MrMultiVI frozen-oracle tests.
- [ ] Refactor only while all new tests are green.
- [ ] Mark RDX-04 complete after scoped Ruff and focused tests pass.

## RDX-05 Adaptive Known-Truth Screen

- [ ] Extend each frozen scenario to the paired-donor DA fixture.
- [ ] Test independent truth/training/evaluation RNG streams.
- [ ] Add atomic staged workflow and exact-grid validation.
- [ ] Run Stage A B1-B3/D0-D5 seed 0, three instances per scenario.
- [ ] Apply hard disqualification gates without retuning.
- [ ] Select at most two redesigns through the frozen rule.
- [ ] Run Stage B B1/B2 plus survivors at seeds 0-2, ten instances per scenario.
- [ ] Verify every immutable run manifest and code/config/data hash.
- [ ] Seal the Stage A/B aggregate and mark RDX-05 complete.

## RDX-07 Milo Bridge

- [ ] Add a small known-result miloR fixture.
- [ ] Export a cell-order-locked SingleCellExperiment contract.
- [ ] Assert named reduced dimensions and exact cell IDs.
- [ ] Implement the frozen graph/neighborhood/count settings.
- [ ] Reorder design rows exactly to neighborhood-count columns.
- [ ] Implement the primary paired GLMM with frozen arguments.
- [ ] Add W22-minus-W00 sign and donor-pairing tests.
- [ ] Add convergence, NA-fit, and separation diagnostics.
- [ ] Run null, DE-only, DA-only, mixed, rare-state, imbalance, continuous, and confounded scenarios.
- [ ] Compute FDP, power, localization, and seed-stability endpoints.
- [ ] Stop a representation on any failed/NA primary fit.
- [ ] Seal the Milo aggregate and mark RDX-07 complete.

## RDX-08 Canonical Human Screen

- [ ] Confirm RDX-01 source/cell/feature/split hashes are unchanged.
- [ ] Train B0-B3 and no more than two redesigns at seeds 0-2.
- [ ] Record held-out prediction, `u`, factual-`z`, rank, runtime, and memory endpoints.
- [ ] Run frozen within-donor timepoint-label permutations.
- [ ] Run frozen human-geometry null, DA-only, and DE-only perturbations.
- [ ] Run Milo without opening factual W22-minus-W00 results.
- [ ] Apply every frozen non-inferiority and DA-safety gate.
- [ ] Seal the human safety aggregate and mark RDX-08 complete or `stop`.

## RDX-09 Selection

- [ ] Disqualify every hard-gate failure.
- [ ] Apply the frozen localization, predictive-loss, stability, and complexity tie-break.
- [ ] Recompute selection with result order permuted.
- [ ] Recompute selection using an independent implementation.
- [ ] Freeze one candidate configuration/code hash or issue `stop`.
- [ ] Record why every nonselected candidate failed or lost the tie-break.
- [ ] Mark RDX-09 complete without inspecting factual human DA.

## RDX-06 Public Mode

- [ ] Proceed only for an RDX-09 `candidate` verdict.
- [ ] If D1 passed, expose only `sample_blind_scaled`.
- [ ] If a D2 family passed, expose only `sample_blind_totalvi`.
- [ ] If only D0 passed, add no new public mode.
- [ ] Add constructor, metadata, and topology-specific loading tests.
- [ ] Refuse unknown and cross-topology semantic overrides.
- [ ] Add save/load and re-save round trips.
- [ ] Update ADR, docstrings, user guide, API docs, examples, and changelog.
- [ ] Run full legacy/v2/MrMultiVI regressions.
- [ ] Mark RDX-06 complete only if every compatibility gate passes.

## RDX-10 Final Evidence

- [ ] If `candidate`, run factual paired W22-minus-W00 Milo once without retuning.
- [ ] If `stop`, skip factual human DA.
- [ ] Run scoped Ruff and compilation.
- [ ] Run full MrTotalVI and MrMultiVI tests.
- [ ] Run benchmark and optional Zarr tests.
- [ ] Build documentation and wheel.
- [ ] Run no-dependency wheel import smoke.
- [ ] Run the normal non-GPU repository suite where feasible.
- [ ] Perform read-only compatibility/API/lineage/statistical review.
- [ ] Address findings and rerun affected gates.
- [ ] Verify all final manifests and hashes after sealing.
- [ ] Update the progress tracker and issue evidence.
- [ ] Write the final `candidate`, `stop`, or `blocked` report.
- [ ] Report completed work, exact validation, blockers, and residual risks to the user.

## Baseline Note — 2026-07-26

Result: `passed`

A sandboxed Zarr probe initially timed out, but the identical operation and the affected test
passed under approved local-only unsandboxed execution without any environment change. The final
baseline is benchmark 13 passed, MrTotalVI 77 passed, and MrMultiVI 65 passed.

Sealed evidence:
`.scratch/mrtotalvi-v2-redesign/executions/20260726T103620Z-baseline/baseline-report.md`

## RDX-00 Note — 2026-07-26

Result: `passed`

The final machine contract freezes 45 endpoints, 24 hard gates, the complete RDX-03 and Stage A/B
grids, convergence, Milo, selection, runtime, human, and RNG controls. The authoritative immutable
governance run is
`.scratch/mrtotalvi-v2-redesign/governance-runs/20260726T113947Z-1773319a2735`.
Its checksum, live file hashes, nested old evidence, and no-`latest` policy passed. The final
focused suite passed 19 tests, the full benchmark suite passed 32 tests, and scoped Ruff and
compilation passed. Independent review closed all findings and is recorded in
`.scratch/mrtotalvi-v2-redesign/reviews/20260726-rdx00-independent-review.md`.

## RDX-02 Note — 2026-07-26

Result: `passed`

The exact 45-endpoint contract now has matched stock scVI/TotalVI runners, distinct named `u` and
factual-`z` exports, independently seeded training/evaluation streams, checkpoint identities bound
to exact validation history plus live and saved state, and lifecycle-aware fail-closed payload
validation. The pure contract suite passed 20 tests under both Python 3.13 and 3.14. The complete
benchmark directory passed 73 tests; scoped Ruff, compilation, checksum, and whitespace checks
passed. Independent re-review closed all four initial findings.

Authoritative evidence:
`.scratch/mrtotalvi-v2-redesign/benchmark-contract-runs/20260726T141348Z-f67b91fcc411`.

Review:
`.scratch/mrtotalvi-v2-redesign/reviews/20260726-rdx02-independent-review.md`.
