# Execution Tasks: MrTotalVI v2 Latent Decoding and Differential Abundance

Use these checkboxes as the execution source of truth. Mark a task complete only after its evidence
is linked from the feature PRD or the frozen plan.

## Stage 0 — Governance, Evidence, Data, and Environment

- [x] Record `HEAD`, branch, full dirty-worktree inventory, and the three frozen packet paths in
  `.scratch/mrtotalvi-v2/PRD.md`.
- [x] Create numbered `.scratch/mrtotalvi-v2/issues/` files in frozen-plan order with canonical
  `Status:` values.
- [x] Add a foundation issue documenting that D-041/C1 evidence is refuted in
  `benchmarks/ANALYSIS_MANIFEST.md`; remove the unsupported empirical claim from active MrTotalVI
  docs/docstrings.
- [x] Inventory all historical MrTotalVI checkpoints/results without promoting any artifact that
  lacks an exact code, data, feature, seed, and configuration manifest.
- [x] Hash and inspect the current human and macaque authoritative H5ADs and the older prepared
  H5AD; record timestamps, shapes, count layers, protein names, and provenance.
- [ ] Build the canonical human W00/W22 QC-pass paired-donor cohort from the authoritative H5AD.
- [ ] Assert unique stable cell IDs, known donor/timepoint, integer raw counts, and complete
  donor-timepoint pairing in the canonical human cohort.
- [ ] Classify the 137 protein columns as biological antibodies or controls and freeze the ordered
  retained protein list with hashes.
- [ ] Freeze HVG-selection rules that fit only inside development-training data without using
  timepoint labels.
- [ ] Write and sign the macaque locked-analysis manifest, including prior-exposure disclosure,
  W08-W00 pairing, feature/homolog mapping, exclusions, and primary endpoint.
- [ ] Estimate environment, GPU, scheduler, runtime, and durable-storage requirements for Stages
  0-5 and publication Stage 7 separately.
- [ ] Create a project-specific `mrtotalvi-v2` Python environment from the repository specification.
- [ ] Export the resolved conda explicit specification, pip freeze, Python session information,
  CUDA/GPU information, and cache settings into the tracked benchmark environment manifest.
- [ ] Record R4_51 `sessionInfo()`, package versions, miloDE source revision, and comparator
  namespace/formal signatures.
- [ ] Run noninteractive Python import, targeted Ruff, and current targeted pytest baselines in the
  declared environment.
- [ ] Run miloR, LEMUR, and miloDE toy/preflight calls and fail the foundation gate on unresolved
  warnings or API mismatch.

## Stage 1 — ADR and Test-First Latent Feasibility

- [ ] Write `docs/adr/0007-mrtotalvi-v2-centered-counterfactuals.md` with the frozen raw-residual
  prior, centering, API, checkpoint, claim, and DA contracts.
- [ ] Add a constructor/load test proving absent `hierarchy_mode` metadata resolves to `legacy`.
- [ ] Freeze expected arrays from a pre-v2 legacy checkpoint and add a numerical regression test.
- [ ] Add failing tests for `mean_sample(eps_centered)=0`, `mean_sample(z)=z_base`, and pairwise
  target differences.
- [ ] Add a failing test for the equal-sample mean raw-residual penalty and its scale.
- [ ] Add failing target-order, target-chunk, batch-size, and common-raw-shift identity tests.
- [ ] Add failing gradient tests for the residual network and every registered embedding row.
- [ ] Add failing save/load tests for explicit v2 metadata and legacy downgrade behavior.
- [ ] Add failing MrMultiVI regression tests before touching any shared hierarchy component.
- [ ] Implement `hierarchy_mode` with `legacy` default and `centered_v2` restricted to MAP
  residuals.
- [ ] Implement exact all-registered-sample raw residual calculation and centered output in the
  MrTotalVI-specific path.
- [ ] Implement the equal-sample raw-residual penalty without altering legacy loss numerics.
- [ ] Add `u_encoder_mode` and implement state-dict-compatible sample-blind bypass behavior.
- [ ] Verify C3 weights equal `N/(S*n_s)`, have mean one, survive save/load, and are invariant to
  within-sample cell duplication.
- [ ] Run a tiny CPU centered-v2 training fixture and record wall time, memory, finite gradients,
  and exact-centering tolerances.
- [ ] Stop or accept Stage 1 based on legacy, centering, gradient, performance, and MrMultiVI gates.

## Stage 2 — Counterfactual Decoders and Descriptive Enrichment

- [ ] Add API tests that freeze latent dataset dimensions, coordinates, variable names, dtypes, and
  attributes.
- [ ] Implement `get_counterfactual_latent()` for registered targets with common posterior draws.
- [ ] Add API tests for RNA library policies and plug-in versus posterior-MC semantics.
- [ ] Add analytic-versus-Monte-Carlo tests for protein background, foreground, mixture
  contributions, and total mean.
- [ ] Implement `get_counterfactual_expression()` with the frozen RNA/protein estimands.
- [ ] Implement observed, specified, and sample-balanced marginal batch/panel/library policies.
- [ ] Implement protein availability masks and fail closed on unsupported targets/panels.
- [ ] Add and test the 512 MiB materialization guard plus chunked Zarr output.
- [ ] Prove target order, feature subset, batch size, target chunk, and storage mode invariance.
- [ ] Prove legacy local-representation and local-distance outputs remain unchanged.
- [ ] Add leave-one-cell-out density and sample-equal aggregation fixture tests.
- [ ] Implement `local_sample_enrichment()` with explicit diagnostics and donor-block contrasts.
- [ ] Preserve `differential_abundance()` numerics and add only the frozen semantic warning.
- [ ] Add seed-ensemble output that separates posterior uncertainty from between-training-seed
  variation.
- [ ] Run Stage 2 targeted tests and save/load round trips.

## Stage 3 — Benchmark, Metric, and Simulation Harness

- [ ] Create `benchmarks/mrtotalvi/` and `tests/benchmarks/mrtotalvi/` package skeletons.
- [ ] Add canonical data-validation and feature-freeze entrypoints.
- [ ] Add typed configuration schemas for C0-C4 and tests proving only preregistered axes differ.
- [ ] Add immutable run-ID generation from timestamp plus code/config/data digests.
- [ ] Add tracked run/publication manifests that reject stale, extra, missing, or hash-mismatched
  artifacts.
- [ ] Define and test the durable artifact-URI policy; treat `.scratch` only as a cache.
- [ ] Write the metric dictionary with exact splits, units, thresholds, CIs, missing/no-call rules,
  ranking, and tie-break.
- [ ] Implement exogenous-truth null, DA-only, DE-only, mixed, rare-state, unequal-cell, continuous,
  and batch-confounding generators.
- [ ] Separate truth, training, and evaluation RNG streams and test deterministic regeneration.
- [ ] Assert truth masks cannot enter model inputs, candidate ranking code, or estimator tuning.
- [ ] Retrain affected models when a scenario changes the training data.
- [ ] Implement cell-level truth projection and zero/zero handling for overlapping neighborhoods.
- [ ] Add CPU end-to-end fixture training, decoding, export, metric, aggregation, and manifest tests.

## Stage 4 — Milo Bridge and Calibration Fixture

- [ ] Export one matched SCE/H5AD fixture with counts, metadata, PCA, TotalVI `z`, C0 `u`, C1 `u`,
  and eligible v2 representations.
- [ ] Store and verify cell-order and embedding checksums for every named reduced dimension.
- [ ] Implement the exact primary Milo graph/neighborhood/count/design/test call from the frozen
  plan.
- [ ] Assert `reduced.dim`/`reduced_dims`, `d`, RNG, BPPARAM, design order, reference level, and
  contrast sign.
- [ ] Record neighborhood membership, counts, overlap, centers, sample counts, normalization
  factors, separation, convergence, and failed/NA fits.
- [ ] Fail the primary Milo fixture if any tested neighborhood fit fails or becomes NA.
- [ ] Reproduce a known miloR toy result in R4_51.
- [ ] Run null, DA-only, DE-only, and mixed exogenous-truth fixture scenarios.
- [ ] Keep Milo inferential metrics separate from descriptive local-enrichment association metrics.
- [ ] Run sensitivity settings as reporting-only outputs without candidate selection.

## Stage 5 — Three-Seed Human Development Screen

- [ ] Reproduce C0 from the canonical cohort and immutable manifest.
- [ ] Reproduce C1; if unsupported, mark it failed and remove the old stability number from active
  evidence.
- [ ] Run C2, C3, and C4 at seed 0 with the frozen configuration and metric dictionary.
- [ ] Disqualify collapsed counterfactuals, invalid decoders, sample-conditioned primary-DA
  candidates, and any model failing reconstruction/biology gates.
- [ ] Select at most one eligible v2 candidate using the frozen ranking and tie-break.
- [ ] Run C0, eligible C1, and the selected v2 candidate at seeds 1 and 2.
- [ ] Aggregate cross-seed geometry and cell-level effect/call stability with confidence intervals.
- [ ] Run the engineering semi-synthetic suite and label it non-publication calibration.
- [ ] Freeze the candidate architecture, hyperparameters, feature rules, and thresholds.
- [ ] Issue and sign exactly one `candidate`, `stop`, or `blocked` report.

## Stage 6 — Optional Differential Expression

- [ ] Gate all DE execution on a passing Stage 5 latent/DA candidate.
- [ ] Freeze LEMUR alignment, embedding dimension, split, contrast, neighborhood, and correction
  settings.
- [ ] Run LEMUR with a donor/timepoint-stratified internal cell split and within-cell-state
  glmGamPoi neighborhood testing.
- [ ] Pin and run miloDE on the same cells/counts/sample IDs/contrast as a development sensitivity
  analysis.
- [ ] Run integer-count within-cell-state donor pseudobulk with minimum donor/cell rules and
  effect-size confidence intervals.
- [ ] Report global pseudobulk separately as a tissue-level composition-confounded estimand.
- [ ] Evaluate simulation truth, effect direction/magnitude, rank, FDR, localization, stability,
  and external/generalization evidence.
- [ ] Promote DE only if its frozen independent gate passes; otherwise document it as unsupported.

## Stage 7 — Publication Calibration and Locked External Validation

- [ ] Write publication compute/storage estimates, scheduler scripts, resume policy, and failure
  handling.
- [ ] Obtain explicit user approval before submitting publication-scale jobs.
- [ ] Run at least 10 training seeds for the frozen candidate and references.
- [ ] Run at least 200 null and 200 DE-only independent replicates, then continue in fixed
  increments until the frozen CI precision target or cap.
- [ ] Run calibrated DA-only, mixed, rare-state, imbalance, and confounding scenarios.
- [ ] Report mean FDP, `P(R>0)`, power, localization, effect calibration, and one-sided CIs.
- [ ] Record an inconclusive verdict if precision is not reached at the cap.
- [ ] Train the frozen architecture separately on the locked macaque cohort and run the predeclared
  W08-W00 endpoint without retuning.
- [ ] Sign separate latent, DA, DE, external-validation, and publication-readiness verdicts.

## Stage 8 — Release Hardening

- [ ] Resolve all MrTotalVI/MrMultiVI/benchmark Ruff findings or document a frozen grandfathered
  baseline with no new findings.
- [ ] Fail closed on rank-deficient, singular, nonconverged, or warning-heavy statistical designs.
- [ ] Run targeted MrTotalVI and MrMultiVI pytest suites.
- [ ] Run benchmark fixture and R comparator tests.
- [ ] Run full repository tests in the declared environment.
- [ ] Run legacy checkpoint, v2 checkpoint, explicit downgrade, chunk/Zarr, and CPU/GPU round trips.
- [ ] Build API documentation, tutorial, model card, limitations, and Milo bridge guide.
- [ ] Reproduce the signed aggregate from tracked manifests and durable artifact hashes.
- [ ] Obtain independent scientific review.
- [ ] Keep `centered_v2` opt-in; require a later ADR before any default change.
- [ ] Produce the final claim-by-claim release and publication-readiness report.

## Validation Checklist

- [ ] `conda run -n mrtotalvi-v2 python -c "import scvi, torch, anndata, scanpy, xarray"`
- [ ] `conda run -n mrtotalvi-v2 python -m ruff check src/scvi/external/mrtotalvi src/scvi/external/mrmultivi tests/external/mrtotalvi tests/external/mrmultivi benchmarks/mrtotalvi`
- [ ] `conda run -n mrtotalvi-v2 python -m pytest tests/external/mrtotalvi -q`
- [ ] `conda run -n mrtotalvi-v2 python -m pytest tests/external/mrmultivi -q`
- [ ] `conda run -n mrtotalvi-v2 python -m pytest tests/benchmarks/mrtotalvi -q`
- [ ] `conda run -n mrtotalvi-v2 python -m benchmarks.mrtotalvi.run_fixture`
- [ ] `/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/Rscript benchmarks/mrtotalvi/tests/test_milo_fixture.R`
- [ ] Full repository pytest and documentation build complete in the release environment.
- [ ] Independent manifest reproduction and scientific sign-off complete.
