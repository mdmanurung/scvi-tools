# MrTotalVI v2 Latent Decoding and Differential Abundance Progress Plan

> **SUPERSEDED**: This plan is superseded by the operative checklist at
> `docs/review-clear-execute/mrtotalvi-v2-redesign/tasks.md`, with current status tracked in
> `todo/TODO_REGISTRY.md`. Real work has progressed well past the "Status: not started" line below
> — do not trust that line. Check the two files above for what is actually done, in progress, or
> blocked.

- Date: 2026-07-25
- Last updated: 2026-07-25
- Project: /exports/para-lipg-hpc/mdmanurung/scvi-tools
- Status: not started

## Objective

Develop an opt-in MrTotalVI v2 that preserves legacy checkpoints, makes the `u -> z_base + eps`
hierarchy identifiable over registered samples, exposes scientifically explicit latent and
RNA/protein counterfactual decoders, and supports reliable differential-abundance analysis through
a matched Milo workflow. Differential expression is secondary and may be omitted from the
publication claim if it does not pass independent validation.

The immediate authorized evidence budget is a fast human CITE-seq screen with three training seeds.
That screen may select or reject one v2 candidate, but it is not publication evidence. Publication
promotion requires the later simulation, seed, and untouched macaque confirmation gates in this
tracker.

## Fixed Decisions

- Preserve current behavior and saved models as `hierarchy_mode="legacy"`.
- Add v2 side by side; do not change the default until all publication and release gates pass.
- Restrict the first counterfactual API and claim to registered sample identities.
- Use `donor_timepoint` as the biological sample unit and keep `donor` and `timepoint` as separate
  design columns.
- Treat counterfactuals as statistical transformations, not causal interventions.
- Use Milo as the primary inferential DA method; retain MrTotalVI aggregated-posterior scoring only
  as descriptive local sample enrichment.
- Use LEMUR as the primary heterogeneous-DE comparator and miloDE as a secondary development
  comparator.

## Review Findings

| Finding | Resolution |
|---|---|
| The draft mixed a three-seed screen with publication-readiness language. | Split candidate selection from publication promotion and gave each distinct gates. |
| `z = z_base + eps` permits arbitrary translation between the base and residual. | Added equal-sample residual centering as the load-bearing v2 invariant. |
| The current `u` encoder is sample-conditioned, so `u` is not demonstrably sample-neutral. | Added conditioned versus sample-blind ablation and a gated conditional-alignment fallback. |
| Counterfactual expression is only an internal DE hook with ambiguous scales. | Added public latent and expression APIs with explicit RNA/protein estimands and technical-covariate policies. |
| The protein hook uses the log-normal median as a background expectation. | Required the analytic mean `exp(alpha + 0.5 * beta^2)` and Monte Carlo verification. |
| Current DA is a per-cell sample-density compatibility score, not replicate-level abundance inference. | Renamed the scientific estimand, prohibited inferential p-values, and made Milo the primary inferential track. |
| Existing comparisons mix cells, feature sets, representations, and estimators. | Required one canonical cell/feature universe and separate representation-versus-estimator comparisons. |
| Current DE can be numerically repeatable while directionally wrong. | Moved DE after the latent/DA gate and required LEMUR, miloDE, and pseudobulk references. |
| Existing benchmark artifacts live under untracked `.scratch` paths and are not a publication source of truth. | Added a tracked benchmark harness, immutable run manifests, and fail-closed promotion. |
| The draft did not specify checkpoint preservation, rollback, or default-promotion policy. | Kept legacy as the default and required round-trip tests and an explicit rollback path. |
| Repository governance requires ADRs and local Markdown PRD/issues. | Added ADR-0007 plus `.scratch/mrtotalvi-v2/PRD.md` and numbered issue creation as Step MRTL-00/01. |
| The root `CONTEXT.md` is CytoANVI-specific and does not define MrTotalVI terminology. | Use ADR-0005 vocabulary until ADR-0007 defines the v2 terms; do not reuse incompatible CytoANVI terms. |
| Ruff is red and the current shell could not reproduce the targeted pytest run. | Made a fresh `environment-lock.yml` release environment and clean lint/test preflight a blocking foundation gate. |

## Progress Summary

- [ ] MRTL-00: Freeze governance, inputs, baselines, and a reproducible environment
- [ ] MRTL-01: Accept the v2 latent and public-API contract in ADR-0007
- [ ] MRTL-02: Build the tracked, estimand-matched benchmark harness
- [ ] MRTL-03: Implement the centered v2 latent hierarchy
- [ ] MRTL-04: Implement and bound the latent/training ablations
- [ ] MRTL-05: Expose counterfactual latent and expression decoders
- [ ] MRTL-06: Separate descriptive local enrichment from inferential DA
- [ ] MRTL-07: Add the Milo bridge and calibrated DA benchmark
- [ ] MRTL-08: Run the adaptive three-seed fast screen and issue a stop/go verdict
- [ ] MRTL-09: Evaluate DE with LEMUR, miloDE, and pseudobulk references
- [ ] MRTL-10: Run publication-grade simulation and untouched macaque confirmation
- [ ] MRTL-11: Complete release validation, documentation, and gated promotion

## Implementation Steps

### [ ] MRTL-00: Freeze governance, inputs, baselines, and a reproducible environment

- **Status:** not started
- **Outcome:** One executable project tracker, one canonical human benchmark cohort, immutable
  baseline manifests, and a supported Python/R environment that can run preflight checks.
- **Actions:**
  - Create `.scratch/mrtotalvi-v2/PRD.md` and numbered issue files following
    `docs/agents/issue-tracker.md`; initialize actionable issues as `Status: ready-for-agent`.
  - Record the Git commit, dirty-worktree inventory, package versions, CUDA/runtime information,
    model configuration, random seed, source data paths, SHA-256 hashes, cell IDs, gene order, and
    protein order in an immutable run manifest.
  - Construct the canonical fast-screen cohort from QC-pass human CITE-seq, restricted to W00 and
    W22, using the exact intersection of cells available to all methods.
  - Select 5,000 Pearson-residual HVGs once from raw counts and retain all 130 proteins; freeze this
    ordered feature universe for every fast-screen model and comparator.
  - Register `sample_key="donor_timepoint"` and retain separate `donor`, `timepoint`, `batch`, and
    protein-panel fields.
  - Define baseline C0 as current LayerNorm + MoG defaults and C1 as LayerNorm + data-initialized,
    frozen VampPrior; do not reuse historical checkpoints unless every manifest field matches.
  - Create a fresh `cytoanvi-release` environment from `environment-lock.yml`, set writable
    Matplotlib/Numba caches as documented in `ENVIRONMENTS_INSTALLATIONS.md`, and install the
    checkout in editable development mode.
  - Pin the R4_51 comparator versions: miloR 2.6.0, LEMUR 1.11.1, miloDE 0.1.0, glmGamPoi 1.22.0,
    edgeR 4.8.2, SingleCellExperiment 1.32.0, and zellkonverter 1.20.1. Record the miloDE source
    revision because it is a development package.
- **Dependencies:** None.
- **Affected areas:** `.scratch/mrtotalvi-v2/`, `environment-lock.yml`,
  `ENVIRONMENTS_INSTALLATIONS.md`, human CITE-seq data manifest.
- **Validation:**
  - `python -c "import scvi, torch, anndata, scanpy"`
  - `python -m ruff check src/scvi/external/mrtotalvi tests/external/mrtotalvi`
  - `python -m pytest tests/external/mrtotalvi -q`
  - R package-version and import preflight under R4_51.
  - Assert unique, order-identical cell IDs and features across all exported method inputs.
- **Acceptance:** Preflight imports and targeted tests complete in the declared environment; lint
  debt and warnings are recorded as explicit issues; no benchmark training starts from an
  unmatched historical artifact.
- **Evidence:** Pending.

### [ ] MRTL-01: Accept the v2 latent and public-API contract in ADR-0007

- **Status:** not started
- **Outcome:** A reviewed architectural contract that extends ADR-0005 without silently changing
  legacy semantics.
- **Actions:**
  - Add `docs/adr/0007-mrtotalvi-v2-centered-counterfactuals.md`.
  - Define `u`, `z_base`, uncentered `eps`, centered `eps`, target-specific `z`, registered sample,
    factual sample, statistical counterfactual, technical batch, protein panel, and support.
  - Specify equal-sample residual centering:
    `eps_centered(u, s) = eps(u, s) - mean_registered_samples(eps(u, .))`.
  - Set `hierarchy_mode="legacy"` as the default and `hierarchy_mode="centered_v2"` as opt-in.
  - Forbid unseen sample identities in v2 and fail closed with an informative error.
  - Fix common-random-number semantics: draw `u` once per cell/draw and reuse it for every target.
  - Define decoder output estimands and the observed/specified/marginal technical-batch and
    protein-panel policies.
  - State that Milo owns primary DA inference for the first publication candidate; a native
    replicate-level test is deferred unless independently calibrated.
- **Dependencies:** MRTL-00.
- **Affected areas:** `docs/adr/0005-mrtotalvi.md`, new ADR-0007, model/API documentation.
- **Validation:** Review the ADR against `_model.py`, `_components.py`, `_module.py`, `_stats.py`,
  saved-model loading, and the canonical W00/W22 design.
- **Acceptance:** No unresolved semantic choice remains for MRTL-03 through MRTL-07; contradictions
  with ADR-0005 are explicitly marked as v2-only changes.
- **Evidence:** Pending.

### [ ] MRTL-02: Build the tracked, estimand-matched benchmark harness

- **Status:** not started
- **Outcome:** Re-runnable Python/R benchmark entrypoints whose outputs can be attributed to one
  immutable input, model, representation, estimator, and contrast.
- **Actions:**
  - Add `benchmarks/mrtotalvi/` with data validation, training, decoding, metric, aggregation, and
    comparator entrypoints; use shared helpers under `benchmarks/common` where contracts match.
  - Store immutable results under `.scratch/mrtotalvi-v2/runs/<run_id>/`; a run ID must contain a
    timestamp and code/config digest.
  - Add a publication manifest that enumerates allowed run IDs. Recursive aggregation of `.scratch`
    is exploratory only.
  - Export one SingleCellExperiment-compatible artifact with counts, metadata, and named reduced
    dimensions for PCA, TotalVI `z`, legacy MrTotalVI `u`, and v2 `u`.
  - Record the same W22-minus-W00 contrast, donor pairing, cells, genes, proteins, and technical
    covariates for every estimator.
  - Implement semi-synthetic scenario generators for null paired swaps, DA-only localized
    downsampling, DE-only expression perturbation with fixed counts, mixed DA+DE, unequal
    cells/sample, rare states, continuous states, and estimable batch confounding.
  - Keep perturbation truth masks and parameters separate from model inputs and aggregation code.
- **Dependencies:** MRTL-00 and the accepted interfaces from MRTL-01.
- **Affected areas:** `benchmarks/mrtotalvi/`, `benchmarks/common/`,
  `.scratch/mrtotalvi-v2/runs/`.
- **Validation:** Fixture tests for cell/feature identity, deterministic scenario generation, design
  rank, truth-mask integrity, immutable manifests, and rejection of stale or extra results.
- **Acceptance:** A CPU-scale fixture completes end to end and aggregation accepts only explicitly
  manifested artifacts.
- **Evidence:** Pending.

### [ ] MRTL-03: Implement the centered v2 latent hierarchy

- **Status:** not started
- **Outcome:** An opt-in hierarchy in which `z_base` is the registered-sample average state and
  centered residuals cannot absorb an arbitrary common translation.
- **Actions:**
  - Add the v2 mode in the MrTotalVI model, module, and `EncoderUZ` path without altering legacy
    state-dict keys or numerical behavior.
  - Compute residuals for all registered samples with equal sample weights; chunk target samples
    for memory while preserving exact results.
  - Use exact centering for the observed-sample v2 release. Do not introduce stochastic target
    subsampling in this version.
  - Apply centered residuals consistently during training, local-sample representation,
    counterfactual decoding, distance calculation, DA export, and effect calculation.
  - Save the resolved hierarchy mode and centering policy in model metadata and run manifests.
- **Dependencies:** MRTL-01 and MRTL-02 fixtures.
- **Affected areas:** `src/scvi/external/mrtotalvi/_components.py`,
  `src/scvi/external/mrtotalvi/_module.py`, `src/scvi/external/mrtotalvi/_model.py`,
  `tests/external/mrtotalvi/`.
- **Validation:**
  - Centered residual mean is zero to `atol=1e-6`.
  - Adding a common vector to all uncentered residuals leaves v2 `z` unchanged.
  - Target=factual, target ordering, chunk size, and batch size do not change results beyond
    `rtol=1e-5, atol=1e-6`.
  - Legacy checkpoint output remains unchanged on a frozen fixture.
  - Forward/backward gradients are finite and centering parameters receive gradients.
- **Acceptance:** All invariance and backward-compatibility tests pass; exact centering is practical
  on the 20 registered W00/W22 donor-timepoint samples.
- **Evidence:** Pending.

### [ ] MRTL-04: Implement and bound the latent/training ablations

- **Status:** not started
- **Outcome:** A small preregistered candidate set tests the root hypotheses without an open-ended
  hyperparameter search.
- **Actions:**
  - Implement C2: C1 plus centered v2 residuals.
  - Implement C3: C2 plus sample-balanced observation weighting.
  - Implement C4: C2 with biological sample identity removed from the `u` encoder while retaining
    declared technical covariates.
  - Keep MAP residuals, latent dimensions, optimizer, epoch budget, KL schedule, batch size, and
    early-stopping policy fixed across C0-C4.
  - Train C0-C4 only for seed 0. Advance C0, C1, and the best eligible v2 configuration to seeds 1
    and 2.
  - Measure reconstruction/predictive deviance, cell-state conservation, within-state donor
    predictability, Procrustes/CKA geometry, kNN overlap, counterfactual distance ranks, and
    displacement-vector similarity.
  - If conditioned `u` retains excessive donor predictability and sample-blind `u` fails the
    biology/reconstruction gate, stop and open a separately reviewed conditional-alignment issue;
    do not silently add an adversarial objective.
- **Dependencies:** MRTL-03.
- **Affected areas:** MrTotalVI encoder/training configuration, benchmark configuration and latent
  metrics.
- **Validation:** Configuration contract tests prove that C0-C4 differ only in preregistered axes;
  all metrics are invariant to raw latent rotation and cell ordering.
- **Acceptance:** At least one v2 configuration reaches the MRTL-08 latent gates, or the latent
  redesign is explicitly stopped with evidence.
- **Evidence:** Pending.

### [ ] MRTL-05: Expose counterfactual latent and expression decoders

- **Status:** not started
- **Outcome:** Public, chunkable APIs return unambiguous latent, RNA, protein, uncertainty, and
  support quantities for registered target samples.
- **Actions:**
  - Add `get_counterfactual_latent(...) -> xr.Dataset` with variables `u`, `z_base`, `eps`, `z`,
    target support, and admissibility.
  - Add `get_counterfactual_expression(...) -> xr.Dataset` with `rna_scale`, `rna_rate`,
    `protein_foreground_mean`, `protein_background_mean`, `protein_total_mean`, and
    `protein_foreground_probability`.
  - Use `E[rate_back] = exp(back_alpha + 0.5 * back_beta**2)` and derive foreground/total
    expectations from the declared TotalVI mixture probabilities.
  - Default to one posterior mean result; support multiple common-random-number draws without
    retaining all draws unless explicitly requested.
  - Default `batch_policy="observed"` and `panel_policy="observed"` so technical context is fixed
    across targets. Implement specified and globally marginalized policies with one declared,
    sample-balanced weighting distribution.
  - Mark proteins absent from a panel as unavailable; never represent imputed values as observed
    measurements.
  - Support cell, gene, protein, target, and draw subsetting plus deterministic chunking.
- **Dependencies:** MRTL-03; API semantics from MRTL-01.
- **Affected areas:** `_model.py`, `_module.py`, tests, API documentation.
- **Validation:**
  - Analytic protein expectations agree with a high-draw Monte Carlo estimate within 1%.
  - Existing factual normalized-expression output agrees with the matching new factual estimand.
  - Target ordering, feature subsetting, batch size, and chunking are invariant.
  - Common-random-number tests show no independent target noise.
  - Unsupported sample, batch, or panel requests fail closed.
- **Acceptance:** Decoder estimands and identities pass tests, and held-out factual predictive
  performance is no more than 2% worse than C1.
- **Evidence:** Pending.

### [ ] MRTL-06: Separate descriptive local enrichment from inferential DA

- **Status:** not started
- **Outcome:** Users cannot mistake latent sample compatibility for replicated differential
  abundance inference.
- **Actions:**
  - Add `local_sample_enrichment(...)` with explicit `log_density` and `log_ratio` outputs,
    leave-own-sample-out behavior, sample-equal group aggregation, and support diagnostics.
  - Preserve the current `differential_abundance` behavior for legacy checkpoints but document it
    as a legacy descriptive alias and issue a semantic warning when grouped output is requested.
  - Replace post hoc donor mean-centering in v2 summaries with explicit within-donor W22-minus-W00
    contrasts and donor-block summaries.
  - Add optional global-posterior shrinkage selected only by leave-one-sample-out predictive score;
    no shrinkage setting becomes default from real-data biological concordance alone.
  - Add seed-ensemble summaries that separate within-model posterior uncertainty from
    between-training-seed variability.
  - Do not return per-cell inferential p-values, gene p-values, or FDR from this API.
- **Dependencies:** MRTL-03 and MRTL-05.
- **Affected areas:** `src/scvi/external/mrtotalvi/_stats.py`, `_model.py`, documentation and tests.
- **Validation:** Synthetic density fixtures, leave-one-sample leakage tests, donor-pairing tests,
  seed-order invariance, and null-label permutation diagnostics.
- **Acceptance:** The API is numerically stable and honestly labeled; failure of reliability gates
  keeps it experimental without blocking Milo-based inferential DA.
- **Evidence:** Pending.

### [ ] MRTL-07: Add the Milo bridge and calibrated DA benchmark

- **Status:** not started
- **Outcome:** MrTotalVI representation quality and DA estimator quality are evaluated separately
  using biological-sample replication and matched inputs.
- **Actions:**
  - Export sparse cell-neighborhood membership, neighborhood-by-sample counts, sample metadata,
    neighborhood centers/overlap, latent name, and complete provenance.
  - Run Milo separately on log-normalized PCA, TotalVI `z`, C0 `u`, C1 `u`, and each eligible v2
    `u`; run v2 `z` only as a sample-leakage diagnostic.
  - For the primary W00/W22 analysis use `buildGraph(k=30, d=20)`,
    `makeNhoods(prop=0.1, k=30, d=20, refined=TRUE)`, biological sample
    `donor_timepoint`, and the paired GLMM `~ timepoint + (1 | donor)`.
  - Use `norm.method="TMM"`, `glmm.solver="Fisher"`, `REML=TRUE`,
    `fdr.weighting="graph-overlap"`, and `SpatialFDR < 0.10`.
  - Compare Milo and `local_sample_enrichment` on the same v2 `u` to isolate estimator behavior.
  - Evaluate null FDR, DA-only power, DE-only false DA, mixed effects, effect-size/sign calibration,
    cell-level localization AUPRC/IoU, region Jaccard, neighborhood-LFC correlation, and
    sensitivity to `k`, `d`, and neighborhood proportion.
  - Treat real human biological concordance as supportive only; semi-synthetic truth determines
    calibration.
- **Dependencies:** MRTL-02 through MRTL-06.
- **Affected areas:** `benchmarks/mrtotalvi/`, R4_51 comparator scripts, result schema and
  aggregation.
- **Validation:** R fixture reproduces a known miloR toy result; sample/design rows match
  neighborhood-count columns exactly; graph and result mapping preserve cell IDs.
- **Acceptance:** Milo runs successfully for every manifested representation, and estimator versus
  representation conclusions can be derived without mixing their estimands.
- **Evidence:** Pending.

### [ ] MRTL-08: Run the adaptive three-seed fast screen and issue a stop/go verdict

- **Status:** not started
- **Outcome:** Exactly one v2 candidate is frozen for expanded validation, or the branch stops with
  a falsifiable failure report.
- **Actions:**
  - Run C0-C4 at seed 0 on the canonical human cohort.
  - Run 10 independently generated instances of each fast-screen semi-synthetic scenario for each
    eligible representation without retuning against scenario truth.
  - Disqualify seed-0 configurations that fail reconstruction, biology, invariance, or DE-only
    false-DA gates.
  - Advance C0, C1, and the highest-ranked eligible v2 configuration to seeds 1 and 2.
  - Freeze the selected architecture and hyperparameters before inspecting the macaque data.
  - Publish an internal `candidate`, `stop`, or `blocked` verdict; never label this fast screen
    publication-ready.
- **Dependencies:** MRTL-00 through MRTL-07.
- **Affected areas:** Immutable human benchmark runs and aggregate decision artifact.
- **Validation and fast-screen gates:**
  - Centering and decoder invariants pass at their specified numerical tolerances.
  - Held-out predictive performance is no more than 2% worse than C1.
  - Cell-state conservation is no more than 0.02 below C1.
  - Median cross-seed `u` kNN Jaccard is at least 0.60 and no worse than C1.
  - Counterfactual target-distance rank Spearman is at least 0.80.
  - Top-region effect-direction agreement across seeds is at least 0.90.
  - Significant-region Jaccard across seeds is at least 0.60.
  - At nominal FDR 0.10, median empirical FDP is at most 0.15 in null and DE-only scenarios.
  - DA power/localization is no more than 0.05 below Milo on PCA/TotalVI, and v2 improves over C0
    on at least one preregistered scenario without losing another by more than 0.05.
- **Acceptance:** All gates pass for one frozen v2 candidate, or the implementation tracker records
  which hypothesis failed and no candidate is promoted.
- **Evidence:** Pending.

### [ ] MRTL-09: Evaluate DE with LEMUR, miloDE, and pseudobulk references

- **Status:** not started
- **Outcome:** Counterfactual expression effects are either validated as a secondary capability or
  explicitly excluded from the release/publication claim.
- **Actions:**
  - Start only after a v2 candidate passes MRTL-08.
  - Rename/reframe cell-level decoded output as descriptive counterfactual effects; do not present
    current cell-by-covariate p-values as gene-level DE.
  - Run LEMUR with `design = ~ donor + timepoint`, `test_fraction=0.5`, W22-minus-W00 contrast, and
    held-out pseudobulk validation grouped by donor and timepoint using glmGamPoi plus
    difference-in-difference.
  - Run miloDE 0.1.0 with `assign_neighbourhoods` and `de_test_neighbourhoods` on the same counts,
    cells, sample IDs, contrast, and reduced dimensions; label it a development/preprint
    sensitivity comparator.
  - Run donor-pseudobulk edgeR/dreamlet globally and within preregistered cell types as the
    independent count-based reference.
  - Compare simulated effect truth, gene sign/rank, neighborhood localization, pathway direction,
    seed stability, and held-out likelihood/calibration.
  - Evaluate RNA first. Protein effects remain descriptive unless a separately justified
    sample-level protein model passes its own calibration.
- **Dependencies:** Passing MRTL-08 and the public decoder from MRTL-05.
- **Affected areas:** `_stats.py`, DE documentation, `benchmarks/mrtotalvi/` R/Python comparator
  track.
- **Validation:** LEMUR train/test separation is preserved; no cells used for neighborhood
  validation leak into selection; all comparators share the canonical input universe.
- **Acceptance:** DE is promoted only if simulated sign/effect calibration passes, real-data
  genome-wide concordance is positive, pathway directions are correct, and seed stability passes.
  Otherwise mark DE experimental/unsupported without blocking latent+DA publication.
- **Evidence:** Pending.

### [ ] MRTL-10: Run publication-grade simulation and untouched macaque confirmation

- **Status:** not started
- **Outcome:** Independent evidence establishes or rejects publication readiness for latent
  decoding and Milo-based DA on MrTotalVI `u`.
- **Actions:**
  - Increase the frozen candidate to at least 10 training seeds.
  - Run at least 20 independent replicates for every calibrated scenario across sample sizes,
    effect sizes, rare-state frequencies, cell-count imbalance, and batch-confounding levels.
  - Add model-based RNA/protein count simulations with known same-cell registered-sample
    counterfactual truth, plus the misspecified human semi-synthetic scenarios.
  - Keep human data as development evidence. Run the already available macaque CITE-seq only after
    configuration freeze and make no architecture/hyperparameter change in response.
  - Validate factual and counterfactual RNA/protein prediction, uncertainty coverage, latent
    geometry, DA FDR/power/localization, and cross-seed reproducibility.
  - Report macaque failure as confirmatory failure; do not replace the dataset or redefine the
    primary endpoint.
- **Dependencies:** Passing MRTL-08; MRTL-09 is optional.
- **Affected areas:** Publication manifests, simulation harness, human and macaque immutable runs.
- **Validation and publication gates:**
  - Nominal DA FDR is 0.10 with upper 95% confidence bound no greater than 0.12.
  - DE-only scenarios do not inflate DA beyond the same bound.
  - Power is no more than 0.05 below Milo reference representations unless localization improves
    by a preregistered, statistically supported margin.
  - Counterfactual predictive intervals achieve preregistered coverage tolerance.
  - Latent, effect, and significant-region seed-stability gates hold across all 10 seeds.
  - Direction and localization reproduce on untouched macaque data without retuning.
- **Acceptance:** A signed go/no-go report distinguishes validated capabilities, descriptive
  capabilities, failed gates, and deferred claims.
- **Evidence:** Pending.

### [ ] MRTL-11: Complete release validation, documentation, and gated promotion

- **Status:** not started
- **Outcome:** A reviewable release candidate preserves legacy behavior and makes no claim beyond
  the evidence established in MRTL-10.
- **Actions:**
  - Resolve all MrTotalVI Ruff findings, including undefined annotations, and adopt a fail-on-new-
    warning policy.
  - Remove or fail closed on rank-deficient/singular statistical designs rather than accepting
    warning-heavy pseudoinverse output as validation.
  - Run targeted, integration, save/load, CPU/GPU, chunk-invariance, and full repository tests in
    the declared release environment.
  - Test old checkpoint loading, unchanged legacy outputs, v2 round trips, and explicit downgrade
    to legacy behavior.
  - Add API reference, tutorial, model card, limitations, registered-sample constraint, statistical
    counterfactual terminology, DA estimand guidance, and Milo bridge instructions.
  - Require an independent scientific review of the immutable aggregate before changing defaults.
  - Promote `centered_v2` to the default only through a new ADR after every publication and release
    gate passes; otherwise keep it opt-in or withdraw it.
- **Dependencies:** MRTL-10 and, if claimed, passing MRTL-09.
- **Affected areas:** MrTotalVI source/tests, documentation, changelog, ADRs, benchmark publication
  manifest.
- **Validation:**
  - `python -m ruff check src/scvi/external/mrtotalvi tests/external/mrtotalvi benchmarks/mrtotalvi`
  - Targeted and full pytest suites in the release environment.
  - Documentation build and example execution.
  - Independent reproduction from immutable manifests.
- **Acceptance:** Clean engineering gates, reproducible artifacts, backward compatibility, and a
  claim-by-claim scientific sign-off. Failed scientific gates cannot be overridden by passing
  software tests.
- **Evidence:** Pending.

## Final Validation

The work is complete only when:

- Legacy checkpoints and outputs remain available and tested.
- V2 latent decomposition satisfies exact centering and counterfactual identity invariants.
- Public decoder estimands, protein expectations, uncertainty, batch/panel policies, and support
  diagnostics are explicit and validated.
- Native local enrichment is clearly descriptive and cannot be confused with replicate-level DA.
- Milo DA uses biological-sample replication, matched cells/features, paired design, and calibrated
  FDR.
- Representation and estimator conclusions are reported separately.
- The fast screen, publication simulations, human development analysis, and untouched macaque
  confirmation have distinct immutable manifests.
- DE is included only if it independently passes; otherwise documentation states it is unsupported.
- Release, methods, and biological-interpretation verdicts are recorded separately.

## Decision and Evidence Log

- 2026-07-25 — Plan reviewed and initialized. Implementation not started.
- 2026-07-25 — User selected side-by-side v2, registered-sample-only counterfactuals, and a
  three-seed fast screen.
- 2026-07-25 — Current DA evidence: LayerNorm + initialized frozen VampPrior reduced the SD of one
  W22 enrichment scalar from 9.46 (older/default-like configuration) to 0.192 across three seeds;
  this is candidate evidence, not calibration.
- 2026-07-25 — Current paired DTP RNA DE evidence: genome-wide Spearman versus PyDESeq2 `-0.240`,
  sign agreement `0.424`, top-100 sign agreement `0.18`, and all 12 inspected IFN genes reversed.
- 2026-07-25 — Comparator stack verified in R4_51: miloR 2.6.0, LEMUR 1.11.1, miloDE 0.1.0,
  glmGamPoi 1.22.0, and edgeR 4.8.2.
- 2026-07-25 — Source preflight found 30 Ruff findings. The default shell failed during
  scanpy/Numba import and did not provide a reproducible targeted pytest run; MRTL-00 is blocking
  before new benchmark execution.

## Risks, Assumptions, and Blockers

- **Blocker:** A reproducible supported Python environment is not currently demonstrated. Clear it
  in MRTL-00 before code or benchmark promotion.
- **Scientific limitation:** Registered-sample counterfactuals do not establish causal effects.
- **Identifiability risk:** Residual centering removes translation ambiguity but does not guarantee
  complete biological disentanglement; invariant geometry and truth-based decoding remain required.
- **Encoder risk:** Sample-blind `u` may remove useful information or degrade reconstruction.
  Conditional alignment is a separately reviewed fallback, not an automatic next tweak.
- **DA risk:** Real-data Milo agreement is not truth. Simulation and semi-synthetic calibration
  govern promotion.
- **Graph risk:** Building DA neighborhoods in `z` may encode the tested sample effect directly;
  `u` is primary and `z` is diagnostic.
- **Comparator risk:** miloDE is a development/preprint package. Pin its source and never make it
  the sole DE validator.
- **Data assumption:** The human and macaque H5ADs, biological pairing, raw counts, protein data,
  and stable cell IDs remain available and legally usable.
- **Compute assumption:** The fast screen is limited to the human W00/W22 cohort and three seeds.
  MRTL-10 requires separately authorized publication-scale compute.
- **Governance:** No failed run may update a promoted/latest pointer, and unmanifested `.scratch`
  results remain exploratory.
