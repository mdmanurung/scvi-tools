# Frozen Execution Plan: MrTotalVI v2 Latent Decoding and Differential Abundance

- Frozen: 2026-07-25
- Repository: `/exports/para-lipg-hpc/mdmanurung/scvi-tools`
- Source plan: `docs/plans/2026-07-25-mrtotalvi-v2-latent-da.md`
- Primary scope: latent `u`/`z` semantics, counterfactual decoding, and calibrated differential
  abundance
- Secondary scope: differential expression, only after the latent and DA gate
- Default behavior: legacy

## Objective

Implement an opt-in, backward-compatible MrTotalVI v2 whose registered-sample latent
counterfactuals have an explicit centered decomposition, whose decoded RNA/protein quantities have
named estimands, and whose primary DA inference is performed with replicate-aware Milo on a
sample-blind `u` representation. Establish scientific reliability with exogenous-truth simulation,
three-seed development screening, precision-driven DA calibration, and a locked external macaque
analysis. Keep DE experimental unless it independently passes simulation and sample-level
validation.

## Repository Facts That Govern Execution

- `HEAD` was `d8c8e997` when this packet was frozen. The worktree contains many unrelated untracked
  CytoANVI and benchmark artifacts. Preserve them and never run a destructive cleanup or reset.
- Current MrTotalVI code is in `src/scvi/external/mrtotalvi/`; `EncoderUZ` is also reused by
  MrMultiVI. V2 behavior must therefore be implemented in a mode-scoped MrTotalVI path unless a
  shared change is proven numerically inert for MrMultiVI.
- Old checkpoints contain no `hierarchy_mode`. Missing mode metadata always means `legacy`; mode
  must never be inferred from tensor values or state-dict shape.
- `benchmarks/ANALYSIS_MANIFEST.md` marks the earlier D-041 VampPrior claim as refuted because its
  supporting artifact was absent. C1 is an unvalidated candidate, not an established baseline. C0
  is the only accepted starting baseline until C1 is reproduced from an immutable manifest.
- The repository-level `environment-lock.yml` is a lower-bound environment specification, not an
  exact lock, and the named `cytoanvi-release` environment is not currently installed.
- The authoritative human and macaque H5ADs exist, but they are newer than
  `human_prepared_10k.h5ad`. The current H5AD headers show 137 protein columns, not the 130 assumed
  by the source plan. No feature count is frozen until the data audit distinguishes biological
  antibodies from controls and hashes the ordered retained set.
- The macaque dataset has already been used by repository benchmark scripts. It is a locked
  external validation dataset, not an untouched dataset. Its observed timepoints are W00, W08, and
  W46 rather than human W22.
- miloR 2.6.0 in R4_51 exposes the required GLMM arguments. The primary call must name the reduced
  dimension explicitly in both `buildGraph()` and `makeNhoods()`.

## Claim Boundaries

- “Counterfactual” means a model-based registered-sample transformation, not a causal
  intervention and not extrapolation to a new donor/sample.
- Centering anchors the output decomposition; it does not prove full parameter identifiability or
  biological disentanglement.
- The native aggregated-posterior score is descriptive local sample compatibility. Milo owns
  replicate-level DA inference. Their p-values, FDR, or power must never be compared as if they
  shared an estimand.
- Human W00-versus-W22 is development evidence. Macaque validation uses a separately trained model
  with the frozen human-selected architecture and a predeclared macaque contrast; it is not
  checkpoint transfer because v2 supports registered samples only.
- Passing tests or a three-seed screen is not publication readiness. Publication claims require
  the later precision-driven simulation and locked external-validation gates.

## Frozen Probabilistic Contract

For a cell-level `u` and `S` registered samples, let the existing attention network return raw MAP
residuals

`r_s = qz_raw(u, s), s = 1, ..., S`.

V2 defines

`eps_centered_s = r_s - (1 / S) * sum_t(r_t)`

and

`z_s = z_base(u) + eps_centered_s`.

The v2 contract is:

1. `hierarchy_mode="legacy"` remains the constructor default and reproduces current numerical
   behavior and state-dict keys.
2. `hierarchy_mode="centered_v2"` is initially supported only with `use_map=True`.
3. The factual likelihood uses the centered residual for the observed registered sample.
4. The v2 residual penalty is the equal-sample mean of the proper raw residual penalties:
   `(1 / S) * sum_s[-log Normal(r_s; 0, exp(pz_scale))]`.
   The denominator keeps its scale comparable to the legacy one-sample penalty while constraining
   the raw common mode and every registered embedding row.
5. `kl_u_weight`, `kl_z_weight`, and the existing global KL annealing apply after the equal-sample
   mean. No unregistered target or stochastic residual-centering approximation is allowed in v2.
6. The API exposes both `eps_raw` and `eps_centered`; the unqualified name `eps` is not used in the
   new dataset.
7. Required identities, per cell and draw, are:
   `mean_sample(eps_centered)=0`,
   `mean_sample(z)=z_base`, and
   `z_s-z_t=eps_centered_s-eps_centered_t`.

Implement the all-target residual and centering helper in MrTotalVI/MrTotalVAE without changing
the default behavior of shared `EncoderUZ.forward()`. Chunking may reduce peak memory but must
produce the same result and gradient as the unchunked calculation.

## Frozen `u` Encoder and Candidate Contract

- Add `u_encoder_mode={"sample_conditioned","sample_blind"}`.
- Legacy defaults to `sample_conditioned`.
- `sample_blind` bypasses both conditional-normalization sample gamma/beta terms and the explicit
  sample embedding in `EncoderXU_TotalVI`; it retains declared technical covariates. Keep legacy
  modules and parameter names present so state dictionaries remain compatible.
- C0: reproduced current LayerNorm + MoG legacy baseline.
- C1: LayerNorm + data-initialized VampPrior pseudoinputs frozen after initialization, with mixture
  logits still trainable. C1 is eligible only after fresh immutable reproduction.
- C2: C1 + `centered_v2`, sample-conditioned `u`; latent-decoding diagnostic only.
- C3: C2 + sample-balanced observations using weights `N / (S * n_s)` under a mean loss, so mean
  weight is one and duplicating cells within one sample does not change sample contribution;
  latent-decoding diagnostic only.
- C4: C2 + `u_encoder_mode="sample_blind"`; this is the only v2 representation eligible for the
  primary Milo DA claim.
- If C4 fails reconstruction/biology gates, primary MrTotalVI-v2 DA stops. Do not substitute a
  sample-conditioned `u`, add adversarial alignment, or tune against real biological concordance.

## Frozen Counterfactual API Contract

Add:

- `get_counterfactual_latent(...) -> xr.Dataset`
- `get_counterfactual_expression(...) -> xr.Dataset`

Latent variables use dimensions:

- `u`: `(draw, cell_name, latent_u_dim)`
- `z_base`: `(draw, cell_name, latent_dim)`
- `eps_raw`, `eps_centered`, `z`: `(draw, cell_name, target_sample, latent_dim)`
- `admissible`, `target_support`: `(cell_name, target_sample)`

Expression variables use:

- RNA: `(draw, cell_name, target_sample, gene)`
- Protein: `(draw, cell_name, target_sample, protein)`

RNA outputs are:

- `rna_scale`: normalized decoder composition, independent of library size
- `rna_rate`: expected count rate under an explicit
  `library_policy={"observed","specified","sample_balanced_marginal"}`

Protein outputs are:

- `protein_background_component_mean`
- `protein_foreground_component_mean`
- `protein_foreground_probability`
- `protein_background_contribution`
- `protein_foreground_contribution`
- `protein_total_mean`
- `protein_available`

The log-normal background component mean is
`exp(back_alpha + 0.5 * back_beta**2)`. Mixture component means and
probability-weighted contributions are distinct variables.

`inference_mode="latent_mean"` is a plug-in decode at `E[u]` and must be labeled as such in dataset
attributes. `inference_mode="posterior_mc"` estimates `E[f(u)]`, uses common posterior draws across
all target samples, and returns declared quantiles/intervals. Counterfactual coverage claims use
`posterior_mc`, never factual reconstruction alone.

Defaults fix technical context across targets with `batch_policy="observed"`,
`panel_policy="observed"`, and `library_policy="observed"`. Specified and sample-balanced marginal
policies must record their exact values/weights. Proteins absent from the fixed panel are
unavailable, not observed or silently imputed.

Materialized output is guarded by an estimated 512 MiB limit. Larger requests must subset or write
chunked Zarr to an explicit path. Chunking and storage mode must not change values.

Existing `get_local_sample_representation()` and `get_local_sample_distances()` remain numerically
unchanged for legacy models. They may call shared internals only after frozen legacy-array tests
pass.

## Frozen Local-Enrichment Contract

Add `local_sample_enrichment()` with:

- per-cell/per-sample `log_density`
- optional sample-equal group `log_ratio`
- leave-own-cell-out density for the factual sample
- `n_reference_cells`, self-exclusion, and finite-support diagnostics
- explicit donor-block W22-minus-W00 summaries
- posterior-draw and between-training-seed summaries reported separately

Keep `differential_abundance()` numerically unchanged as a legacy descriptive API. A semantic
warning may be added for grouped inferential-sounding use, but it must not silently change output.
Neither API returns inferential p-values or FDR.

## Data, Metric, and Simulation Contract

Before training seed 0, create tracked manifests and a metric dictionary that freeze:

- canonical source paths, SHA-256 hashes, modification times, cell IDs, integer-count checks,
  inclusion/exclusion rules, donor/timepoint pairing, protein control exclusions, ordered genes,
  ordered proteins, and technical covariates
- human development cohort: QC-pass, known-timepoint W00/W22 cells with complete paired donor
  blocks; derive it from the current authoritative H5AD rather than the older prepared H5AD
- feature selection: fit Pearson-residual HVGs inside the development-training partition without
  timepoint labels; every comparator receives the same ordered cells/features
- macaque external-validation manifest: prior-exposure disclosure, homolog/protein mapping,
  paired contrast (prefer W08-W00 because both species contain it), sample unit, exclusions, and
  primary endpoint, signed before any new macaque model/result inspection
- metric definitions, direction, aggregation unit, split, threshold, CI, missing/no-call rule, and
  tie-break for reconstruction, biological conservation, kNN overlap, CKA/Procrustes geometry,
  counterfactual truth, effect variance, DA calibration, localization, and stability

Semi-synthetic truth must be defined in an exogenous simulator latent, preregistered annotation, or
trajectory coordinate that is not any evaluated representation. Each model is retrained whenever
the scenario changes the training data. Truth RNG, training RNG, and evaluation RNG streams are
independent. Perturbations preserve raw integer-count/library/protein constraints unless that
violation is the declared scenario.

Overlapping-neighborhood calls are projected to cells for localization scoring. Cross-seed
stability uses cell-level effect/call maps or a frozen reference partition, never unmatched Milo
neighborhood IDs. Zero/zero call sets, ties, NAs, and no-discovery FDP are defined in the metric
dictionary.

FDR is estimated as mean FDP across independent datasets, with `FDP=0` when there are no
discoveries. Global-null calibration also reports `P(R>0)`. Use paired scenario-level differences
and a predeclared one-sided CI. Publication calibration starts with at least 200 null and 200
DE-only independent replicates and continues in fixed increments until CI half-width is at most
0.02 or the preregistered cap is reached. Record an inconclusive result at the cap; never replace
mean FDR with median FDP.

The three-seed screen may use a smaller engineering scenario suite but cannot make an FDR or
publication-readiness claim.

## Milo Contract

For each representation, store its exact cell order, embedding checksum, graph, neighborhood
membership, neighborhood-by-sample counts, sample design, and results. Use the same cells and
biological sample unit for all representations, but acknowledge that rebuilding a graph changes
the tested neighborhood family.

Primary human call:

```r
set.seed(42)
milo <- buildGraph(
  milo,
  k = 30,
  d = 20,
  reduced.dim = latent_name
)
milo <- makeNhoods(
  milo,
  prop = 0.1,
  k = 30,
  d = 20,
  refined = TRUE,
  reduced_dims = latent_name,
  refinement_scheme = "graph"
)
milo <- countCells(
  milo,
  meta.data = data.frame(colData(milo)),
  samples = "donor_timepoint"
)
design_df <- design_df[colnames(nhoodCounts(milo)), , drop = FALSE]
results <- testNhoods(
  milo,
  design = ~ timepoint + (1 | donor),
  design.df = design_df,
  fdr.weighting = "graph-overlap",
  glmm.solver = "Fisher",
  REML = TRUE,
  norm.method = "TMM",
  fail.on.error = FALSE,
  BPPARAM = BiocParallel::SerialParam()
)
```

Before testing, assert unique design rows, exact row/count-column identity, the intended reference
level and W22-minus-W00 sign, full fixed-effect rank, `d <= ncol(reducedDim)`, nonempty biological
samples, and finite positive normalization factors. Record `checkSeparation()`, convergence, NA
fits, and per-neighborhood sample counts. `fail.on.error=FALSE` permits diagnostics to finish, but
the primary benchmark fails if any tested neighborhood has a failed/NA fit. Sensitivity values for
`k`, `d`, and `prop` are reporting-only and never choose the candidate.

Representation comparisons use common cell-level simulation truth. `local_sample_enrichment` is
reported only as descriptive association/effect direction on the same `u`; it is excluded from
Milo FDR/power tables.

## Execution Stages and Gates

### Stage 0 — Governance, evidence, data, and environment

Create `.scratch/mrtotalvi-v2/PRD.md` and numbered issues, freeze the current Git/dirty-worktree
inventory, correct the unsupported C1 evidence claim, audit canonical data, lock the macaque
analysis, and create a project-specific `mrtotalvi-v2` Python environment. Export an exact conda
explicit specification, pip freeze, Python/R session information, CUDA driver/GPU details, and
comparator source revisions. No implementation proceeds until targeted imports, Ruff baseline,
targeted pytest baseline, and R toy calls are reproducible.

### Stage 1 — ADR-0007 and test-first latent feasibility

Write and accept `docs/adr/0007-mrtotalvi-v2-centered-counterfactuals.md` with the contracts above.
Add failing tests for checkpoint defaults, raw/centered residual identities, all-target raw
penalty, gradients, deterministic chunking, sample-blind bypass, and MrMultiVI non-regression.
Implement the smallest CPU fixture and one tiny centered-v2 training run. Stop if exact all-target
centering is not computationally practical at 20 samples.

### Stage 2 — Public decoder and descriptive enrichment

Implement the two counterfactual dataset APIs, memory guard/Zarr path, explicit RNA/protein
estimands, and registered-target failures. Then implement `local_sample_enrichment()` and preserve
the legacy DA API. Pass analytic-versus-Monte-Carlo protein tests, common-random-number tests,
factual-identity tests, leave-one-cell-out tests, and save/load tests before benchmark use.

### Stage 3 — Tracked benchmark and simulation harness

Add `benchmarks/mrtotalvi/` with deterministic data validation, configuration, training, decoding,
simulation, Milo export, aggregation, and report entrypoints. Large raw artifacts may be cached in
`.scratch`, but publication evidence requires content hashes and durable artifact URIs in tracked
manifests; `.scratch` paths alone are not durable evidence. Run IDs contain timestamp, code digest,
config digest, and data-manifest digest. Aggregation accepts only explicitly listed run IDs.

### Stage 4 — Milo bridge and calibration fixture

Export matched SCE/H5AD inputs for PCA, TotalVI `z`, C0 `u`, reproduced C1 `u`, and eligible v2
representations. Prove the named-reduced-dimension and design contracts on a toy fixture. Run
exogenous-truth null, DA-only, DE-only, and mixed fixture scenarios. Do not launch the human
three-seed screen until the R bridge completes with zero failed primary fits.

### Stage 5 — Three-seed human development screen

Run C0-C4 at seed 0. Advance C0, reproduced C1, and at most one eligible v2 configuration to seeds
1 and 2 using the frozen metric dictionary and tie-break. C4 is required for a primary DA
candidate. Require noncollapsed counterfactual variance and truth-based decoder recovery in
addition to reconstruction, biology, geometry, and stability gates. Issue only `candidate`,
`stop`, or `blocked`; never `publication-ready`.

### Stage 6 — Optional DE evaluation

Start only after a latent/DA candidate passes Stage 5. Use LEMUR 1.11.1 with an explicit
donor/timepoint-stratified internal cell split, frozen alignment settings, declared contrast, and
within-cell-state glmGamPoi neighborhoods. Use miloDE 0.1.0 as a pinned development sensitivity
analysis. Use integer-count, within-cell-state donor pseudobulk as the primary count-based
reference; global pseudobulk is a separate composition-confounded tissue-level estimand. Same-donor
methods are triangulation, not independent biological validation. Promote DE only from simulation
truth plus external/generalization evidence; otherwise document it as unsupported.

### Stage 7 — Publication-scale calibration and locked external validation

Before scheduler submission, write resource/time/storage estimates and obtain explicit approval for
publication-scale compute. Freeze the candidate, then run at least 10 model-training seeds,
precision-driven simulation replicates, and the locked macaque architecture replication. No
architecture or threshold changes follow macaque inspection. The signed report separates software,
latent-decoding, DA, DE, and external-validation verdicts.

### Stage 8 — Release hardening

Resolve MrTotalVI Ruff findings, fail closed on singular statistical designs, run targeted and full
tests, verify legacy and v2 save/load/downgrade behavior, build documentation/tutorials, and obtain
independent scientific review. `centered_v2` remains opt-in. Changing the default requires a later
ADR after every scientific and release gate passes.

## Stop Conditions

Stop and report evidence when any of the following occurs:

- canonical cell/protein lineage or integer raw counts cannot be established
- an old checkpoint differs from the frozen legacy arrays
- centered identities, raw-residual penalty, gradients, or MrMultiVI non-regression fail
- exact all-target centering exceeds the frozen CPU/GPU memory or runtime budget
- C1 cannot be reproduced; continue with C0 but do not cite the refuted stability claim
- C4 fails reconstruction/biology or calibrated null DA; do not promote conditioned `u`
- Milo design, named embedding, convergence, separation, or sample-count contracts fail
- the simulation cap is reached without the required precision; report inconclusive
- publication-scale compute, durable storage, or external-validation authorization is absent
- DE fails its independent gate; omit DE without blocking a validated latent/DA result

## Validation Commands

Run from the repository root with writable cache paths:

```bash
conda run -n mrtotalvi-v2 python -c "import scvi, torch, anndata, scanpy, xarray"
conda run -n mrtotalvi-v2 python -m ruff check \
  src/scvi/external/mrtotalvi \
  src/scvi/external/mrmultivi \
  tests/external/mrtotalvi \
  tests/external/mrmultivi \
  benchmarks/mrtotalvi
conda run -n mrtotalvi-v2 python -m pytest tests/external/mrtotalvi -q
conda run -n mrtotalvi-v2 python -m pytest tests/external/mrmultivi -q
conda run -n mrtotalvi-v2 python -m pytest tests/benchmarks/mrtotalvi -q
conda run -n mrtotalvi-v2 python -m benchmarks.mrtotalvi.run_fixture
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/Rscript \
  benchmarks/mrtotalvi/tests/test_milo_fixture.R
```

Before full release, also run the repository test suite, documentation build, legacy checkpoint
round trip, v2 round trip, chunk/Zarr invariance, CPU/GPU parity, and independent manifest
reproduction.

