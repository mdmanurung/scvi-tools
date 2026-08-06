# Frozen Execution Plan: MrTotalVI Stable Latent and DA Redesign

Frozen: 2026-07-26  
Repository: `/exports/para-lipg-hpc/mdmanurung/scvi-tools`  
Base commit: `d8c8e997a67997a53f55923eb3ab14e6cf06f94c`  
Source tracker:
`docs/plans/2026-07-26-mrtotalvi-stable-latent-da-redesign.md`  
Source-tracker SHA-256:
`b7ee438f41c7c2b8cf29b140bc226f2044c9c47d8040183aeaeb71f50d6f2dcc`

## Objective

Determine whether an opt-in, TotalVI-compatible sample-blind MrTotalVI encoder can produce:

1. a stable and useful shared `u`;
2. a factual `z` that is non-inferior to matched TotalVI; and
3. calibrated replicate-level Milo differential abundance.

The terminal scientific result is exactly one eligible opt-in candidate or a documented `stop`.
No result may change defaults or establish a publication claim.

## Frozen Boundaries

- Preserve `legacy`, `centered_v2`, `sample_conditioned`, and `sample_blind` behavior and evidence.
- Preserve the original MrTotalVI packet:
  - plan SHA-256 `4011a896...`;
  - tasks SHA-256 `4395026f...`;
  - handoff SHA-256 `c56e3d44...`.
- Do not overwrite immutable pilot, engineering, historical-comparator, or oracle artifacts.
- Do not run DE, LEMUR, miloDE, macaque validation, new-sample surgery, non-MAP residual work,
  minified-mode work, default promotion, publication, commit, or push.
- Do not tune from factual human W22-minus-W00 DA results.
- External environment writes, network access, GPU jobs, or scheduler submission require a
  separate approval when reached.
- Preserve all unrelated dirty-worktree changes.

## Frozen Human Lineage

For MrTotalVI human development only:

- preserve current harmonized-human order;
- intersect harmonized immune IDs with the final-assembly parent;
- join `pass_qc` from the final parent and retain only `pass_qc=True`;
- retain W00/W22 and donors present at both timepoints;
- expect exactly 46,817 cells and 10 complete donors;
- never modify or reclassify either source H5AD or the upstream global selected-run state;
- stop on source-hash drift, count drift, cell mismatch, count disagreement, or missing metadata;
- hash-split within donor-timepoint;
- compute 5,000 Pearson-residual HVGs on model-training cells only; and
- retain the ordered 130 biological, non-isotype proteins.

The derived object and manifest must be written to a new immutable run directory with no `latest`
pointer.

## Frozen Comparators and Redesigns

| ID | Definition |
|---|---|
| B0 | stock scVI `z`, RNA-only contextual comparator |
| B1 | stock TotalVI `z`, primary factual-`z` comparator |
| B2 | legacy MrTotalVI C0 `u` and factual `z` |
| B3 | centered, sample-conditioned C2 `u` and factual `z` |
| D0 | current C4; convergence control |
| D1 | D0 with exact TotalVI per-modality input normalization only |
| D2 | TotalVI-normalized, TotalVI-`FCLayers`, sample-blind encoder; frozen initialized VampPrior |
| D3 | D2 with trainable MoG |
| D4 | D2 with sample-equal weighting |
| D5 | D2 with trainable MoG and sample-equal weighting |

All nonlisted axes are held fixed. D1-D5 remain package-private during selection.

## Ordered Execution

### 1. Freeze governance and regression state

- Record exact hashes for the source tracker, old packet, validation report, blocker, existing
  benchmark runs, package source, and regression fixtures.
- Add a redesign amendment/ADR and typed D0-D5 configuration contract.
- Add formal verdict validation for `candidate`, `stop`, and `blocked`.
- Run the current narrow package and benchmark suites before source changes.

### 2. Seal the human development cohort

- Implement a fail-closed derivation utility and tests.
- Read source H5ADs without modifying them.
- Verify exact cell, QC, count, metadata, feature, and panel lineage.
- Create the hashed train/held-out split and training-only 5,000-HVG ranking.
- Seal the derived cohort, feature manifest, source/code/environment hashes, and lineage amendment.
- If the expected source hashes or 46,817-cell contract do not hold, record `blocked` and continue
  only with synthetic/package work.

### 3. Extend the benchmark contract

- Add B0/B1 matched runners and separate `u`/factual-`z` exports.
- Add convergence histories, best-checkpoint identity, reconstruction/KL terms, posterior scales,
  latent effective rank, residual magnitude, gradients, parameters, time, and memory.
- Add rotation-appropriate CKA/Procrustes, kNN stability, state conservation, sample leakage,
  technical-batch mixing, modality-specific held-out loss, and predictive calibration.
- Never rank scVI on protein or multimodal ELBO.

### 4. Diagnose current C4

- Run B1-B3 and D0 at seeds 0-2 on the mixed, unequal-cell, sealed 500-cell, and—if available—
  canonical human fixtures.
- Use common convergence controls: check every five epochs, minimum 50 epochs, maximum 400 epochs,
  patience 30 checks, and best-checkpoint restoration.
- Do not extend or retune only one candidate.
- If D0 ultimately passes all latent and DA gates, retain current `sample_blind` and do not create a
  redundant public mode.

### 5. Implement D1-D5 test-first

- First add one failing public-behavior test for the TotalVI-equivalent input transform, then its
  minimal implementation.
- Next add one failing behavior test for a sample-blind TotalVI-`FCLayers` posterior, then its
  minimal implementation.
- Add technical-covariate, gradient, centering, checkpoint, legacy, and MrMultiVI tests
  incrementally.
- Keep `EncoderUZ.forward()` and all old encoder branches unchanged.
- Construct D1-D5 only through benchmark-private configuration until selection.

### 6. Run the adaptive known-truth screen

- Stage A: B1-B3 and D0-D5, seed 0, three independent instances of every frozen scenario.
- Disqualify contract, convergence, nonfinite, collapse, leakage, or factual-`z` failures.
- Stage B: B1, B2, and at most two eligible redesigns, seeds 0-2, ten independent instances of
  every frozen scenario.
- Use separate truth, training, and evaluation RNG streams.
- Write atomic, exact-grid, hash-verified runs and stop downstream rules on gate failure.

### 7. Build and validate Milo

- Export one cell-order-locked SingleCellExperiment contract with PCA, B0/B1, B2 `u`/`z`, and
  eligible redesign `u`/`z`.
- Use `buildGraph(k=30, d=20)`.
- Use `makeNhoods(prop=0.1, k=30, d=20, refined=TRUE,
  refinement_scheme="graph")`.
- Use biological replicate column `donor_timepoint`.
- Reorder design rows exactly to neighborhood-count columns.
- Use primary paired GLMM `~ timepoint + (1|donor)` with graph-overlap FDR, Fisher solver, REML,
  TMM normalization, and `fail.on.error=FALSE`.
- Require zero failed or NA primary fits; fixed-donor GLM is reporting-only.

### 8. Run the human non-inferiority screen

- Train B0-B3 and at most two eligible redesigns on the sealed human cohort at seeds 0-2,
  20 latent dimensions, identical convergence controls, 5,000 genes, and 130 proteins.
- Evaluate held-out prediction, `u` leakage/state/stability, factual-`z` state/stability, rank,
  runtime, and memory.
- Run frozen within-donor label permutations and human-geometry null, DA-only, and DE-only
  semi-synthetic checks.
- Do not inspect factual W22-minus-W00 Milo results.

### 9. Select or stop

- Disqualify any hard-gate failure.
- Rank survivors by median known-truth Milo localization at SpatialFDR 0.10.
- Treat differences below 0.02 as tied.
- Break ties by factual-`z` predictive loss relative to TotalVI, then `u` cross-seed kNN Jaccard,
  then fewer trainable parameters.
- Reproduce selection independently from per-run results and test result-order invariance.
- Freeze configuration and code hash before any factual human DA result is opened.

### 10. Expose only the frozen eligible public mode

If a D2-family candidate passes every synthetic, Milo, and human safety gate, expose:

```python
u_encoder_mode: Literal[
    "sample_conditioned",
    "sample_blind",
    "sample_blind_totalvi",
] = "sample_conditioned"
```

`sample_blind_totalvi` requires `centered_v2`, performs TotalVI modality normalization, uses a
TotalVI-style sample-blind posterior trunk, and retains declared technical covariates. Its topology
must load only from matching metadata. Cross-topology semantic overrides are refused.

If D1 alone passes, expose `sample_blind_scaled`. If only D0 passes, expose nothing new. A `stop` or
`blocked` verdict exposes no mode.

### 11. Post-freeze report and verification

- For `candidate`, run factual paired W22-minus-W00 Milo without retuning.
- For `stop`, skip factual human DA.
- Run package/benchmark regressions, Ruff, compilation, Zarr tests, docs, wheel, import smoke, and
  normal non-GPU tests where feasible.
- Perform a read-only compatibility, lineage, API, and statistical review.
- Seal a report separating software facts, synthetic mechanism evidence, DA calibration, human
  descriptive evidence, failures, and blocked claims.

## Hard Gates

An eligible redesign must satisfy all of:

- centered identity maximum absolute error at most `1e-6`;
- `u` within-state sample predictability no greater than the permutation-null 95th percentile plus
  `0.02`;
- `u` state and kNN-state accuracy no more than `0.02` below TotalVI;
- `u` cross-seed 15-neighbor Jaccard at least `0.60` and no more than `0.05` below TotalVI;
- factual-`z` total held-out predictive loss no more than 2% worse than TotalVI in every seed;
- factual-`z` RNA and protein held-out losses each no more than 3% worse than TotalVI;
- factual-`z` state metrics no more than `0.02` below TotalVI;
- factual-`z` cross-seed Jaccard at least `0.60`;
- effective latent rank at least half the configured dimension;
- every registered residual row trained with finite nonzero gradients;
- median Milo FDP at most `0.15` in null and DE-only scenarios at SpatialFDR 0.10;
- Milo power/localization no more than `0.05` below the best PCA/TotalVI reference; and
- improvement over B2 of at least `0.05` in one preregistered DA scenario without a loss greater
  than `0.05` in another.

## Stop Conditions

- Source data or plan hashes drift from the frozen values.
- The 46,817-cell human lineage cannot be reproduced exactly.
- Shared cell counts or covariates disagree between declared sources.
- A required dependency or permission is absent.
- A requested action would overwrite an existing immutable run or unrelated dirty-worktree file.
- Primary Milo fits contain failed or NA fits.
- No redesign passes the hard gates.
- A step would require factual-human outcome-guided tuning, macaque access, DE, publication, default
  promotion, scheduler submission, external writes, or network access beyond current authority.

## Required Terminal Artifacts

- updated progress tracker and issue evidence;
- redesign amendment/ADR and frozen candidate schema;
- immutable human lineage/feature/split manifest or explicit blocker;
- tested D0-D5 benchmark contract;
- immutable synthetic and, if unblocked, human runs;
- Milo export, design, diagnostics, and calibrated aggregate;
- formal `candidate`, `stop`, or `blocked` report;
- package/build verification for any exposed mode; and
- independent review record.
