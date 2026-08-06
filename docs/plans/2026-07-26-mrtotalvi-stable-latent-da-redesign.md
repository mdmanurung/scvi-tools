# MrTotalVI Stable-Latent and Differential-Abundance Redesign

> **SUPERSEDED**: This plan is superseded by the operative checklist at
> `docs/review-clear-execute/mrtotalvi-v2-redesign/tasks.md`, with current status tracked in
> `todo/TODO_REGISTRY.md`. Real work has progressed well past the "Status: not started" line below
> — do not trust that line. Check the two files above for what is actually done, in progress, or
> blocked.

Date: 2026-07-26  
Last updated: 2026-07-26  
Project: `/exports/para-lipg-hpc/mdmanurung/scvi-tools`  
Status: not started

## Objective

Develop and evaluate a new opt-in MrTotalVI encoder variant that preserves useful shared `u`
geometry, makes factual `z` non-inferior to matched TotalVI on multimodal representation and
prediction endpoints, and supports calibrated replicate-level differential abundance (DA) through
Milo.

The existing `legacy`, `centered_v2`, `sample_conditioned`, and `sample_blind` semantics,
checkpoints, package evidence, and negative/inconclusive benchmark results remain frozen. The work
may select one new experimental mode or end with a `stop` verdict. It must not change defaults,
claim that centered v2 is better before the gates pass, or use human biological discoveries to tune
the model.

This tracker is a redesign amendment to, not a replacement for, the frozen MrTotalVI packet and
ADR-0007. Macaque confirmation, DE, default promotion, publication claims, and release publication
remain out of scope.

## Decisions

- Primary objective: stable, sample-neutral `u` and calibrated Milo DA.
- Co-primary safety objective: factual `z` must remain useful and be non-inferior to matched
  TotalVI; scVI is an RNA-only contextual comparator, not a multimodal likelihood comparator.
- Compatibility: test new behavior under a separately named opt-in encoder mode; never revise
  `sample_blind` in place.
- Human cell universe: use harmonized-immune cell IDs re-gated by final-parent `pass_qc`, restricted
  to complete W00/W22 donor pairs. The current expected size is exactly 46,817 cells.
- Human selection boundary: use convergence, latent-quality, leakage, stability, and null/synthetic
  safety metrics only. Inspect factual W22-minus-W00 DA results only after the architecture and
  hyperparameters are frozen.
- Counterfactual expression, DE, conditional adversarial alignment, teacher distillation,
  new-sample mapping, minified mode, vectorization, and non-MAP residuals are not part of this
  redesign.

## Review Findings

| Finding | Resolution |
|---|---|
| The historical C4 comparison used three epochs, had low absolute state accuracy, and recorded no convergence history. | Add a convergence-controlled diagnostic before attributing instability to the architecture. |
| Current `sample_blind` removes sample embeddings but still uses `log1p` raw concatenated counts and a custom MLP; TotalVI scales RNA and protein separately and uses its standard posterior trunk. | Test input normalization and encoder topology as separate preregistered ablations. |
| Existing evaluation emphasizes `u`, while users also need a useful factual `z`. | Score `u` and factual `z` separately, with endpoint-specific interpretations and gates. |
| ELBO values are not directly comparable between RNA-only scVI and multimodal models. | Compare TotalVI and MrTotalVI on matched RNA/protein likelihood endpoints; restrict scVI comparison to RNA prediction and latent/DA metrics. |
| Human W00/W22 has three conflicting cell universes. | Select the 46,817-cell harmonized-ID/final-parent-QC intersection and require a signed immutable derivation; reject count or digest drift. |
| Human factual DA has no known truth and could invite outcome-guided model selection. | Select on exogenous-truth simulations and human null/non-inferiority checks; reveal factual human DA only after freeze. |
| An unconditional sample adversary can remove real condition biology needed for DA. | Do not add adversarial or label-conditioned alignment in this phase. Open a separate issue only after this redesign fails. |
| A single composite score could hide a serious failure. | Use hard non-inferiority/calibration gates first, then a fixed lexicographic tie-break. |
| The repository has a large dirty worktree and immutable prior evidence. | Add isolated files, preserve all existing artifacts, use atomic run directories, and never update a `latest` pointer. |

## Candidate and Comparator Matrix

The diagnostic implementations remain package-private until a winner passes every gate.

| ID | Model / representation | Encoder input and trunk | `u` prior | Weighting | Role |
|---|---|---|---|---|---|
| B0 | scVI `z` | RNA-only stock scVI | stock | cell-equal | RNA-only contextual comparator |
| B1 | TotalVI `z` | stock TotalVI multimodal encoder | standard Normal | cell-equal | primary factual-`z` comparator |
| B2 | MrTotalVI C0 `u`, factual `z` | legacy sample-conditioned | MoG | cell-equal | accepted legacy baseline |
| B3 | MrTotalVI C2 `u`, factual `z` | legacy sample-conditioned | frozen initialized VampPrior | cell-equal | centered conditioned control |
| D0 | current C4 | current sample-blind raw-`log1p` MLP | frozen initialized VampPrior | cell-equal | convergence control |
| D1 | transform-only ablation | TotalVI per-modality scaling, current sample-blind MLP | frozen initialized VampPrior | cell-equal | isolates input transform |
| D2 | TotalVI-style sample-blind | TotalVI scaling and `FCLayers` posterior trunk; no biological sample input | frozen initialized VampPrior | cell-equal | principal redesign |
| D3 | D2 prior ablation | same as D2 | trainable MoG | cell-equal | tests prior instability |
| D4 | D2 weighting ablation | same as D2 | frozen initialized VampPrior | sample-equal | tests imbalance |
| D5 | D2 joint ablation | same as D2 | trainable MoG | sample-equal | tests prior-weight interaction |

All models use the same cells, ordered features, latent dimension, technical covariates, decoder
capacity where applicable, train/validation split, optimizer, learning-rate schedule, KL schedule,
early-stopping policy, and random seeds. Parameter counts and wall time are reported rather than
silently treated as matched.

## Other Improvement Families and Disposition

This is the complete disposition of the repository-recorded improvement families relevant to the
next decision; it is not a claim to enumerate every conceivable model.

| Improvement family | Disposition in this plan |
|---|---|
| Longer training and convergence diagnostics | Execute as D0 before attributing failure to architecture. |
| TotalVI-compatible modality scaling | Execute as D1. |
| TotalVI-compatible sample-blind posterior trunk | Execute as D2-D5. |
| MoG versus initialized frozen VampPrior | Execute as D2/D3 and D4/D5 paired ablations. |
| Cell-equal versus sample-equal loss | Execute as D2/D4 and D3/D5 paired ablations. |
| Centered full-sample residuals and raw penalty | Retain unchanged; package mechanism already verified. |
| Conditional/adversarial sample alignment | Defer to a separate issue; it can erase real sample-associated biology needed for DA. |
| TotalVI teacher or relational distillation | Defer; first test direct TotalVI-compatible inputs/trunk without a second trained model. |
| Global-posterior shrinkage for local enrichment | Defer; it does not repair encoder or factual-`z` quality and needs separate predictive selection. |
| Counterfactual RNA/protein calibration | Keep descriptive and report secondary diagnostics; it cannot select this latent/DA redesign. |
| Milo replicate-level DA | Execute as the primary inferential endpoint. |
| LEMUR, miloDE, and pseudobulk DE | Defer until a latent/DA candidate passes; DE cannot rescue a failed representation. |
| New-sample/reference-query surgery | Defer to a separate registered-versus-unseen-sample contract. |
| Minified mode and vectorized decoding/training | Defer until model semantics are selected; require numerical-equivalence work. |
| Non-MAP centered residual posterior | Defer to a new probabilistic derivation and calibration ADR. |
| Default promotion | Prohibited; this plan can produce only an opt-in candidate or a stop verdict. |

### Eligible public API

Only an empirically eligible encoder is exposed:

```python
MrTotalVI(
    ...,
    u_encoder_mode: Literal[
        "sample_conditioned",
        "sample_blind",
        "sample_blind_totalvi",
    ] = "sample_conditioned",
)
```

`sample_blind_totalvi`, if selected, means:

- normalize RNA and protein separately exactly as the matched TotalVI inference path does before
  `log1p`;
- use a TotalVI-style `FCLayers` posterior trunk and Normal output;
- exclude the biological sample key from every encoder input and normalization layer;
- retain explicitly registered technical batch and extra categorical/continuous covariates; and
- require `hierarchy_mode="centered_v2"`.

The new mode has topology-specific checkpoint keys. It must round-trip only with matching metadata;
cross-topology semantic overrides are refused. Missing metadata continues to mean `legacy` plus
`sample_conditioned`. If D0 is the only eligible design, no new public mode is added. If D1 alone
wins, name and document it as `sample_blind_scaled` rather than misrepresenting it as a TotalVI
trunk.

## Progress Summary

- [ ] RDX-00: Freeze governance, hypotheses, and run contracts
- [ ] RDX-01: Resolve and seal the human W00/W22 lineage
- [ ] RDX-02: Extend benchmark diagnostics and matched comparators
- [ ] RDX-03: Determine whether current C4 was merely undertrained
- [ ] RDX-04: Implement encoder ablations behind package-private test hooks
- [ ] RDX-05: Run the adaptive known-truth redesign screen
- [ ] RDX-06: Implement the selected public opt-in mode
- [ ] RDX-07: Build and validate the named-representation Milo bridge
- [ ] RDX-08: Run the canonical human non-inferiority screen
- [ ] RDX-09: Freeze one candidate or issue a stop verdict
- [ ] RDX-10: Run post-freeze human DA and final independent review

## Implementation Steps

### [ ] RDX-00: Freeze governance, hypotheses, and run contracts

- **Status:** not started
- **Outcome:** A reviewed amendment and issue state authorize only this bounded redesign.
- **Actions:** Preserve prior plans, checkpoints, manifests, and reports byte-for-byte; record their
  hashes. Add a redesign ADR amendment defining the new topology and checkpoint boundary. Freeze
  candidate IDs D0-D5, endpoints, thresholds, adaptive pruning, tie-breaks, environment, seeds, and
  formal verdicts (`candidate`, `stop`, `blocked`) before new fits.
- **Dependencies:** None.
- **Affected areas:** MrTotalVI ADRs, `.scratch/mrtotalvi-v2/`, benchmark schemas.
- **Validation:** Hash comparison against the frozen packet and all three immutable pilot runs;
  configuration tests reject extra candidates, axes, metrics, or verdict strings.
- **Acceptance:** Existing evidence is unchanged and the redesign can neither overwrite C0-C4 nor
  silently expand into DE, macaque, publication, or default-promotion work.
- **Evidence:** Pending.

### [ ] RDX-01: Resolve and seal the human W00/W22 lineage

- **Status:** not started
- **Outcome:** One immutable, QC-lineage-valid human development cohort and feature manifest.
- **Actions:** Derive cells in current harmonized-object order by intersecting the harmonized human
  immune IDs with the final-assembly parent, joining source `pass_qc`, keeping `pass_qc=True`, W00
  or W22, and donors present at both timepoints. Do not modify either source H5AD. Verify counts and
  shared metadata against both parents; stop on disagreement. Hash-split cells within every
  donor-timepoint into model-train and held-out sets. Select 5,000 Pearson-residual HVGs using only
  model-train raw counts and retain the ordered 130 non-isotype proteins. Seal source paths,
  source hashes, ordered cell/gene/protein hashes, split hashes, covariate levels, and derivation
  code in a project-specific lineage amendment and immutable run directory.
- **Dependencies:** RDX-00.
- **Affected areas:** Human derivation utility and immutable manifests; no upstream source
  replacement.
- **Validation:** Expect 46,817 cells and 10 complete donors. Require unique ordered IDs, integer
  nonnegative RNA/protein counts, exact parent-QC provenance, no retained parent-QC failures,
  no `pass_qc` overwrite, 5,000 unique genes, 130 unique proteins, and complete train/held-out
  coverage. Any count or digest difference produces `blocked`.
- **Acceptance:** The signed amendment explicitly selects this universe for MrTotalVI human
  development without reclassifying the stale joint object or the upstream global selected run.
- **Evidence:** Pending.

### [ ] RDX-02: Extend benchmark diagnostics and matched comparators

- **Status:** not started
- **Outcome:** A tested metric dictionary evaluates optimization, `u`, factual `z`, prediction, and
  DA without mixing estimands.
- **Actions:** Add stock scVI and TotalVI runners on identical splits and compatible covariates.
  Record full epoch histories, best checkpoint, modality-specific reconstruction loss, KL terms,
  gradient norms, posterior scale, latent variance/effective rank, residual magnitude, parameter
  count, wall time, and peak memory. Export `u` and factual `z` separately. Add rotation-invariant
  CKA/Procrustes diagnostics, cross-seed kNN Jaccard, state conservation, within-state biological
  sample prediction, technical-batch mixing, RNA/protein held-out negative log likelihood, and
  posterior predictive calibration. Use evaluation annotations only in metrics, never training.
- **Dependencies:** RDX-00; RDX-01 only for human fixtures.
- **Affected areas:** `benchmarks/mrtotalvi/`, `tests/benchmarks/mrtotalvi/`, metric dictionary.
- **Validation:** Unit fixtures prove cell-order invariance, rotation/reflection invariance where
  claimed, correct chance/permutation nulls, modality normalization, no train/validation leakage,
  and rejection of incomparable ELBO aggregation.
- **Acceptance:** TotalVI versus MrTotalVI uses matched RNA/protein likelihood metrics; scVI is not
  included in multimodal ELBO rankings.
- **Evidence:** Pending.

### [ ] RDX-03: Determine whether current C4 was merely undertrained

- **Status:** not started
- **Outcome:** Optimization failure is separated from architectural failure before adding a model.
- **Actions:** Refit B1, B2, B3, and D0 on the existing mixed fixture, unequal-cell fixture, sealed
  500-cell engineering fixture, and canonical human train split when available. Use identical
  convergence-controlled training: validation check every five epochs, at least 50 epochs, at most
  400 epochs, patience 30 checks, best-checkpoint restoration, and no candidate-specific retuning.
  Run seeds 0, 1, and 2. Record whether each fit reaches a stable validation plateau and whether
  latent rank or posterior variance collapses.
- **Dependencies:** RDX-02; canonical-human portion also depends on RDX-01.
- **Affected areas:** Benchmark runner and immutable diagnostic runs.
- **Validation:** Re-running with cell batches and target chunks changed must preserve exported
  representations and scores within declared tolerances. Non-convergence at 400 epochs is a failed
  run, not a reason to extend only one candidate.
- **Acceptance:** If D0 passes every final `u`, factual-`z`, and DA gate after convergence, retain
  current `sample_blind` and do not expose a redundant new mode. Otherwise continue to D1-D5 with
  the failure decomposition recorded.
- **Evidence:** Pending.

### [ ] RDX-04: Implement encoder ablations behind package-private test hooks

- **Status:** not started
- **Outcome:** D1-D5 differ only along the preregistered transform, trunk, prior, and weighting
  axes.
- **Actions:** Implement a reusable TotalVI-normalized input transform and a new sample-blind
  `FCLayers` Normal encoder. Biological sample indices must never reach either path. Technical
  covariates must follow stock TotalVI semantics. Keep the existing encoder and shared
  `EncoderUZ.forward()` untouched. Add benchmark-only construction hooks for D1-D5; do not yet add
  a public constructor value.
- **Dependencies:** RDX-00 and RDX-02.
- **Affected areas:** MrTotalVI components/module internals and focused tests.
- **Validation:** Test exact TotalVI input-transform equality; sample-index invariance; technical
  covariate sensitivity; finite/nonzero core gradients; absent biological-conditioning
  parameters/gradients in the new topology; centered identities; all-sample residual gradients;
  checkpoint round trips; and unchanged legacy/C2/C4/MrMultiVI frozen arrays.
- **Acceptance:** D1 isolates only preprocessing, D2 only adds the trunk change, and D3-D5 change
  only declared prior/weighting axes.
- **Evidence:** Pending.

### [ ] RDX-05: Run the adaptive known-truth redesign screen

- **Status:** not started
- **Outcome:** At most two redesigns advance without outcome-guided hyperparameter search.
- **Actions:** Use the eight existing scenario families with an expanded paired-donor fixture
  suitable for DA. Stage A runs D0-D5 and B1-B3 at seed 0 on three independently generated
  instances per scenario. Disqualify contract, convergence, collapse, factual-`z`, or leakage
  failures. Stage B runs the remaining best two redesigns plus B1 and B2 at seeds 0-2 on ten
  independently generated instances per scenario. Training, truth, and evaluation RNG streams
  remain independent. A single orchestrated workflow writes atomic run directories and stops
  downstream rules when a gate fails.
- **Dependencies:** RDX-03 and RDX-04.
- **Affected areas:** Simulation configurations, workflow, immutable benchmark runs.
- **Validation:** Exact grid aggregation rejects missing, duplicate, stale, or unregistered fits.
  All result readers verify code/config/data hashes and never discover runs recursively.
- **Acceptance:** A redesign advances only if:
  - centered identities remain at most `1e-6`;
  - `u` within-state sample predictability is no greater than the 95th percentile of its
    preregistered permutation null plus `0.02`;
  - `u` state and kNN-state accuracy are each no more than `0.02` below B1;
  - `u` cross-seed 15-neighbor Jaccard is at least `0.60` and no more than `0.05` below B1;
  - factual `z` total held-out predictive loss is no more than 2% worse than TotalVI in every seed;
  - factual `z` RNA and protein losses are each no more than 3% worse than TotalVI;
  - factual `z` state metrics are no more than `0.02` below TotalVI and its cross-seed Jaccard is
    at least `0.60`;
  - no latent coordinate is nonfinite, effective latent rank is at least half the configured
    dimension, and no registered residual row is untrained.
- **Evidence:** Pending.

### [ ] RDX-06: Implement the selected public opt-in mode

- **Status:** not started
- **Outcome:** The winning encoder, if any, is an auditable package mode with no legacy drift.
- **Actions:** Expose only the eligible D1 or D2-family encoder under the name defined in the API
  contract. Store its metadata in `init_params_` and plain attributes. Refuse missing, unknown, or
  topology-incompatible loading overrides. Update counterfactual/local-representation routing,
  docstrings, ADR, user guide, and changelog without changing centered-v2 formulas.
- **Dependencies:** Passing RDX-05. If only D0 passes, record “no new mode” and skip source exposure.
- **Affected areas:** MrTotalVI model/module/components, documentation, and tests.
- **Validation:** Legacy oracle, old missing-metadata checkpoint, current v2 round trips,
  new-topology round trip, refused cross-topology override, MrMultiVI regression, package build,
  and wheel import smoke.
- **Acceptance:** Old keys/shapes/numerics are unchanged; the new topology loads only from explicit
  metadata and remains opt-in.
- **Evidence:** Pending.

### [ ] RDX-07: Build and validate the named-representation Milo bridge

- **Status:** not started
- **Outcome:** The same replicate-level DA estimator runs on every named representation with exact
  cell/design lineage.
- **Actions:** Export PCA, scVI `z`, TotalVI `z`, B2 `u`/factual `z`, and each eligible redesign
  `u`/factual `z` into one cell-order-locked SingleCellExperiment contract. Use
  `buildGraph(k=30, d=20)`, `makeNhoods(prop=0.1, k=30, d=20, refined=TRUE,
  refinement_scheme="graph")`, `countCells(samples="donor_timepoint")`, and exact reordered design
  rows. For paired data use `testNhoods` with `~ timepoint + (1|donor)`,
  `fdr.weighting="graph-overlap"`, `glmm.solver="Fisher"`, `REML=TRUE`,
  `norm.method="TMM"`, and `fail.on.error=FALSE`. A fixed-donor GLM is reporting-only.
- **Dependencies:** RDX-02 and an eligible representation from RDX-05.
- **Affected areas:** Python export, R4_51 Milo runner, schemas, fixtures.
- **Validation:** Reproduce a miloR toy fixture; assert design rows equal neighborhood-count
  columns; preserve cell IDs; require no NA/failed primary fits; test separation diagnostics and
  W22-minus-W00 sign convention.
- **Acceptance:** On known-truth scenarios at nominal SpatialFDR 0.10, median FDP is at most 0.15
  for null and DE-only scenarios; power and localization are no more than 0.05 below the best of
  PCA/TotalVI; and the redesign improves over B2 by at least 0.05 on one preregistered DA scenario
  without losing another by more than 0.05.
- **Evidence:** Pending.

### [ ] RDX-08: Run the canonical human non-inferiority screen

- **Status:** not started
- **Outcome:** Synthetic eligibility survives realistic human dimensions and sample structure.
- **Actions:** Train B0-B3 and at most two eligible redesigns on the sealed 46,817-cell cohort at
  seeds 0-2, latent dimension 20, identical convergence settings, 5,000 genes, and 130 proteins.
  Evaluate held-out prediction, `u` leakage/state/stability, factual-`z` state/stability, latent
  rank, runtime, and memory. Run preregistered within-donor timepoint-label permutations and
  human-geometry semi-synthetic null, DA-only, and DE-only perturbations. Do not inspect the factual
  W22-minus-W00 Milo result.
- **Dependencies:** RDX-01, RDX-06 if a public mode exists, and RDX-07.
- **Affected areas:** Immutable human development runs and aggregate report.
- **Validation:** Every model uses the exact cell/feature/split hashes; no source `pass_qc` field is
  overwritten; no candidate-specific epochs, dimensions, or hyperparameters; all runs restore the
  best validation checkpoint.
- **Acceptance:** All RDX-05 latent gates and RDX-07 DA safety gates pass on the human fixture.
  Failure yields `stop`, not retuning on human annotations or DA effects.
- **Evidence:** Pending.

### [ ] RDX-09: Freeze one candidate or issue a stop verdict

- **Status:** not started
- **Outcome:** Exactly one candidate, or no candidate, is selected reproducibly.
- **Actions:** Disqualify every hard-gate failure. Among remaining candidates, rank by median
  known-truth Milo localization at SpatialFDR 0.10. Treat differences below 0.02 as tied; break ties
  by lower factual-`z` predictive loss relative to TotalVI, then higher `u` cross-seed kNN
  Jaccard, then fewer trainable parameters. Freeze architecture, prior, weighting, preprocessing,
  hyperparameters, feature universe, and code hash before factual human DA is revealed.
- **Dependencies:** RDX-08.
- **Affected areas:** Signed aggregate, candidate manifest, issue statuses.
- **Validation:** A second implementation of the selection function reproduces the result from
  per-run metrics; perturbing result order does not change it.
- **Acceptance:** Formal verdict is exactly `candidate` or `stop`. “Candidate” means eligible for
  post-freeze human description, not generally superior, publication-ready, or default-worthy.
- **Evidence:** Pending.

### [ ] RDX-10: Run post-freeze human DA and final independent review

- **Status:** not started
- **Outcome:** A complete evidence report states what improved, what did not, and what remains
  blocked.
- **Actions:** Only after RDX-09, run the factual paired W22-minus-W00 Milo analysis for the frozen
  representations. Report effect directions, neighborhood localization, donor influence,
  convergence/separation, and cross-seed stability without using them to revise the model. Run
  scoped Ruff, full MrTotalVI/MrMultiVI and benchmark suites, optional Zarr tests, docs and wheel
  builds, and normal non-GPU tests. Perform a read-only independent compatibility, API, lineage,
  and statistical review; address findings and rerun affected gates.
- **Dependencies:** RDX-09 `candidate` verdict. A `stop` verdict skips factual human DA and proceeds
  directly to the negative report and review.
- **Affected areas:** Final immutable report, package verification, issue evidence links.
- **Validation:** Re-read every manifest and hash after sealing; verify source-code hashes still
  match; distinguish pre-existing unrelated failures; prove no factual result changed the frozen
  candidate.
- **Acceptance:** The report separately labels software facts, synthetic mechanism evidence,
  calibrated DA evidence, human descriptive evidence, failures, and blocked claims. Macaque and
  publication stages remain blocked pending a separately signed panel/homolog-map protocol and
  new authorization.
- **Evidence:** Pending.

## Final Validation

The redesign is complete only when one of these terminal outcomes is recorded:

1. **Candidate:** one opt-in configuration passes all compatibility, convergence, `u`, factual
   `z`, Milo calibration, human non-inferiority, package, and independent-review gates; or
2. **Stop:** no configuration passes, the negative result and failure decomposition are sealed,
   existing modes/defaults remain unchanged, and no further model is improvised.

Passing requires all of the following:

- exact legacy and existing-v2 regression preservation;
- a signed 46,817-cell human lineage manifest or an explicit `blocked` outcome;
- convergence-controlled matched scVI/TotalVI/MrTotalVI fits;
- useful, stable `u` and factual `z` under their separate gates;
- calibrated known-truth Milo DA with no hidden estimator changes;
- no human biological outcome-guided tuning;
- immutable manifests and reproducible selection;
- package/documentation/build verification; and
- independent review.

## Risks, Assumptions, and Blockers

- The expected 46,817 count is derived from current source hashes. Any source drift blocks the run
  until the lineage amendment is recomputed and re-signed.
- Human L2 annotations are evaluation-only and may themselves be imperfect; report sensitivity to
  L1.5/L2 granularity without choosing the better-looking result.
- TotalVI is the primary factual-`z` comparator because it uses both modalities. scVI cannot be
  ranked on protein likelihood.
- A sample-aware factual `z` is not expected to be sample-neutral. Its safeguards are predictive
  quality, cell-state preservation, seed stability, and DA false-positive control.
- Cross-seed latent coordinates are not directly identified; use neighbor, CKA, Procrustes, and
  downstream stability metrics rather than coordinate-wise correlation.
- The proposed Milo GLMM requires zero failed/NA primary fits. Separation or convergence failures
  block that representation rather than being silently dropped.
- GPU execution and external environment changes require separate runtime approval. The plan does
  not authorize scheduler submission, dependency installation, publication, commit, or push.
- Macaque remains blocked by the 74-versus-130 panel choice and exact RNA homolog map.

## Decision and Evidence Log

- 2026-07-26 — Existing bounded evidence reviewed: centering and leakage mechanisms worked, but C4
  cross-seed human-fixture kNN Jaccard was 0.194 and synthetic target-distance recovery was poor.
- 2026-07-26 — User selected redesign-first, a new opt-in compatibility boundary, latent/DA as the
  primary goal, explicit factual-`z` comparison with TotalVI/scVI, and human-lineage resolution now.
- 2026-07-26 — Human plan default fixed to harmonized immune IDs re-gated by final-parent QC
  (expected 46,817 W00/W22 cells); macaque remains blocked.
- 2026-07-26 — Plan reviewed, patched, converted to a progress tracker, and initialized.
  Implementation has not started.
