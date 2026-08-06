# ADR-0008: Bounded MrTotalVI stable-latent and DA redesign

## Status

Accepted as an experimental and selection contract on 2026-07-26.

No encoder has been selected by this ADR. It does not add a public mode, alter a checkpoint,
promote a model, change a default, or establish a scientific or publication claim.

## Context

ADR-0005 defines the original MrTotalVI hierarchy and ADR-0007 defines the opt-in centered-v2 and
current sample-blind package contracts. The frozen redesign evidence found that the current C4
sample-blind comparison was not sufficient for a scientific choice: it was trained for only three
epochs, did not record a convergence history, had weak absolute state recovery, and showed unstable
cross-seed geometry. The current encoder also combines raw RNA and protein counts under one
`log1p` path and a custom MLP, whereas TotalVI normalizes the modalities separately and uses an
`FCLayers` posterior trunk.

The redesign therefore needs to distinguish undertraining from input-transform, posterior-topology,
prior, and observation-weighting effects. It must evaluate shared `u` and factual `z` separately,
use stock TotalVI as the primary multimodal factual-`z` comparator, and test replicate-level Milo
DA without selecting on factual human W22-minus-W00 results.

The detailed executable contract is the
[RDX-00 governance amendment](../../.scratch/mrtotalvi-v2-redesign/governance-amendment.md). This
ADR fixes the architectural and checkpoint decision boundary. It is additive to ADR-0005 and
ADR-0007 and does not reopen their accepted behavior.

The machine-readable selection authority is
`benchmarks.mrtotalvi.redesign_contract.redesign_run_contract()`, sealed as
`redesign-run-contract.json` in each completed governance run. It freezes the exact endpoint IDs,
hard thresholds, Stage A/B grids, seeds, convergence controls, Milo estimator, tie-break,
environment boundary, and human dimensions. The older pilot metric dictionary remains
no-selection historical evidence and cannot replace this contract.

## Decision

### Bounded experiment

The authorized references are:

- B0: stock scVI `z`, RNA-only contextual comparator;
- B1: stock TotalVI `z`, primary multimodal factual-`z` comparator;
- B2: legacy sample-conditioned MrTotalVI C0 `u` and factual `z`;
- B3: centered sample-conditioned MrTotalVI C2 `u` and factual `z`.

The authorized D-series evaluation rows are exactly:

| ID | Input transform | Posterior trunk | `u` prior | Weighting |
|---|---|---|---|---|
| D0 | existing raw-count `log1p` | existing sample-blind MLP | frozen initialized VampPrior | cell-equal |
| D1 | TotalVI per-modality normalization | existing sample-blind MLP | frozen initialized VampPrior | cell-equal |
| D2 | TotalVI per-modality normalization | sample-blind TotalVI-style `FCLayers` | frozen initialized VampPrior | cell-equal |
| D3 | TotalVI per-modality normalization | sample-blind TotalVI-style `FCLayers` | trainable MoG | cell-equal |
| D4 | TotalVI per-modality normalization | sample-blind TotalVI-style `FCLayers` | frozen initialized VampPrior | sample-equal |
| D5 | TotalVI per-modality normalization | sample-blind TotalVI-style `FCLayers` | trainable MoG | sample-equal |

The TotalVI transform normalizes RNA and protein separately exactly as the declared matched
TotalVI inference path does before `log1p`. The D2-D5 posterior never receives biological sample
identity, but it retains explicitly registered technical covariates under stock TotalVI semantics.

Cells, ordered features, latent dimension, technical covariates, decoder capacity where
applicable, splits, optimizer, schedules, convergence controls, and random streams are held fixed.
The implementation must reject unknown D-series IDs, unknown axis values, and any realized
configuration that differs from its declared row. D0 is evaluated only as the existing convergence
control and is not eligible for a new-mode `candidate` verdict. D1-D5 remain package-private until
selection.

### Representation and selection contract

Shared `u` and factual `z` are exported and judged separately. scVI is not ranked on protein
likelihood or multimodal ELBO. Hard convergence, finiteness, rank, gradient, leakage, state,
stability, prediction, Milo-fit, false-discovery, power, and localization gates precede any
tie-break; a gain on one endpoint cannot compensate for a hard failure on another.

At most two D1-D5 redesigns advance from the preregistered Stage A screen. Survivors are selected
by median known-truth Milo localization at SpatialFDR 0.10, with differences below `0.02` treated
as tied, followed by factual-`z` predictive loss relative to TotalVI, `u` cross-seed 15-neighbor
Jaccard, and trainable parameter count. The result must be invariant to input order and reproduced
by an independent selection implementation.

Before factual human DA is available, the selected ID, all four axes, architecture, preprocessing,
prior, weighting, training controls, features, data/split identities, code hash, and configuration
hash are frozen. After the required compatibility implementation is also frozen, factual paired
W22-minus-W00 Milo may be run once for the candidate and reported descriptively. It cannot change
the candidate. A `stop` or `blocked` result never opens factual human DA.

### Topology and checkpoint boundary

All existing modes retain their accepted contracts:

```python
hierarchy_mode = "legacy"       # existing default
u_encoder_mode = "sample_conditioned"  # existing default
```

Existing `legacy`, `centered_v2`, `sample_conditioned`, and `sample_blind` semantics, keys, shapes,
numerics, and loading behavior remain frozen. D0 uses the current `sample_blind` implementation
unchanged. `EncoderUZ.forward()`, existing encoder branches, and MrMultiVI remain unchanged.

D1 uses the existing posterior topology with a different, explicitly named input-transform
semantic. If D1 alone is selected and later passes the public-mode gates, its only eligible public
name is `sample_blind_scaled`.

D2-D5 introduce a TotalVI-style `FCLayers` Normal posterior topology. If one is selected and later
passes the public-mode gates, its only eligible public name is `sample_blind_totalvi`, and it
requires `hierarchy_mode="centered_v2"`.

Any selected new mode must store explicit encoder-semantic and topology metadata. D2-D5 use
topology-specific checkpoint keys. Matching tensor shapes do not establish compatibility, and
cross-topology or cross-semantic loading overrides into or out of a new mode are refused. Missing
metadata continues to resolve only to `legacy` plus `sample_conditioned`; modes are never inferred
from tensor values, state-dict keys, or shapes. ADR-0007's existing same-topology semantic-override
contract remains unchanged and does not cross this boundary.

The selected candidate manifest also retains prior and weighting values; an encoder mode name alone
is not the full training provenance.

### Verdicts

The formal verdict enum contains only:

- `candidate`: exactly one frozen D1-D5 configuration passed the applicable gates. At selection
  time this means eligibility for post-freeze human description, not general superiority or
  default-worthiness. Terminal completion additionally requires compatibility, package/build, and
  independent-review evidence.
- `stop`: the valid scientific screen completed and no D1-D5 redesign passed. This includes a
  D0-only pass. Seal the negative or no-new-mode result, retain existing behavior, add no mode, skip
  factual human DA, and do not improvise another architecture.
- `blocked`: integrity, lineage, dependency, permission, or compute constraints prevent a valid
  candidate/stop decision. Record the failed gate and do not substitute a data universe, estimator,
  dependency, or environment.

Non-convergence, nonfinite output, or a failed/`NA` primary Milo fit under the authorized controls
disqualifies that representation; it is not a `blocked` result. If a complete valid screen leaves
no eligible D1-D5 redesign, the verdict is `stop`.

If D0 passes but no D1-D5 redesign does, the terminal verdict is `stop`; current `sample_blind` is
retained and no redundant mode is added. A D1 result can only make `sample_blind_scaled` eligible
for later implementation. A D2-D5 result can only make `sample_blind_totalvi` eligible. No result
from this ADR changes the default.

### Human and permission boundaries

The only permitted human development cohort is the immutable harmonized-immune-ID/final-parent-QC
intersection with exactly 46,817 W00/W22 cells and 10 complete donors. A source, cell, count,
feature, or metadata mismatch yields `blocked`; the 47,709-cell and 51,174-cell universes are not
fallbacks.

The experiment does not include DE analysis, LEMUR, miloDE, macaque validation, causal or
publication claims, default promotion, adversarial alignment, teacher distillation, new-sample
surgery, minified mode, vectorization, or non-MAP residuals. The preregistered DE-only
semi-synthetic DA-safety perturbation is not a DE analysis. The experiment does not authorize
dependency installation, network access, external-environment writes, GPU or scheduler execution,
commit, push, or publication. Those actions require separate authority. Existing artifacts are
immutable, new runs are atomically sealed without a `latest` pointer, and unrelated dirty-worktree
changes are preserved.

## Consequences

### Positive

- Undertraining is tested before an architectural failure is inferred.
- Input transform, posterior topology, prior, and weighting effects are separable and
  preregistered.
- Factual `z` safety cannot be hidden by a favorable shared-`u` or DA result.
- The factual-human result cannot influence architecture or hyperparameter selection.
- New encoder semantics cannot be silently loaded through an old or shape-compatible checkpoint.
- A valid negative or blocked outcome leaves all existing modes and defaults unchanged.

### Limitations

- Acceptance of this ADR is governance evidence, not empirical evidence for D0-D5.
- `candidate` is a bounded experimental verdict, not a claim of general biological validity.
- The 46,817-cell human cohort is development evidence only and has no known factual DA truth.
- Registered-sample counterfactual and causal limitations from ADR-0007 remain.
- Macaque confirmation, DE, publication, and default promotion require separate plans and
  authorization.

## Rejected interpretations

- Existing `sample_blind` is not silently replaced by D1 or D2.
- A TotalVI-like input transform is not described as a TotalVI trunk.
- Shape-compatible checkpoints are not assumed to share semantics.
- Human factual DA is not a tuning or tie-breaking endpoint.
- A D0-only pass is not recorded as `candidate`.
- A missing dependency or lineage mismatch is not converted into a scientific `stop`.
- Passing synthetic or package tests alone does not select or promote a mode.
