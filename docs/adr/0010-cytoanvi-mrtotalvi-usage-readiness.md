# ADR-0010: CytoANVI and MrTotalVI usage-readiness boundary

## Status

Accepted for local engineering remediation on 2026-08-07. Scientific protocol approval,
scheduler execution, result adjudication, capability promotion, publication, and release remain
separate human-authorized actions.

## Context

The source review at commit `297769d3c62b9228244a05469dc8349a55e4174c` found a usable core
beside unsafe or ambiguous surfaces. The existing `cytoanvi 0.1.0` wheel predates material source
fixes while reporting the same version. Source presence, source tests, installed-artifact tests,
scientific evidence, and promotion therefore need independent state and evidence.

## Decision

### Version and artifact authority

- The breaking candidate version is `0.2.0`.
- `dist/cytoanvi-0.1.0-py3-none-any.whl` remains byte-for-byte immutable and is logically
  quarantined by `docs/artifacts/cytoanvi-0.1.0-quarantine.json`.
- A `0.2.0` wheel may be built once, only from a clean reconstruction of the scoped P1 commit.
  A materially different rebuild needs a new version and decision.
- Tracked authority is the manifest, inventory, dependency-authority digest, and acceptance receipt
  under `docs/artifacts/cytoanvi-0.2.0/`; an ignored wheel is not authority by itself.
- Installed acceptance must run outside the checkout with `PYTHONPATH` unset, no editable install,
  no `scvi-tools` distribution, and both `scvi` and `cytoanvi` owned by the candidate distribution.

### Fail-closed API and migration contracts

- CytoANVI rejects every non-null `adversarial_classifier`, continual state without replay, stable
  TTA novelty entry points, invalid TTA calibration, and unsupported mapQC versions before the
  affected training or inference path. `control_adata` is required for the implemented continual
  objective. Empirical priors and class weights are training-split quantities.
- MrTotalVI resolves `u_prior` to exactly `standard`, `mog`, or `vamp`; the deprecated boolean is a
  migration input only. Supervision is a separate explicit `none` or `labels` choice and defaults
  to `none` with zero weight. Contradictory checkpoint/API states fail closed.
- MrTotalVI exhaustively validates raw RNA/protein counts before AnnData mutation, validates sample
  metadata and subsets before statistics, refuses public biological DE, refuses `use_vmap=True`,
  requires exact multi-file protein identity/order, and does not export an incomplete streaming
  training surface.
- ADR-0005 remains historical authority for legacy hierarchy semantics. ADR-0007 remains authority
  for centered-v2 registered-sample descriptive semantics, subject to the stricter public DE and
  input contracts above. ADR-0008 and ADR-0009 scientific-selection policy is unchanged.

The exact compatibility table is in
`docs/migration/cytoanvi-mrtotalvi-0.2.0.md`.

### Mandatory capabilities

The capability decision matrix contains exactly these 19 IDs:

1. `cytoanvi.core`
2. `cytoanvi.mapping.same_panel`
3. `cytoanvi.mapping.panel_divergent`
4. `cytoanvi.hierarchy`
5. `cytoanvi.integration_clustering`
6. `cytoanvi.tta_ood`
7. `cytoanvi.continual`
8. `cytoanvi.mapqc`
9. `cytoanvi.artifact`
10. `mrtotalvi.core`
11. `mrtotalvi.embeddings`
12. `mrtotalvi.prior_choice`
13. `mrtotalvi.label_supervision`
14. `mrtotalvi.da`
15. `mrtotalvi.legacy_de`
16. `mrtotalvi.centered_v2`
17. `mrtotalvi.streaming`
18. `mrtotalvi.new_sample_inference`
19. `mrtotalvi.artifact`

Rows may record negative or inconclusive results. None may be omitted. Engineering execution,
scientific result, and promotion are separate fields.

### P2 and P3 authority

- P2 protocols freeze artifact, cohort, biological unit, split/leakage boundary, independent truth,
  RNG streams, compute budget, representation semantics, endpoint, numeric margin, donor-level
  uncertainty, multiplicity, controls, no-call policy, immutable outputs, and independent reviews.
- Missing scientific choices stay `draft_unfrozen` or `blocked`; prior outcomes cannot select them.
- The preserved 42/48 RDX-03 run is retrospective only and cannot be spliced. Effective rank is an
  ADR-0009 alert, not a terminal v2 integrity gate.
- An agent cannot sign protocol approval, adjudication, P3 promotion, publication, or release.
  Promotion requires terminal exact-artifact P2 evidence and a named human approver/date.

## Consequences

Passing P1 source tests can establish source engineering correctness only. A built wheel without a
clean dependency authority remains engineering-unaccepted. Synthetic protocols and fixtures do not
establish biological validity. The authoritative status page must preserve those distinctions and
keep no-go and blocked capabilities visible.
