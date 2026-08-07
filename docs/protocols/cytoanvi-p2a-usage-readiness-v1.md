# CytoANVI P2A usage-readiness protocol v1

Status: `draft_unfrozen`; execution is not authorized.

The machine contract is
`benchmarks/cytoanvi/usage_readiness_contract_v1.json`. It contains all nine mandatory CytoANVI
capabilities and is validated against
`docs/artifacts/schemas/scientific-protocol.schema.json`. Artifact identity remains null until the
single 0.2.0 candidate is sealed. Independent pre-run review, scheduler submission, result review,
and promotion remain human actions.

## Frozen minimum structure

- Training seeds are at least `[0, 1, 2]`; split, training, perturbation/operating-point, and
  bootstrap streams are separate named streams.
- The biological unit for scientific endpoints is the donor. Cell-level summaries cannot replace
  donor-level uncertainty.
- Every terminal run must use the terminal-manifest schema, the exact
  `compute_budget.evaluation_grid × rng_streams.training_seeds` run IDs, immutable artifact identity,
  retained negative results, and scheduler/accounting evidence (or an explicitly justified local
  backend where scientifically appropriate). A terminal manifest cannot declare its own smaller
  grid.
- Synthetic examples and fake-scHPL tests are engineering fixtures only.
- Stable TTA OOD, continual updating without approved replay/control authorities, automatic mapQC
  gating, and an unaccepted artifact are no-go.

## Capability freeze blockers

| Capability | State | Missing freeze authority |
| --- | --- | --- |
| Core | `draft_unfrozen` | Independent second cohort/labels; exact split; endpoint/margin; donor uncertainty; controls; compute |
| Same-panel mapping | `draft_unfrozen` | Independent query truth; rare-class/calibration/failure limits; controls |
| Panel-divergent mapping | `draft_unfrozen` | Authoritative backbone/mask algorithm; second-cohort panel truth; margins/controls |
| Hierarchy | `draft_unfrozen` | Independent ontology and target labels; hierarchy endpoint/margin; external tutorial repair |
| Integration/clustering | `draft_unfrozen` | Graph operating points; joint biology/batch endpoints; second cohort; multiplicity |
| TTA OOD | `blocked` | Historical result is below chance; replacement algorithm/truth/margins are unfrozen |
| Continual | `blocked` | Replay/control authorities, utility formula, minimum gain, and maximum forgetting are unfrozen |
| mapQC | `blocked` | Historical automatic-gate evidence is negative; false-accept/reject ceilings are unfrozen |
| Artifact | `blocked` | Candidate tuple and verified offline dependency authority are unavailable |

No missing threshold, cohort, label, algorithm, or control is inferred from a historical outcome.
All positive and negative controls remain empty in the draft machine entries so the validator will
refuse any accidental `frozen` transition. An independent reviewer must freeze those controls and
every other required field before one exact scheduler packet may be proposed. Approval requires a
named and dated reviewer, a declared independence rule, and capability-specific pre/post review
receipts bound to that reviewer.
