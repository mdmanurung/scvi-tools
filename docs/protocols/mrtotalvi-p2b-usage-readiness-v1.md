# MrTotalVI P2B usage-readiness protocol v1

Status: `draft_unfrozen`; execution is not authorized.

The machine contract is
`benchmarks/mrtotalvi/usage_readiness_contract_v1.json`. All ten mandatory MrTotalVI capability
rows are present. Artifact identity, independent pre-run review, scheduler submission, result
adjudication, and promotion remain pending human authorities.

## Reused verified lineage only

- Lineage manifest:
  `.scratch/mrtotalvi-v2-redesign/human-lineage-runs/20260731T081355Z-991ec740-b50f4e3a-e6ce6542/lineage-manifest.json`,
  SHA-256 `8712246ed716a7b8af78eb29dadda0247623d1e7b597ecbd0a22e71b55e45fc6`.
- Human universe: 46,817 cells, 10 complete donors, 20 donor-timepoint samples, 5,000
  training-only HVGs, and 130 non-isotype proteins.
- Sealed split: 37,447 training and 9,370 held-out cells; split digest
  `68c2c95a74650ed02cac88053de63de030becf88a7448ec13a0534021cb286b9`.
- Governance contract:
  `.scratch/mrtotalvi-v2-redesign/governance-runs/20260726T113947Z-1773319a2735/redesign-run-contract.json`,
  SHA-256 `6f81c023b37c9dc8003cbf31ec3870c2bbd90c96f82c1702ade5e167ec90be48`.
- Evaluation grid is exactly B0-B3 and D0-D5 with training seeds `[0, 1, 2]` and no
  candidate-specific retuning. Terminal run IDs must be the exact ordered cross-product
  `{arm}-seed-{seed}`; a terminal manifest cannot declare its own smaller grid.
- Effective rank follows ADR-0009 alert-only semantics. It is recorded, but it cannot terminate an
  otherwise complete grid.
- The preserved 42/48 run is retrospective evidence only. It is never spliced into this protocol.

## Capability boundaries

| Capability | State | Boundary/blocker |
| --- | --- | --- |
| Core | `draft_unfrozen` | Endpoint priority, margins, uncertainty, multiplicity, artifact, and independent review remain open |
| Embeddings | `draft_unfrozen` | `u` and factual `z` stay separate; legacy `u` is sample-conditioned; rank is alert-only; metric priorities/margins remain open |
| Prior choice | `blocked` | Standard/MoG/Vamp may be prespecified sensitivity arms; no prior is generally recommended |
| Label supervision | `draft_unfrozen` | Labels alone are metadata; explicit supervised scientific role and margins remain unapproved |
| DA | `blocked` | Package DA is descriptive; historical seed instability prevents inferential use; factual human DA remains locked |
| Legacy DE | `blocked` | Public API refuses; historical eps-space results are negative; use donor-pseudobulk |
| Centered v2 | `draft_unfrozen` | Registered-sample descriptive semantics only; real-data invariance tolerances remain open |
| Streaming | `blocked` | Private registry adapter is not end-to-end training or a stable export |
| New-sample inference | `blocked` | Fixed 20-sample registered universe; no projection/surgery algorithm |
| Artifact | `blocked` | Candidate tuple and exact offline dependency authority are unavailable |

Positive and negative controls already present in the sealed redesign authority are carried forward
where applicable (B/reference arms, known-truth DA, centering/gradient identities, label permutation,
geometry null, DE-only, and unknown-target refusals). Their presence does not freeze the protocol:
capability-specific endpoints, margins, donor-level uncertainty, multiplicity, artifact identity,
and independent review must all close first.

No scheduler/GPU command is included here. A freeze requires a named and dated independent-review
authority, an independence rule, and matching pre-run review receipts for each frozen capability;
post-run approval likewise requires a matching receipt. Scheduler submission still requires a
separate, explicit authorization for that exact artifact and protocol-derived grid.
