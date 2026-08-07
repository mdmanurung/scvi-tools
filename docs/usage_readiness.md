# CytoANVI and MrTotalVI usage readiness

This page is the authoritative capability-status surface for this fork. Shorter READMEs, guides,
docstrings, and tutorials must link here and must not promote a capability beyond this table. The
machine authority is
[`docs/artifacts/usage-readiness-matrix-v1.json`](artifacts/usage-readiness-matrix-v1.json).

The five evidence states are separate: source present; source engineering tests terminal; exact
installed artifact engineering-accepted; capability-specific scientific evidence terminal; named
human promotion signed. A later state is never inferred from an earlier one.

Current artifact state: `cytoanvi 0.2.0` source remediation is in progress. The prior 0.1.0 wheel is
[quarantined](artifacts/cytoanvi-0.1.0-quarantine.json). Isolated 0.2.0 installed acceptance is
`blocked_dependency_authority` until a hash-locked local dependency authority exists. No row is
promoted and no P2 run is authorized by this page.

The root-owned treeArches engineering fixture is
`vignettes/cytoanvi_treearches_synthetic.py`. The richer tutorial directory is an uninitialized
external gitlink and is excluded from this artifact; its repair is explicitly
`blocked_external_submodule` in the frozen packet handoff.

| Capability ID | Engineering | Scientific evidence | Current boundary |
| --- | --- | --- | --- |
| `cytoanvi.core` | Pending P1/source and wheel acceptance | Limited historical support | Pending independent two-cohort protocol and human promotion |
| `cytoanvi.mapping.same_panel` | Pending | Limited historical support | Mapping agreement is not target accuracy |
| `cytoanvi.mapping.panel_divergent` | Pending | Limited historical support | Requires an independently validated shared marker backbone |
| `cytoanvi.hierarchy` | Pending | Experimental only | Existing near-leaf hierarchy is not biological validation |
| `cytoanvi.integration_clustering` | Pending | Limited historical support | Dataset-specific biology/batch checks required |
| `cytoanvi.tta_ood` | Stable APIs fail closed | Historical negative | **No-go**; existing TTA AUROC was below chance; replacement is unfrozen |
| `cytoanvi.continual` | EWC plus replay only; pending | Experimental only | No consequential update; replay and required controls must be explicit |
| `cytoanvi.mapqc` | Exact 0.1.1 compatibility only; pending | Historical negative | **No-go** as an automatic acceptance gate |
| `cytoanvi.artifact` | Blocked installed acceptance | Blocked | **No-go** until exact wheel receipt passes |
| `mrtotalvi.core` | Pending P1/source and wheel acceptance | Limited historical support | Expert exploratory in-memory use only after engineering acceptance |
| `mrtotalvi.embeddings` | Pending | Limited historical support | `z` is factual/sample-aware; legacy `u` is sample-conditioned |
| `mrtotalvi.prior_choice` | Explicit schema pending | Experimental only | No generally recommended prior; prespecify and sensitivity-check |
| `mrtotalvi.label_supervision` | Explicit opt-in pending | Experimental only | Labels alone must not change the objective |
| `mrtotalvi.da` | Descriptive path pending | Historical negative | **No-go** for inference or single-fit decisions |
| `mrtotalvi.legacy_de` | Public refusal pending | Historical negative | **No-go**; use donor-pseudobulk PyDESeq2/edgeR/dreamlet |
| `mrtotalvi.centered_v2` | Pending | Experimental only | Registered-sample descriptive, non-causal, no new-sample inference |
| `mrtotalvi.streaming` | Stable export removal pending | Blocked | **No-go** as a model-training surface |
| `mrtotalvi.new_sample_inference` | Explicit refusal boundary | Unsupported | **No-go**; fixed registered sample universe |
| `mrtotalvi.artifact` | Blocked installed acceptance | Blocked | **No-go** until exact wheel receipt passes |

P2 protocols live in `docs/protocols/` with machine contracts under `benchmarks/`. Missing cohorts,
independent labels, algorithms, controls, or numeric margins remain `draft_unfrozen`/`blocked`.
Scheduler/GPU execution, independent review, result adjudication, and promotion require separate
human authorization.
