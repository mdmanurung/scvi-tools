# 06 — B4 continual case-control update + B6 λ tuning [SUPERSEDED]

Status: wontfix
Blocked-by: a dataset with an explicit case/control axis

> Unblocked and superseded by issue 10 (full Roider with rLN vs lymphoma entities).

## Task

Benchmark the continual case-control update (`load_query_data_with_replay`) and tune
`ewc_importance` (λ) for CytoVI's intensity likelihood — the paper's λ=100 was for scANVI/RNA and
won't transfer (documented in INVESTIGATION.md).

Needs a dataset with healthy **controls** present in both reference and query (the EWC term is
`F_reference ∘ F_query_ctrl`). The Roider BNHL data is all-tumor — no built-in control — so this is
deferred pending a decision:

- (a) a real case/control cytometry dataset Mikhael points to, or
- (b) a pseudo-control split (validates plumbing, not biology).

## Acceptance (once a dataset exists)

- Sweep λ ∈ {0, 1, 10, 100, 1000}; plot reference-latent drift vs case-signal recovery.
- A λ exists where reference/control latent is preserved (low drift) and a known case-enriched
  population is recovered (`differential_abundance`).
- Record the chosen λ as the CytoVI-specific default and update INVESTIGATION.md / the ADR note.

## Comments

- Deferred per the 2026-06-10 planning discussion (chose "defer B4 for now"). The harness
  (`benchmarks/cytoanvi/tasks.py`) does not yet implement B4/B6 — add when a dataset is chosen.
