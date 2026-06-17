# 10 — Run B4 + B6 continual update on full Roider (rLN case-control)

Status: ready-for-agent
Blocked-by: cytovi-benchmark/02, cytovi-benchmark/03

## Task

Full Roider now has disease entities — use **rLN (resected control)** as reference/replay and
lymphoma entities as case query.

- Reference: rLN patients (panel 1 labelled)
- Replay buffer: rLN healthy-control cells
- Query: case lymphoma patients (panel 2 or mixed per design)
- Sweep `ewc_importance` ∈ {0, 1, 10, 100, 1000}
- Metrics: replay latent drift (L2 pre/post); DA cluster recovery for known entity-enriched population
- `max_epochs=1000` for reference; query epochs per continual plan defaults

## Acceptance

- Drift vs λ curve JSON
- Documented λ knee as CytoVI-specific default
- B4 pass: low drift + case signal recovered at some λ

## Comments

Unblocks former issue 06 (deferred for lack of case/control on vignette subsample).
