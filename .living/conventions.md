# Active Conventions

Conventions crystallized from recurring learnings. Each entry links to the source learnings.

## Format

```
### C-NNN — Title
**Pack**: [convention pack name, or "local"]
**Source**: L-NNN, L-NNN (learnings that produced this)
**Rule**: The convention in one sentence.
**Details**: When it applies and how to follow it.
```

---

### C-001 — Always mask uncertainty at per-cell level, not per-panel
**Pack**: local
**Source**: L-001
**Rule**: When computing per-cell statistics on multi-panel data, apply `nan_layer` masking per cell, not a global panel mask.
**Details**: Applies to TTA uncertainty, reconstruction loss logging, and any aggregation over marker dimensions. Global masks silently inflate estimates for cells with missing panels.

### C-002 — Raise ValueError at construction when n_labels < 1
**Pack**: local
**Source**: L-002, L-017
**Rule**: `CytoANVI` raises `ValueError` at construction when `n_labels < 1`; the forward pass does NOT silently return uniform priors.
**Details**: Applies at model construction time. The "uniform prior fallback" approach described in L-002 was superseded — the current implementation fails fast at `__init__` so that silent mis-configuration is impossible. Do not add fallback logic in the forward pass; fix callers to supply at least one labeled category.

### C-003 — Subsample to ≤10k cells for Fisher/EWC importance computation
**Pack**: local
**Source**: L-003
**Rule**: Fisher importance matrices are computed on a ≤10k cell subsample with a log-progress callback.
**Details**: Applies to all `ContinualUpdate` construction paths. The 10k limit prevents OOM on GPU for large atlases. Log the subsample fraction for auditability.
