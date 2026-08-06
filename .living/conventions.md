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

### C-004 — Never compare embeddings with a statistic whose denominator or operating point encodes the effect under test
**Pack**: local
**Source**: L-094, L-097, and the 2026-07-28 review (F1)
**Rule**: When comparing two representations, use a measure that is free of the quantity being compared. Pair every variance-share or fixed-setting statistic with a denominator-free counterpart before drawing a conclusion.
**Details**: Three separate confident-but-wrong conclusions in this project came from the same class of error:
- **L-094** — multivariate η² = trace(between)/trace(total). The denominator is each embedding's own total variance, which is exactly what compression changes. Fix: kNN label recovery (no denominator).
- **L-097** — best-matching-cluster F1 at a single Leiden resolution. Different embeddings peak at different resolutions, so a fixed setting measures where each sits on its curve. Fix: sweep and report best-achievable.
- **Review F1 (2026-07-28)** — kNN purity for `timepoint` where same-sample neighbours are trivially same-timepoint. The statistic contained the sample-mixing quantity inside it. Fix: decompose, `(knnP_a − knnP_ab)/(1 − knnP_ab)`.

Checklist before reporting any cross-embedding comparison:
1. What is in the denominator, and does it differ between the arms? If yes, the number is not comparable.
2. Is this a single operating point (resolution, k, threshold)? If yes, sweep it.
3. Does the statistic contain a trivially-satisfied component (same-sample, same-batch)? If yes, decompose it out.
4. Is there a rank-based or recovery-based alternative? Prefer it, or report both.
Also record the detection floor: a global pooled kNN statistic at k only resolves subpopulations above roughly 1/k of the total, so it cannot be used to rule out rare states.
