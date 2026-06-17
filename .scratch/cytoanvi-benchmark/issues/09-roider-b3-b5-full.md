# 09 — Run B3 + B5 on full Roider (scib, epochs=1000, holdout sweep)

Status: ready-for-agent
Blocked-by: cytovi-benchmark/02, cytovi-benchmark/03

## Task

On **full** Roider cohort (B-D1):

- **B3:** panel-1 reference → panel-2 query; holdout on panel-1 for hard F1; concordance vs k-NN on panel-2
- **B5:** `--holdout-sweep` over all cell types; report best/worst/mean AUROC
- `max_epochs=1000`, seeds 0, 1, 2

Extend `run.py` with `--holdout-sweep` and `--multiseed 0,1,2`.

## Acceptance

- B3: concordance mean ≥ 0.70; holdout F1 ≥ k-NN
- B5: ∃ holdout type with AUROC > 0.70 (or document all types in sweep table)

## Comments

### 2026-06-17 — vignette prototype done; full cohort pending

Vignette B3+B5 smoke on tutorial `.h5ad` validates harness (`--holdout-type`, B5 sweep, 3-seed B1).
All PRD smoke targets pass. **Full-cohort run still blocked** on `cytovi-benchmark/02` (63-patient
ingest) and `cytovi-benchmark/03` (scib infra). Re-run with `max_epochs=1000` when data ready.
