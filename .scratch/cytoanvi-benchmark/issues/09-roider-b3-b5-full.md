# 09 — Run B3 + B5 on full Roider (scib, epochs=1000, holdout sweep)

Status: ready-for-agent
Blocked-by: —

## Task

On **full** Roider cohort (B-D1):

- **B3:** panel-1 reference → panel-2 query; holdout on panel-1 for hard F1; concordance vs k-NN on panel-2
- **B5:** `--holdout-sweep` over all cell types; report best/worst/mean AUROC
- `max_epochs=1000`, seeds 0, 1, 2

Panel-1 ``cell_type`` = **Leiden clusters** (``r=1.0``, cached under ``data/roider_full/``); not manual
gating names.

## Acceptance

- B3: concordance mean ≥ 0.70; holdout F1 ≥ k-NN
- B5: ∃ holdout type with AUROC > 0.70 (or document all types in sweep table)

## Comments

### 2026-06-17 — vignette e1000 B3 complete

`results/e1000/roider_e1000_b3_multiseed.json`: p1 holdout macro-F1 **0.917 ± 0.018**; p2 concordance
**0.877 ± 0.012** (3 seeds, epochs=1000). **Pass** PRD vignette targets. B5 sweep + full cohort still
pending (`cytovi-benchmark/02`).

### 2026-06-17 — vignette prototype done; full cohort pending

Vignette B3+B5 smoke on tutorial `.h5ad` validates harness (`--holdout-type`, B5 sweep, 3-seed B1/B2
with scib). All PRD smoke targets pass or documented (B2: better bio, worse batch). **Full-cohort
run blocked** on `cytovi-benchmark/02` (63-patient ingest). Re-run with `max_epochs=1000` when data
ready. Data helper: `python -m benchmarks.common.fetch_data --list-full-cohort`.
