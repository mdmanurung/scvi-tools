# PRD: Track A — CYTOVI paper-faithful benchmarks

Status: ready-for-agent
Owner: mdmanurung
Branch: feat/cytoanvi
Created: 2026-06-17

## Problem

scvi-tools ships `CYTOVI` as the official implementation of the CytoVI paper, but there is no
automated benchmark suite that reproduces the paper's quantitative claims against the reference
repositories ([cytovi-reproducibility](https://github.com/YosefLab/cytovi-reproducibility),
[cytovi-reference-implementation](https://github.com/YosefLab/cytovi-reference-implementation)).

## Approach

Build `benchmarks/cytovi/` mirroring the paper's evaluation axes (A1–A6) on **full cohorts**,
`max_epochs=1000`, **scib-metrics** for all integration tasks.

Master plan: `notes/2026-06-17-cytovi-cytoanvi-benchmark-plan.md`.

## Tasks

| ID | Paper | Dataset | Priority |
|----|-------|---------|----------|
| A1 | Fig S2 PPC | Nuñez flow, mass cyt, CITE protein | P3 |
| A2 | Fig 2E integration | Nuñez batch replicate | **P1** |
| A3 | Fig S4 imputation | Nuñez 50k semi-synthetic | P2 |
| A4 | Fig 3 multi-panel | Nuñez + Kreutmair | P4 |
| A5 | Fig 4 Roider DA | Full 63-patient cohort | P4 |
| A6 | Fig 5 CLL clinical | 95 patients | P6 (data gated) |

## Success criteria

- **A2:** CytoVI scib aggregate ≥ best baseline (Harmony/cyCombine) on min-max preproc; within ±0.05
  of reference repo on same split.
- **A3:** Mean imputation Pearson within ±0.05 of reference per-marker table.
- **A5:** Panel ICC ≥ 0.95 for DA cluster frequencies.

## Artifacts

```
benchmarks/cytovi/          # to be created
benchmarks/common/          # shared scib + training helpers
.scratch/cytovi-benchmark/results/
```

## Blockers

- Nuñez FCS + manual labels (issue 01)
- Full Roider raw (issue 02) — not vignette `.h5ad`
- cyCombine baseline via **[cyCombinePy](https://github.com/mdmanurung/cyCombinePy)** (Python; batch correction only, not imputation)
- CLL data on request (issue 07)
