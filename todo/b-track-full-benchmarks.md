**Title**: Full Nuñez/Roider benchmarks at max_epochs=1000
**Status**: open
**Priority**: critical
**Category**: analysis
**Date**: 2026-06-10
**Author**: mdmanurung

## Description

Run all Track-B benchmark tasks (B1–B6, B8, B9) on full Nuñez and Roider cohorts at max_epochs=1000 with ≥3 seeds.

## Motivation

Prior results used vignette subsamples (100 epochs) — not publication-grade. scib metrics require full-cohort cell counts. Gate publication on these results (see D-002).

## Acceptance criteria

- [ ] B1: CytoANVI macro-F1 ≥ CytoVI k-NN + 0.03 (mean over ≥3 seeds)
- [ ] B2: scib bio score ≥ CytoVI, batch score within 0.05
- [ ] B3: Panel-1 → Panel-2 mapping concordance ≥ 0.80
- [ ] B4: Continual update F1 within 0.03 of fresh retrain
- [ ] B5: Novelty AUROC ≥ 0.70 on held-out types
- [ ] B6: λ sweep identifies stable default
- [ ] B8: HCE ≥ flat CE on Roider
- [ ] B9: mapQC passes on query controls

## Notes

See PRD: `.scratch/cytoanvi-benchmark/PRD.md`
Benchmark code: `benchmarks/cytoanvi/`
Issues: `.scratch/cytoanvi-benchmark/issues/08-*.md` through `13-*.md`
