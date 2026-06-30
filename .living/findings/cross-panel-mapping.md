# Topic: Cross-Panel Mapping and Novelty Detection (B3/B5)

**Question**: Can CytoANVI transfer labels across panels (B3) and detect novel/held-out cell types (B5)?
**Dataset**: Roider BNHL (panel-1 labeled, panel-2 unlabeled; 2 panels, 10 backbone markers)
**Source docs**: `.scratch/cytoanvi-benchmark/issues/09-roider-b3-b5-full.md`, `results/roider_multiseed_summary.json`

---

## F-006 — B3 panel-1 holdout macro-F1: 0.941 ± 0.012
**Date**: 2026-06-28
**Status**: NOT publication-grade (roider smoke — `--dataset roider`, not `roider-full`)
**Validity caveat**: `roider-smoke` — `--dataset roider` (vignette-scale Roider), 3 seeds. Full-cohort `roider-full` pending.
**Claim**: Panel-1 holdout macro-F1 **0.941 ± 0.012** (seeds 0/1/2); panel-2 concordance vs k-NN **0.862 ± 0.009**.
**Meets target (concordance ≥0.80)**: yes (0.862)
**Source**: `.scratch/cytoanvi-benchmark/issues/09-roider-b3-b5-full.md`, `results/final_summary.json`
**Note**: Panel-1 cell_type labels = Leiden clusters at resolution 1.0 (not manual gating) — benchmark is cluster concordance, not biological ground truth.

## F-007 — B5 novelty AUROC: best 0.909, mean 0.490 (full 13-type holdout sweep)
**Date**: 2026-06-28
**Status**: NOT publication-grade (roider-smoke)
**Validity caveat**: `roider-smoke`, seed-0 only for sweep
**Claim**: Held-out type sweep over 13 Leiden clusters. Best AUROC (seed-0): Tfh **0.875**, Treg CD69+ **0.779**, Ttox EM3 **0.742**. Overall best **0.909**, mean **0.490** across all types. Several types AUROC > 0.70 (pass target); others <0.50 (fail). Full sweep table in `results/roider_multiseed_summary.json`.
**Meets target (AUROC ≥0.70 for held-out types)**: partially — passes for high-uncertainty types, fails for ambiguous clusters.
**Source**: `results/roider_multiseed_summary.json`
**Implications**: Novelty detection works for well-separated types; ambiguous Leiden clusters (possibly heterogeneous) show poor discrimination. Consider whether biological or cluster identity matters for B5 framing.
