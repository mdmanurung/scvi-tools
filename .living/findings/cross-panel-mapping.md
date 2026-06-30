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

## F-007 — B5 novelty AUROC 3-seed e1000: best 0.833 ± 0.122, mean 0.462 ± 0.075 (Roider 13-type holdout sweep)
**Date**: 2026-06-30 (updated from 2026-06-28 smoke; now 3-seed e1000)
**Status**: NOT publication-grade (roider-e1000, not roider-full)
**Validity caveat**: `roider-e1000` — `--dataset roider` (≈5k cells), max_epochs=1000, seeds 0/1/2; `roider-full` pending.
**Claim**: 13-type T-cell holdout sweep on Roider BNHL. 3-seed headline:
- best_auroc: **0.833 ± 0.122** (per-seed: 0.744, 0.783, 0.972)
- mean_auroc: **0.462 ± 0.075** (per-seed: 0.467, 0.385, 0.534)
- n_fdr_significant: **5.0 mean** (per-seed: 5, 3, 7)
Most reliable detectable types (mean AUROC across 3 seeds): Ttox EM3 **0.776 ± 0.071**, Tfh **0.724 ± 0.258** (high variance), Tpr **0.693 ± 0.027**. Consistently undetectable: Treg CD69- **0.120 ± 0.010**, Ttox EM1 **0.369 ± 0.259**, Ttox EM2 **0.258 ± 0.299**.
**Meets target (AUROC ≥0.70 mean across seeds)**: 2/13 types pass (Ttox EM3, Tfh); 1 near-threshold (Tpr 0.693). Bimodal distribution confirmed — 5 types consistently near-chance.
**Source**: `results/e1000/roider_e1000_b5_multiseed.json`
**Implications**: Novelty detection robust for immunologically distinct T-cell subsets (effector memory, follicular helper); fails for naive/regulatory subtypes that share marker profiles with training labels. High Tfh variance (0.457–0.972) warrants investigation.

---

## F-010 — B5 per-type AUROC table (3-seed e1000, Roider 13 T-cell types)
**Date**: 2026-06-30
**Status**: NOT publication-grade (roider-e1000)
**Validity caveat**: `roider-e1000`, seeds 0/1/2
**Claim**: Per-type mean AUROC ± std across 3 seeds:

| Type | Mean AUROC | Std | FDR sig (mean n seeds) | Notes |
|------|-----------|-----|------------------------|-------|
| Ttox EM3 | **0.776** | 0.071 | 3/3 | Most consistent; low variance |
| Tfh | **0.724** | 0.258 | 2/3 | High variance (0.457–0.972); seed-dependent |
| Tpr | 0.693 | 0.027 | 3/3 | Consistently near threshold |
| Treg CD69+ | 0.604 | 0.137 | 2/3 | Moderate; variable |
| Th CM2 | 0.529 | 0.133 | 1/3 | Borderline |
| Tdn | 0.480 | 0.025 | 0/3 | Near chance |
| Naive CD4 T | 0.415 | 0.123 | 0/3 | Below chance |
| Naive CD8 T | 0.408 | 0.215 | 0/3 | High variance |
| Tdp | 0.396 | 0.219 | 0/3 | High variance (n=16 novel cells) |
| Ttox EM1 | 0.369 | 0.259 | 0/3 | High variance |
| Ttox EM2 | 0.258 | 0.299 | 0/3 | Very high variance |
| Th CM1 | 0.230 | 0.045 | 0/3 | Consistently poor |
| Treg CD69- | **0.120** | 0.010 | 0/3 | Worst; model actively assigns to existing label |

**Interpretation**: Immunologically distinct effector types (EM3) are consistently detectable; naive and regulatory subtypes (CD4/CD8 naive, Th CM1, Treg CD69-) share marker profiles with training cells and are not detectable. High variance for Tfh, Tdp, Ttox EM1/EM2 may reflect small holdout size (n=277, 16, 266, 351) or stochastic embedding.
**Source**: `results/e1000/roider_e1000_b5_multiseed.json`
**Implications**: B5 novelty detection claim should focus on Ttox EM3 and Tpr (low variance, consistent); Tfh result is interesting but variable. Publication framing: "detects immunologically distinct effector subtypes; fails for phenotypically overlapping naive/regulatory clusters."
