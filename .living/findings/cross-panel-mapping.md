# Topic: Cross-Panel Mapping and Novelty Detection (B3/B5)

**Question**: Can CytoANVI transfer labels across panels (B3) and detect novel/held-out cell types (B5)?
**Dataset**: Roider BNHL (panel-1 labeled, panel-2 unlabeled; 2 panels, 10 backbone markers)
**Source docs**: `.scratch/cytoanvi-benchmark/issues/09-roider-b3-b5-full.md`, `results/roider_multiseed_summary.json`

---

## F-006 — B3 panel-1 holdout macro-F1: 0.941 ± 0.012 ⚠️ SUPERSEDED by F-012
**Date**: 2026-06-28
**Status**: SUPERSEDED — `roider-smoke` (≈5k cells, 100 epochs); use F-012 for full-cohort results.
**Validity caveat**: `roider-smoke` — `--dataset roider` (vignette-scale Roider), 3 seeds. Numbers are NOT representative of full-cohort performance.
**Claim**: Panel-1 holdout macro-F1 **0.941 ± 0.012** (seeds 0/1/2); panel-2 concordance vs k-NN **0.862 ± 0.009**.
**Meets target (concordance ≥0.80)**: yes (0.862) — but this is the *smoke* number, superseded by 0.671 at full cohort (F-012).
**Source**: `.scratch/cytoanvi-benchmark/issues/09-roider-b3-b5-full.md`, `results/final_summary.json`
**Note**: Panel-1 cell_type labels = Leiden clusters at resolution 1.0 (not manual gating) — benchmark is cluster concordance, not biological ground truth.

## F-007 — B5 novelty AUROC 3-seed e1000: best 0.833 ± 0.122, mean 0.462 ± 0.075 (Roider 13-type holdout sweep) ⚠️ SUPERSEDED by F-013
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
**Status**: SUPERSEDED — `roider-e1000` (≈5k cells); use F-013 for full-cohort 11-type results.

---

## F-010 — B5 per-type AUROC table (3-seed e1000, Roider 13 T-cell types) ⚠️ SUPERSEDED by F-013
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
**Status**: SUPERSEDED by F-013 (full-cohort 11-type sweep). e1000 subset numbers are not representative.

---

## F-012 — B3 Roider FULL-COHORT 3-seed (publication-grade) ✅
**Date**: 2026-07-05
**Status**: Publication-grade (roider-full, max_epochs=1000, seeds 0/1/2, jobs 25145151/52/53)
**Claim**:
- Panel-1 holdout supervised macro-F1: **0.828 ± 0.015** (defensible headline — evaluates CytoANVI predictions against held-out labeled cells)
- Panel-2 inter-method agreement vs CytoVI-kNN: **0.671 ± 0.008** (concordance between two models, NOT ground-truth accuracy)
**Gate status**: p2 concordance gate (≥0.80) is **NOT met** at full cohort (0.671). p1 macro-F1 (0.828) is the honest supervised result and should be the primary claim.
**Supersedes**: F-006 (roider-e1000 0.877; smoke 0.862 — both from small subsets)
**Critical limitation**: p2 "concordance" is agreement between CytoANVI and CytoVI-kNN — two methods that share the CytoVI encoder for the baseline. This is NOT independent validation against expert-gated ground truth.
**Source**: `results/roider_full_b3_s{0,1,2}.json`, `results/roider_full_b3_multiseed.json`, `results/publication_summary.json`

---

## F-013 — B5 Roider FULL-COHORT 3-seed (publication-grade) ✅ NEGATIVE RESULT
**Date**: 2026-07-05
**Status**: Publication-grade (roider-full, max_epochs=1000, seeds 0/1/2, inductive calibrated, 11 Leiden types)
**Claim**:
- CytoANVI TTA-uncertainty mean_auroc: **0.484 ± 0.019 (below chance — NEGATIVE)**
- CytoVI kNN-distance OOD baseline: **0.775 ± 0.002** (substantially better)
- Gap: CytoVI kNN-OOD outperforms CytoANVI TTA-uncertainty by **0.291 ± 0.021**
- best_auroc 0.833 is the max over types (a single cherry-picked type); it is NOT the summary statistic.
**Interpretation**: CytoANVI's Bregman-Information TTA uncertainty scores do not reliably flag novel cell types at full-cohort scale. A simple unsupervised kNN-distance score in CytoVI latent substantially outperforms. This is a **robust negative result** (consistent across 3 seeds, 0.484 ± 0.019).
**Open question (diagnostic pending, jobs 25149032/33/34)**: Is this a failure of the TTA *method* (Bregman-Information) or a failure of CytoANVI's *latent space* as an OOD space? The re-run adds `cytoanvi_knn_mean_auroc` — if CytoANVI-kNN ≈ CytoVI-kNN then TTA is the weak link; if both fail then the latent is weaker.
**Supersedes**: F-007, F-010 (roider-e1000 subset)
**Source**: `results/roider_full_b5_sweep_s{0,1,2}.json`, `results/roider_full_b5_sweep_multiseed.json`, `results/publication_summary.json`
