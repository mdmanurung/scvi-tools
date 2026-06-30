# Topic: Label-Transfer Accuracy (B1/B2)

**Question**: Does CytoANVI achieve better cell-type label transfer than CytoVI k-NN baseline?
**Datasets**: Nuñez PBMCs (2 batches, 11 cell types), Roider BNHL (2 panels)
**Source docs**: `.scratch/cytoanvi-benchmark/issues/08-nunez-b1-b2-full.md`, `results/roider_multiseed_summary.json`, `results/final_summary.json`

---

## F-001 — B1 Roider macro-F1: CytoANVI +0.115 vs CytoVI k-NN (vignette-100)
**Date**: 2026-06-18
**Status**: NOT publication-grade (vignette subsample, epochs=100)
**Validity caveat**: `vignette-100` — subsampled Roider, 100 epochs. Full-cohort e1000 result pending (publication gate).
**Claim**: CytoANVI macro-F1 **0.925 ± 0.009** vs CytoVI k-NN **0.810 ± 0.025** (Δ **+0.115**) over 3 seeds.
**Meets target (≥+0.03)**: yes
**Source**: `results/roider_multiseed_summary.json`

## F-002 — B1 Roider macro-F1: vignette e1000 (epochs=1000, scib)
**Date**: 2026-06-27
**Status**: NOT publication-grade (vignette subsample, not full `roider-full`)
**Validity caveat**: `vignette-e1000` — subsampled Roider, 1000 epochs, scib integration.
**Claim**: CytoANVI macro-F1 **0.908 ± 0.008** vs CytoVI k-NN **0.787 ± 0.039** (Δ **+0.121**, 3 seeds).
**Meets target (≥+0.03)**: yes
**Source**: `results/final_summary.json`, publication_manifest.json phase 2

## F-003 — B1 Nuñez macro-F1: CytoANVI slightly worse than k-NN at e1000 (reversal)
**Date**: 2026-06-27
**Status**: NOT publication-grade (vignette subsample, 5 seeds)
**Validity caveat**: `vignette-e1000` — subsampled Nuñez. This reversal is the key negative result to watch.
**Claim**: CytoANVI **0.954 ± 0.032** vs k-NN **0.967 ± 0.004** (Δ **−0.013**, seeds 0–4). Seeds 3–4 show Δ+0.020; seeds 0–2 pull it down. Clean Nuñez (well-separated types, strong k-NN) may be a regime where CytoANVI does not improve on k-NN.
**Meets target (≥+0.03)**: no
**Source**: `results/final_summary.json`
**Implications**: B1 success criterion requires Δ≥+0.03 on full Nuñez. This pre-result suggests it may not pass. Monitor closely when full-cohort jobs complete.

## F-004 — B2 Roider e1000 3-seed: CytoANVI bio +0.108, batch Δ−0.006 (passes both gates)
**Date**: 2026-06-29 (updated from vignette-e1000 to roider-e1000 3-seed)
**Status**: NOT publication-grade (roider-e1000 ≈5k cells, not roider-full)
**Validity caveat**: `roider-e1000` — `--dataset roider` (vignette-scale Roider), max_epochs=1000, 3 seeds
**Claim**: CytoANVI bio=**0.737 ± 0.003** vs CytoVI **0.628 ± 0.013** (Δ **+0.108**); batch=**0.792 ± 0.014** vs CytoVI **0.798 ± 0.007** (Δ **−0.006**).
**Meets target**: YES — bio +0.108 (strong), batch −0.006 (within ±0.05 tolerance)
**Source**: `results/e1000/roider_e1000_multiseed.json`, publication_manifest.json
**Implications**: At e1000 on vignette-scale data, CytoANVI clearly wins on bio conservation without measurable batch-mixing loss. This supersedes earlier vignette-100 result (F-004 prev: bio +0.099, batch −0.040 — batch gate missed at 100 epochs).

## F-005 — B2 Nuñez r0.05 e1000 3-seed: CytoANVI bio +0.009, batch Δ−0.005 (passes both gates)
**Date**: 2026-06-29
**Status**: NOT publication-grade (nunez-r005-e1000 subsample, not nunez-full)
**Validity caveat**: `nunez-r005-e1000` — Nuñez subsampled r=0.05, max_epochs=1000, seeds 0/1/2
**Claim**: CytoANVI bio=**0.769 ± 0.010** vs CytoVI **0.760 ± 0.011** (Δ **+0.009**); batch=**0.799 ± 0.002** vs CytoVI **0.804 ± 0.001** (Δ **−0.005**).
**Meets target**: YES — bio +0.009 (marginal), batch −0.005 (within ±0.05 tolerance)
**Source**: `results/e1000/nunez_r005_e1000_multiseed.json`
**Implications**: Bio improvement is small (+0.009) on Nuñez — dataset's clean separation means CytoVI already achieves strong integration. Batch mixing nearly neutral. Both PRD B2 gates pass. Full-cohort confirmation needed.
