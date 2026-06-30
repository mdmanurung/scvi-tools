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

## F-004 — B2 scib: better bio, worse batch (fails batch target)
**Date**: 2026-06-27
**Status**: NOT publication-grade (vignette-e1000)
**Validity caveat**: `vignette-e1000`
**Claim**: CytoANVI bio score **+0.099** vs CytoVI; batch score **−0.040**. Both B2 PRD flags `false`.
**Meets target**: no (batch target ±0.05 breached)
**Source**: `results/roider_multiseed_summary.json`, `results/final_summary.json`
**Implications**: Batch mixing worse with CytoANVI — classifier pressure may over-separate batches. Watch B2 on full cohort.

## F-005 — Nuñez B2 seed-2 one-off: CytoANVI ≈ CytoVI (essentially tied)
**Date**: 2026-06-29
**Status**: single-seed recovery run, not in publication manifest
**Validity caveat**: one seed, one-off job 25104249
**Claim**: CytoANVI total scib **0.7739** vs CytoVI **0.7747** — essentially tied.
**Source**: PRD.md phase log
