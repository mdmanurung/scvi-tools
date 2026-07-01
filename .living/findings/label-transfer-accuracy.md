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

## F-003 — B1 Nuñez full inductive e1000 3-seed: CytoANVI +0.017 vs CytoVI kNN (ceiling)
**Date**: 2026-07-01 (supersedes 2026-06-27 entry)
**Status**: provisionally complete (nunez-full-inductive-e1000-3seed, ≥3 seeds, inductive kNN)
**Validity caveat**: `nunez-full-inductive-e1000-3seed` — full Nuñez cohort (~100k cells, 11 types), max_epochs=1000, seeds 0/1/2, inductive kNN annotation (L-022 fix). Phenograph and FlowSOM unavailable (pip install errors); not used.
**Claim**:
| Method | macro-F1 | n_held |
|--------|----------|--------|
| CytoANVI | **0.9751 ± 0.0003** | 20001/seed |
| CytoVI kNN | **0.9581 ± 0.0007** | 20001/seed |
| XGBoost | **0.9722 ± 0.0008** | 20001/seed |
| Raw marker kNN | **0.9010 ± 0.0028** | 20001/seed |
| Harmony kNN | **0.9111 ± 0.0026** | 20001/seed |

Δ (CytoANVI vs CytoVI kNN) = **+0.0170** (0.9751 − 0.9581).
**Meets target (≥+0.03)**: FORMALLY NO — but ceiling effect: both methods near-perfect on clean 11-type PBMC; CytoANVI still wins. Nuñez is not the primary comparison dataset.
**Source**: `.scratch/cytoanvi-benchmark/results/e1000/nunez_inductive_b1_multiseed.json` (PID 3186154 v3, commit e8a6d5f9 harmony fix)
**Implications**: Prior transductive result (Δ−0.013, seeds 0–4) was invalid due to leaky Leiden clustering (L-022). After fixing to inductive kNN, CytoANVI consistently outperforms all baselines even on clean PBMC data, just by a smaller margin (+0.017) than on heterogeneous clinical cytometry (Roider Δ+0.121, F-002). The ceiling effect is scientifically expected: well-separated 11-type PBMC leaves little room for semi-supervised improvement over strong unsupervised kNN. XGBoost (no batch info) scores 0.9722 vs CytoVI kNN 0.9581, so CytoANVI's ELBO-based embedding slightly outperforms both supervised and unsupervised competitors.

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
