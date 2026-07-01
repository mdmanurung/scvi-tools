# Analysis Manifest

Tracks benchmark tasks and their current status.

## CytoANVI benchmarks (`benchmarks/cytoanvi/`)

| Task | File | Measures | Baseline | Status | Result |
|------|------|----------|----------|--------|--------|
| B1: Label transfer | `run.py --task b1` | Macro-F1 | CytoVI k-NN | roider-e1000 3-seed ✓; nunez-full-inductive-e1000 ✓ (PID 3186154 v3, done 2026-07-01 05:58); roider-full PENDING (after B3/B5 full-cohort) | Roider Δ+0.121±0.040 ✅ gate; Nuñez: CytoANVI 0.9751±0.0003 vs kNN 0.9581±0.0007 (Δ+0.017, ceiling); prior reversal Δ−0.013 was transductive leakage L-022; F-003 updated |
| B2: Integration | `run.py --task b2` | scib bio/batch | CytoVI latent | roider-e1000 3-seed ✓; nunez-r005-e1000 3-seed ✓; roider-full PENDING | Roider batch Δ−0.006 ✅; bio +0.108 gain; Nuñez batch Δ−0.005 ✅ (F-004, F-005) |
| B3: Cross-panel mapping | `run.py --task b3` | Concordance | CytoVI k-NN | roider-e1000 3-seed ✓; roider-full RUNNING (job 25132400, 3 seeds, 1000 epochs; smoke gate ✅ job 25129287: 3.49s/epoch CytoANVI, 6.55s CytoVI, no NaN); ETA ~13h | p2 concordance 0.877±0.012 ✅ gate (F-006); smoke 20-epoch concordance 0.641 |
| B4: Continual update | `run.py --task b4` | F1 drift | Static CytoVI | roider-smoke only | drift 0.0 plumbing only (F-008); blocked by real case/control data |
| B5: Novelty detection | `run.py --task b5` | AUROC | — | roider-e1000 3-seed ✓; roider-full sweep RUNNING (job 25132401, seed 0, 47 clusters × ~29 min; ETA ~24h; --time=48:00:00) | best_auroc 0.833±0.122; mean_auroc 0.462±0.075 (roider-e1000); full-cohort sweep pending (F-007, F-010) |
| B6: λ sweep | `run.py --task b6` | F1 vs λ | — | roider-smoke only | λ=1.0 best (0.888); plumbing only (F-008) |
| B8: HCE vs flat CE | `run.py --task b8` | Macro-F1 | Flat CE | nunez-full-e1000 3-seed ✓ (job-25128164 complete 18:45 CEST Jun 30) | Δ_hier_vs_flat = +0.0862±0.0027 ✅ pub-gate; flat_ce 0.9783±0.0011; direct HCE −0.0984±0.0851 (expected) (F-011) |
| B9: mapQC | `run.py --task b9` | mapqc_score | Low control | BLOCKED (mapqc not installed) | — |

## CytoVI benchmarks (`benchmarks/cytovi/`)

| Task | Status |
|------|--------|
| Vignette smoke | passing |

## Common utilities (`benchmarks/common/`)

- `training.py` — shared training loop with checkpoint/resume
- `aggregate_results.py` — JSON result aggregation across seeds

## Prior smoke results (epochs=100, NOT publication-grade)

| Task | Result |
|------|--------|
| B1 | CytoANVI +0.115 macro-F1 vs CytoVI k-NN (Roider, 3 seeds) |
| B3 | Panel-2 concordance 0.86 |
| B5 | Several holdout types AUROC > 0.70 |
