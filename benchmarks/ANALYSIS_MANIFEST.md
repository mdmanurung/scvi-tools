# Analysis Manifest

Tracks benchmark tasks and their current status.

## CytoANVI benchmarks (`benchmarks/cytoanvi/`)

| Task | File | Measures | Baseline | Status | Result |
|------|------|----------|----------|--------|--------|
| B1: Label transfer | `run.py --task b1` | Macro-F1 | CytoVI k-NN | roider-e1000 3-seed ✓; nunez-inductive-e1000 RUNNING (PID 1520357); roider-full PENDING | Roider Δ+0.121±0.040 ✅ gate; Nuñez Δ−0.013 ❌ FAILS on leaky labels (F-003, L-022); re-run in progress with inductive kNN annotation fix |
| B2: Integration | `run.py --task b2` | scib bio/batch | CytoVI latent | roider-e1000 3-seed ✓; nunez-r005-e1000 3-seed ✓; roider-full PENDING | Roider batch Δ−0.006 ✅; bio +0.108 gain; Nuñez batch Δ−0.005 ✅ (F-004, F-005) |
| B3: Cross-panel mapping | `run.py --task b3` | Concordance | CytoVI k-NN | roider-e1000 3-seed ✓; roider-full PENDING | p2 concordance 0.877±0.012 ✅ gate (F-006) |
| B4: Continual update | `run.py --task b4` | F1 drift | Static CytoVI | roider-smoke only | drift 0.0 plumbing only (F-008); blocked by real case/control data |
| B5: Novelty detection | `run.py --task b5` | AUROC | — | seed-0 ✓ (FDR-patched); seeds 1+2 RUNNING (PIDs 1411611/1418691) | best 0.744 ✅ gate (Tfh); mean 0.467 concerning; bimodal 4/13 types ≥0.70 (F-007, F-010); multiseed pending |
| B6: λ sweep | `run.py --task b6` | F1 vs λ | — | roider-smoke only | λ=1.0 best (0.888); plumbing only (F-008) |
| B8: HCE vs flat CE | `run.py --task b8` | Macro-F1 | Flat CE | seed-0 ✓ seed-1 ✓; seed-2 RUNNING nvidia2 epoch~400/1000 (PID 1054336) | seeds 0/1 done; multiseed pending seed-2 completion |
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
