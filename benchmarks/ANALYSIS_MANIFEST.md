# Analysis Manifest

Tracks benchmark tasks and their current status.

## CytoANVI benchmarks (`benchmarks/cytoanvi/`)

| Task | File | Measures | Baseline | Status | Result |
|------|------|----------|----------|--------|--------|
| B1: Label transfer | `run.py --task b1` | Macro-F1 | CytoVI k-NN | roider-e1000 3-seed ✓; nunez-inductive-e1000 RUNNING (PID 3186154, seed-0 epoch~98/1000 @21:32 CEST Jun 30; ~14s/epoch; ETA all-3 seeds ~08:45 CEST Jul 1); roider-full PENDING | Roider Δ+0.121±0.040 ✅ gate; Nuñez rerun after harmony fix (L-025, commit e8a6d5f9); prior runs (1520357/2539861) crashed on harmonypy 0.2.0 Z_corr transposition |
| B2: Integration | `run.py --task b2` | scib bio/batch | CytoVI latent | roider-e1000 3-seed ✓; nunez-r005-e1000 3-seed ✓; roider-full PENDING | Roider batch Δ−0.006 ✅; bio +0.108 gain; Nuñez batch Δ−0.005 ✅ (F-004, F-005) |
| B3: Cross-panel mapping | `run.py --task b3` | Concordance | CytoVI k-NN | roider-e1000 3-seed ✓; roider-full PENDING (smoke test job 25129287 RUNNING, Leiden=47 clusters); B3 submission gated on smoke epoch timing | p2 concordance 0.877±0.012 ✅ gate (F-006) |
| B4: Continual update | `run.py --task b4` | F1 drift | Static CytoVI | roider-smoke only | drift 0.0 plumbing only (F-008); blocked by real case/control data |
| B5: Novelty detection | `run.py --task b5` | AUROC | — | roider-e1000 3-seed ✓ (seeds 0/1/2 complete; multiseed JSON written) | best_auroc 0.833±0.122; mean_auroc 0.462±0.075; n_fdr_sig 5.0; 2/13 types pass ≥0.70 (Ttox EM3 0.776±0.071, Tfh 0.724±0.258); Tpr near-threshold 0.693±0.027; bimodal confirmed (F-007, F-010) |
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
