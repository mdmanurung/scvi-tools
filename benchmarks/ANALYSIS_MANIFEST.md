# Analysis Manifest

Tracks benchmark tasks and their current status.

Publication evidence must be generated through
`.scratch/cytoanvi-benchmark/publication_manifest.json` and
`benchmarks/common/aggregate_results.py --manifest`. Recursive summaries and old Roider vignette
JSONs are exploratory/provenance only. As of 2026-07-03, required full-cohort Roider B3/B5
artifacts are still running and the package is not yet final-publication ready.

## CytoANVI benchmarks (`benchmarks/cytoanvi/`)

| Task | File | Measures | Baseline | Status | Result |
|------|------|----------|----------|--------|--------|
| B1: Label transfer | `run.py --task b1` | Macro-F1 | CytoVI k-NN | roider-e1000 3-seed ✓; nunez-full-inductive-e1000 ✓ (PID 3186154 v3, done 2026-07-01 05:58); roider-full PENDING (after B3/B5 full-cohort) | Roider Δ+0.121±0.040 ✅ gate; Nuñez: CytoANVI 0.9751±0.0003 vs kNN 0.9581±0.0007 (Δ+0.017, ceiling); prior reversal Δ−0.013 was transductive leakage L-022; F-003 updated |
| B2: Integration | `run.py --task b2` | scib bio/batch | CytoVI latent | roider-e1000 3-seed ✓; nunez-r005-e1000 3-seed ✓; roider-full PENDING | Roider batch Δ−0.006 ✅; bio +0.108 gain; Nuñez batch Δ−0.005 ✅ (F-004, F-005) |
| B3: Cross-panel mapping | `run.py --task b3` | Inter-method agreement (NOT accuracy) | CytoVI k-NN | roider-e1000 3-seed ✓; roider-full RUNNING (rerun job 25140597 on res-hpc-gpu11, publication recipe; supersedes failed job 25132400) | p2 inter-method agreement vs kNN 0.877±0.012 ✅ concordance gate (F-006, roider-e1000). **LIMITATION: this is agreement between CytoANVI and CytoVI-kNN (two methods sharing the CytoVI encoder), NOT ground-truth accuracy — no independent panel-2 labels exist.** Smoke concordance 0.641; full-cohort rerun pending. |
| B4: Continual update | `run.py --task b4` | F1 drift | Static CytoVI | roider-smoke only | **PLUMBING ONLY — pseudo-batch split, NOT real case/control cytometry data; drift 0.0 does NOT constitute evidence of catastrophic-forgetting mitigation** (F-008); blocked by real case/control data |
| B5: Novelty detection | `run.py --task b5` | AUROC (mean over types is PRIMARY) | — | roider-e1000 3-seed ✓; roider-full sweep RUNNING (rerun job 25140598 on res-hpc-gpu11, publication recipe, inductive calibrated B5; supersedes failed job 25132895) | **PRIMARY: mean_auroc 0.462±0.075 (near chance)** — unweighted mean over all held-out cell types. best_auroc 0.833±0.122 is max over types (single cherry-picked type) and is NOT the headline summary statistic. (roider-e1000, F-007, F-010); full-cohort sweep pending |
| B6: λ sweep | `run.py --task b6` | F1 vs λ | — | roider-smoke only | **PLUMBING ONLY — pseudo-batch split, NOT real case/control cytometry data; λ=1.0 best (0.888) is NOT evidence of catastrophic-forgetting mitigation** (F-008) |
| B8: HCE vs flat CE | `run.py --task b8` | Macro-F1 | Flat CE | nunez-full-e1000 3-seed ✓ (job-25128164 complete 18:45 CEST Jun 30) | Δ_hier_vs_flat = +0.0862±0.0027 ✅ pub-gate; flat_ce 0.9783±0.0011; direct HCE −0.0984±0.0851 (expected) (F-011). **CAVEAT: Nuñez hierarchy has only ONE internal node (Dendritic cells → Plasmacytoid dendritic cells); all other 9 cell types are leaves. The +0.086 gain reflects hierarchical decoding on a shallow tree — this is NOT evidence that HCE training improves latent features in general.** |
| B9: mapQC | `run.py --task b9` | mapqc_score | Low control | **BLOCKED — optional dependency `scvi-tools[cytoanvi-mapping-qc]` not installed; task logic not validated end-to-end** | — |

## Label provenance & limitations

**This section must be read before interpreting B1–B3 results.**

### Nuñez dataset

Labels derive from the **CytoVI tutorial workflow**: train CytoVI → compute Leiden clusters → apply manual cluster-to-cell-type map. They are NOT independent manual gating performed by a biologist blind to the model output. Consequence: the CytoVI encoder's latent structure is baked into the cluster assignments used as "ground truth."

### Roider dataset (full cohort)

Labels are **Leiden clusters at resolution r=1.0**, not manual gating. No independent expert annotation has been applied to the full cohort. Cluster identities may conflate biologically distinct populations or split a single population into multiple clusters.

### Circular comparison in B1 and B3

The **CytoVI-kNN baseline** uses the same CytoVI encoder that generated the Leiden clusters used as reference labels. This creates a partially circular comparison: the baseline has privileged access to the same latent structure that defined the labels. B1 and B3 results overstate the advantage of CytoANVI relative to a truly independent annotator. Treat these benchmarks as internal consistency checks, not independent validation.

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
