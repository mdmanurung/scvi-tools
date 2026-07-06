# Analysis Manifest

Tracks benchmark tasks and their current status.

Publication evidence must be generated through
`.scratch/cytoanvi-benchmark/publication_manifest.json` and
`benchmarks/common/aggregate_results.py --manifest`. Recursive summaries and old Roider vignette
JSONs are exploratory/provenance only. As of 2026-07-05, full-cohort Roider B3 (3 seeds) and B5
(3 seeds) are complete and `publication_summary.json` is produced (commit 66a1b806). The honest
full-cohort numbers supersede all e1000/smoke figures below.

## CytoANVI benchmarks (`benchmarks/cytoanvi/`)

| Task | File | Measures | Baseline | Status | Result |
|------|------|----------|----------|--------|--------|
| B1: Label transfer | `run.py --task b1` | Macro-F1 | CytoVI k-NN | roider-full 3-seed ✅ COMPLETE (jobs 25149326/27/28); nunez-full-inductive-e1000 ✓ | **ROIDER FULL-COHORT (FINAL):** CytoANVI **0.9317±0.0022** vs CytoVI-kNN **0.8928±0.0034**, Δ **+0.0388±0.0018** ✅ gate (≥+0.03). XGBoost 0.9516. Nuñez: CytoANVI 0.9751±0.0003 vs kNN 0.9581±0.0007 (Δ+0.017, near-ceiling). Prior Roider e1000 Δ+0.121±0.040 superseded by full-cohort. Aggregate: `aggregate_b1_roider_multiseed.py` → `roider_full_b1_multiseed.json`. |
| B2: Integration | `run.py --task b2` | scib bio/batch | CytoVI latent | roider-e1000 3-seed ✓; nunez-r005-e1000 3-seed ✓; roider-full PENDING | Roider batch Δ−0.006 ✅; bio +0.108 gain; Nuñez batch Δ−0.005 ✅ (F-004, F-005) |
| B3: Cross-panel mapping | `run.py --task b3` | Inter-method agreement (NOT accuracy) | CytoVI k-NN | roider-full 3-seed ✅ COMPLETE (jobs 25145151/52/53) | **FULL-COHORT (FINAL):** p1 holdout macro-F1 **0.828±0.015** (supervised headline); p2 inter-method agreement **0.671±0.008** (concordance with CytoVI-kNN, NOT accuracy — see F-012). Gate (≥0.80) on concordance is NOT met at full cohort; p1 F1 is the defensible supervised result. **LIMITATION: p2 concordance is agreement between CytoANVI and CytoVI-kNN (two methods sharing the CytoVI encoder), NOT ground-truth accuracy — no independent panel-2 labels exist.** Prior e1000 numbers (0.877±0.012) were on 5k-cell subset — superseded by F-012. |
| B4: Continual update | `run.py --task b4` | F1 drift | Static CytoVI | roider-smoke only | **PLUMBING ONLY — pseudo-batch split, NOT real case/control cytometry data; drift 0.0 does NOT constitute evidence of catastrophic-forgetting mitigation** (F-008); blocked by real case/control data |
| B5: Novelty detection | `run.py --task b5` | AUROC (mean over types is PRIMARY) | CytoVI kNN-distance OOD | roider-full 3-seed ✅ COMPLETE (jobs 25145151/52/53); diagnostic re-run with CytoANVI-kNN baseline PENDING (jobs 25149032/33/34) | **FULL-COHORT (FINAL — NEGATIVE RESULT):** mean_auroc **0.484±0.019 (below chance)** vs CytoVI kNN-distance OOD baseline **0.775±0.002**. CytoANVI TTA-uncertainty is far worse than plain kNN-distance in CytoVI latent. See F-013. best_auroc 0.833±0.122 (e1000 subset max/cherry-picked) is NOT the headline. Diagnostic re-run (PENDING) will add CytoANVI-latent kNN to separate TTA-method weakness from latent-space weakness. |
| B6: λ sweep | `run.py --task b6` | F1 vs λ | — | roider-smoke only | **PLUMBING ONLY — pseudo-batch split, NOT real case/control cytometry data; λ=1.0 best (0.888) is NOT evidence of catastrophic-forgetting mitigation** (F-008) |
| B8: HCE vs flat CE | `run.py --task b8` | Macro-F1 | Flat CE | nunez-full-e1000 3-seed ✓ (job-25128164 complete 18:45 CEST Jun 30) | Δ_hier_vs_flat = +0.0862±0.0027 ✅ pub-gate; flat_ce 0.9783±0.0011; direct HCE −0.0984±0.0851 (expected) (F-011). **CAVEAT: Nuñez hierarchy has only ONE internal node (Dendritic cells → Plasmacytoid dendritic cells); all other 9 cell types are leaves. The +0.086 gain reflects hierarchical decoding on a shallow tree — this is NOT evidence that HCE training improves latent features in general.** |
| B9: mapQC | `run.py --task b9` | mapqc_score | Low control | FAILED (job 25149329; mapqc 0.1.1 installed but crashes) | **BLOCKED — mapqc 0.1.1 library bug:** `IndexError: single positional indexer is out-of-bounds` in `_get_per_cell_filtering_info` when mode of per-cell neighborhood filter is empty. Triggered by Nuñez dataset. Not reportable until mapqc is patched or dataset workaround found. |

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

## Prior smoke results (epochs=100, NOT publication-grade, SUPERSEDED)

These numbers are superseded by the full-cohort 3-seed results above. Retained for provenance.

| Task | Result | Superseded by |
|------|--------|---------------|
| B1 | CytoANVI +0.115 macro-F1 vs CytoVI k-NN (Roider, 3 seeds) | F-002 (roider-e1000), F-003 (nunez-full), full-cohort: +0.0388±0.0018 ✅ |
| B3 | Panel-2 concordance 0.86 (5k-cell subset smoke) | F-012: full-cohort 0.671±0.008 |
| B5 | Several holdout types AUROC > 0.70 (cherry-picked best_auroc framing) | F-013: full-cohort mean_auroc 0.484±0.019 (NEGATIVE) |
