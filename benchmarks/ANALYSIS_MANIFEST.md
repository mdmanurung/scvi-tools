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
| B8: HCE vs flat CE | `run.py --task b8` | Macro-F1 | Flat CE | nunez-full-e1000 3-seed ✓ (job-25128164); Hao CITE-seq ADT mixed-granularity sweep (multiseed in progress) | Nuñez Δ_hier_vs_flat = +0.0862±0.0027; flat_ce 0.9783±0.0011 (F-011). **CORRECTED FRAMING (L-046):** HCE is *mathematically identical to flat CE for leaf-only labels* (a leaf's subtree mass = its own prob; loss+gradients match to 1e-7 — `test_hce_equals_flat_ce_for_leaf_only_targets`). It only engages when some cells are labelled at INTERNAL (coarse) nodes — partial/mixed-granularity annotation. Nuñez's +0.086 came from a *populated* coarse node (Dendritic cells has cells AND parents pDC), exactly that regime. A first Hao test with all-leaf labels showed a spurious −0.03; validation traced it to (a) an all-leaf setup where HCE is a no-op and (b) a **bf16 numerical bug** in the HCE loss (`finfo(bf16).eps≈8e-3` distorted the log), now FIXED (commit fcfff064, fp32). The correct mixed-granularity test (Hao ADT, fp32, controlled seed, **3 seeds**) at coarse_frac 0.0/0.4/0.7: leaf macro-F1 +0.0002±0.0008 / +0.0094±0.0085 / +0.0059±0.0110 (**noisy, ~1σ — NOT a robust fine-accuracy win**); cross-lineage error −0.0001±0.0000 / −0.0008±0.0004 / −0.0011±0.0004 (**consistent, all 3 seeds, ~2.75σ**). **Verdict: HCE's value is LINEAGE COHERENCE, not fine accuracy — report it with a lineage-level / hierarchical metric, not leaf macro-F1. Effect is small (~9% rel. cross-lineage-error reduction; lineage acc already ~99% on rich ADT = ceiling). Frame around partial/mixed-granularity annotation; HCE is a provable no-op on all-fine labels.** |
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

## MrTotalVI / MrMultiVI benchmarks (`schisto CITE-seq`)

Empirical validation on the schistosomiasis CITE-seq dataset (human, 97,954 cells; 6 batches,
29 cell types; capped at 50,000 cells for scoring). All models trained at `n_latent=20`,
`max_epochs` per `train_model.py` defaults, 10,000 HVGs. **Single seed (SEED=0) only** — treat
as preliminary; multi-seed needed for publication. Runner:
`.scratch/mr-schisto-benchmark/run_mr_multimodal_benchmark.py`, latents in
`schisto_citeseq/analysis/integration/mr_multimodal_publication/outputs/`.

### scIB integration scores (human, seed 0 — see F-015)

| Model | Bio conservation | Batch correction | **Total** |
|-------|-----------------|-----------------|-----------|
| Unintegrated | 0.632 | 0.521 | 0.588 |
| TotalVI | 0.576 | **0.684** | **0.619** |
| MrTotalVI_u | 0.577 | 0.640 | 0.602 |
| MrTotalVI_z | 0.599 | 0.620 | 0.607 |
| MultiVI | 0.585 | 0.555 | 0.573 |
| **MrMultiVI_u** | 0.594 | 0.663 | **0.622** ← best overall |
| MrMultiVI_z | 0.592 | 0.655 | 0.617 |

### kNN label transfer (held-out 20%, k=15, seed 0 — see F-015)

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Unintegrated | **0.863** | **0.666** ← best (integration hurts kNN) |
| TotalVI | 0.823 | 0.586 |
| MrTotalVI_u | 0.826 | 0.583 |
| MrTotalVI_z | 0.834 | 0.582 |
| MultiVI | 0.816 | 0.564 |
| MrMultiVI_u | 0.809 | 0.562 |
| MrMultiVI_z | 0.809 | 0.584 |

### Interpretation

**MrMultiVI vs MultiVI:** ✅ Batch-correction win — +0.047 total (3-seed), +0.128±0.008 batch correction. MR
hierarchical donor latent substantially improves batch mixing; bio conservation is unchanged
(Δbio = −0.006±0.010, mean ≈ 0 across 3 seeds; the single-seed +0.009 estimate from F-015 was
noise that did not replicate). MultiVI baseline is single-seed (SEED=0 fixed), so error bars are
asymmetric: gap of +0.047 is ~5× MR's within-model std (±0.009). Label transfer negative (kNN
regresses — see F-015; probable over-mixing of rare/single-batch cell types). See F-038 for
3-seed bio/batch decomposition.

**MrTotalVI vs TotalVI:** ❌ No scIB win — MrTotalVI_z −0.012 / MrTotalVI_u −0.017 total vs
TotalVI. TotalVI batch correction (0.684) notably better than MrTotalVI_u (0.640) or _z (0.620).
MrTotalVI_z slightly better on label transfer accuracy (+0.011) with similar F1.

**Unintegrated dominates label transfer (all models):** Integration uniformly hurts kNN label
transfer on this dataset. Probable cause: 6/29 cell types (DC1, DC2, HSC_erythroid, RBC,
B_non-switched_memory, B_switched_memory) are single-batch or too small — kBET skips them.
Integration mixes batches but disrupts the local cluster structure those rare types need for kNN.
The kNN metric is misleading here; scIB total score is the appropriate headline.

**Status:** human scIB complete (3 seeds both models). Multi-seed details:
- **MrTotalVI 3-seed scIB (F-022, 2026-07-12):** MrTotalVI_u 0.634±0.007, MrTotalVI_z 0.628±0.004 vs TotalVI 0.639. Gap vs TotalVI = 0.005 (<1%), within 1 std — not a clinically meaningful difference. Single-seed s0 was atypically low (0.607); multi-seed mean is more representative.
- **MrMultiVI 3-seed scIB (F-024, 2026-07-12):** MrMultiVI_u **0.640±0.009**, MrMultiVI_z 0.634±0.006 vs MultiVI 0.593. **+0.047 win over MultiVI** ✅ clear and consistent. MrMultiVI_u ties TotalVI (0.639). Full 3-seed ranking: MrMultiVI_u 0.640 ≥ TotalVI 0.639 > MrTotalVI_u 0.634 > MrMultiVI_z/MrTotalVI_z 0.628–0.634 > MultiVI 0.593.
- Macaque pending (no latents yet).

**DE / DA schisto results (2026-07-11–12):**
- **MrTotalVI W22 multi-seed (F-023):** median lfc_std=0.011 (stable). Top hits = Y-chr genes (sex confound, CV≈70%) + **IFN "suppression"** (IFITM3, IFIT3, IFI44L; cv<35%). ⚠️ IFN direction is a model artifact — see F-029.
- **Sex-adjusted DE MrTotalVI W22 — NULL RESULT (F-027 artifact, F-028 multi-seed, COMPLETE ✅):** `donor_key='sex'` multi-seed (jobs 25211187/207/208, seeds 0+1+2). **Spearman rho (sex_adj multi-seed vs naive multi-seed) = 1.000**. Reason: sex is orthogonal to timepoint in the balanced paired design (all 10 donors at both W00+W22), so WLS sex beta is algebraically independent of the timepoint beta. Corrected from prior "n=5 donors" — actual n=10 donors (4M+6F). See L-075 and L-076.
- **MrMultiVI W22 multi-seed (F-025):** median lfc_std=0.050 (4.5× noisier than MrTotalVI). Top hits dominated by Ig V-gene segments and lncRNAs — no IFN signal in top-20. IFN 6/9 concordant with MrTotalVI in sign — also artifacts per F-029.
- **Cross-model W22 Spearman multi-seed (F-026):** rho=0.289 (vs single-seed 0.104 from F-018). The shared IFN signal (6/9 concordant) is the most coherent cross-model signal but represents a shared artifact (both models' eps contract IFN genes negatively).
- **Donor pseudobulk calibration — IFN UPREGULATED at W22, model artifact confirmed (F-029, COMPLETE ✅):** 10-donor paired t-test (df=9, BH). 0 genes pass genome-wide padj<0.05 (top: PROK2 padj=0.082). All 9 IFN genes POSITIVE direction in unstratified pseudobulk. Cell-type-stratified (12 types): IFN consistently UP within every cell type (STAT1 positive 12/12, IFITM3 9/12). Classical monocytes: 2 significant genes (U62317.4, PSMB10), ISG-dominated top-20. Y-chr negative control: DDX3Y pseudobulk ≈ 0 (padj=0.814) vs model +0.441 — confirms model's Y-chr signal is sex-composition confound. **⚠️ MODEL IFN-DOWN NARRATIVE IS WRONG.** Correct narrative: IFN pathway upregulation at W22 (post-treatment immune recovery). Files: `results/pseudobulk_de_W22_results.tsv`, `results/pseudobulk_stratified_summary.json`.
- Permutation null calibration (F-021): 20/20 perms complete; per-cell chi² test at n=10 donors produces frac_below_0.05 sd≈0.47 → uninformative. Formal calibration achieved via pseudobulk (F-029).

**DA/DE parity with MRVI (2026-07-11, COMPLETE):** Full remediation B1–B5 landed.
- `store_lfc=True` now returns decoded gene/protein `lfc`, `lfc_std`, `pde` (when `delta`
  provided), and optional `baseline_expression` for both MrTotalVI and MrMultiVI.
- Protein background is deterministic on the LFC contrast path (D-021; `rate_back=exp(back_alpha)`).
- Feature coords split as `"gene"` / `"protein"` at the model level (D-022).
- Both models default `use_vmap=False`; MrTotalVI uses BatchNorm (vmap-incompatible), D-023.
- 5/5 MrTotalVI LFC tests + 6/6 MrMultiVI LFC tests pass; backward-compat tests pass.
- ADR-0005/0006 and `mr_multimodal.md` updated; L-061 added for BatchNorm×vmap gotcha.

### DTP retraining results (donor_timepoint, 3 seeds — 2026-07-12)

Rationale: `sample_key="donor"` makes temporal DE/DA invalid in paired designs (L-076–L-078). DTP retrain uses `sample_key="donor_timepoint"` (20 samples = 10 donors × 2 timepoints), fixing the design matrix. Jobs 25211393–25211398.

**scIB (F-032):**

| Model | Total |
|-------|-------|
| MrMultiVI_u_dtp | **0.648±0.006** ← +0.057 vs MultiVI ✅ |
| MrMultiVI_z_dtp | 0.641±0.005 |
| MultiVI (baseline) | 0.591 |
| MrTotalVI_u_dtp | 0.625±0.012 |
| MrTotalVI_z_dtp | 0.624±0.003 |
| TotalVI (baseline) | 0.634 |

MrMultiVI DTP integration win is preserved and confirmed. MrTotalVI DTP does not improve over non-DTP.

**DE concordance vs PyDESeq2 gold standard (F-033, F-036):**

| Model | Spearman rho | Sign agree (all) | Sign agree (top-100) | IFN direction |
|-------|-------------|-----------------|---------------------|---------------|
| MrTotalVI_dtp | −0.240 | 0.424 | 0.220 | 12/12 wrong; abs_max=0.013 (near-zero) |
| MrMultiVI_dtp | +0.036 | 0.514 | n/a | near-random |
| Old donor model | −0.095 | ~0.16 (top-100) | — | wrong (invalid design matrix) |
| **MRVI (reference)** | **−0.138** | **0.450** | **0.370** | **0/7 correct; abs_max=0.032 (near-zero)** |

**Key finding (F-036, 2026-07-12)**: MRVI itself is anti-concordant with PyDESeq2 (rho=−0.138, sign agree on sig genes=0.324). The eps-space DE limitation (L-079) applies to the reference MRVI implementation, not only MrTotalVI/MrMultiVI. F-020 biological narrative ("IFN activation → decline → W22 suppression") is **retracted** — the cross-model IFN-suppression convergence is a shared artifact. All eps-space models show near-zero IFN LFCs; the signal is absent, not merely inverted. Script: `.scratch/mr-schisto-benchmark/compare_mrvi_pydeseq2.py`.

**DA stability (F-034):**
- MrTotalVI_dtp: mean W22 enrichment = +1.12 ± **9.46** (s0=+2.74, s1=−9.04, s2=+9.67). Completely unstable; not usable.

**Conclusion**: DTP retraining fixes the design-matrix validity issue (required) and improves MrMultiVI scIB integration. It does NOT fix the fundamental eps/u partitioning limitation for treatment-level DE. For temporal DE, use PyDESeq2 pseudobulk (F-031). The model contribution is in integration quality and cell-type-level resolution, not temporal DE.

### Per-cell-type DE concordance (CPU, global model LFC vs stratified pseudobulk — F-037, 2026-07-12)

**Question**: does the eps-space DE limitation (L-079) vary by cell type? Are there any of the 12 cell types where global model LFC directionally agrees with cell-type-specific pseudobulk?

**Method**: compared pre-computed global DTP-model LFC (W22 contrast) against 12 cell-type-stratified pseudobulk t-test results (paired, 10 donors). Script: `.scratch/mr-schisto-benchmark/compare_per_celltype_pb.py`. Output: `results/per_celltype_pb_concordance.{tsv,json}`.

**Result — eps-space limitation is cell-type-universal:**

| Model | Spearman rho range | IFN pairs correct |
|-------|-------------------|------------------|
| MRVI | −0.098 to +0.027 | 15/84 (17.9%, below chance) |
| MrTotalVI_dtp | −0.126 to −0.008 (all negative) | 31/156 (19.9%, below chance) |
| MrMultiVI_dtp | −0.037 to +0.012 (near-zero) | 67/156 (42.9%, near-random) |

No cell type is rescued. Classical monocytes (strongest pseudobulk IFN signal) give MRVI rho=−0.098 and 0/7 IFN genes correct. MrMultiVI's best performers (Tcm/Naive cytotoxic T, Naive B cells: 8/13 IFN correct each) are still below random chance. All 12 × 3 Spearman rho values are near-zero or negative. **The eps-space DE limitation is not cell-type-specific** — it is universal across all cell types in this dataset.

**Track 2 (GPU per-cell-type MRVI DE)**: script `run_mrvi_de_per_celltype.py` + SLURM job `slurm/submit_mrvi_de_per_celltype.sh` are ready. Will run actual `model.differential_expression(adata=adata_ct)` per cell type. Main risk: cell types missing donors from full cohort of 38 will be skipped. **Not yet submitted** — Track 1 already closes the question architecturally.

---

## Publication Confidence Review — 2026-07-12

### Scope

Package-level readiness of MrTotalVI and MrMultiVI for a methods publication. No manuscript draft exists; review is against the package as a tool that others would use and cite.

### Positive confidence factors

| Item | Evidence | Confidence |
|------|----------|------------|
| MrMultiVI **batch-correction** win over MultiVI | +0.047 total scIB 3-seed (F-024, F-032); Δbatch +0.128±0.008, Δbio −0.006±0.010 ≈ flat (F-038) | ✅ High (batch only) |
| DA stability (MrMultiVI DTP) | mean +0.96 ± 0.13, 3-seed (F-035); 4.6× MRVI baseline | ✅ High |
| Test suite | 98/98 passing (session 45) | ✅ High |
| IFN artifact retracted, documented | F-029, F-031, F-036, F-037; F-020 PARTIALLY RETRACTED | ✅ Complete |
| Design matrix fix (DTP) | L-076–L-078; DTP scIB improves MrMultiVI | ✅ Done |
| ADRs and user guide updated | ADR-0005/0006; mr_multimodal.md | ✅ Done |

### Risk register (publication-blocking or significant)

**P0 — FIXED THIS SESSION:**
- `mr_multimodal.md` described `store_lfc=True` as a working feature with no disclosure that eps-space LFC is empirically anti-concordant with pseudobulk across all 12 cell types (F-037). A user running DE would receive numbers that contradict the biological ground truth with no warning. **Fixed**: empirical eps-space DE disclaimer + DA stability caveat added to v1 Limitations.
- FINDINGS_REGISTRY contained stale cross-references: F-023 status "pending sex-adjustment" (resolved by F-028 null result), F-021 pointed to retracted F-020/F-017 as evidence basis, F-026 labeled convergent IFN artifact as "most robust shared signal", F-017 status did not note IFN retraction. **Fixed**: all four entries updated.

**P1 — FIXED (session 49/50):**
1. ✅ **Batch vs bio breakdown now in `mr_multimodal.md`.** (Session 50) Added "Integration benchmarks (single dataset, batch-correction only)" block: total scIB +0.047 (3-seed); **Δbatch +0.128±0.008 dominant; Δbio = −0.006±0.010 (flat, within noise)**. Single-seed +0.009 estimate (F-015) did not replicate — corrected in manifest + docs. MultiVI baseline is single-seed; asymmetric error bars disclosed. F-038 registered.
2. ✅ **MrTotalVI DA instability caveat with concrete numbers now in docs.** (Session 49) Added to v1 Limitations: s0=+2.74, s1=−9.04, s2=+9.67 numeric range; contrasted with MrMultiVI stable (mean +0.96 ± 0.13).
3. ✅ **Single-dataset limitation now disclosed in docs.** (Session 50) Integration benchmarks block explicitly states validation is on one CITE-seq dataset (schisto, n=10 donors, 12 cell types); multi-dataset validation (macaque) pending.

**P1 — Still open:**
4. **kNN regression framing.** Integration uniformly hurts kNN label transfer (F-015). The manifest explanation ("kBET-skipped types disrupt local structure") is credible but asserted, not formally tested. A reviewer will ask for kBET-removed replication or cell-type-stratified kNN. Skeptic audit (F-038) confirms this is a genuine limitation, not explained away by the 6/29-small-type argument.

**Skeptic audit verdict (session 50, F-038):** No show-stoppers. Win is real and consistent. Required reframings for publication:
- "batch-correction win," not "integration win"
- Disclose asymmetric error bars (MultiVI baseline is single-seed)
- "MrMultiVI_u reaches parity with TotalVI within noise" (0.640 vs 0.639, not "ties")
- MrTotalVI does NOT beat TotalVI (0.634 vs 0.639)
- kNN regression reported as a genuine limitation alongside scIB Total

**P2 — Lower priority:**
5. **F-021 permutation null still claims "Biological evidence must rely on F-020/F-017"** — that sentence is stale. Status updated with warning, but the body text should be corrected when the findings registry is next formally revised.
6. **DE parity section (lines 109-116) reads as a success** — technically correct (code works) but juxtaposed with the DTP results section showing anti-concordance. The sequencing is confusing without a reader note distinguishing "API is implemented" from "biological validity."

### Overall confidence rating (updated session 50, post-F-038)

**Moderate-High with focused framing.** P0 and P1 items are addressed. The package supports a focused methods paper that frames:
- MrMultiVI as a **batch-correction tool** with a consistent +0.047 Total scIB improvement over MultiVI (Δbatch +0.128±0.008; bio conservation flat); reaching parity with TotalVI within noise
- eps-space DE as **not suitable for biological conclusions** without pseudobulk cross-validation (cross-validate with PyDESeq2/edgeR before drawing conclusions)
- MrTotalVI DA at high sample counts as unstable; MrMultiVI DA as the reliable pathway
- kNN label transfer regression as a genuine limitation (over-mixing possible)

**Remaining honest caveat**: validation is single-dataset (schisto, human only). Macaque replication pending.

**The IFN-suppression retraction is complete and well-documented.** No surviving claim of IFN suppression exists in user-facing docs.

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
