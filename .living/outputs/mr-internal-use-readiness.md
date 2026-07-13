# MrTotalVI / MrMultiVI — Internal Scientific Use Readiness Review

**Date**: 2026-07-13  
**Reviewer**: Multi-agent adversarial audit (V1–V5 verifiers + inline code review)  
**Scope**: Internal scientific use (single-dataset evidence acceptable; not publication readiness)  
**Discriminating question**: *If an internal scientist runs this method on their own data, will they get a correct answer or be misled?*

> This review supersedes the "Publication Confidence Review — 2026-07-12" in `benchmarks/ANALYSIS_MANIFEST.md` for the **internal use** framing. Cross-reference that document for the publication framing (Moderate-High with focused framing).

---

## Capability Matrix

| Capability | MrTotalVI | MrMultiVI | Guardrail / Notes |
|---|---|---|---|
| **Latent integration (u-space)** | Ready-with-guardrail | Ready-with-guardrail | Batch-correction only (F-038: Δbatch +0.128, Δbio ≈ 0). Don't claim biological variation is preserved. |
| **Latent integration (z-space = u + donor eps)** | Ready | Ready | `give_z=True` in `get_latent_representation`; donor-drop base-class bug fixed (V5-006). |
| **Differential abundance (DA)** | **Not ready** for high-sample studies | Ready-with-guardrail | MrTotalVI DA: std=9.46 across 3 seeds, mean=1.12 — catastrophic instability (F-034). MrMultiVI DA: std=0.126, mean=0.956 — stable (F-035). Use MrMultiVI as the DA path. |
| **Differential expression (eps-space, store_lfc=True)** | Not ready for bio conclusions | Not ready for bio conclusions | Anti-concordant with pseudobulk ground truth, all 12 cell types (Spearman ρ −0.24 to +0.04; L-079, F-029/F-031/F-036/F-037). Architectural: eps absorbs batch, not biology. Cross-validate with PyDESeq2/edgeR before trusting any specific gene. |
| **DE sign/directionality (store_lfc=True)** | Ready for sanity-check use | Ready for sanity-check use | Positive control confirmed (V4-006): 20× inflation → mean_lfc > 0. CRN fix reduces lfc_std (V2-001/L-082). Use for directional sanity checks, not biology conclusions. |
| **Protein differential expression (MrTotalVI)** | Ready-with-guardrail | Ready-with-guardrail | Deterministic protein background reconstruction correct (D-021/V2-002). Same eps-space caveats apply. |
| **Counterfactual / local sample distances** | Ready | Ready | `get_outlier_cell_sample_pairs` confirmed (V2-001 CRN). `get_aggregated_posterior` returns correct MixtureSameFamily. |
| **Save / load roundtrip** | Ready | Ready | Custom hyperparams correctly serialized via `init_params_` (V5-004). MrMultiVI requires MuData at load, not AnnData (V5-001 — see guardrail). |
| **ATAC differential accessibility** | N/A | Not ready (stub) | `differential_accessibility` raises NotImplementedError (V5-003). Documented in user guide. |
| **ATAC + DE** | N/A | Not ready | `differential_expression` raises NotImplementedError when n_regions > 0 (V5-002). Documented. |

---

## Verified-Correct Core Mechanisms

The following were confirmed by direct code audit (V1/V2 verifiers):

- **Jensen-gap correction**: `logsumexp(log_probs_k) - log(K)` at `_stats.py:197` — correct (V1-001).
- **BH FDR**: flatten → BH → reshape at `_stats.py:766` — correct (V1-002).
- **WLS estimator**: `(X^T W X)^{-1} X^T W` with per-cell admissibility weighting at `_stats.py:601-619` — correct (V1-003).
- **Two-level KL (kl_u + kl_z)**: Both KL terms in the ELBO. `kl_div_z` in `_loss_with_mc_samples()` is a placeholder that `loss()` **replaces** with the correct `kl_u_weight*kl_u + kl_z_weight*kl_z` (V1-005 refuted "kl_u only" hypothesis).
- **CRN u_anchor**: null decode and contrast decode share the same u sample per MC draw; lfc_std anti-conservatism is fixed (V2-001, L-082).
- **Deterministic protein background (D-021)**: `rate_back_det = exp(back_alpha)` in both modules — correct (V2-002).
- **get_latent_representation donor-drop fix**: `give_z=True` correctly returns `z_base + eps` including donor residual (V5-006 confirmed fix).

### Chi2 df design note (V1-004)

The Wald statistic Chi2 df uses `n_per_cell` (admissible samples per cell), matching MRVI reference (`mrvi_torch/_model.py:1372`). The V1 verifier flagged this as non-standard (textbook would use n_latent). This is an intentional design choice from the MRVI paper — in practice n_per_cell >> n_latent, making p-values conservative (fewer false positives). **Not a bug; note in analysis documentation.**

---

## Adversarially Tested Claims

| Claim | Status | Evidence |
|---|---|---|
| D-041: VampPrior+frozen → 18.7% DA variance reduction | **REFUTED** | `outputs/validate_freeze_prior/variance_comparison.json` does not exist. `outputs/` directory never created. Numbers in decisions.md have no artifact backing. |
| F-034: MrTotalVI DA catastrophically unstable (std=9.46) | **CONFIRMED** | `.scratch/mr-schisto-benchmark/results/da_mrtotalvi_dtp_summary.json` — s0=+2.74, s1=−9.04, s2=+9.67 |
| F-035: MrMultiVI DA stable (std=0.126, mean=0.956) | **CONFIRMED** | `.scratch/mr-schisto-benchmark/results/da_mrmultivi_dtp_summary.json` |
| eps-space DE IFN narrative | **RETRACTED** | F-017/020/026/027 retracted; L-079 anti-concordance across all 12 cell types |
| lfc_std CRN fix | **CONFIRMED** | Code audit: u_samples shared across null/contrast decode in `_stats.py:583-731` (V2-001) |

---

## P0 / P1 / P2 Backlog

See `todo/TODO_REGISTRY.md` for the actionable items. Summary:

### P0 — Must fix before any internal use

| ID | Item | Status |
|---|---|---|
| P0-001 | `mrtotalvi/_model.py:192` spurious `.to_numpy()` on numpy array in VampPrior init | ✅ FIXED (this session) |
| P0-002 | `pyproject.toml` missing `pythonpath = ["src"]` — tests fail without manual PYTHONPATH | ✅ FIXED (this session) |
| P0-003 | `test_vamprior_has_correct_parameters` stale assertion (pre-protein_in_encoder default change) | ✅ FIXED (this session) |
| P0-004 | `test_mrmultivi_encode_covariates_expands_qu_input` stale assertion (same) | ✅ FIXED (this session) |
| P0-005 | DA tested only on untrained models in CI (V4-001): no trained-model DA correctness path | ✅ FIXED (this session) — `test_differential_abundance_trained_model_smoke` added to both test files |

### P1 — Fix before regular reliance on results

| ID | Item | Status |
|---|---|---|
| P1-001 | DA multi-seed calibration: no test asserts DA stability across seeds in CI (V4-002) | ✅ FIXED — `test_mrmultivi_da_multiseed_stability` added (2026-07-13); see L-092 for synthetic-data limitation |
| P1-002 | `n_labels==0` / unlabeled_category path untested (V4-003) | ✅ FIXED — `test_mrtotalvi_n_labels_zero_mog_prior_smoke` and `test_mrmultivi_n_labels_zero_mog_prior_smoke` added |
| P1-003 | Statistical tests lack fixed seeds → flakiness risk in `test_donor_axis_separation`, `test_mrtotalvi_lfc_sign_known_positive_control`, etc. (V4-004) | ✅ FIXED — explicit `scvi.settings.seed = 0` added to all three statistical-assert tests |
| P1-004 | Backward-compat guard untested: new flags at defaults → latents match pre-change baseline (V4-005) | ✅ FIXED — `test_mrtotalvi_default_latent_is_deterministic` and `test_mrmultivi_default_latent_is_deterministic` added |
| P1-005 | MrMultiVI load() error message for AnnData is confusing (V5-001): says "provide n_genes/n_regions" instead of "use MuData" | ✅ FIXED — `isinstance(mdata, MuData)` guard added before `super().__init__` in `MrMultiVI.__init__`; `test_mrmultivi_mudata_guard_raises_typeerror` verifies it |
| P1-006 | MrTotalVI DA instability root-cause investigation (why std=9.46 vs MrMultiVI std=0.126) | ⏸ OPEN — compute/data-blocked (requires GPU + real schistosomiasis data) |

### P2 — Improvement / documentation

| ID | Item | Status |
|---|---|---|
| P2-001 | Add per-seed DA numbers (s0/s1/s2) to `mr_multimodal.md` (V2-004 — range present, breakdown absent) | ✅ FIXED — per-seed breakdown added to DA block |
| P2-002 | Document Chi2 df = n_per_cell (admissible samples, not n_latent) in analysis notes (V1-004) | ✅ FIXED — added to Objective (methods note) block |
| P2-003 | No-intercept design matrix: note in docs that eps pre-centering absorbs the intercept (V1-006) | ✅ FIXED — added to Objective (methods note) block |
| P2-004 | `differential_accessibility` stub: add explicit mention in user guide that `differential_accessibility` is a named method that raises `NotImplementedError` (V5-003) | ✅ FIXED — wording tightened in LFC section |
| P2-005 | Multi-dataset validation (macaque CITE-seq replication) — pending (V2-006) | ⏸ OPEN — compute/data-blocked (requires macaque dataset + training) |

---

## Documentation Status

All critical caveats ARE present in `docs/user_guide/models/mr_multimodal.md`:
- eps-space DE anti-concordance (L120-130) ✅
- MrTotalVI DA instability range −9.0 to +9.7 (L133-138) ✅  
- Batch-correction-only integration (L140-150) ✅  
- Single-dataset validation disclosure (L148-150) ✅  
- ATAC NotImplementedError (L106-107, L117-118) ✅

Gap: per-seed breakdown (s0/s1/s2) not in docs (V2-004 — P2-001).

---

## Reconciliation with Publication Review (2026-07-12)

The publication review rated overall confidence **Moderate-High with focused framing** for:
- MrMultiVI DA (stable, schisto-validated)
- eps-space DE directional sign only (with strong caveats)
- Integration as batch-correction

This internal-use review **agrees** with those verdicts and adds:
1. MrTotalVI DA is **Not Ready** even for internal use at high sample counts — route to MrMultiVI.
2. The eps-space DE caveats are even more important for internal users who may not read the limitations section.
3. Five P0 test/infra fixes were needed and are now resolved.

**Verdict: MrMultiVI is ready for careful internal use** with the above guardrails. **MrTotalVI** is ready for integration and DE use but **not DA** at high sample counts.

---

*Evidence files: `.scratch/mr-schisto-benchmark/results/da_{mrtotalvi,mrmultivi}_dtp_summary.json`*  
*Tests run: 112 passing (P0 session) + 5 new P1 tests (P1-001: 1, P1-002: 2, P1-004: 2) + 2 new code-review tests (S61: guard + interaction) = 119 target total as of 2026-07-13*
