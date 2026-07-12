# Findings: DA/DE Parity — MRVI vs MrTotalVI / MrMultiVI

**Date:** 2026-07-12 (updated session-42)
**Status:** verified; CRN fix implemented — LFC is now an unbiased posterior-marginalized estimator
**Author:** session-24 (plan), session-25 (LFC gap closed), session-41 (estimand difference documented), session-42 (CRN fix implemented)

---

## Summary

MrTotalVI and MrMultiVI expose the same `differential_abundance` (DA) and
`differential_expression` (DE) APIs as the MRVI reference. Both DA and the
latent-space DE core reach **full algorithmic parity** with MRVI, and add a
`donor_key` superset feature not present in MRVI. The only gap is the
gene/protein-space LFC path in DE, which is an **intentional v1 cut** documented
in ADR-0005/0006 and `mr_multimodal.md` — it is now the active remediation target.

---

## Parity Matrix

| Area | Verdict | Notes |
|------|---------|-------|
| DA algorithm | **Full parity + superset** | All MRVI DA features present; `donor_key` added |
| DE latent-space WLS | **Equivalent (modulo `u` semantics)** | Same estimator, structurally equivalent code |
| DE gene/protein LFC | **Fully implemented** | `compute_h_from_x_eps` in both modules; `store_lfc`, `lfc_std`, `pde`, `store_baseline` all available; batch-marginalized |
| MrMultiVI ATAC DE | **Documented stub** | Unconditional stub; separate DA recommended |

---

## 1. Differential Abundance — Full Parity + Superset

**MRVI reference:** `mrvi_torch/_model.py:850-1034`  
**Mr* shared engine:** `mrtotalvi/_stats.py:86-219`

### Algorithm correspondence

Both implement u-space aggregated-posterior DA:
1. Sample `u` from `q(u|x)` per cell per MC draw.
2. Evaluate `log_prob(u; sample_posterior_k)` under each sample's fitted Normal.
3. Logsumexp over samples → per-cell assignment probability.
4. Design matrix + OLS → per-covariate DA statistics.
5. BH FDR.

**All load-bearing MRVI DA features confirmed present in `_stats.py`:**
- `omit_original_sample` — param `:94`, applied `:170-177`: omits the index-sample
  from the denominator logsumexp to avoid self-assignment bias.
- `compute_log_enrichment` — param `:93`, applied `:192,:213`: returns per-cell
  log-enrichment over the null mixture rather than just the log probability.
- Logsumexp aggregation — `:176-177`.

**Superset feature:** `donor_key` within-donor centering of log-probs (`_stats.py:133-147`).
Not present in MRVI; used to control for donor-level effects before fitting the
sample-covariate design matrix.

**Implementation difference (not a defect):** MRVI reads `self.sample_info` (via
`update_sample_info`); `_stats.py` builds `sample_info` inline from `adata.obs`
(`:166`). Functionally equivalent.

**Inherent caveat:** `u` has different semantics per base model — MULTIVAE uses a
mixed-modality posterior (`mrmultivi/_module.py:257`) while TOTALVI uses a
RNA+protein encoder (`mrtotalvi/_module.py`). Same algorithm, different latent
geometry. Results are not numerically comparable across models but the algorithm is
identical.

---

## 2. DE Latent-Space WLS — Equivalent Computation (Modulo `u` Semantics)

**MRVI reference:** `mrvi_torch/_model.py:1146-1585`  
**Mr* shared engine:** `mrtotalvi/_stats.py:300-546`

### Estimator correspondence

Not byte-identical code, but the same estimator:

| Step | MRVI line | `_stats.py` line | Notes |
|------|-----------|------------------|-------|
| eps standardization | `:1360-1363` | `:498-501` | mean/std over sample dim |
| WLS via Amat | `:1365-1368` | `:505-508` | `einsum(Amat, eps_norm)` |
| sqrtm Wald prefactor | `:1369-1370` | `:511-512` | `einsum(prefactor, betas)` |
| Chi² test statistic | `:1371` | `:514` `ts = (betas_norm**2).sum(-1).mean(0)` | **square-then-mean-over-MC** — applied per draw, then averaged |
| `df = clamp(n_admissible, min=1)` | `:1372` | `:517` | — |
| BH FDR | `:1576` | `:542` | `false_discovery_control(..., method="bh")` |

**Corrected note on MC collapse ordering:** `betas_rescaled = (betas*eps_std).mean(0)`
at `:521` collapses the MC axis only for the reported `beta` coefficient (effect
direction, not the p-value). The chi² test statistic at `:514` already applies
square-then-mean-over-MC per draw. There is **no early-MC-collapse defect** in the
test statistic — the p-value computation is correct.

**Superset features** (not in MRVI):
- `donor_key` nuisance dummies: extra columns added to design matrix; only
  `n_fixed` betas reported upstream.
- `lambd` ridge regularization: added to the WLS diagonal.

**Implementation difference (equivalent):** mr* loops `qz(u, cf_sample)` per
donor to collect eps (`_stats.py:460-471`); MRVI extracts eps from
`inference(cf_sample)` for all samples at once (`_model.py:1335-1358`). The `qz`
call in both mr modules returns a **3-tuple** `(z_base, eps, eps_dist)` — not
MRVI's 2-tuple — reflecting the richer mr* latent hierarchy.

---

## 3. DE Gene/Protein-Space LFC — Fully Implemented

**MRVI reference:** `mrvi_torch/_module.py:~691-714` (`compute_h_from_x_eps`),
consumed by `mrvi_torch/_model.py:1378-1454`.

**NOTE:** The session-24 version of this document described the LFC path as a
"documented gap." It has since been implemented (committed in feat(mrtotalvi,mrmultivi)
working-tree commit). The description below reflects the current implemented state.

### Current state in `_stats.py` and module files

- `store_lfc=True` guard at `:384-388`: raises `NotImplementedError` **only if**
  `model.module.compute_h_from_x_eps` is absent. Both modules implement it.
- `delta`, `store_baseline`, `eps_lfc`, `lfc_std`, `pde` are all fully wired
  (`_stats.py:560-687`).
- Batch marginalization over unique batch values in each mini-batch (`:590-643`).
- `compute_h_from_x_eps` in `mrtotalvi/_module.py:618-736`;
  `compute_h_from_x_eps` in `mrmultivi/_module.py:626+`.

### MRVI's reference is RNA-only; protein path is novel design

`mrvi_torch/_module.py:~691-714` computes `h = px.mean / library_exp` — RNA only.
**There is no MRVI reference for protein LFC.** The implementation therefore has two parts:
1. **RNA path** — ported from MRVI: `px_["scale"]` contrast, log2 fold-change.
2. **Protein path** — novel design; resolved the "TotalVI decoder contrast
   semantics not finalized" question from ADR-0005.

### Design decisions resolved (D-020–D-024)

- **D-020** — protein contrast: `py_["scale"]` (L1-normalized foreground, per
  `DecoderTOTALVI` docstring; `_base_components.py:962`). ✅ implemented.
- **D-021** — deterministic protein background: `py_["scale"]` is stochastic via
  `rate_back = exp(Normal(back_alpha, back_beta).rsample())`. Independent background
  draws in x_1/x_0 don't cancel → LFC contaminated by background noise unless fixed.
  Implementation uses `rate_back_det = exp(back_alpha)` (log-mean) to reconstruct
  `py_scale_det` on the contrast path, leaving ELBO training path untouched
  (`_module.py:728-736`). ✅ implemented.
- **D-022** — feature layout: `compute_h_from_x_eps` returns
  `concat(px_scale, py_scale_det)` (`_module.py:736`). ✅ implemented.
- **D-023** — RNA scale: softmax `scale` (MRVI-h analog). ✅ implemented.
- **D-024** — vmap policy: `use_vmap` param in `_stats.py` is currently ignored;
  the LFC path uses explicit Python loops (safe with BatchNorm decoders). ✅ implemented.

### Estimand: posterior-marginalized LFC via CRN (implemented 2026-07-12)

Mr* now uses **Common Random Numbers (CRN)** to estimate the posterior-marginalized LFC:

> E_u[log2 dec(u + β_k)] − E_u[log2 dec(u + β_null)]

**Implementation:** `_stats.py` stores `u_samples` (one sampled u per MC draw) alongside `eps_batch`. In the LFC block, both x_0 and x_1 for MC draw `mc_idx` receive `u_anchor=u_samples[mc_idx]`. x_0 is now computed inside the MC loop (not once outside). Both `compute_h_from_x_eps` hooks accept `u_anchor: torch.Tensor | None` — when provided, it overrides `qu.mean` as the latent anchor for `qz(u, cf_sample)`.

**Correctness proof:** Because `EncoderUZ` with `use_map=True` (default) is deterministic given u, `qz(u_anchor, cf)` → `(u_anchor, eps_det, None)` — so `z_base = u_anchor` exactly. When `extra_eps` for x_0 and x_1 are both equal to `eps_mean_cell`, the two decode calls are bit-identical → LFC == 0 exactly (verified by `test_mrtotalvi_crn_identity` / `test_mrmultivi_crn_identity`).

**Variance:** `lfc_mc_cov.var(1)` now captures both β_k regression uncertainty (across MC) and u-posterior uncertainty (u_anchor varies per draw). `lfc_std` is no longer anti-conservative.

**Legacy path:** `u_anchor=None` → `qu.mean` (Jensen-biased) is retained for standalone callers (D-021 determinism tests). `_stats.py` always provides `u_anchor`. See D-034.

**Comparison to MRVI reference:** MRVI uses independent u draws in separate vmap slices (unbiased but high-variance, Cov=0). Mr* CRN is lower-variance and unbiased — a genuine improvement over both the original `qu.mean` path and MRVI's independent-draws approach.

---

## 4. MrMultiVI ATAC — Documented Stub

`mrmultivi/_model.py:482-487` rejects ATAC-containing models for DE; `:505-510`
is an unconditional `NotImplementedError` for `differential_accessibility`. This
is documented behavior — ATAC effects use separate DA over `u` rather than being
mixed into RNA/protein DE semantics.

---

## Empirical benchmark status (as of 2026-07-11)

Empirical cross-model comparison is blocked at two levels:
1. **MRVI checkpoint incompatibility**: Checkpoint trained with old `MultiheadAttention`
   API (`embed_dim_proj_query`, `attention.in_proj_weight`) and `mlp_eps` hidden dim 128;
   current `TorchMRVAE` uses `q_proj`/`k_proj`/`v_proj` and hidden dim 64.
   Mismatch is architectural — not loadable with current code. MRVI skipped gracefully
   in benchmark script.
2. **GPU compatibility**: SLURM `gpu` partition assigned TITAN Xp (sm_61); current
   PyTorch build requires sm_75+. Fix: resubmit with `--partition=gpu-long --gres=gpu:L40S:1`.
   Job resubmitted (pending).

Consequence: empirical Spearman(LFC_MRVI, LFC_MrTotalVI) is permanently unavailable
until MRVI is retrained on current architecture. MrTotalVI standalone LFC output
(RNA + protein) can be evaluated once GPU issue is resolved.
