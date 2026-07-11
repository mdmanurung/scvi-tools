# ADR-0006: MrMultiVI — MrVI-style hierarchical donor latent space on MultiVI

**Status:** Accepted  
**Date:** 2026-07-08  
**Author:** @mdmanurung

---

## Context

MultiVI jointly models RNA, ATAC, and optional protein data in a single latent space, but treats
donor identity as a batch covariate to be removed — not a biological axis to be modelled.  MrVI
decomposes cell state into a *sample-unaware* base `u` and a *sample-aware* residual
`z = z_base + eps`, where `eps` is produced by an attention block over a per-donor embedding.
This ADR records the decision to graft that hierarchy onto MULTIVAE, yielding MrMultiVI.

---

## Decision

Subclass `MULTIVAE` (not copy-adapt) into `MrMultiVAE`, and subclass `MULTIVI` into `MrMultiVI`.
The reusable `EncoderUZ` + `AttentionBlock` components come from the pre-existing
`mrtotalvi/_components.py` (no duplication).

### Grafting point

MULTIVAE's `inference` method produces a mixed latent `z` from per-modality encoders via
`mix_modalities`.  That `z` becomes `u0` — the multimodal base.  `MrMultiVAE.inference`
(not `_regular_inference`, which does not exist on MULTIVAE) intercepts it and applies a
two-stage hierarchy:

1. **Sample-conditioned u-encoder** (`EncoderXU_MultiVI`): `u0` is fed into the same
   `fc → ConditionalNormalization → act → fc → ConditionalNormalization → act → (+sample_embed) →
   NormalDistOutputNN` pipeline as `EncoderXU_TotalVI` (no `log1p` since `u0` is already a
   latent, not raw counts).  Output: `qu = Normal(mu_u, σ_u)`, `u = qu.rsample()`.
   `qz_m` / `qz_v` are overridden with `qu.loc` / `qu.scale**2` so MULTIVAE's `loss()` computes
   `kl_u = KL(qu, N(0,1))` automatically.
2. **Donor residual** (`EncoderUZ`): `z_base, eps = qz(u, cf_sample)`; `z = z_base + eps`.

The decoder receives `z` with the same shape as before — **unchanged**.

### Key differences from MrTotalVI

| Axis | MrTotalVAE | MrMultiVAE |
|------|-----------|-----------|
| Override target | `_regular_inference` | `inference` (no underscore) |
| u-encoder input | `log1p([x_rna, x_prot])` (raw counts) | `u0` = MULTIVAE mixed latent (no log1p) |
| KL interception | `out["qz"] = qu` → TotalVI loss reads it | `out["qz_m"] = qu.loc`, `out["qz_v"] = qu.scale**2` |
| KL key in loss | `"kl_div_z"` | `"kl_divergence_z"` |
| `setup_*` method | `setup_anndata` | `setup_mudata` (full re-implementation) |

### Two-level KL

MULTIVAE's parent loss is still used for reconstruction and paired-modality penalties, but
`kl_divergence_z` is replaced with custom `kl_u + kl_z`. `kl_u` is computed against either a
learned mixture-of-Gaussians prior over `u` or an analytic Gaussian prior. `kl_z =
-log N(0, exp(pz_scale))(eps)` is included when `z_u_prior=True`; setting `z_u_prior=False`
omits the residual prior penalty.

### Configurable `u` dimensionality

`n_latent_u=None` preserves the original isomorphic setting and calls `EncoderUZ` with
`n_latent_u=None`, so `z_base = u`. When `n_latent_u != n_latent`, `EncoderXU_MultiVI` emits the
requested `u` dimension and `EncoderUZ` projects `u -> z`. The MULTIVAE decoders continue to
receive `z` with the stock `n_latent` dimension.

### u prior

`u_prior_mixture=True` registers learned `u_prior_logits`, `u_prior_means`, and
`u_prior_scales`, and computes Monte Carlo `KL(q(u) || p(u))`. If `labels_key` is registered and
`n_labels > 1`, the prior uses one component per label and biases the matching component logits by
`u_prior_label_weight`. Setting `u_prior_mixture=False` uses an analytic Gaussian KL instead.

### `setup_mudata` re-implementation

Can't call `super().setup_mudata()` and append a field after the fact — `register_manager` is
called at the end of the base implementation and freezes the field list.  Instead, the full
MULTIVI `setup_mudata` body is copied and the `SAMPLE_KEY` field (a
`MuDataCategoricalObsField` with `mod_key=None`) is inserted into the list before
`register_fields`.  `INDICES_KEY` is NOT re-added — it is already in MULTIVI's field list.

### `get_latent_representation` manual implementation

MULTIVI's `get_latent_representation(give_mean=True, modality="joint")` returns `qz_m` — the
mean of `u`, not `z`.  For `give_z=True, give_mean=True`, the method re-runs `EncoderUZ` with
`u_mean = out["qz_m"]` and returns `z_base(u_mean) + eps(u_mean, sample)`.  For
`give_z=False`, it returns `out["qz_m"]` (u_mean) directly.

---

## Conscious cuts (v1 scope)

- **Minified-mode inference**: off.  MULTIVAE minified mode is rare; deferred.
- **ArchesMixin new-donor surgery**: embedding table is fixed at `n_sample` at training time.
  Adding new donors post-training is out of scope.
- **ATAC differential expression is explicitly unsupported**: `differential_expression` rejects
  ATAC-containing models with `NotImplementedError`; ATAC effects should use a future
  `differential_accessibility` API instead of mixing semantics.
- **Decoded RNA/protein LFC**: `differential_expression` now supports `store_lfc=True` for
  RNA-only or RNA+protein bimodal models. Returns `lfc`, `lfc_std`, optional `pde` (when `delta`
  is provided), and optional `baseline_expression`. Feature coordinates are labelled `"gene"` /
  `"protein"` (D-022). ATAC-containing models still reject `store_lfc` with `NotImplementedError`.
  `use_vmap=False` is the default; LayerNorm in MULTIVAE makes `use_vmap=True` safe as an opt-in.

---

## Verification plan

Five tests in `tests/external/mrmultivi/test_mrmultivi.py`:

1. **Smoke**: trains with finite ELBO; counterfactual path (`cf_sample`) also finite.
2. **Reconstruction**: MrMultiVI recon loss ≤ MULTIVI × 1.05 at equal epochs.
3. **Latent repr**: `give_z=True` ≠ `give_z=False` — hierarchy not silently discarded.
4. **Local sample**: `get_local_sample_representation` / `get_local_sample_distances`
   return `xr.DataArray` with correct dims `(n_cell, n_donor, n_latent)` /
   `(n_cell, n_donor, n_donor)`.
5. **Hierarchy-collapse contrast** (L-6): zeroing `module.qz.embedding.weight` collapses
   cross-donor distances.  Trained distances must exceed zeroed baseline.
6. **Prior/statistical parity**: non-isomorphic `u`, label-conditioned MoG priors, analytic
   Gaussian prior fallback, `z_u_prior=False`, aggregated posterior, DA, outlier scoring, and
   explicit ATAC DE rejection.

---

## Consequences

- MrMultiVI adds ~350 LoC across `_module.py`, `_model.py`, and the `setup_mudata` re-impl.
- `EncoderUZ` is shared with MrTotalVI — no new component code.
- `scvi.external.MrMultiVI` and `MrMultiVAE` are exported from `src/scvi/external/__init__.py`.
- MultiVI's existing API is **unchanged** — MrMultiVI is a strict extension.
