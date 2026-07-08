# MrTotalVI — grafting MrVI's hierarchical donor latent onto TotalVI

## Status

Accepted.

## Context

TotalVI models multimodal gene + protein expression data with a flat `N(0,1)` prior on a shared
latent `z`. Donor/sample identity is injected only as a nuisance covariate (batch cat_list), not
as a modeled axis. This means TotalVI cannot ask "what would cell *i* look like in donor *d*?" — a
capability MrVI provides for single-modality RNA.

MrVI's approach: decompose the latent space into a sample-*unaware* `u` (the encoder output) and a
sample-*aware* `z = z_base(u) + eps`, where `eps` is computed by attending over a per-sample
embedding table via `EncoderUZ`. This additive hierarchy is the load-bearing design: because `u` and
`z` live in the same Euclidean space, the decoder is unchanged, and counterfactual donor queries
reduce to substituting a different donor index into the attention block.

## Decision

Graft MrVI's `u→z` hierarchy onto TotalVI as `MrTotalVI` (placed in `scvi.external`):

### Key design choices

**TotalVI's encoder output becomes `u` verbatim.**
Unlike MrVI, which has a dedicated `EncoderXU` conditioned on sample, `MrTotalVI` uses TotalVI's
existing `EncoderTOTALVI` without modification. All sample-specific signal is isolated in `eps`. This
is a conscious choice (not a porting bug): the u-encoder is deliberately sample-unaware so the
sample axis lives entirely in the residual.

**Isomorphic dims (`n_latent_u == n_latent`).**
`EncoderUZ.fc is None` and `z_base == u`. This means the decoder input dimension is identical to
stock TotalVI — zero decoder changes are needed. The hierarchy adds capacity without reshaping any
existing weight matrices.

**`latent_distribution` forced to `"normal"`.**
Under `"ln"` (softmax normalisation), `u` is simplex-constrained. An additive Euclidean residual on
a simplex is mathematically invalid. The code asserts loudly at model construction time.

**`use_map=True` in `EncoderUZ`.**
Emits one deterministic `eps` per (u, sample) pair. The correct KL term is then
`kl_z = −log p(eps) = −log N(0, exp(pz_scale))(eps)`, not a full encoder KL. This matches MrVI's
loss formulation.

**Two-level KL loss.**
- `kl_u = KL(q_u ‖ N(0,1))` — TotalVI's existing `kl_div_z` line, unchanged.
- `kl_z = −log p(eps)` — new second-level term.
- Both fold into the existing `kl_local["kl_div_z"]` slot for compatibility with the training loop.

**Three categorical axes kept distinct.**
- `sample_key` → `EncoderUZ` embedding table (new).
- `batch_key` → decoder `cat_list` (existing).
- `panel_key` → protein background prior (existing).
Conflating any two of these would corrupt the biological interpretation.

**`_setup_hierarchy` deferred pattern.**
`MrTotalVAE.__init__` accepts `n_sample=0` as a placeholder. `MrTotalVI.__init__` calls
`super().__init__()` first (which populates `self.summary_stats`), then calls
`self.module._setup_hierarchy(self.summary_stats.n_sample, ...)`. This avoids needing to
pre-compute `n_sample` before the registry is built.

**`scvi.external`, not a top-level package.**
v1 is an extension, not an independent framework. The placement mirrors `mrvi_torch` (TorchMRVI)
which follows the same pattern.

### Explicit cuts (v1)

- **No ArchesMixin** — the per-sample embedding is fixed at `n_sample`. New-donor surgery would
  require embedding surgery or a projection module (future work).
- **No `latent_distribution="ln"`** — architecturally invalid; asserted rather than silently ignored.
- **No `differential_expression` / `differential_abundance`** — deferred to v2 after the
  counterfactual API is validated.
- **No minified-mode inference** — the hierarchy requires the full encoder path.
- **u prior = `N(0,1)` (MoG/label-conditioned deferred)** — MrVI supports a mixture-of-Gaussians
  prior conditioned on cell type labels; this is not ported in v1.

### MultiVI path (future)
`EncoderUZ` is factored into `mrtotalvi/_components.py` (not `_module.py`) so MultiVI can reuse
it. The insertion point for MultiVI is after `mix_modalities` in `_multivae.py`, where the mixed
`qz_m` becomes `u`.

## Consequences

**Positive:**
- All TotalVI functionality (protein background prior, dispersion, likelihood, batch correction)
  is inherited unchanged — zero decoder modifications.
- Counterfactual donor queries (`get_local_sample_representation`, `get_local_sample_distances`)
  work by re-running `EncoderUZ` with a substituted donor index, a cheap O(n_cells × n_donors)
  loop over the attention block.
- Five deepening-invariance tests verify: finite ELBO + `give_z` wiring check, fragile counterfactual
  path (cf_sample + mc_samples), non-degenerate embedding, and donor-axis separation with
  before/after contrast (zeroed-embedding collapse).

**Negative / risks:**
- Shape reconciliation under `(mc_samples, batch, n_latent)` leading dimension is the highest
  runtime risk; mitigated by the mc_samples=2 smoke test.
- The embedding table is not transferable across datasets with different donor sets (no
  ArchesMixin support).
- `kl_z` is a fixed-prior log-probability, not a full distributional KL — users must understand
  `pz_scale` controls the regularisation strength on the donor residual.
