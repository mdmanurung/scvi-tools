# MrTotalVI — grafting MrVI's hierarchical donor latent onto TotalVI

## Status

Accepted.

## Context

TotalVI models multimodal gene + protein expression data with a flat `N(0,1)` prior on a shared
latent `z`. Donor/sample identity is injected only as a nuisance covariate (batch cat_list), not
as a modeled axis. This means TotalVI cannot ask "what would cell *i* look like in donor *d*?" — a
capability MrVI provides for single-modality RNA.

MrVI's approach: decompose the latent space into a sample-*conditioned* `u` (from a dedicated
`EncoderXU` that applies per-sample normalization) and a sample-*aware* residual
`z = z_base(u) + eps`, where `eps` is computed by attending over a per-sample embedding table via
`EncoderUZ`. This additive hierarchy is the load-bearing design: because `u` and `z` live in the
same Euclidean space, the decoder is unchanged, and counterfactual donor queries reduce to
substituting a different donor index into the attention block.

## Decision

Graft MrVI's `u→z` hierarchy onto TotalVI as `MrTotalVI` (placed in `scvi.external`):

### Key design choices

**Sample-conditioned u-encoder (`EncoderXU_TotalVI`) — matches MrVI's `EncoderXU`.**
`MrTotalVI` ports MrVI's sample-conditioned encoder to accept concatenated RNA + protein counts.
Architecture: `log1p([x_rna, x_prot]) → fc1 → ConditionalNormalization(sample) → act → fc2 →
ConditionalNormalization(sample) → act → (+sample_embed) → NormalDistOutputNN → Normal(mu_u, σ_u)`.
Sample identity is woven into `u` itself via per-donor `gamma`/`beta` embeddings — not just the
`eps` residual. This means the two-level hierarchy carries sample information at both levels.

**Configurable `u` dimensionality.**
`n_latent_u=None` preserves the original isomorphic setting (`n_latent_u == n_latent`,
`EncoderUZ.fc is None`, `z_base == u`). When `n_latent_u != n_latent`, `EncoderXU_TotalVI`
emits the requested `u` dimension and `EncoderUZ` projects `u -> z` before the unchanged TotalVI
decoder. This keeps the decoder contract stable while allowing non-isomorphic MrVI-style
hierarchies.

**`latent_distribution` forced to `"normal"`.**
Under `"ln"` (softmax normalisation), `u` is simplex-constrained. An additive Euclidean residual on
a simplex is mathematically invalid. The code asserts loudly at model construction time.

**`use_map=True` in `EncoderUZ`.**
Emits one deterministic `eps` per (u, sample) pair. The correct KL term is then
`kl_z = −log p(eps) = −log N(0, exp(pz_scale))(eps)`, not a full encoder KL. This matches MrVI's
loss formulation.

**Two-level KL loss with MrVI parity.**
- `kl_u` is computed explicitly against either a learned mixture-of-Gaussians prior over `u` or an
  analytic Gaussian prior.
- If `labels_key` is registered and `n_labels > 1`, the MoG prior uses one component per label and
  biases the matching component logits by `u_prior_label_weight`.
- `kl_z = -log p(eps)` is added when `z_u_prior=True`; `z_u_prior=False` drops the residual prior
  penalty while retaining `kl_u`.
- The custom `kl_u + kl_z` replaces TotalVI's parent `kl_div_z` value inside
  `kl_local["kl_div_z"]`.

**Covariate-aware `qu`.**
When `encode_covariates=True`, `EncoderXU_TotalVI` appends one-hot batch/categorical covariates and
continuous covariates to the RNA+protein encoder input before its first linear layer. The donor
sample axis remains separate and continues to enter through conditional normalization and the
`EncoderUZ` embedding table.

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
- **No minified-mode inference** — the hierarchy requires the full encoder path.
- **Decoded RNA/protein LFC storage is not implemented yet** — `differential_expression` returns
  latent-space `beta`, `effect_size`, and `pvalue` from the local counterfactual linear model, but
  does not yet store decoded RNA/protein LFC arrays.

### MultiVI path (implemented)
`EncoderUZ`, `ConditionalNormalization`, and `NormalDistOutputNN` are factored into
`mrtotalvi/_components.py` so MultiVI can reuse them. `EncoderXU_MultiVI` (no `log1p`, single
latent input) is grafted into `MrMultiVAE` at the same level — see ADR-0006.

## Consequences

**Positive:**
- All TotalVI functionality (protein background prior, dispersion, likelihood, batch correction)
  is inherited unchanged — zero decoder modifications.
- Counterfactual donor queries (`get_local_sample_representation`, `get_local_sample_distances`)
  work by re-running `EncoderUZ` with a substituted donor index, a cheap O(n_cells × n_donors)
  loop over the attention block.
- `get_aggregated_posterior`, `differential_abundance`, and
  `get_outlier_cell_sample_pairs` operate over `u` using TorchMRVI-style aggregated posteriors.
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
