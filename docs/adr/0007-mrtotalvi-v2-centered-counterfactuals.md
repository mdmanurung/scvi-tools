# ADR-0007: Opt-in centered MrTotalVI counterfactuals

## Status

Accepted as a package contract on 2026-07-25.

Scientific validation is explicitly excluded. This ADR does not resolve the human/macaque data
lineage blocker, establish a preferred model, or authorize biological or publication claims.

## Context

ADR-0005 introduced the legacy MrTotalVI hierarchy

`z = z_base(u) + eps(u, observed_sample)`.

The legacy hierarchy is useful but leaves a common residual shift unconstrained in its reported
sample-specific decomposition. It also conditions `u` on the biological sample through
conditional-normalization embeddings and an explicit sample embedding. A package-level v2 needs an
opt-in, explicitly centered decomposition, a sample-blind encoder option, named deterministic
decoder estimands, bounded materialization, and checkpoint semantics that cannot silently change
old models.

The frozen scientific execution packet remains historical. The package-development amendment under
`.scratch/mrtotalvi-v2/` authorizes only synthetic package development and one bounded,
non-biological engineering fixture.

## Decision

### Modes and checkpoint semantics

`MrTotalVI` adds:

```python
hierarchy_mode: Literal["legacy", "centered_v2"] = "legacy"
u_encoder_mode: Literal["sample_conditioned", "sample_blind"] = "sample_conditioned"
```

- `legacy` remains the default and retains its existing state-dict keys, shapes, and numerics.
- `centered_v2` requires `use_map=True` and `z_u_prior=True`.
- Either encoder mode is legal with either hierarchy mode.
- Modes are plain attributes and constructor metadata in `init_params_`; they are not parameters or
  persistent buffers.
- A checkpoint without mode metadata resolves only to `legacy` and `sample_conditioned`.
- Unknown stored or requested values fail.
- Modes are never inferred from tensor values, parameter shapes, or state-dict keys.
- `EncoderUZ.forward()` remains unchanged because MrMultiVI shares it.

`MrTotalVI.load()` accepts optional hierarchy and encoder overrides. A differing override is
refused unless `allow_semantic_override=True`. An allowed change emits a warning, records loaded
and resolved modes, and updates `init_params_` so a subsequent save writes the resolved semantics.

### Centered hierarchy and training penalty

For every cell or posterior draw, the module evaluates raw residuals for the full registered sample
universe in registry order:

```text
eps_raw[s]      = qz_raw(u, s)
eps_centered[s] = eps_raw[s] - mean_t(eps_raw[t])
z[s]            = z_base + eps_centered[s]
```

The factual likelihood gathers the centered residual for the observed registered sample. Centering
never uses only requested targets, minibatch samples, or observed samples.

The residual penalty is evaluated before the factual gather:

```text
mean_registered_sample(
    sum_latent(-log Normal(eps_raw; 0, exp(pz_scale)))
)
```

`kl_z_weight` and global KL annealing apply after the equal-sample mean. With one registered sample,
`eps_centered` is exactly zero while the raw residual penalty remains active.

Target chunking may change memory use but not values or gradients. Every registered residual
embedding row must receive finite, nonzero gradients even when its sample is absent from the
factual minibatch.

For `scale_observations=True`:

- legacy retains its exact `1 / n_s` behavior;
- centered v2 uses `N / (S * n_s)`, whose full-data mean is the mean of per-sample mean losses.

### Sample-blind encoder

`sample_blind` retains all modules and parameter names but bypasses:

- both conditional-normalization gamma/beta embedding lookups; and
- the explicit `sample_embed`.

It retains normalization layers, linear layers, activations, output heads, and all explicitly
registered technical covariates. Unused conditioning parameters remain in checkpoints and may have
absent gradients. This mode changes only `EncoderXU_TotalVI`; MrMultiVI behavior is frozen.

### Counterfactual latent dataset

`get_counterfactual_latent()` is available only for `centered_v2` and returns:

- `u(draw, cell_name, latent_u_dim)`
- `z_base(draw, cell_name, latent_dim)`
- `eps_raw`, `eps_centered`, `z(draw, cell_name, target_sample, latent_dim)`
- `admissible`, `target_support(cell_name, target_sample)`
- `observed_sample(cell_name)`

`target_support` means that a target is registered and retains at least one reference cell after
factual self-exclusion. `admissible` is the separate aggregated-posterior threshold result.

### Counterfactual expression dataset

`get_counterfactual_expression()` is available only for `centered_v2` and returns:

- `rna_scale`, `rna_rate(draw, cell_name, target_sample, gene)`
- `protein_background_component_mean`
- `protein_foreground_component_mean`
- `protein_foreground_probability`
- `protein_background_contribution`
- `protein_foreground_contribution`
- `protein_total_mean`
- `protein_batch_efficiency(draw, cell_name, target_sample, protein)`
- `protein_available(cell_name, target_sample, protein)`

Protein expectations are deterministic:

```text
background = efficiency * exp(back_alpha + 0.5 * back_beta**2)
foreground = background * fore_scale
p_foreground = 1 - sigmoid(mixing)
background_contribution = (1 - p_foreground) * background
foreground_contribution = p_foreground * foreground
total = background_contribution + foreground_contribution
```

The public path does not use the decoder's stochastic `rate_back`. The private deterministic
decoder-parameter path must not alter default TotalVI numerics.

### Technical-context policies

- `observed` holds each query cell's factual batch, panel, extra covariates, and effective library
  fixed across targets.
- `specified` requires one registered batch/panel label and a positive library scalar or positive
  cell-aligned vector.
- `sample_balanced_marginal` weights biological samples equally and empirical joint technical
  contexts within each sample.
- If batch and panel are both registered, marginalizing only one fails rather than constructing
  unsupported combinations.
- Extra categorical and continuous covariates remain observed.
- A registered size-factor key is controlled by the selected library policy.
- Under marginalization, a protein is available only if measured in every positive-weight
  technical context; otherwise all corresponding protein outputs are `NaN`.

### Validation, draws, and storage

- `target_samples=None` uses registry order; explicit requested order is retained.
- Unknown or duplicate targets/features and unsupported technical categories fail before inference.
- `latent_mean` requires exactly one draw.
- `posterior_mc` requires at least two draws and returns raw draws plus posterior means and
  quantiles.
- Quantiles are unique, strictly increasing, and lie inside `(0, 1)`.
- Posterior noise is counter-based and keyed by seed, cell ID, draw, and latent coordinate. It is
  common across targets and invariant to batching, target/feature subsetting, and storage.
- Attributes record schema version, modes, inference mode, RNG, policies, formulas, context-table
  hash, dtype, chunks, estimated bytes, registered-target limitation, and non-causal meaning.

In-memory materialization has a hard 512 MiB estimate including summaries and 20% overhead. Larger
requests require subsetting or explicit Zarr output. Zarr writes direct regions into a temporary
sibling store, refuses an existing destination, atomically renames only on success, and returns a
lazy dataset. Missing Zarr/Dask dependencies produce the existing `parallel`-extra installation
hint.

### Descriptive enrichment

`local_sample_enrichment()` returns per-cell/per-sample log densities, reference counts,
self-exclusion, finite-support diagnostics, optional equal-sample group densities and log ratios,
and optional paired donor summaries.

- Group density is equal-sample `logmeanexp` across sample-specific densities.
- Only the query cell is excluded from its factual sample mixture.
- A singleton factual reference returns `NaN`, zero references, and `finite_support=False`.
- Paired contrasts require sample-constant covariates and exact donor pairing.
- Posterior-draw and between-training-seed summaries remain separate.

`differential_abundance()` stays numerically unchanged and may add only a descriptive,
non-inferential warning for grouped use. Legacy `differential_expression()` fails closed on a
centered-v2 model because v2 DE validation is outside package scope.

`combine_mrtotalvi_seed_results()` concatenates `training_seed`, retains `draw`, and reports
within-seed posterior mean/SD separately from between-seed mean/SD. It never pools these uncertainty
sources.

## Consequences

### Positive

- Existing users retain the exact default hierarchy and checkpoint topology.
- V2 counterfactuals have an explicit registered-sample-centered decomposition.
- The raw penalty constrains the common residual mode and trains every registered sample embedding.
- Decoder outputs have named deterministic estimands and explicit technical contexts.
- Large requests fail before accidental materialization and have atomic optional storage.

### Limitations

- Targets are limited to registered samples; there is no new-sample surgery or causal
  extrapolation.
- Centering anchors the reported decomposition but does not prove full identifiability or biological
  disentanglement.
- Counterfactual means are model-based registered-sample transformations, not causal interventions.
- Local enrichment is descriptive and non-inferential.
- Passing synthetic tests or the bounded comparator smoke does not show that v2 is scientifically
  better than legacy MrTotalVI.

## Package acceptance boundary

Acceptance requires legacy/MrMultiVI regression, analytic hierarchy and gradient tests, API/storage
contracts, documentation, package builds, and the bounded engineering smoke. Milo, simulation,
candidate selection, multi-seed scientific runs, macaque validation, publication jobs, and
biological interpretation remain outside this ADR's package acceptance.

