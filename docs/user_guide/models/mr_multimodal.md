# Mr multimodal models

`MrTotalVI` and `MrMultiVI` extend TotalVI and MultiVI with the MrVI hierarchical
sample latent:

1. `u` is the base representation used for integration and aggregated posterior
   statistics. It is sample-conditioned by default in both legacy models.
2. `z = z_base(u) + eps` is the sample-aware representation used for local
   counterfactual donor queries.

Both models keep the stock TotalVI/MultiVI decoders unchanged. The hierarchy changes the
encoder and latent prior, not the RNA/protein/ATAC likelihood definitions.

The {doc}`/usage_readiness` matrix is authoritative. MrTotalVI engineering coverage does not
promote a prior, representation claim, DA/DE inference, streaming, or new-sample inference.

## Setup

Register a `sample_key` for the donor or sample axis. RNA (the registered layer or `X`) and protein
(`obsm`) inputs must be finite, non-negative, integer-like **raw counts**. Setup exhaustively checks
all values before mutating AnnData state.

`labels_key` records annotations but does not change the default unsupervised objective. To opt into
label supervision, set `u_prior_supervision="labels"` and a finite positive
`u_prior_label_weight`; disclose that choice in every comparison.

```python
scvi.external.MrTotalVI.setup_anndata(
    adata,
    protein_expression_obsm_key="protein_expression",
    sample_key="donor",
    batch_key="batch",
    labels_key="cell_type",
)

model = scvi.external.MrTotalVI(
    adata,
    sample_key="donor",
    n_latent=20,
    n_latent_u=10,
    u_prior_supervision="none",  # default even when labels_key is registered
)
```

The historical behavior remains the default. Opt into the package-only v2
semantics explicitly:

```python
model_v2 = scvi.external.MrTotalVI(
    adata,
    sample_key="donor",
    n_latent=20,
    n_latent_u=10,
    hierarchy_mode="centered_v2",
    u_encoder_mode="sample_blind",
    use_map=True,
    z_u_prior=True,
)
```

`centered_v2` requires `use_map=True` and `z_u_prior=True`. The
`sample_blind` encoder bypasses biological sample embeddings while retaining
explicitly registered technical covariates. It does not change parameter names
or shapes. MrMultiVI remains unchanged.

```python
scvi.external.MrMultiVI.setup_mudata(
    mdata,
    sample_key="donor",
    batch_key="batch",
    labels_key="cell_type",
    modalities={
        "rna_layer": "rna",
        "atac_layer": "accessibility",
        "protein_layer": "protein_expression",
        "labels_key": None,
    },
)
```

## Prior Options

New MrTotalVI calls resolve `u_prior` to exactly `"standard"`, `"mog"`, or `"vamp"`; the default
is `"mog"` for continuity, not because it is scientifically preferred.

- `n_latent_u=None` preserves the original isomorphic hierarchy, so `u` and `z` have the same
  dimensionality.
- `n_latent_u < n_latent` learns a lower-dimensional `u` and projects it into `z`.
- `u_prior="mog"` uses learned mixture logits, means, and scales.
- `u_prior="standard"` uses an analytic Gaussian prior over `u`.
- `u_prior="vamp"` uses pseudoinputs; data initialization is restricted to frozen training indices
  and records its seed and ordered-index digest.
- `u_prior_supervision="labels"` is the only label-conditioned mode and requires registered labels
  plus a finite positive weight. The default is `"none"` with weight `0.0`.
- `z_u_prior=False` omits the residual `eps` prior penalty while keeping the `u` prior.

The deprecated `u_prior_mixture` boolean is accepted only for the exact migration combinations in
{doc}`/migration/cytoanvi-mrtotalvi-0.2.0`; contradictory combinations fail before module
construction.

`latent_distribution="ln"` is rejected because the additive `u -> z` hierarchy is Euclidean.

### Choosing a prior: unsettled

No prior is generally recommended. The analysis historically used to justify MoG is unrecoverable
and used non-authoritative geometry. One-dataset Vamp results are insufficient for promotion.
Prespecify prior choice, include MoG/Vamp sensitivity where relevant, and do not choose after seeing
the biological outcome. The following is an experimental sensitivity arm, not a preferred recipe.

The two flags are not gated the same way. `init_prior_from_data` applies only to `u_prior="vamp"`.
When enabled, initialization uses only the frozen training indices and persists the seed and
ordered-index digest; validation/test cells cannot influence pseudoinputs. `freeze_prior_after_init`
also exists for historical sensitivity work, but no combination of these flags is a recommended
scientific default.

```python
model = scvi.external.MrTotalVI(
    adata,
    u_prior="vamp",
    u_prior_mixture_k=20,
    init_prior_from_data=True,
    freeze_prior_after_init=True,
    use_batch_norm="none",
    use_layer_norm="both",
)
```

This code block is therefore an explicitly prespecified sensitivity arm. It is not evidence that
Vamp is preferable to MoG or standard Gaussian on a new dataset.

## Representations

Use `give_z=False` for the `u` representation and `give_z=True` for the donor-aware `z`
representation.

```python
u = model.get_latent_representation(give_z=False)
z = model.get_latent_representation(give_z=True)
local_z = model.get_local_sample_representation()
local_d = model.get_local_sample_distances()
```

The default `u` encoder is sample-conditioned, so do not describe legacy `u` as
sample-unaware or donor-neutral. `u_encoder_mode="sample_blind"` removes the
implemented sample-conditioning paths, but this software property alone does
not establish biological disentanglement. Use `z` for registered-sample model
transformations and local sample distances.

## Batch representation — MrTotalVI only

This section applies to {class}`~scvi.external.MrTotalVI`. MrMultiVI has no
`batch_representation` option.

By default the batch covariate is one-hot encoded, so every batch adds a column to the
u-encoder input and to each decoder layer. With many batches that becomes the dominant
input width. `batch_representation="embedding"` replaces the one-hot code with a single
learned embedding table shared by the u-encoder and the decoder:

```python
model = MrTotalVI(
    adata,
    sample_key="donor",
    encode_covariates=True,             # required for batch to reach the u-encoder
    batch_representation="embedding",
    batch_embedding_kwargs={"embedding_dim": 5},
)
model.train()
batch_vectors = model.get_batch_representation()   # (n_cells, embedding_dim)
```

The input width then grows by `embedding_dim` instead of by `n_batch`. Note
`encode_covariates` defaults to `False` on this model, and batch only enters the
u-encoder when it is `True`; the decoder uses the representation either way.

Per-batch *parameter tables* stay one-hot indexed in both modes — gene and protein
dispersion, `log_per_batch_efficiency`, the protein background prior and the library-size
priors. Those are lookup tables rather than network inputs, matching {class}`~scvi.module.VAE`.

`"one-hot"` remains the default and is bit-for-bit unchanged, so existing checkpoints are
unaffected. `"embedding"` changes the architecture, so a model trained with it cannot be
reloaded as `"one-hot"`.

## Opt-in centered counterfactual datasets

The v2 APIs are available only when `hierarchy_mode="centered_v2"`.
Residuals are evaluated for every registered sample, centered over that full
registry, and only then subset to requested targets.

```python
latent = model_v2.get_counterfactual_latent(
    indices=query_indices,
    target_samples=["donor_a", "donor_b"],
    inference_mode="posterior_mc",
    n_draws=32,
    random_state=0,
)

expression = model_v2.get_counterfactual_expression(
    indices=query_indices,
    target_samples=["donor_a", "donor_b"],
    gene_list=["IL7R", "LST1"],
    protein_list=["CD3", "CD14"],
    batch_policy="observed",
    panel_policy="observed",
    library_policy="observed",
)
```

The latent dataset names `u`, `z_base`, `eps_raw`, `eps_centered`, and `z`.
The expression dataset names RNA scale/rate and deterministic protein mixture
components, contributions, totals, batch efficiency, and panel availability.
Posterior Monte Carlo output retains raw `draw` values and adds posterior means
and quantiles.

Technical contexts are explicit:

- `observed` holds each query cell's factual batch, panel, extra covariates,
  and effective library fixed across targets. The effective library is the
  registered size factor when present, otherwise the observed RNA total under
  the default TotalVI setting, or the posterior log-normal mean when
  `use_observed_lib_size=False`.
- `specified` requires registered batch/panel labels and a positive library
  scalar or cell-aligned vector.
- `sample_balanced_marginal` weights biological samples equally and empirical
  joint batch/panel contexts within each sample. Batch and panel cannot be
  marginalized separately.

In-memory output has a hard 512 MiB estimate including summaries and overhead. Requests above that
limit must be reduced through explicit cell, target, or feature subsetting. Public `zarr_path` /
`zarr_chunks` streaming is quarantined and fails before filesystem access.

All targets are limited to registered samples. These datasets are model-based,
non-causal transformations. Centering defines the reported decomposition; it
does not prove identifiability, biological neutrality, or superiority over the
legacy model.

For descriptive local densities:

```python
enrichment = model_v2.local_sample_enrichment(
    group_key="condition",
    contrast=("treated", "control"),
    donor_key="donor",
)
```

Group densities use equal-sample `logmeanexp`; only a query cell's factual
reference mixture excludes that cell. These outputs and grouped
`differential_abundance()` results are descriptive and non-inferential.

## Statistical APIs

Both multimodal Mr models expose aggregated-posterior and outlier utilities. MrTotalVI also exposes
descriptive DA:

```python
ap = model.get_aggregated_posterior()
da = model.differential_abundance(sample_cov_keys=["condition"])
outliers = model.get_outlier_cell_sample_pairs()
```

`MrTotalVI.differential_abundance()` is descriptive and non-inferential. Its public
`differential_expression()` fails closed for both legacy and centered-v2 models. For biological
DE, aggregate raw counts at the donor-by-condition level and use a replicate-aware method such as
PyDESeq2, edgeR, or dreamlet:

```python
model.differential_expression(sample_cov_keys=["condition"])
# RuntimeError: use donor-pseudobulk PyDESeq2/edgeR/dreamlet.
```

`MrMultiVI.differential_expression()` supports `store_lfc=True` for RNA-only or RNA+protein
bimodal models. ATAC-containing models raise `NotImplementedError`; a `differential_accessibility`
method exists on `MrMultiVI` but currently raises `NotImplementedError` (users who call it receive
a clear error rather than `AttributeError`; the implementation is deferred to v2).

## v1 Limitations

These limitations are intentional cuts documented in ADR-0005 / ADR-0006.

**Differential expression:**
- MrTotalVI accepts only `use_vmap=False`; `use_vmap=True` raises before inference/statistics.
  MrMultiVI behavior is unchanged by the 0.2.0 MrTotalVI hardening.
- `MrMultiVI.differential_expression` rejects ATAC-containing models. Use `differential_abundance`
  for sample-level DA over `u` instead.

**Differential expression — empirical validation note (eps-space limitation):**
The LFC values returned by `store_lfc=True` are decoded from the `eps = z − u` residual, which
captures the *donor-specific* deviation from the sample-conditioned base `u`. Because the `u` encoder
absorbs cell-state variation (including treatment-induced states such as IFN activation) as part of
cell identity, `eps` carries minimal cell-state treatment signal. On one CITE-seq dataset
(schistosomiasis, n=10 donors, 12 cell types), eps-space LFC from all three models (MRVI,
MrTotalVI, MrMultiVI) was **anti-concordant** with pseudobulk DE gold standard (PyDESeq2,
Spearman rho −0.24 to +0.04 across all cell types). This limitation is architectural and
cell-type-universal: no cell type showed positive concordance between model LFC and pseudobulk
direction. These historical internals remain private for reproduction; the public MrTotalVI method
does not return p-values or decoded LFC. Use donor-pseudobulk for biological inference.

**Differential abundance — sample-count sensitivity:**
`MrTotalVI.differential_abundance` is unreliable when `sample_key` maps to many samples relative
to the prior capacity. In one experiment with 20 donor-timepoint samples (schistosomiasis DTP
cohort, 3 seeds), MrTotalVI DA varied from −9.0 to +9.7 across seeds (per-seed: s0 = +2.74,
s1 = −9.04, s2 = +9.67; std 9.46 >> mean 1.12). MrMultiVI DA at the same setup was stable
(per-seed: s0 = +0.84, s1 = +0.94, s2 = +1.09; mean +0.96 ± 0.13). If using
`differential_abundance` with a `sample_key` that expands the sample count significantly (e.g.
`sample_key="donor_timepoint"`), do not interpret the package output as inferential evidence.
Use an independently validated, replicate-aware compositional method for biological DA.

**Integration benchmarks (single dataset, batch-correction only):**
MrMultiVI's integration improvement over MultiVI has been benchmarked on one CITE-seq dataset
(schistosomiasis, n=10 donors, 12 cell types; human only). Across three seeds, MrMultiVI achieves a
total scIB score of 0.640±0.009 vs. MultiVI 0.593 (Δ+0.047; note the MultiVI baseline is a single
seed — seed variance for MultiVI is unmeasured). The gain is entirely attributable to batch
correction: across the same three seeds, the scIB batch sub-score improves by Δ+0.128±0.008, while
bio conservation is flat (Δbio = −0.006±0.010, indistinguishable from zero). A single-seed
preliminary estimate suggested Δbio ≈ +0.009, but this does not replicate in the 3-seed analysis.
Interpret the MrMultiVI integration claim specifically as improved batch mixing; bio conservation is
not improved. kNN label transfer is worse under MrMultiVI (local structure regresses). Multi-dataset
validation (macaque CITE-seq) is pending.

**ArchesMixin / scArches surgery:**
Neither model supports reference-query surgery or the `ArchesMixin` protocol.

**Streaming and new-sample inference:**
`MrTotalVIBatchDataModule` is not a stable public training API, and 0.2.0 does not claim end-to-end
streaming training or inference for biological samples absent from the registered training set.

**Objective (methods note):**
The default training objective (`use_map=True`) is a MAP/cross-entropy penalty on the residual
`eps`, not a strict ELBO. For the legacy hierarchy, the analytic ELBO — including the `q(eps)`
entropy term — is available via `use_map=False` at model construction. Centered v2 requires
`use_map=True`.
```python
model = scvi.external.MrTotalVI(adata, sample_key="donor", use_map=False)
```

The `kl_u` term uses a single-sample Monte-Carlo estimate of `KL(q(u)‖p(u))` (unbiased, higher
variance than analytic). The MrMultiVI mixed-posterior variance is regularized only via the
`kl_div_paired` penalty, not through a KL-to-prior on the per-modality encoders.

The Wald test (`differential_abundance`, `differential_expression`) uses a Chi² distribution
whose degrees of freedom equal the number of admissible samples per cell (`n_per_cell`,
`_stats.py:627`), matching the MRVI reference implementation. This is conservative (n_per_cell
≫ n_latent) and is an intentional design choice, not a bug.

The design matrix (`_construct_design_matrix`, `_stats.py:291`) is intentionally intercept-free.
The response `eps` is mean-centred per cell across samples (`_stats.py:610`) before regression,
making a constant term non-identifiable. The Wald test therefore operates on covariate slopes
only; intercept-like shifts are absorbed by the per-cell centering step.
