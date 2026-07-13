# Mr multimodal models

`MrTotalVI` and `MrMultiVI` extend TotalVI and MultiVI with the MrVI hierarchical
sample latent:

1. `u` is the sample-conditioned base representation used for integration and
   aggregated posterior statistics.
2. `z = z_base(u) + eps` is the sample-aware representation used for local
   counterfactual donor queries.

Both models keep the stock TotalVI/MultiVI decoders unchanged. The hierarchy changes the
encoder and latent prior, not the RNA/protein/ATAC likelihood definitions.

## Setup

Register a `sample_key` for the donor or sample axis. Optionally register `labels_key` to use one
mixture-prior component per label.

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
)
```

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

By default the models use a learned mixture-of-Gaussians prior over `u`.

- `n_latent_u=None` preserves the original isomorphic hierarchy, so `u` and `z` have the same
  dimensionality.
- `n_latent_u < n_latent` learns a lower-dimensional `u` and projects it into `z`.
- `u_prior_mixture=True` uses learned mixture logits, means, and scales.
- `labels_key` with more than one label switches the mixture count to `n_labels` and biases each
  cell toward its observed label component.
- `u_prior_mixture=False` uses an analytic Gaussian prior over `u`.
- `z_u_prior=False` omits the residual `eps` prior penalty while keeping the `u` prior.

`latent_distribution="ln"` is rejected because the additive `u -> z` hierarchy is Euclidean.

## Representations

Use `give_z=False` for the `u` representation and `give_z=True` for the donor-aware `z`
representation.

```python
u = model.get_latent_representation(give_z=False)
z = model.get_latent_representation(give_z=True)
local_z = model.get_local_sample_representation()
local_d = model.get_local_sample_distances()
```

Use `u` for donor-neutral integration and abundance statistics. Use `z` for donor-aware
counterfactual biology and local sample distances.

## Statistical APIs

Both multimodal Mr models expose:

```python
ap = model.get_aggregated_posterior()
da = model.differential_abundance(sample_cov_keys=["condition"])
outliers = model.get_outlier_cell_sample_pairs()
```

`MrTotalVI.differential_expression()` returns latent-space `beta`, `effect_size`, and `pvalue`
from the local counterfactual linear model, and optionally decoded gene/protein-space LFC arrays
when `store_lfc=True`:

```python
ds = model.differential_expression(
    sample_cov_keys=["condition"],
    store_lfc=True,          # adds lfc, lfc_std, feature coords
    delta=0.5,               # adds pde = P(|lfc| >= delta)
    store_baseline=True,     # adds baseline_expression
)
# ds["lfc"].coords["feature"] values: "gene" or "protein"
```

`MrMultiVI.differential_expression()` supports `store_lfc=True` for RNA-only or RNA+protein
bimodal models. ATAC-containing models raise `NotImplementedError`; a `differential_accessibility`
method exists on `MrMultiVI` but currently raises `NotImplementedError` (users who call it receive
a clear error rather than `AttributeError`; the implementation is deferred to v2).

## v1 Limitations

These limitations are intentional cuts documented in ADR-0005 / ADR-0006.

**Differential expression:**
- Both models default to `use_vmap=False`. MrTotalVI's inherited TotalVI decoder uses BatchNorm
  (incompatible with `torch.vmap`). MrMultiVI uses LayerNorm and may support `use_vmap=True` as
  an opt-in in a future release.
- `MrMultiVI.differential_expression` rejects ATAC-containing models. Use `differential_abundance`
  for sample-level DA over `u` instead.

**Differential expression — empirical validation note (eps-space limitation):**
The LFC values returned by `store_lfc=True` are decoded from the `eps = z − u` residual, which
captures the *donor-specific* deviation from the sample-unaware base `u`. Because the `u` encoder
absorbs cell-state variation (including treatment-induced states such as IFN activation) as part of
cell identity, `eps` carries minimal cell-state treatment signal. On one CITE-seq dataset
(schistosomiasis, n=10 donors, 12 cell types), eps-space LFC from all three models (MRVI,
MrTotalVI, MrMultiVI) was **anti-concordant** with pseudobulk DE gold standard (PyDESeq2,
Spearman rho −0.24 to +0.04 across all cell types). This limitation is architectural and
cell-type-universal: no cell type showed positive concordance between model LFC and pseudobulk
direction. Cross-validate eps-space LFC against pseudobulk (e.g. PyDESeq2 or edgeR) before
drawing biological conclusions from `store_lfc=True` output.

**Differential abundance — sample-count sensitivity:**
`MrTotalVI.differential_abundance` is unreliable when `sample_key` maps to many samples relative
to the prior capacity. In one experiment with 20 donor-timepoint samples (schistosomiasis DTP
cohort, 3 seeds), MrTotalVI DA varied from −9.0 to +9.7 across seeds (per-seed: s0 = +2.74,
s1 = −9.04, s2 = +9.67; std 9.46 >> mean 1.12). MrMultiVI DA at the same setup was stable
(per-seed: s0 = +0.84, s1 = +0.94, s2 = +1.09; mean +0.96 ± 0.13). If using
`differential_abundance` with a `sample_key` that expands the sample count significantly (e.g.
`sample_key="donor_timepoint"`), validate stability across seeds before reporting results. Prefer
MrMultiVI for DA when both modalities are available.

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

**Objective (methods note):**
The default training objective (`use_map=True`) is a MAP/cross-entropy penalty on the residual
`eps`, not a strict ELBO. The analytic ELBO — including the `q(eps)` entropy term — is available
via `use_map=False` at model construction:
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
