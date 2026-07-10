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
from the local counterfactual linear model. Decoded RNA/protein LFC storage is not implemented
yet.

`MrMultiVI.differential_expression()` rejects ATAC-containing models. ATAC effects should use a
separate differential-accessibility API rather than being mixed into RNA/protein DE semantics.
