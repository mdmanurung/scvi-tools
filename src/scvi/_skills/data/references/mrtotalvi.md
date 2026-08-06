# MrTotalVI

`scvi.external.MrTotalVI` — sample-aware TotalVI for CITE-seq (genes + proteins)
with a two-level latent hierarchy (cell-level `z`, donor-level `u`).

## Minimal working example

```python
import scvi
from scvi.external import MrTotalVI

scvi.settings.seed = 0

MrTotalVI.setup_anndata(
    adata,
    layer="counts",
    protein_expression_obsm_key="protein",
    sample_key="donor",
    batch_key="batch",
    labels_key="celltype",
)
model = MrTotalVI(adata, sample_key="donor", n_latent=20, n_hidden=256)
model.train(
    max_epochs=400,
    early_stopping=True,
    batch_size=512,
    accelerator="gpu",
    devices=1,
    plan_kwargs={"lr": 1e-3, "n_epochs_kl_warmup": 40},
)
u = model.get_latent_representation(give_z=False)  # donor-space
z = model.get_latent_representation(give_z=True)   # cell-space
model.save("path/to/model", overwrite=True)
```

---

## `__init__` key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `sample_key` | — | required; must match the key used in `setup_anndata` |
| `n_latent` | 20 | cell-level latent dim |
| `n_latent_u` | None | donor-level latent dim; defaults to `n_latent` |
| `n_latent_sample` | 16 | sample embedding dim |
| `u_prior` | `"mog"` | `"mog"` (mixture-of-Gaussians) or `"vamp"` (VampPrior) |
| `u_prior_mixture_k` | 20 | number of mixture components / VampPrior pseudo-inputs |
| `init_prior_from_data` | False | k-means init of VampPrior pseudo-inputs (vamp only) |
| `freeze_prior_after_init` | False | freeze pseudo-inputs after init (vamp only) |
| `use_batch_norm` | `"both"` | pass `"none"` for small-N cohorts (≤20 donors) |
| `use_layer_norm` | `"none"` | pass `"both"` when disabling BatchNorm |

### LayerNorm vs BatchNorm — when to switch

BatchNorm running statistics are unstable when `n_samples ≤ 20`. For small
cohorts, use:

```python
model = MrTotalVI(
    adata, sample_key="donor",
    use_batch_norm="none",
    use_layer_norm="both",
)
```

This reduces DA variance by ~10× at the cost of ~5% slower training.

### VampPrior stabilised configuration (D-038 recipe)

All four flags must be set together for the anchored variant:

```python
model = MrTotalVI(
    adata, sample_key="donor",
    u_prior="vamp",
    u_prior_mixture_k=20,
    init_prior_from_data=True,   # k-means init from ≤10k cells
    freeze_prior_after_init=True, # freeze pseudo-inputs post-init
)
```

**Pre-registered success criterion**: W22-enrichment std ≤ 0.30 across seeds
(>65% reduction from LN-MoG baseline std=0.875). See D-041 in `.living/decisions.md`.

---

## Saving and loading

```python
model.save("path/to/model", overwrite=True)
loaded = MrTotalVI.load("path/to/model", adata=adata)
```

A valid save directory contains `model.pt`, `attr.pkl`, `model_params.pt`,
`var_names.csv`, and `registry_`. If any of these are missing (e.g., only
`model.pt` present), `MrTotalVI.load()` will fail. This happens when training
was interrupted or the model was saved with raw `torch.save` instead of
`model.save()`.

---

## Footguns

1. **`latent_distribution` must be `"normal"`** — the additive `u→z` hierarchy
   is mathematically invalid under `"ln"` (softmax normalisation). Passing
   `latent_distribution="ln"` raises `ValueError` immediately.

2. **Zero-cell donors** — every unique value in `sample_key` must have ≥1 cell.
   Missing donors raise `ValueError` listing the integer code positions.

3. **sample_key vs batch_key** — these are different obs columns. `sample_key`
   is the donor identity; `batch_key` is the technical batch/run. Do not pass the
   same column for both.

4. **`setup_anndata` before `__init__`** — calling `MrTotalVI(adata, ...)` without
   first running `MrTotalVI.setup_anndata(adata, ...)` raises `RegistryError`.

5. **DTP (donor_timepoint) for valid DA** — using `sample_key="donor"` and then
   calling `differential_abundance(sample_cov_keys=["timepoint"])` does NOT work;
   the `timepoint` covariate is only available when each sample maps to exactly one
   timepoint. Use `sample_key="donor_timepoint"` at training time. See
   `references/differential-abundance.md`.

6. **`init_prior_from_data` only fires for `u_prior="vamp"`** — setting it with
   `u_prior="mog"` silently does nothing.

7. **`freeze_prior_after_init` without `init_prior_from_data`** — freezes random
   initial pseudo-inputs, which is never useful. Always set both together.

8. **VampPrior pseudo-input shape** — pseudo-inputs live in the raw data space
   (genes + proteins), not the latent space. Their shape is `(k, n_input_genes +
   n_proteins)` unless `protein_in_encoder=False`, in which case it is
   `(k, n_input_genes)`.
