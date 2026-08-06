# Latent Representations

How to extract embeddings from MrTotalVI, MrMultiVI, and CytoANVI.

---

## give_z — cell-space vs donor-space

All Mr* models have a two-level hierarchy: cell-level `z` and donor-level `u`.

| Call | Returns | Shape | Use for |
|------|---------|-------|---------|
| `model.get_latent_representation(give_z=True)` | `z` | `(n_cells, n_latent)` | UMAP, clustering, integration |
| `model.get_latent_representation(give_z=False)` | `u` | `(n_cells, n_latent_u)` | DA, donor-level analysis |

`u` is constant per donor: all cells from the same donor have identical `u`.

```python
u = model.get_latent_representation(give_z=False)  # donor-space
z = model.get_latent_representation(give_z=True)   # cell-space

# Store in adata
adata.obsm["X_mrtotalvi_u"] = u
adata.obsm["X_mrtotalvi_z"] = z
```

---

## Local sample representation

`get_local_sample_representation()` returns the per-cell sample embedding
(the learned additive offset that maps from `u` to the local cell neighbourhood).

```python
eps = model.get_local_sample_representation()
# shape: (n_cells, n_latent_sample)
adata.obsm["X_mrtotalvi_eps"] = eps
```

This is lower-dimensional than `z` (controlled by `n_latent_sample`, default 16)
and captures within-donor batch/covariate effects.

---

## Local sample distances

`get_local_sample_distances()` computes pairwise distances between sample
embeddings (donors), averaged across cells.

```python
dist = model.get_local_sample_distances(
    sample_key="donor_timepoint",  # which samples to compare
    use_mean=True,                 # use posterior mean (faster than MC sampling)
)
# dist is a pandas DataFrame: (n_samples × n_samples) distance matrix
```

Useful for MDS / hierarchical clustering of donors.

---

## CytoANVI — annotation and soft labels

CytoANVI also exposes a `predict()` method that returns hard cell-type labels
and `get_normalized_expression()` for reconstructed protein values.

```python
# Hard label prediction
labels = model.predict(adata)         # np.ndarray of str labels
soft   = model.predict(adata, soft=True)  # (n_cells, n_labels) probabilities

# Reconstructed protein expression
prot_hat = model.get_normalized_expression(adata)
# DataFrame: (n_cells, n_proteins), library-size-normalized
```

---

## Subsetting to a query

All representation methods accept an explicit `adata` argument to evaluate on
a subset or a query dataset:

```python
# Evaluate trained reference model on query cells (surgery)
u_query = model.get_latent_representation(
    adata=query_adata,
    give_z=False,
)
z_query = model.get_latent_representation(
    adata=query_adata,
    give_z=True,
)
```

Passing `adata=None` (the default) evaluates on the training data.

---

## Footguns

1. **`give_z` is a kwarg, not a positional arg** — `get_latent_representation(True)`
   does NOT set `give_z=True`; the first positional arg is `adata`. Always pass
   `give_z=` explicitly.

2. **u is per-donor, not per-cell** — all cells from the same donor have the
   same `u`. Do not average u across cells within a donor; they are already
   identical. Clustering on `u` directly segments donors, not cell types.

3. **`get_local_sample_representation()` is lower-dim than `z`** — default
   `n_latent_sample=16` regardless of `n_latent`. Do not concatenate `eps`
   and `z` as if they have the same dimension.

4. **`get_normalized_expression` is protein only for MrTotalVI** — it returns
   the decoded protein output, not the gene deconvolution. If you need gene
   expression, use `model.get_normalized_expression(transform_batch=...)` from
   the TotalVI parent (genes are in `model.get_normalized_expression()[0]`,
   proteins in `[1]`).

5. **Surgery query requires `prepare_query_anndata` first** — passing raw
   query cells to `model.get_latent_representation(adata=query_adata)` on a
   base (non-surgery) model will silently succeed if var_names match, but will
   fail to project unseen markers correctly. Use `load_query_data` then call
   `surgery_model.get_latent_representation()`.

6. **`n_latent_u` defaults to `n_latent`** — if you set `n_latent=30` but did
   not explicitly set `n_latent_u`, `u` has shape `(n_cells, 30)`. Check
   `model.n_latent_u` before asserting shape.
