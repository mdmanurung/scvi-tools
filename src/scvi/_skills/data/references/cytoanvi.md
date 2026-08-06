# CytoANVI

`cytoanvi.CytoANVI` — semi-supervised, annotation-aware VAE for cytometry
(mass cytometry, flow cytometry, CITE-seq protein-only). Operates on
**transformed protein values** (arcsinh / logicle), not raw counts.

## Minimal working example

```python
import scvi
from cytoanvi import CytoANVI

scvi.settings.seed = 0

CytoANVI.setup_anndata(
    adata,
    labels_key="celltype",         # required; column with cell-type annotations
    unlabeled_category="unknown",  # cells without labels
    layer=None,                    # None → adata.X (arcsinh-transformed values)
    batch_key="batch",
    sample_key="donor",
)
model = CytoANVI(
    adata,
    n_latent=20,
    n_hidden=256,
    n_layers=2,
    encode_covariates=True,
)
model.train(
    max_epochs=400,
    early_stopping=True,
    batch_size=512,
    accelerator="gpu",
    devices=1,
)
z = model.get_latent_representation()
model.save("path/to/model", overwrite=True)
```

---

## `__init__` key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_latent` | 20 | latent dimensionality |
| `n_hidden` | 256 | hidden layer width |
| `n_layers` | 2 | encoder/decoder depth |
| `n_labels` | auto | inferred from `labels_key`; must be ≥1 (raises if 0) |
| `encode_covariates` | True | incorporate batch/sample covariates in encoder |
| `deeply_inject_covariates` | True | inject covariates at every layer, not just the first |
| `use_batch_norm` | `"both"` | pass `"none"` for small cohorts |
| `use_layer_norm` | `"none"` | pass `"both"` when disabling BatchNorm |

---

## Surgery / query transfer (scArches-style)

Query transfer allows annotating a new cytometry panel against a trained reference
model, even when the query panel is missing some markers.

```python
# 1. Train reference on full backbone panel
CytoANVI.setup_anndata(ref_adata, ...)
ref_model = CytoANVI(ref_adata, ...)
ref_model.train(...)
ref_model.save("reference_model")

# 2. Prepare query — only backbone markers
CytoANVI.prepare_query_anndata(
    query_adata,
    reference_model="reference_model",
    inplace=True,
)

# 3. Load reference and create surgery model for query
ref_model = CytoANVI.load("reference_model", adata=ref_adata)
surgery_model = CytoANVI.load_query_data(
    query_adata,
    reference_model=ref_model,
    freeze_dropout=True,
)
surgery_model.train(
    max_epochs=100,
    plan_kwargs={"weight_decay": 0.0},
)
query_z = surgery_model.get_latent_representation(query_adata)
```

---

## Continual update (Phase 2 — EWC + replay)

For case-control atlas updates: train a new model on new data using EWC
(elastic weight consolidation) to preserve old representations.

```python
from cytoanvi import CytoANVI
from cytoanvi.continual import ewc_importance

# Compute importance weights from reference model + reference data
importance = ewc_importance(ref_model, ref_adata, n_samples=10_000)

# Train new model with EWC penalty
new_model = CytoANVI(new_adata, ...)
new_model.train(
    max_epochs=200,
    plan_kwargs={
        "ewc_lambda": 100.0,
        "ewc_importance": importance,
        "ewc_reference_params": dict(ref_model.module.named_parameters()),
    },
)
```

EWC importance is Fisher information estimated on ≤10k cells (C-003).

---

## Footguns

1. **`layer=None` means arcsinh/logicle-transformed values, not raw counts** —
   CytoANVI expects transformed protein values. Passing raw event counts will
   train without error but produce nonsensical embeddings and classifications.

2. **`unlabeled_category` remapped to last integer code** — the unlabeled class
   is always placed at index `n_labels - 1`. If your `labels_key` column contains
   values sorted alphabetically, the last label alphabetically is NOT necessarily
   the unlabeled one; explicitly set `unlabeled_category` to the exact string.

3. **`n_labels == 0` guard** — if `setup_anndata` is called without `labels_key`,
   the classifier is disabled (C-002). Calling `model.predict()` or accessing
   `model.history["train_classification_loss"]` in this case will `KeyError`.
   Always provide `labels_key`.

4. **`nan_layer` for missing panels** — if some donors have missing markers,
   store a binary mask in `adata.layers["_nan_mask"]` (1 = missing) and pass
   `nan_layer="_nan_mask"` to `setup_anndata`. Without this, missing values
   propagate through the reconstruction loss and bias imputed expressions.

5. **`prepare_query_anndata` before `load_query_data`** — skipping
   `prepare_query_anndata` causes a shape mismatch in the surgery encoder because
   the query var_names won't match the reference.

6. **EWC lambda scale** — `ewc_lambda` is on the order of 100–1000 for typical
   cytometry cohorts. Values < 1.0 give no penalty; values > 10,000 freeze the
   model and suppress new-data learning.

7. **`freeze_dropout=True` in `load_query_data`** — this freezes the VAE encoder
   dropout layers so the reference representaton is preserved during surgery. Do
   not set `freeze_dropout=False` unless you explicitly want unconstrained
   surgery (few-shot fine-tuning, not transfer).
