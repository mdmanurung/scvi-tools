# CytoANVI

**CytoANVI** (Python class {class}`~scvi.external.CytoANVI`) is a semi-supervised extension of
{class}`~scvi.external.CYTOVI` for antibody-based single-cell data (flow cytometry, mass cytometry,
CITE-seq protein). It follows the same design pattern as {class}`~scvi.model.SCANVI` extends
{class}`~scvi.model.SCVI`: a shared CytoVI protein encoder/decoder plus a classifier head and a
partially observed label objective (M1+M2 hierarchy), while keeping CytoVI's batch correction,
missing-marker masking, and scArches query mapping.

The advantages of CytoANVI are:

- Transfers cell-type labels to unlabeled cells with a trained classifier, not only k-NN in latent space.
- Integrates labeled reference and unlabeled query in one model (semi-supervised training).
- Inherits CytoVI panel-aware query prep, imputation, differential abundance/expression, and latent integration.
- Supports uncertainty scores for novel or ambiguous cells via test-time augmentation.
- Supports continual reference updates with optional EWC replay (experimental).

The limitations of CytoANVI include:

- Requires at least some labeled cells per type you want to predict (or an unlabeled category for unknowns).
- The classifier reads the **backbone** latent; types separated only by panel-specific markers may be under-resolved.
- Effectively requires a GPU for training on large cytometry panels.
- Does **not** use CytoVI's label-conditioned mixture-of-Gaussians prior (semi-supervised M1+M2 replaces it).

```{topic} Related tutorials:
- {doc}`/tutorials/notebooks/cytometry/CytoANVI_tutorial` (label transfer, panel mapping, uncertainty)
- {doc}`/tutorials/notebooks/cytometry/CytoVI_batch_correction_tutorial` (CytoVI preprocessing & Nuñez data)
- {doc}`/tutorials/notebooks/cytometry/CytoVI_advanced_tutorial` (multi-panel Roider mapping)
- {doc}`/user_guide/models/cytovi` (shared cytometry preprocessing and tasks)
- {doc}`/user_guide/models/scanvi` (semi-supervised VI background)
```

## Preliminaries

CytoANVI expects the same inputs as CytoVI:

- A transformed protein matrix (typically arcsinh + min-max scaled) in a layer (default `"scaled"`).
- A `batch_key` for batch / replicate correction.
- A `labels_key` with a dedicated **unlabeled category** string (e.g. `"Unknown"`) for cells without annotations.

For overlapping antibody panels, set up the reference with a `nan_layer` (see
{func}`scvi.external.cytovi.merge_batches`) so panel-specific markers can be masked during query
mapping.

## Relation to CytoVI

| Feature | CytoVI | CytoANVI |
|---------|--------|----------|
| Latent integration | yes | yes |
| Label transfer | k-NN in latent (`impute_categories_from_reference`) | `predict()` classifier |
| Semi-supervised training | optional label-informed **prior** only | classifier + partial labels |
| Query / scArches | `load_query_data` | `prepare_query_anndata` + `load_query_data` (panel-aware nan mask) |
| Uncertainty | — | `get_uncertainty()` |
| Continual update | — | `load_query_data_with_replay()` (EWC + replay) |

You can warm-start from a trained CytoVI model with {meth}`~scvi.external.CytoANVI.from_cytovi_model`.

## Quick start (label transfer)

```python
import scvi
from scvi.external import CytoANVI

CytoANVI.setup_anndata(
    adata,
    layer="scaled",
    batch_key="batch",
    labels_key="cell_type",
    unlabeled_category="Unknown",
)

model = CytoANVI(adata, y_prior="empirical")  # optional: class-imbalance prior
model.train(max_epochs=1000)

adata.obsm["X_CytoANVI"] = model.get_latent_representation()
adata.obs["pred_cell_type"] = model.predict()
```

Hold out a fraction of labels by setting those cells to `"Unknown"` before `setup_anndata`, then
compare `model.predict()` on held-out cells to ground truth.

## Training details

- Default training uses {class}`~scvi.train.SemiSupervisedTrainingPlan` (labeled + unlabeled minibatches).
- `classification_ratio` (default `50`, set via `train(plan_kwargs={"classification_ratio": ...})`)
  balances the semi-supervised classification loss against the ELBO. Higher values emphasize label
  transfer accuracy; lower values can improve batch mixing in the latent (benchmark B2 tradeoff).
- `y_prior="empirical"` sets the label prior from observed label frequencies (Laplace-smoothed); use
  for imbalanced panels. Default is uniform.
- Only `latent_distribution="normal"` is supported.
- For overlapping panels, ensure `encode_backbone_only=True` (default when a nan mask is present).

### Integration vs label transfer (batch–bio tradeoff)

CytoANVI shapes the latent with both reconstruction and the classifier on labeled cells. On real
cytometry vignette data this often **improves biological conservation** (cell types separate more
clearly) at a small cost to **batch mixing** versus unsupervised CytoVI. Tune
`classification_ratio` and `y_prior` if batch correction is the primary goal.

## Tasks

### Dimensionality reduction

```python
latent = model.get_latent_representation()
adata.obsm["X_CytoANVI"] = latent
```

Same API as CytoVI; the latent is shaped by both reconstruction and the classifier on labeled cells.

### Cell-type prediction

```python
pred = model.predict()
prob = model.predict(soft=True)  # per-class probabilities
```

Unlabeled cells (category `unlabeled_category`) receive predictions; labeled cells can be evaluated
by masking labels during training.

### Warm-start from CytoVI

```python
from scvi.external import CYTOVI, CytoANVI

cytovi_model = CYTOVI(adata)
cytovi_model.train(max_epochs=1000)

anvi = CytoANVI.from_cytovi_model(
    cytovi_model,
    unlabeled_category="Unknown",
    labels_key="cell_type",
)
anvi.train(max_epochs=200)  # fine-tune classifier + semi-supervised head
```

### Panel-divergent query mapping (Roider-style)

When the query panel differs from the reference (shared backbone + panel-specific markers):

```python
query = CytoANVI.prepare_query_anndata(query_adata, reference_model=model)
query_model = CytoANVI.load_query_data(query, model)
query_model.train(max_epochs=200)
query.obs["pred"] = query_model.predict()
```

`prepare_query_anndata` pads missing markers and writes a **nan mask** so padded zeros are not
treated as real intensities. The reference must have been trained with a genuine backbone/panel split.

### Uncertainty / novelty detection

```python
unc = model.get_uncertainty()  # per-cell Bregman information (higher = more uncertain)
```

Useful for flagging held-out cell types or low-confidence predictions (see benchmark task B5).

### Continual update (experimental)

For updating a reference with new query cohorts while limiting catastrophic forgetting (cscanvi-style):

```python
from scvi.external import CytoANVI

# ~20% of reference cells, selected by uncertainty (paper default)
replay = CytoANVI.select_replay_by_uncertainty(model, reference_adata, fraction=0.2)
# healthy controls from the query (~5–10%)
controls = query_adata[query_adata.obs["status"] == "healthy"].copy()

updated = CytoANVI.load_query_data_with_replay(
    query_adata,
    reference_model=model,
    replay_adata=replay,
    control_adata=controls,  # required
)
updated.train(max_epochs=200, plan_kwargs={"ewc_importance": 100.0})  # λ — retune for CytoVI
```

`ewc_importance` (= λ) is **not** a constructor argument; pass it at train time. The paper used
`λ=100` for scANVI/RNA; CytoVI's intensity likelihood has different Fisher magnitudes, so λ must be
retuned (see benchmark task B6).

See ADR `docs/adr/0002-cytoanvi-continual-follows-paper-not-not-code.md` for design notes.

## Nuñez PBMC labels (benchmark D2)

Vignette Nuñez FCS files do **not** include cell-type annotations. For paper-aligned eleven-type
labels, use the CytoVI tutorial workflow (train CytoVI → Leiden on latent → manual cluster map):

```bash
PYTHONPATH=src:. python -m benchmarks.cytoanvi.annotate_nunez \
  --data-dir data --out data/nunez_annotated.h5ad --max-epochs 100
```

Then point CytoANVI at `obs["cell_type"]` from that file. For publication-grade benchmarks, pass
`--require-annotated-nunez` to the benchmark CLI so Leiden proxy labels are not used.

## Inherited CytoVI methods

CytoANVI subclasses CytoVI and retains (among others):

- {meth}`~scvi.external.CYTOVI.get_normalized_expression` — denoised / batch-corrected protein expression
- {meth}`~scvi.external.CYTOVI.differential_expression`
- {meth}`~scvi.external.CYTOVI.differential_abundance`

Multi-panel references are built with {func}`scvi.external.cytovi.merge_batches` before
`CytoANVI.setup_anndata`.

Refer to {doc}`/user_guide/models/cytovi` for preprocessing cofactors and mathematical background.
