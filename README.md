# CytoANVI

> **This is the CytoANVI fork of [scvi-tools](https://github.com/scverse/scvi-tools).**
> CytoANVI adds semi-supervised, annotation-aware variational inference for antibody-based
> single-cell cytometry (mass cytometry, flow cytometry, CITE-seq protein).
> The upstream scvi-tools package is maintained by the [scverse community](https://scverse.org).
>
> **Status: research/internal use — not yet on PyPI.** This fork bundles a *modified* `scvi`
> (CytoVI lives at `scvi.external.cytovi`), so it ships both the `scvi` and `cytoanvi` import
> packages. Install it **from source into a clean environment** and do **not** install it alongside
> the upstream `scvi-tools` package (they collide on the `scvi` import). Public packaging is
> deferred pending resolution of that namespace decision.

The [usage-readiness matrix](docs/usage_readiness.md) is the authoritative capability status.
Source tests, installed-wheel acceptance, scientific validation, and human promotion are separate
states; shorter examples on this page never override that authority.

CytoANVI is a semi-supervised variational autoencoder for antibody-based single-cell cytometry. It
extends [CytoVI](https://scvi-tools.org/) (protein-intensity VAE) with the scANVI-style M1+M2
latent hierarchy and a label classifier, giving you a single model that jointly learns a batch-
corrected latent space **and** transfers cell-type labels from a partially annotated reference to
unlabeled cells — while respecting the realities of cytometry data (few markers, arcsinh-transformed
intensities rather than counts, and panels that don't share all markers).

## What CytoANVI does

- **Semi-supervised label transfer** — train on a mix of labeled and unlabeled cells; predict labels
  (hard or soft/probabilistic) for the rest.
- **Batch-corrected latent space** — a shared `z1` latent suitable for UMAP, clustering, kNN.
- **Novelty / out-of-distribution research** — the former TTA estimator is quarantined after a
  below-chance multi-seed result; no stable novelty API is currently supported.
- **Hierarchical label transfer** — supply a cell-type tree (or learn one with scHPL/treeArches) and
  predict with scores propagated through the hierarchy.
- **Panel-divergent reference→query mapping** — scArches-style surgery that pads and masks markers
  missing from the query panel (backbone/missing-marker handling).
- **Continual case-control atlas updates** — add new (e.g. disease) batches with EWC + experience
  replay to limit catastrophic forgetting of the reference.
- **Warm start from a trained CytoVI model** — reuse an existing CytoVI encoder/decoder.
- **Query-mapping QC** — mapQC scoring of how well query cells embed into the reference.

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Preprocessing cytometry data](#preprocessing-cytometry-data)
- [Feature guide](#feature-guide)
  - [1. Semi-supervised label transfer](#1-semi-supervised-label-transfer)
  - [2. Latent space, normalized expression, DE](#2-latent-space-normalized-expression-and-differential-expression)
  - [3. Novelty / OOD status](#3-novelty--ood-status)
  - [4. Warm start from a CytoVI model](#4-warm-start-from-a-cytovi-model)
  - [5. Hierarchical label transfer](#5-hierarchical-label-transfer)
  - [6. Reference → query mapping (panel-divergent surgery)](#6-reference--query-mapping-panel-divergent-surgery)
  - [7. Continual case-control atlas update](#7-continual-case-control-atlas-update)
  - [8. Query-mapping QC (mapQC)](#8-query-mapping-qc-mapqc)
  - [9. Missing markers / multi-panel data](#9-missing-markers--multi-panel-data)
  - [10. Save and load](#10-save-and-load)
- [Key training parameters](#key-training-parameters)
- [Optional dependencies](#optional-dependencies)
- [Performance and scale](#performance-and-scale)
- [Limitations and status](#limitations-and-status)
- [Citation](#citation)
- [License](#license)

## Installation

Install from a checkout of this repository, into a **fresh** environment (do not co-install with
`scvi-tools` — see the status note above). Python ≥ 3.12 is required.

```bash
git clone https://github.com/mdmanurung/scvi-tools.git
cd scvi-tools
python -m venv .venv && source .venv/bin/activate    # or a fresh conda env
pip install .
python -c "from cytoanvi import CytoANVI; import cytoanvi; print('cytoanvi', cytoanvi.__version__)"
```

Install a [PyTorch](https://pytorch.org) build matching your CUDA/CPU setup. The core label-transfer
and save/load APIs need no extra dependencies; optional backends are listed under
[Optional dependencies](#optional-dependencies).

## Quick start

The following is the minimal end-to-end workflow (train → predict → latent → save/load). Replace
the synthetic array with your own arcsinh-transformed marker intensities.

```python
import numpy as np
import anndata as ad
from cytoanvi import CytoANVI

# adata.X: cells × markers, arcsinh-transformed intensities (see preprocessing below).
# adata.obs["cell_type"]: ground-truth labels for SOME cells; unlabeled cells get "Unknown".
rng = np.random.default_rng(0)
adata = ad.AnnData(np.abs(rng.normal(2, 1, size=(600, 18))).astype("float32"))
adata.var_names = [f"marker_{i}" for i in range(18)]
adata.obs["batch"] = rng.choice(["b0", "b1"], size=600)
labels = rng.choice(["Tcell", "Bcell", "NK", "Mono"], size=600).astype(object)
labels[rng.random(600) < 0.4] = "Unknown"          # 40% unlabeled
adata.obs["cell_type"] = labels

CytoANVI.setup_anndata(
    adata,
    labels_key="cell_type",
    unlabeled_category="Unknown",
    batch_key="batch",
)
model = CytoANVI(adata)
model.train(max_epochs=1000, batch_size=4096)       # publication default; use fewer epochs to iterate

adata.obs["pred"] = model.predict()                 # hard labels for every cell
adata.obsm["X_cytoanvi"] = model.get_latent_representation()

model.save("cytoanvi_model", overwrite=True)
model = CytoANVI.load("cytoanvi_model", adata=adata)
```

## Preprocessing cytometry data

CytoANVI expects **arcsinh-transformed** intensities (not raw counts, not log — cytometry intensities
are continuous). The CytoVI preprocessing utilities live in `scvi.external.cytovi`:

```python
from scvi.external import cytovi

adata = cytovi.read_fcs("sample.fcs")               # FCS -> AnnData (writes a "raw" layer)
cytovi.transform_arcsinh(adata)                     # "raw" -> "transformed" (cofactor via global_scaling_factor=5)
cytovi.scale(adata)                                 # "transformed" -> "scaled" (min-max to [0, 1])

# Combine samples/panels into one object with a batch key. merge_batches works on the "scaled"
# layer and, by default, auto-registers the per-cell missing-marker mask into the "_nan_mask" layer:
adata = cytovi.merge_batches([sample_a, sample_b], batch_key="batch")
```

The preprocessing chain leaves the model-ready values in the `scaled` layer, so point the model at
it with `setup_anndata(..., layer="scaled")` (or copy it to `adata.X`). See the CytoVI user guide
(`docs/user_guide/models/cytovi.md`) for arcsinh cofactor choices and the panel/backbone-marker
conventions. For multi-panel data with markers missing in some batches, use the `_nan_mask` layer
(see [Missing markers](#9-missing-markers--multi-panel-data)).

## Feature guide

### 1. Semi-supervised label transfer

Set up the model with a `labels_key` and the `unlabeled_category` that marks cells without a
ground-truth label. CytoANVI learns from the labeled cells and marginalizes over labels for the
unlabeled ones.

```python
CytoANVI.setup_anndata(adata, labels_key="cell_type", unlabeled_category="Unknown", batch_key="batch")
model = CytoANVI(adata, y_prior="uniform", class_weighting="none")
model.train(max_epochs=1000)

preds = model.predict()                 # np.ndarray of hard labels
proba = model.predict(soft=True)        # DataFrame: per-cell probability for each label
```

- `y_prior="empirical"` sets the label prior to the observed training-label frequencies (use with
  care for query prediction — the reference frequencies may not match the query population).
- `class_weighting="inverse_frequency"` up-weights rare cell types in the classification loss.

### 2. Latent space, normalized expression, and differential expression

```python
z = model.get_latent_representation()               # (n_cells, n_latent) batch-corrected latent
expr = model.get_normalized_expression()            # decoded/denoised protein intensities
de = model.differential_expression(groupby="cell_type", group1=["Tcell"], group2=["Bcell"])
```

The latent `z` is the input for UMAP/clustering/kNN. `get_normalized_expression` and
`differential_expression` are inherited from the CytoVI/scVI machinery and respect the cytometry
likelihood.

### 3. Novelty / OOD status

The former stable TTA estimator and threshold helper are **no-go**. Across three seeds its mean OOD
AUROC was below chance, while a latent-space kNN diagnostic performed well. Stable
`get_uncertainty()` and the indirect TTA replay selector therefore fail closed in 0.2.0.

No replacement novelty workflow is promoted. A future latent-distance/density method must freeze
reference-only calibration, independent biological OOD truth, prevalence, and operating-point
errors before use. The explicitly named experimental TTA surface exists only for historical method
reproduction and is not a supported decision API.

### 4. Warm start from a CytoVI model

If you already trained a CytoVI model, initialize CytoANVI with its encoder/decoder weights instead
of training from scratch.

```python
from cytoanvi import CytoANVI

model = CytoANVI.from_cytovi_model(
    cytovi_model,                      # a trained scvi.external.CYTOVI model
    unlabeled_category="Unknown",
    labels_key="cell_type",
    adata=adata,
)
model.train(max_epochs=400)            # fine-tune the classifier + hierarchy on top of CytoVI
```

### 5. Hierarchical label transfer

Provide a cell-type tree as a dict of `parent -> [children]`, then predict with scores propagated
through the hierarchy. Set the hierarchy before training so the hierarchical cross-entropy (HCE)
objective is used.

```python
edges = {
    "Lymphoid": ["Tcell", "Bcell", "NK"],
    "Tcell": ["CD4T", "CD8T"],
}
model.set_hierarchy(edges)
model.train(max_epochs=1000)

leaf = model.predict_hierarchical(leaf_only=True)          # leaf-level labels
scores = model.predict_hierarchical(soft=True)             # DataFrame of per-node scores
```

You can also **learn** a hierarchy from the data with scHPL/treeArches (requires the
`cytoanvi-hierarchy` extra). `learn_hierarchy` operates on a latent AnnData across batches:

```python
from cytoanvi import hierarchy

# latent_adata: obs has the batch and cell-type columns; X/obsm holds the CytoANVI latent.
edges = hierarchy.learn_hierarchy(
    latent_adata,
    batch_key="batch",
    batch_order=["b0", "b1"],       # order in which batches are merged into the tree
    cell_type_key="cell_type",
)
model.set_hierarchy(edges)
```

Run the tracked engineering fixture (`vignettes/cytoanvi_treearches_synthetic.py`) and use the
`cytoanvi.hierarchy` module (`learn_hierarchy`, `run_tree_arches_pipeline`, `update_hierarchy`) for
the reference→query hierarchy pipeline. The richer tutorial lives in an external uninitialized
gitlink; its repair is `blocked_external_submodule` in the usage-readiness packet and is not part of
the 0.2.0 artifact.

### 6. Reference → query mapping (panel-divergent surgery)

Map a new (query) dataset onto a trained reference model with scArches-style surgery. When the query
panel is missing markers the reference used, `prepare_query_anndata` pads **and** masks them so the
encoder ignores the absent markers.

```python
from cytoanvi import CytoANVI

CytoANVI.prepare_query_anndata(query_adata, reference_model)   # pad + mask missing markers in place
q_model = CytoANVI.load_query_data(query_adata, reference_model)
q_model.train(max_epochs=200)                                  # surgery: update only new parameters

query_adata.obs["pred"] = q_model.predict()
query_adata.obsm["X_cytoanvi"] = q_model.get_latent_representation()
```

### 7. Continual case-control atlas update

Grow a reference atlas with new case/control batches while limiting catastrophic forgetting, using a
modified EWC penalty plus experience replay of reference cells.

```python
q_model = CytoANVI.load_query_data_with_replay(
    query_adata,
    reference_model,
    replay_adata=reference_adata,      # reference cells rehearsed during the update
    control_adata=control_adata,       # required query controls for the control Fisher
    freeze_classifier=True,
)
q_model.train(max_epochs=200, plan_kwargs={"ewc_importance": 1.0})   # tune ewc_importance for your data
```

> **Experimental/no-go for consequential updates.** Replay and controls must be predeclared, and a
> reloaded continual model cannot train until replay is explicitly reconstructed. The EWC
> importances are a diagonal empirical Fisher computed by
> `cytoanvi._continual.fisher_importances`. λ is **not** portable from RNA-domain EWC — tune it for
> a frozen protocol without using the final evaluation set. There is no promoted value.

### 8. Query-mapping QC (mapQC)

Score how well query cells embed into the reference neighborhood structure (requires the
`cytoanvi-mapping-qc` extra, pinned to the only tested compatibility version, mapQC 0.1.1).

```python
joint = model.score_query_mapping(
    reference_adata,
    query_adata,
    sample_key="sample",
    n_nhoods=500,
    k_min=15,
    k_max=100,
)
joint.obs["mapqc_score"]           # per-cell mapping-quality score
```

mapQC is advisory/experimental, not an automatic gate: one benchmark rejected every Nuñez query
despite strong label transfer, and no independent false-accept/false-reject threshold is frozen.

### 9. Missing markers / multi-panel data

When markers are absent in some batches/panels, register a boolean `nan_layer` (True = observed,
False = missing) so the reconstruction loss masks unobserved markers per cell — the model imputes
them rather than treating missing as zero.

`merge_batches` registers this mask automatically; you can also build it explicitly with
`register_nan_layer`. The mask lives in the `_nan_mask` layer by default — pass that name to
`setup_anndata(nan_layer=...)`.

```python
from scvi.external import cytovi

cytovi.register_nan_layer(adata)   # writes the per-cell observed-marker mask into the "_nan_mask" layer
CytoANVI.setup_anndata(
    adata,
    labels_key="cell_type",
    unlabeled_category="Unknown",
    batch_key="batch",
    layer="scaled",                # model-ready values from the preprocessing chain
    nan_layer="_nan_mask",         # per-cell missing-marker mask
)
```

### 10. Save and load

```python
model.save("my_model", overwrite=True)
model = CytoANVI.load("my_model", adata=adata)
```

Save/load round-trips the panel-aware marker mask and hierarchy, so a reloaded core model predicts
identically and is ready for query surgery. Continual replay batches are intentionally not saved;
reconstruct them explicitly before any further continual training.

## Key training parameters

`model.train(...)` (defaults shown):

| Parameter | Default | Notes |
|---|---|---|
| `max_epochs` | `1000` | Publication default; use far fewer (e.g. 20–50) while iterating. |
| `batch_size` | `4096` | Sized for cytometry scale (often ≫100k cells). |
| `lr` | `1e-3` | Adam learning rate. |
| `n_epochs_kl_warmup` | `400` | KL warm-up length (epochs); set `n_steps_kl_warmup` to warm up by steps instead. |
| `early_stopping` | `True` | With `early_stopping_patience=30`. |
| `train_size` | `0.9` | Train/validation split. |
| `accelerator` / `devices` | `"auto"` | GPU used automatically when available. |

Model constructor highlights: `y_prior` (`"uniform"`/`"empirical"`/tensor), `class_weighting`
(`"none"`/`"inverse_frequency"`/`"sqrt_inverse_frequency"`), `protein_likelihood`
(`"normal"` default / `"beta"`), `linear_classifier`, `hierarchy_edges`.

## Optional dependencies

The core model needs no extras. Install a backend only if you use its feature:

```bash
pip install ".[cytoanvi-hierarchy]"     # scHPL / treeArches hierarchy learning
pip install ".[cytoanvi-mapping-qc]"    # mapQC query-mapping QC (score_query_mapping)
pip install ".[cytoanvi-baselines]"     # FlowSOM benchmark baseline
pip install ".[cytoanvi-annbatch]"      # experimental large-data benchmark loader
```

## Performance and scale

- CytoANVI is tuned for cytometry-scale data (hundreds of thousands to millions of cells); the
  default `batch_size=4096` reflects that. A CUDA-capable GPU is strongly recommended for full runs.
- The likelihood is **Normal** over arcsinh-transformed intensities by default (`protein_likelihood`;
  a `"beta"` option exists for (0, 1)-scaled data). CytoANVI deliberately does **not** use count
  likelihoods (NB/ZINB) — those are inappropriate for antibody intensity data.
- For debugging NaNs you can enable per-step finite-value checks; for maximum throughput they can be
  disabled with `CYTOANVI_DISABLE_FINITE_CHECKS=1` (the numeric variance clamp always stays on).

## Limitations and status

CytoANVI is a research candidate with capability-specific status. Core label transfer, latent,
save/load, and mapping remain conditional on an independently annotated target holdout. Hierarchy
and integration are exploratory. Stable TTA novelty is no-go; continual update and automatic mapQC
gating are not usage-ready. The exact 0.2.0 wheel is not engineering-accepted until the isolated
receipt passes. See [the authoritative matrix](docs/usage_readiness.md); do not co-install with
`scvi-tools`, publish, or release based on source tests alone.

## Citation

If you use CytoANVI, please cite:

> **CytoANVI: annotation-aware variational inference for antibody-based single-cell cytometry.**
> Manurung et al. Manuscript in preparation (2026).

CytoANVI builds on CytoVI and scANVI, and is a fork of scvi-tools. Please also cite the scvi-tools
and scverse publications:

> **A Python library for probabilistic analysis of single-cell omics data.** Gayoso, Lopez, Xing,
> et al. _Nature Biotechnology_ 2022. doi:
> [10.1038/s41587-021-01206-w](https://doi.org/10.1038/s41587-021-01206-w).

> **The scverse project provides a computational ecosystem for single-cell omics data analysis.**
> Virshup, Bredikhin, Heumos, et al. _Nature Biotechnology_ 2023. doi:
> [10.1038/s41587-023-01733-8](https://doi.org/10.1038/s41587-023-01733-8).

## License

BSD 3-Clause. This fork retains the upstream scvi-tools copyright and adds the CytoANVI copyright:

```
Copyright (c) 2020, The scvi-tools development team
Copyright (c) 2026, CytoANVI contributors (Mikhael Manurung)
```

scvi-tools is part of the scverse® project ([website](https://scverse.org)). CytoANVI is an
independent fork and is not endorsed by the scverse project.
