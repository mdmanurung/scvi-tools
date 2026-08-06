---
name: cytoanvi
description: Use for any task involving scvi-tools, MrTotalVI, MrMultiVI, or CytoANVI —
  sample-aware variational inference for single-cell CITE-seq, cytometry, and multi-modal
  omics data. Triggers on setup_anndata, setup_mudata, differential_abundance, get_latent_representation,
  get_local_sample_representation, donor-timepoint DA, VampPrior, freeze_prior_after_init,
  LayerNorm vs BatchNorm choice, surgery/query transfer, continual updates, or any
  MrTotalVI/MrMultiVI/CytoANVI model construction, training, or inference task.
  This is a router skill — read the relevant file under references/ for details and
  footguns before writing code.
---

# cytoanvi / scvi-tools

This repository is a fork of [scvi-tools](https://scvi-tools.org/) extended with
three sample-aware models for multi-donor single-cell omics:

| Model | Data type | Key class |
|-------|-----------|-----------|
| **MrTotalVI** | CITE-seq (genes + proteins), AnnData | `scvi.external.MrTotalVI` |
| **MrMultiVI** | RNA + protein (no ATAC), MuData | `scvi.external.MrMultiVI` |
| **CytoANVI** | Cytometry proteins only | `cytoanvi.CytoANVI` |

All three share the **M1+M2 hierarchy**: a cell-level latent `z` and a
donor-level latent `u`, where `u` is drawn from a mixture-of-Gaussians (MoG)
or VampPrior and informs `z` via a learned additive offset.

This skill is a **router**. Each topic below maps to a reference file with
full API signatures, concrete examples, and footguns.

---

## Task → reference file

| Task | Reference |
|------|-----------|
| Register AnnData or MuData, set up layers/obs keys | `references/setup-patterns.md` |
| Construct and train MrTotalVI | `references/mrtotalvi.md` |
| Construct and train MrMultiVI | `references/mrmultivi.md` |
| Construct and train CytoANVI, surgery/query, continual updates | `references/cytoanvi.md` |
| Run differential abundance (`differential_abundance`) | `references/differential-abundance.md` |
| Get latent embeddings, local sample distances | `references/latent-representations.md` |

---

## Cross-cutting concepts (read these first if unsure)

- **sample_key vs batch_key** — `sample_key` identifies the donor/sample (one row in the
  per-donor embedding table); `batch_key` is the technical batch/plate. They are different
  columns in `adata.obs`. Never pass the same key for both unless donors and batches coincide.

- **give_z vs give_u** — `get_latent_representation(give_z=False)` returns `u` (donor-space,
  shape `(n_cells, n_latent_u)`); `give_z=True` returns `z` (cell-space,
  shape `(n_cells, n_latent)`). For integration and UMAP, use `z`. For DA, use `u`.

- **setup before __init__** — `setup_anndata`/`setup_mudata` must always be called as a
  classmethod **before** constructing the model. Calling `__init__` first raises an error.

- **Upstream base classes are unmodified** — `TOTALVI`, `MULTIVI` signatures are preserved.
  MrTotalVI/MrMultiVI add kwargs via `**model_kwargs` pass-through. Never assume a kwarg
  added to the Mr* model is available on the upstream class directly.
