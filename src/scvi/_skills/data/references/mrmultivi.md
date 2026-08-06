# MrMultiVI

`scvi.external.MrMultiVI` — sample-aware MultiVI for RNA + protein (and optionally
ATAC) multi-modal data with a two-level latent hierarchy. Takes **MuData**, not AnnData.

## Minimal working example

```python
import mudata as mu
import anndata as ad
import numpy as np
import scvi
from scvi.external import MrMultiVI

scvi.settings.seed = 0

# Build MuData from an AnnData that has adata.obsm["protein"]
rna = adata.copy()
prot = ad.AnnData(
    X=np.asarray(adata.obsm["protein"]).astype(np.float32),
    obs=adata.obs[["batch", "donor", "celltype"]].copy(),
)
prot.var_names = list(adata.obsm["protein"].columns)
mdata = mu.MuData({"rna": rna, "protein": prot})
for col in ["batch", "donor", "celltype"]:
    mdata.obs[col] = adata.obs[col].astype(str).to_numpy()

MrMultiVI.setup_mudata(
    mdata,
    sample_key="donor",
    batch_key="batch",
    labels_key="celltype",
    rna_layer="counts",
    modalities={
        "rna_layer": "rna",
        "protein_layer": "protein",
        "batch_key": "rna",
        "labels_key": None,
    },
)
model = MrMultiVI(
    mdata,
    sample_key="donor",
    n_genes=adata.n_vars,   # required — number of gene features
    n_regions=0,             # required — set 0 if no ATAC
    n_latent=20,
    n_hidden=256,
)
model.train(
    max_epochs=400,
    early_stopping=True,
    batch_size=512,
    accelerator="gpu",
    devices=1,
)
u = model.get_latent_representation(give_z=False)  # donor-space
z = model.get_latent_representation(give_z=True)   # cell-space
model.save("path/to/model", overwrite=True)
```

---

## `__init__` key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `sample_key` | — | required; must match the key used in `setup_mudata` |
| `n_genes` | — | required; pass `adata.n_vars` (or the RNA modality's n_vars) |
| `n_regions` | — | required; pass `0` if no ATAC modality |
| `n_latent` | 20 | cell-level latent dim |
| `n_latent_u` | None | donor-level latent dim; defaults to `n_latent` |
| `n_latent_sample` | 16 | sample embedding dim |
| `u_prior` | `"mog"` | `"mog"` or `"vamp"` |
| `protein_in_encoder` | True | if True, proteins widen the encoder input; set False to isolate gene-only VampPrior pseudo-input shape |
| `modality_weights` | `"equal"` | `"equal"` or `"cell"` (per-cell learned weights) |

---

## Saving and loading with MuData

```python
model.save("path/to/model", overwrite=True)
# Reload: pass the same mdata (or a compatible one) as adata=
loaded = MrMultiVI.load("path/to/model", adata=mdata)
```

---

## Footguns

1. **Takes MuData, not AnnData** — passing `AnnData` to `MrMultiVI.__init__`
   raises `TypeError` with the message "MrMultiVI requires a MuData object."

2. **`n_genes` and `n_regions` are required** — they are not inferred from the
   MuData. Omitting them raises the upstream `AssertionError:
   "n_genes and n_regions must be provided if using AnnData"`.

3. **`modalities` cannot be None** — `setup_mudata` raises `ValueError` immediately.

4. **`modalities["batch_key"]` is a modality name, not the batch column name** —
   it tells the registry which modality's `.obs` holds the `batch_key` column
   (typically `"rna"`).

5. **`labels_key=None` in modalities** — means the labels column is looked up in
   global `mdata.obs`, not in a specific modality. This is correct for most setups.

6. **Modality order** — `setup_mudata` reorders to rna → protein internally, but
   it is safest to build the MuData in that order (`mu.MuData({"rna":..., "protein":...})`).

7. **`differential_accessibility` is a stub** — it raises `NotImplementedError`.
   Call `differential_abundance` instead for sample-level DA.

8. **`protein_in_encoder=True` (default) changes pseudo-input shape** — if using
   VampPrior, the pseudo-inputs span the combined gene+protein space. If you need
   pure gene-latent pseudo-inputs (e.g., for unit-test assertions on pseudo-input
   shape), pass `protein_in_encoder=False`.

9. **`give_z=False` returns u (donor-space), `give_z=True` returns z (cell-space)**
   — same convention as MrTotalVI; the parameter name is the same.
