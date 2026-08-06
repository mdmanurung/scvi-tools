# Setup patterns — `setup_anndata` and `setup_mudata`

## MrTotalVI — `setup_anndata`

```python
from scvi.external import MrTotalVI

MrTotalVI.setup_anndata(
    adata,
    layer="counts",                      # raw integer count layer (required)
    protein_expression_obsm_key="protein", # obsm key for protein matrix
    sample_key="donor",                  # or "donor_timepoint" for valid DA
    batch_key="batch",                   # technical batch (plate/run) — may be None
    labels_key="celltype",               # optional; enables label-conditioned u-prior
)
```

**Footguns**

1. `layer=` must point to **raw integer counts**, not log-normalised values. Pass
   `layer=None` only if `adata.X` itself holds raw counts.
2. `protein_expression_obsm_key` must be a key in `adata.obsm`, not `adata.layers`.
   MrTotalVI reads proteins from `obsm`, not a layer.
3. `sample_key` must already be a column in `adata.obs` before calling this. The
   categorical mapping is frozen at setup time; adding new donors to the AnnData
   later (surgery/query) requires `prepare_query_anndata` instead.
4. Every unique value of `sample_key` must have ≥1 cell. Zero-cell donors cause a
   `ValueError` inside `MrTotalVI.__init__` with a message listing the missing indices.
5. `labels_key` is optional; omit it to train without label conditioning.

---

## MrMultiVI — `setup_mudata`

MrMultiVI takes a `MuData` object (not `AnnData`). Build it first:

```python
import mudata as mu
import anndata as ad
import numpy as np
from scvi.external import MrMultiVI

rna  = adata.copy()                          # genes
prot = ad.AnnData(
    X=np.asarray(adata.obsm["protein"]).astype(np.float32),
    obs=adata.obs[["batch", "donor", "celltype"]].copy(),
)
prot.var_names = list(adata.obsm["protein"].columns)
mdata = mu.MuData({"rna": rna, "protein": prot})
# propagate obs columns to global mdata.obs:
for col in ["batch", "donor", "celltype", "donor_timepoint"]:
    if col in adata.obs.columns:
        mdata.obs[col] = adata.obs[col].astype(str).to_numpy()

MrMultiVI.setup_mudata(
    mdata,
    sample_key="donor",          # or "donor_timepoint"
    batch_key="batch",
    labels_key="celltype",
    rna_layer="counts",          # layer key inside mdata["rna"]
    modalities={
        "rna_layer": "rna",      # modality name for RNA
        "protein_layer": "protein", # modality name for protein
        "batch_key": "rna",      # which modality's obs holds batch_key
        "labels_key": None,      # None → global mdata.obs
    },
)
model = MrMultiVI(
    mdata,
    sample_key="donor",
    n_genes=adata.n_vars,        # number of gene features (required)
    n_regions=0,                 # set 0 if no ATAC modality
    n_latent=20,
    n_hidden=256,
)
```

**Footguns**

1. `modalities` cannot be `None` — it raises `ValueError` immediately.
2. `n_genes` and `n_regions` must be passed explicitly to `MrMultiVI.__init__`;
   they are not inferred from the MuData automatically.
3. The `modalities` dict keys (`rna_layer`, `protein_layer`, `batch_key`, etc.)
   must match the exact modality names you used when building the MuData
   (`mu.MuData({"rna": ..., "protein": ...})`).
4. `batch_key` in `modalities` points to the **modality name** whose `.obs` holds
   the batch column (usually `"rna"`), not the batch column name itself.
5. Modality order in the MuData matters: `setup_mudata` reorders to rna → atac →
   protein internally, but it is safest to build the MuData in that order.
6. Passing an `AnnData` to `MrMultiVI.__init__` raises `TypeError` with a clear
   message directing you to `setup_mudata`.

---

## CytoANVI — `setup_anndata`

CytoANVI operates on the **protein** (antibody) layer only, not genes.

```python
from cytoanvi import CytoANVI

CytoANVI.setup_anndata(
    adata,
    labels_key="celltype",       # required; column with cell-type labels
    unlabeled_category="unknown", # value used to mark unlabeled cells
    layer=None,                  # None → use adata.X (pre-transformed protein values)
    batch_key="batch",
    sample_key="donor",
    nan_layer="_nan_mask",       # optional binary mask for missing antibody panels
)
```

**Footguns**

1. `labels_key` and `unlabeled_category` are **required** (positional-like) — CytoANVI
   is always semi-supervised; fully supervised training is not supported.
2. `unlabeled_category` is remapped to the **last integer code** internally. Cells with
   this label value contribute to reconstruction loss but not the classifier loss.
3. `layer=None` means `adata.X` — CytoANVI expects arcsinh- or logicle-transformed
   protein values, **not** raw counts. Do not pass a raw count layer.
4. `nan_layer` enables missing-marker imputation for overlapping panels. If a layer
   named `"_nan_mask"` exists in `adata.layers` but `nan_layer` is omitted, a
   `UserWarning` is issued and the layer is auto-registered.
5. `sample_key` is optional for CytoANVI but required for surgery/query transfer.
