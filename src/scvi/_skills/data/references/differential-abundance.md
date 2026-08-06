# Differential Abundance

`model.differential_abundance()` — sample-level Bayesian DA test for
MrTotalVI and MrMultiVI. Returns per-cell enrichment log-probabilities
across sample covariates (e.g., timepoint).

---

## DTP setup — the donor_timepoint pattern

DA requires each sample to map to **exactly one** covariate value. When the
covariate is `timepoint`, each donor appears multiple times (W00, W22, ...),
so using `sample_key="donor"` breaks the one-sample-one-timepoint requirement.
The fix is to create a `donor_timepoint` column and use it as the sample key.

```python
# Build the donor_timepoint column
adata.obs["donor_timepoint"] = (
    adata.obs["donor"].astype(str) + "_" + adata.obs["timepoint"].astype(str)
)

# Optional: restrict to timepoints with known group membership
# (e.g., exclude "W06" if it's transitional and not assigned to a group)
modeling_keep = adata.obs["timepoint"].isin(["W00", "W22"])
adata_model = adata[modeling_keep].copy()

# Train with donor_timepoint as sample_key
MrTotalVI.setup_anndata(
    adata_model,
    layer="counts",
    protein_expression_obsm_key="protein",
    sample_key="donor_timepoint",     # <-- DTP key
    batch_key="batch",
    labels_key="celltype",
)
model = MrTotalVI(adata_model, sample_key="donor_timepoint", ...)
model.train(...)
```

---

## Running differential_abundance

```python
da = model.differential_abundance(
    sample_cov_keys=["timepoint"],    # list of obs columns that are sample covariates
    donor_key="donor",                # obs column identifying the donor (not the DTP key)
    n_mc_samples=250,                 # MC samples for posterior estimation (default 250)
    batch_size=512,
)
```

`da` is an `xarray.Dataset`. Key data variable: `log_probs`.

---

## Extracting W22 enrichment

```python
import numpy as np

# log_probs shape: (n_cells, n_timepoints)
tp_lp = da["log_probs"]             # xarray DataArray

# Per-cell enrichment score: W22 vs W00 log-probability difference
enrichment = tp_lp.sel(timepoint="W22") - tp_lp.sel(timepoint="W00")

# Summary across cells
mean_enrich = float(enrichment.mean())
std_enrich  = float(enrichment.std())

# Collect across seeds for stability assessment
# seed 0: mean_enrich = X.XXX
# seed 1: mean_enrich = Y.YYY
# seed 2: mean_enrich = Z.ZZZ
cross_seed_std = np.std([X, Y, Z])  # substitute actual values
```

---

## Accessing the full xarray Dataset

```python
da                        # xarray.Dataset
da.data_vars              # lists available variables
da["log_probs"]           # DataArray: (n_cells, n_covariates)
da["log_probs"].dims      # ('cells', 'timepoint') or similar
da["log_probs"].coords    # coordinate labels for each dim

# To convert to numpy:
lp_np = da["log_probs"].values   # shape (n_cells, n_timepoints)

# To convert to pandas DataFrame (one column per timepoint):
lp_df = da["log_probs"].to_pandas()
```

---

## Pre-registered success criterion (D-041, VampPrior validation)

For MrTotalVI-LN + VampPrior + `freeze_prior_after_init=True`:

- **Pass**: W22-enrichment std ≤ 0.30 across 3 seeds AND all 3 seeds positive
- **Fail**: std > 0.30 OR any seed negative → VampPrior does not rescue LN DA
- Baseline (MrTotalVI-LN MoG): std = 0.875, mean = 1.821
- See `.living/decisions.md` D-041 for full rationale

---

## Footguns

1. **`sample_key="donor"` breaks multi-timepoint DA** — if a donor has cells at
   W00 AND W22, both map to the same sample embedding. The `timepoint` covariate
   then has no variance within a sample, so `differential_abundance` returns all
   zeros or NaN. Use `donor_timepoint` as the sample key.

2. **`donor_key` must be the raw donor column, not the DTP column** —
   `donor_key="donor_timepoint"` causes the test to treat each
   donor-timepoint pair as a separate donor, inflating the denominator.

3. **`sample_cov_keys` must be present in `adata.obs`** — the key must already
   exist in the obs used at `setup_anndata` time. It is not inferred from
   the DTP column; you must keep a separate `timepoint` column.

4. **`n_mc_samples` ≥ 100 for publication** — `n_mc_samples=10` is fine for
   smoke tests but produces unstable estimates. Use ≥ 250 for any reported result.

5. **Result is xarray, not pandas** — many analysis scripts expect `.to_pandas()`
   or `.values`; don't try to index with `da["log_probs"]["W22"]` (use `.sel()`).

6. **Jensen-gap bias correction** — `differential_abundance` applies an
   `n_mc_samples` dependent Jensen-gap correction to reduce bias in the ELBO
   estimator. Do not average the per-sample ELBO across seeds before correcting;
   each seed's result is already corrected independently.

7. **Zero-cell donor-timepoint combinations** — any `donor_timepoint` value that
   has no cells in the modeling AnnData will raise during the aggregation step.
   Filter `adata_model` to cells that have the DTP key before training.
