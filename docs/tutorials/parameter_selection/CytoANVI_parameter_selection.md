---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# CytoANVI: choosing parameters

This tutorial is about **how to pick values**, not about what the arguments mean. For the API
surface see {doc}`/user_guide/models/cytoanvi`; tracked runnable engineering workflows live at
`vignettes/cytoanvi_example_reference_query.py` and
`vignettes/cytoanvi_treearches_synthetic.py`.

The {doc}`/usage_readiness` capability table is authoritative. This guide cannot promote a
capability that remains no-go or blocked there.

Most CytoANVI defaults were chosen for a reason, but the reasons differ enormously in strength.
Some are backed by multi-seed benchmarks on full cohorts; others are conventions inherited from the
source paper that have never been tuned here; a few are plain assertions with nothing behind them.
Treating those as equivalent is how people end up over-trusting a default. So every recommendation
below carries an explicit evidence tier.

```{admonition} Evidence tiers used in this tutorial
:class: important

| Tier | What it means |
|---|---|
| **Benchmark-backed** | A finding with numbers and ≥3 seeds behind it. Trust the direction and roughly the magnitude. |
| **Convention** | Held fixed across all benchmarks so results stay comparable. Never actually swept — a reasonable starting point, not an optimum. |
| **Assertion** | Stated in a docstring, with no experiment behind it. Treat as a hypothesis. |
| **Contested / negative** | Evidence is missing, unrecoverable, or points the other way. Do not rely on it. |
```

+++

## Setup

Synthetic data throughout, so this page is self-contained. For real flow or mass cytometry data,
apply the arcsinh + min-max preprocessing described in {doc}`/user_guide/models/cytoanvi` before
`setup_anndata`.

```{code-cell} ipython3
import numpy as np
import scvi
from cytoanvi import CytoANVI

adata = scvi.data.synthetic_iid()
adata.layers["scaled"] = adata.X.copy()
adata.obs["celltype"] = adata.obs["labels"].astype(str)
# hold out some labels so the semi-supervised path is exercised
rng = np.random.default_rng(0)
unlabeled_idx = rng.choice(adata.n_obs, size=adata.n_obs // 3, replace=False)
adata.obs.loc[adata.obs_names[unlabeled_idx], "celltype"] = "Unknown"

CytoANVI.setup_anndata(
    adata,
    layer="scaled",
    labels_key="celltype",
    unlabeled_category="Unknown",
    batch_key="batch",
)
```

+++

## Capacity: `n_latent`, `n_hidden`, `n_layers`

**Tier: Convention.** No sweep of these exists for CytoANVI. `n_latent=None` triggers a heuristic
on the number of input features; `n_hidden=128` and `n_layers=1` are inherited defaults.

Cytometry panels are small — typically 20–50 markers, versus thousands of genes in scRNA-seq — so
the usual scRNA-seq intuitions do not transfer. A latent dimension approaching the number of
markers leaves the model free to learn something close to an identity map.

```{code-cell} ipython3
model = CytoANVI(adata, n_latent=10)
print("markers:", adata.n_vars, "| latent:", model.module.n_latent)
```

Start with the heuristic. Only raise `n_layers` if the reconstruction is clearly underfitting; the
depth mostly buys you nothing on a 30-marker panel and costs training stability.

+++

## `batch_size`: the one training default with a hard reason

**Tier: Benchmark-backed (as a floor, not an optimum).** `train()` defaults to `batch_size=4096`,
which is far larger than typical scRNA-seq settings. This is deliberate: cytometry cohorts run to
10⁵–10⁶ cells, and **small batches caused NaN divergence** during development.

Lower it only if you are memory-constrained, and watch the loss for NaNs if you do.

`max_epochs=1000` is **Convention** — fixed across the benchmark suite because 100 epochs was
demonstrably insufficient for scIB metrics to converge. It is a floor that was shown to be enough,
not a tuned value. Early stopping (on by default, patience 30) usually stops well short of it.

+++

## `classification_ratio`: the integration ↔ label-transfer dial

**Tier: Assertion (direction only).** Default 50, passed through `plan_kwargs`. It weights the
classifier loss against the ELBO.

The documented direction — higher favours label-transfer accuracy, lower favours batch mixing — is
the B2 tradeoff, and it is directionally sound. What does **not** exist is any sweep establishing
*how much* to move it for a given effect. If you change it, treat it as an experiment on your own
data with your own held-out labels, not as following a recommendation.

```{code-cell} ipython3
# The tradeoff dial. There is no benchmark-backed value other than the default.
model.train(max_epochs=2, plan_kwargs={"classification_ratio": 50}, accelerator="cpu")
```

+++

## `y_prior` and `class_weighting`

**Tier: Assertion.** "Use `y_prior='empirical'` for imbalanced panels" appears in the docstring and
is plausible, but no benchmark in this repository supports it. The same is true of
`class_weighting='inverse_frequency'` and `class_weight_clip=10.0`.

These are reasonable things to try on an imbalanced panel. They are not validated defaults, and
this tutorial will not pretend otherwise. In 0.2.0 empirical priors and class weights are resolved
only from the actual training split and that boundary is persisted; held-out labels cannot affect
them. Measure per-class recall on held-out labels after fitting; do not tune from those labels.

+++

## Continual update and `ewc_importance` (λ): no promoted recipe

**Tier: Benchmark-backed (as a negative result about portability).**

```{admonition} λ does not transfer between datasets or codebases
:class: warning

The source paper's λ=100 was derived for scANVI on RNA counts. CytoVI's arcsinh-intensity
likelihood puts the Fisher information on a completely different scale, so that value carries no
meaning here. Worse, the Fisher computation in this codebase *changed*: an earlier
batch-mean-gradient `(E[grad])²` approximation was replaced by an exact per-sample Fisher
(`per_sample=True`, now the default), which shifts the absolute scale again.

**Consequence:** any λ copied from the paper, from an older run, or from a different dataset is
meaningless. More importantly, continual updating is no-go without an explicitly predeclared
reference replay set and an external matched control. A replay-less reload refuses before trainer
construction; uncertainty-selected replay is not a stable path. This guide does not recommend a λ
or a sweep as a substitute for the missing P2 protocol.
```

+++

## Uncertainty / novelty detection — do not tune this, do not use it

**Tier: Contested / negative.** This is the most important section on the page.

The historical experimental TTA Bregman-Information score had a **mean AUROC near 0.484 — below
chance** on the full Roider cohort across three seeds. Legacy summaries disagree on the reported
dispersion, so no spread estimate is treated as sealed evidence. Stable `get_uncertainty()` and the
indirect uncertainty replay selector therefore fail closed in 0.2.0.

The diagnostic matters: a kNN out-of-distribution score computed *in CytoANVI's own latent space*
performed better in the same retrospective analysis. That comparator is not independently reviewed
or promoted; it only localizes the historical TTA failure and remains an unfrozen P2 candidate.

```{admonition} There is no parameter that fixes this
:class: danger

`tta_rep` (default 50) only trades variance for compute — more repetitions of a broken estimator
give you a more precise broken estimate. `mask_percentage` (0.5, the paper's convention) is **not
exposed** on any public method and has never been swept.

Do not treat novelty detection as a tuning problem. Latent-space kNN is a useful preregistered P2
comparator, but it is not a promoted automatic acceptance rule in this release.
```

```{code-cell} ipython3
# Research comparator only: kNN distance in the CytoANVI latent, not stable package TTA.
latent = model.get_latent_representation()
print("latent shape:", latent.shape)
# Build a reference kNN index on known cells, then score query cells by distance to it.
```

+++

## Hierarchical cross-entropy (`hierarchy_edges` / `reachability_matrix`)

**Tier: Convention.** HCE helps when your label set is genuinely hierarchical and confusions are
concentrated between siblings — mislabelling a CD4 T cell as a CD8 T cell is cheaper than calling
it a B cell, and HCE encodes that. It does nothing for a flat, well-separated panel.

The two arguments are mutually exclusive: pass edges *or* a precomputed matrix, not both.

+++

## Batch norm vs layer norm — an inconsistency worth knowing

CytoANVI's module defaults to `use_batch_norm="both"`, `use_layer_norm="none"`. MrTotalVI defaults
to the **opposite** (`"none"` / `"both"`), on the rationale that layer normalisation avoids
confounding donor effects with batch statistics.

No document in this repository explains why the two models diverge here, and no experiment compares
them. Flagged so you know it is an unexamined inconsistency rather than a considered difference.

+++

## Summary

| Parameter | Default | Tier | Guidance |
|---|---|---|---|
| `n_latent` | `None` (heuristic) | Convention | Small panels ⇒ small latent; start at the heuristic |
| `n_hidden` / `n_layers` | 128 / 1 | Convention | Rarely worth changing on a 20–50 marker panel |
| `batch_size` | 4096 | Benchmark-backed floor | Large on purpose — small batches diverged to NaN |
| `max_epochs` | 1000 | Convention | A sufficiency floor; early stopping usually fires first |
| `classification_ratio` | 50 | Assertion (direction only) | Higher = label transfer, lower = batch mixing; magnitude unvalidated |
| `y_prior`, `class_weighting` | `uniform`, `none` | Assertion | Plausible for imbalance; unvalidated. Measure per-class recall |
| `ewc_importance` | — | Benchmark-backed (non-portability) | No promoted value; replay plus external control are mandatory before P2 use |
| `tta_rep`, `mask_percentage` | Experimental only | Contested / negative | Stable TTA refuses; historical method is below chance (AUROC 0.484) |
| HCE hierarchy | `None` | Convention | Only for genuinely hierarchical labels |

**The general rule:** the further down this table you go, the less the default is protecting you.
Anything marked *Assertion* or *Contested* deserves an experiment on your own data before you rely
on it in a published claim.
