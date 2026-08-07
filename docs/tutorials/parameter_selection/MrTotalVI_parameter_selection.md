---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# MrTotalVI: choosing parameters

MrTotalVI extends TOTALVI with an MrVI-style two-level latent space: a sample-conditioned base `u`
in the legacy default, and
`z = z_base(u) + eps` where `eps` carries per-donor residual structure. This tutorial is about
**how to choose values** for its parameters. For what each argument means, see
{doc}`/user_guide/models/mr_multimodal`.

The {doc}`/usage_readiness` matrix is authoritative. No prior, DA/DE inference, streaming, or
new-sample inference capability is promoted by this parameter guide.

Several MrTotalVI defaults rest on weaker evidence than their status as defaults implies — one of
them on an analysis whose source file no longer exists. Rather than presenting all defaults as
settled, every recommendation below is tagged with an evidence tier.

```{admonition} Evidence tiers used in this tutorial
:class: important

| Tier | What it means |
|---|---|
| **Benchmark-backed** | A finding with numbers and ≥3 seeds behind it |
| **Convention** | Held fixed across benchmarks for comparability; never swept |
| **Assertion** | Docstring claim with no experiment behind it |
| **Contested / negative** | Evidence missing, unrecoverable, or pointing the other way |
```

+++

## Setup

```{code-cell} ipython3
import numpy as np
import scvi
from scvi.external import MrTotalVI

adata = scvi.data.synthetic_iid()
adata.obs["sample"] = np.array([f"donor_{i % 4}" for i in range(adata.n_obs)])

MrTotalVI.setup_anndata(
    adata,
    protein_expression_obsm_key="protein_expression",
    sample_key="sample",
    batch_key="batch",
)
model = MrTotalVI(
    adata,
    sample_key="sample",
    n_latent=20,
    n_latent_u=10,
    u_prior_supervision="none",
)
```

RNA and protein inputs must be finite, non-negative, integer-like raw counts. Setup validates the
complete registered matrices, not a sampled prefix.

+++

## `labels_key` is metadata unless supervision is explicit

```{admonition} Labels alone do not change the objective
:class: important

In 0.2.0, passing `labels_key` to `setup_anndata` records annotations but leaves the model
unsupervised. The new-call default is `u_prior_supervision="none"` with
`u_prior_label_weight=0.0`.

To opt in, set `u_prior_supervision="labels"` and a finite positive
`u_prior_label_weight`. This requires registered labels and must be disclosed in comparisons.

Old checkpoints are migrated explicitly; contradictory legacy/new supervision fields fail closed.
```

+++

## `u_prior`: the default is contested

**Tier: Contested.** The resolved choices are exactly `"standard"`, `"mog"`, and `"vamp"`.
`"mog"` remains the default for compatibility, not because it is scientifically preferred.

The analysis that established `"mog"` as the default compared cluster dispersion on a 57k-cell
T/NK subset. Two problems:

1. **The source analysis is unrecoverable.** No surviving file exists anywhere in this project.
2. **It measured the wrong geometry.** It compared UMAP/centroid dispersion — but Leiden clustering
   operates on the kNN graph, not on UMAP coordinates. On graph-native metrics the ranking
   *reverses*: VampPrior `u` beats MoG `u` on donor mixing (0.225 vs 0.273) and on cross-seed
   stability (Jaccard 0.189 vs 0.145), at equal cell-type purity.

Those historical results are insufficient to recommend any prior: the source analysis is
unrecoverable, the geometry was non-authoritative, and the comparison was not validated across
independent cohorts. Prespecify the prior and sensitivity arms before looking at outcomes.

```{code-cell} ipython3
# Experimental sensitivity arm; not a preferred recipe.
model_vamp = MrTotalVI(
    adata,
    sample_key="sample",
    n_latent=20,
    n_latent_u=10,
    u_prior="vamp",
    init_prior_from_data=True,
    freeze_prior_after_init=True,
)
```

```{admonition} freeze_prior_after_init only makes sense with vamp
:class: warning

Vamp data initialization is restricted to the frozen training indices. The seed and ordered
training-index digest are persisted so validation/test cells cannot affect the pseudoinputs.
Freezing a randomly initialized MoG prior is not a validated recommendation.
```

+++

## Differential abundance: route around MrTotalVI

**Tier: Benchmark-backed (negative).**

```{admonition} MrTotalVI DA is not ready for use
:class: danger

Across 3 seeds on the same cohort, MrTotalVI's DA enrichment came out at **+1.12 ± 9.46** — the
standard deviation is roughly nine times the mean, with individual seeds at +2.74, −9.04, and
+9.67. The sign of the result changes with the seed.

MrMultiVI on the identical setup gave **+0.956 ± 0.126**, which is stable.

The VampPrior + frozen-prior configuration is the only mitigation candidate, and while the
underlying measurement is real (std 0.192), it was produced from a dirty checkout without
data/environment/config hashes and is **refuted for promotion** under this project's reproduction
standard. There is currently **no validated fix.**

Treat package DA as descriptive only. For biological DA, use a prespecified, replicate-aware
compositional method with donor-level uncertainty outside this API.
```

+++

## Differential expression: public refusal

**Tier: Benchmark-backed (negative).**

eps-space DE was compared against a stratified pseudobulk gold standard across all 12 cell types.
Every cell type gave a **negative** Spearman correlation (MrTotalVI: −0.126 to −0.008). Direction
of interferon-gene changes was called correctly 19.9% of the time — below the 50% you would get by
guessing. No cell type rescues it, including the one with the strongest pseudobulk signal.

Critically, **the reference MRVI implementation fails the same way** (ρ = −0.138). This is a
property of eps-space DE as a method, not a defect introduced here.

```{admonition} Not a "use with caution"
:class: danger

These historical outputs are anti-concordant with pseudobulk ground truth. MrTotalVI therefore
fails closed for public `differential_expression()` in both legacy and centered-v2 modes. Aggregate
raw counts at the donor-by-condition level and use PyDESeq2, edgeR, or dreamlet; there is no public
escape flag for latent p-values or decoded LFC.
```

+++

## Operational no-go boundaries

`use_vmap=True` is rejected before inference/statistics. `MrTotalVIBatchDataModule` is not a stable
public training export, and this release does not support end-to-end streaming training or
inference for biological samples absent from the registered training set.

+++

## Latent sizing: `n_latent`, `n_latent_u`, `n_latent_sample`

**Tier: Convention.** `n_latent=20`, `n_latent_sample=16`, `n_latent_u=None` (isomorphic with `z`).
Every benchmark in this project holds `n_latent=20` fixed for comparability. It was never swept, so
it is a starting point rather than an optimum.

The one structural fact worth knowing: with `n_latent_u=None`, `u` and `z` are isomorphic, so
`z = u + eps` exactly. Measured on real data, `eps` carries only 2–17% of `z`'s variance, and
`rank(z) ≈ rank(u) + 1`. Setting `n_latent_u` smaller than `n_latent` forces a genuine bottleneck
into `u` and is the more interesting configuration if you want `u` to be meaningfully compressed.

+++

## `batch_representation`

**Tier: Convention.** `"one-hot"` by default; `"embedding"` learns a dense batch embedding instead.
Prefer `"embedding"` when you have many batches, where one-hot becomes wide and sparse. Set the
width with `batch_embedding_kwargs={"embedding_dim": 5}`. No sweep establishes a recommended
`embedding_dim`; the default follows the upstream convention.

+++

## Parameters with no guidance at all

These are exposed and have defaults, but nothing in this project informs how to choose them:
`kl_u_weight`, `kl_z_weight` (both 1.0), `z_u_prior_scale`, `learn_z_u_prior_scale`,
`scale_observations`, `u_prior_mixture_k` (20), `qz_kwargs`/`qu_kwargs`. Treat any change as an
experiment. `use_batch_norm="none"` / `use_layer_norm="both"` is justified by one sentence — layer
norm avoids confounding donor effects with batch statistics — and note that CytoANVI uses the
**opposite** convention, unexplained anywhere.

+++

## A methodological note before you compare configurations

```{admonition} Do not compare embeddings with a statistic that encodes the effect under test
:class: important

This project drew four separate false conclusions from this exact error — a variance-share
statistic whose denominator is what compression changes, a cluster F1 at a single Leiden
resolution, and a kNN purity that trivially contained the quantity being tested.

Before believing any comparison between two parameter settings: check whether the denominator moves
with the effect, sweep the operating point rather than fixing one, and pair every variance-share
metric with a denominator-free counterpart (rank- or recovery-based).
```

+++

## Summary

| Parameter | Default | Tier | Guidance |
|---|---|---|---|
| `labels_key` / `u_prior_supervision` | `None` / `"none"` | Explicit contract | Labels remain metadata unless supervision is explicitly enabled with a positive weight |
| `u_prior` | `"mog"` | Contested | Compatibility default only; no prior is scientifically recommended |
| `init_prior_from_data` + `freeze_prior_after_init` | `False`, `False` | Experimental | Vamp initialization uses training indices only; treat as a prespecified sensitivity arm |
| `n_latent` / `n_latent_u` | 20 / `None` | Convention | Fixed for comparability, never swept. `n_latent_u < n_latent` for a real bottleneck |
| `batch_representation` | `"one-hot"` | Convention | Prefer `"embedding"` with many batches |
| `differential_abundance` | — | Benchmark-backed (negative) | Descriptive only; use a replicate-aware external method for inference |
| `differential_expression` | Refused | Benchmark-backed (negative) | Public API fails closed; use donor-pseudobulk |
| `kl_*_weight`, `scale_observations`, … | various | No guidance | Treat any change as an experiment |
