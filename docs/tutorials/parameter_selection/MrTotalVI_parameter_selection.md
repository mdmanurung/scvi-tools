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

MrTotalVI extends TOTALVI with an MrVI-style two-level latent space: a sample-blind base `u`, and
`z = z_base(u) + eps` where `eps` carries per-donor residual structure. This tutorial is about
**how to choose values** for its parameters. For what each argument means, see
{doc}`/user_guide/models/mr_multimodal`.

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
model = MrTotalVI(adata, sample_key="sample", n_latent=20, n_latent_u=10)
```

+++

## `labels_key` is not metadata — read this before using it

```{admonition} Registering labels_key silently makes the prior label-supervised
:class: danger

This is undocumented in both the constructor docstring and the user guide, and it has already
caused one benchmarking error in this project.

Passing `labels_key` to `setup_anndata` does **not** merely record an annotation column. It
activates the label-conditioned branch of the mixture prior: the number of mixture components is
overridden to `n_labels`, and `u_prior_label_weight` (default **10.0**) biases every cell's KL term
toward the prior component matching its ground-truth label.

**Consequence:** a model trained with `labels_key` registered is *partially supervised*. Comparing
it against a model trained without one — or against stock TOTALVI — is a supervised-vs-unsupervised
comparison, and any integration or bio-conservation metric from that comparison is confounded.

If you want labels recorded but not used by the prior, keep them in `adata.obs` and do not pass
them to `setup_anndata`.
```

+++

## `u_prior`: the default is contested

**Tier: Contested.** The default is `"mog"` (mixture of Gaussians); the alternative is `"vamp"`.

The analysis that established `"mog"` as the default compared cluster dispersion on a 57k-cell
T/NK subset. Two problems:

1. **The source analysis is unrecoverable.** No surviving file exists anywhere in this project.
2. **It measured the wrong geometry.** It compared UMAP/centroid dispersion — but Leiden clustering
   operates on the kNN graph, not on UMAP coordinates. On graph-native metrics the ranking
   *reverses*: VampPrior `u` beats MoG `u` on donor mixing (0.225 vs 0.273) and on cross-seed
   stability (Jaccard 0.189 vs 0.145), at equal cell-type purity.

So `"mog"` remains the default for continuity, not because it won a defensible comparison. If your
downstream step is graph-based clustering, VampPrior is at least as defensible.

```{code-cell} ipython3
# VampPrior, data-initialised and frozen -- the configuration with the best DA-stability evidence.
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

Freezing is meaningful only when the prior was *data-anchored first*, which is what
`init_prior_from_data=True` does for VampPrior. Setting `freeze_prior_after_init=True` with
`u_prior="mog"` freezes a **randomly initialised** prior and provides no benefit.
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

If both modalities are available, route differential abundance through MrMultiVI.
```

+++

## Differential expression with `store_lfc=True`

**Tier: Benchmark-backed (negative).**

eps-space DE was compared against a stratified pseudobulk gold standard across all 12 cell types.
Every cell type gave a **negative** Spearman correlation (MrTotalVI: −0.126 to −0.008). Direction
of interferon-gene changes was called correctly 19.9% of the time — below the 50% you would get by
guessing. No cell type rescues it, including the one with the strongest pseudobulk signal.

Critically, **the reference MRVI implementation fails the same way** (ρ = −0.138). This is a
property of eps-space DE as a method, not a defect introduced here.

```{admonition} Not a "use with caution"
:class: danger

These outputs are anti-concordant with the pseudobulk ground truth. Do not draw biological
conclusions from them without cross-validating against a pseudobulk method such as PyDESeq2. The
model's contribution is integration quality and cell-type-level resolution — not temporal DE.
```

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
| `labels_key` | `None` | **Undocumented side effect** | Registering it makes the prior label-supervised. Omit it for unsupervised comparisons |
| `u_prior` | `"mog"` | Contested | Source analysis unrecoverable and measured the wrong geometry; VampPrior wins on graph metrics |
| `init_prior_from_data` + `freeze_prior_after_init` | `False`, `False` | Benchmark-backed | Use together, and only with `u_prior="vamp"` |
| `n_latent` / `n_latent_u` | 20 / `None` | Convention | Fixed for comparability, never swept. `n_latent_u < n_latent` for a real bottleneck |
| `batch_representation` | `"one-hot"` | Convention | Prefer `"embedding"` with many batches |
| `differential_abundance` | — | Benchmark-backed (negative) | std 9.46 across seeds. Route through MrMultiVI |
| `store_lfc=True` | `False` | Benchmark-backed (negative) | Anti-concordant with pseudobulk in all 12 cell types. Cross-validate |
| `kl_*_weight`, `scale_observations`, … | various | No guidance | Treat any change as an experiment |
