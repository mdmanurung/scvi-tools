"""Query-novelty uncertainty for CytoANVI via test-time augmentation (TTA).

Per-cell Bregman-Information uncertainty: encode each cell several times under random feature
masking and measure how unstable its latent embedding is. High scores flag cells whose embedding
is sensitive to which markers are seen — a proxy for novelty / out-of-distribution query cells
(e.g. disease states absent from the reference). This is independent of the EWC continual update
(see :mod:`~cytoanvi._continual`); it only shares the cscanvi paper as a source.
"""

from __future__ import annotations

import torch


def mask_augment(
    x: torch.Tensor,
    mask_percentage: float = 0.5,
    nan_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly zero a fixed fraction of features per cell.

    ``mask_percentage`` is the fraction of features zeroed in both branches.
    When ``nan_mask`` is provided (1 = observed, 0 = missing), only **observed** features are
    candidates for masking — per-cell missing backbone markers are never perturbed.

    Default 0.5 matches the paper's TTA (mask 50% of genes per perturbation).
    ``generator`` is forwarded to ``torch.randperm`` so TTA draws are reproducible when a seeded
    generator is supplied (e.g. in tests).
    """
    if nan_mask is None:
        _, feature_dim = x.shape
        num_masked = max(1, int(mask_percentage * feature_dim))
        mask = torch.cat(
            [
                torch.zeros(num_masked, dtype=torch.bool),
                torch.ones(feature_dim - num_masked, dtype=torch.bool),
            ]
        )
        mask = mask[torch.randperm(mask.size(0), generator=generator)]
        mask = mask.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        return x * mask

    # Vectorized: assign random noise to observed entries; unobserved get -1 so they
    # can never be selected as the top-k slots to mask.
    noise = torch.rand_like(x)
    noise = noise.masked_fill(nan_mask <= 0, -1.0)
    n_obs = (nan_mask > 0).sum(dim=1)
    k = (mask_percentage * n_obs.float()).clamp(min=1).long()
    k = k * (n_obs > 0).long()  # all-unobserved rows: k=0 → mask all-True → pass through
    # rank[i, j] = rank of column j within row i (0 = highest noise = to be masked)
    order = noise.argsort(dim=1, descending=True)
    ranks = order.argsort(dim=1)
    mask = ranks >= k.unsqueeze(1)  # True = keep, False = zero
    return x * mask


def bregman_information_lse(zs: torch.Tensor, axis: int = 0, class_axis: int = -1) -> torch.Tensor:
    """Bregman Information of ``zs`` under the log-sum-exp generator.

    ``BI[Z] = E[LSE(Z)] - LSE(E[Z])``, estimated over the sample ``axis``. Taken from
    https://github.com/MLO-lab/Uncertainty_Estimates_via_BVD. Higher = more uncertain / novel.
    """
    e_of_lse = zs.logsumexp(dim=class_axis).mean(axis)
    lse_of_e = zs.mean(axis).unsqueeze(axis).logsumexp(dim=class_axis).squeeze(axis)
    return e_of_lse - lse_of_e


def compute_uncertainty_scores(
    inference_inputs: dict,
    module,
    tta_rep: int = 10,
    nan_mask: torch.Tensor | None = None,
    mode: str = "latent",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Per-cell Bregman-Information uncertainty over ``tta_rep`` mask-augmented draws.

    Parameters
    ----------
    mode
        ``"latent"`` (default): BI computed over the ``n_latent``-dimensional encoder
        mean — the original formulation. ``"logit"``: BI computed over the
        ``n_labels``-dimensional classifier logits (canonical BVD-on-logits). Both use
        the LSE generator; the difference is which representation is stacked.
    generator
        Optional ``torch.Generator`` for reproducible TTA masking.  When ``None`` (default),
        draws are non-deterministic (production behaviour).
    """
    input_x = inference_inputs["x"]
    all_draws = []
    for _ in range(tta_rep):
        aug_x = mask_augment(input_x, nan_mask=nan_mask, generator=generator)
        if mode == "logit":
            draw = module.classify(
                aug_x,
                batch_index=inference_inputs.get("batch_index"),
                cont_covs=inference_inputs.get("cont_covs"),
                cat_covs=inference_inputs.get("cat_covs"),
            )
        else:
            aug_inputs = dict(inference_inputs)
            aug_inputs["x"] = aug_x
            draw = module.inference(**aug_inputs)["z"]
        all_draws.append(draw)
    draws_out = torch.stack(all_draws).detach()  # (tta_rep, batch, dim) — stays on device
    return bregman_information_lse(draws_out)
