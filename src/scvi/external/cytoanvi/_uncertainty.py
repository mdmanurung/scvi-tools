"""Query-novelty uncertainty for CytoANVI via test-time augmentation (TTA).

Per-cell Bregman-Information uncertainty: encode each cell several times under random feature
masking and measure how unstable its latent embedding is. High scores flag cells whose embedding
is sensitive to which markers are seen — a proxy for novelty / out-of-distribution query cells
(e.g. disease states absent from the reference). This is independent of the EWC continual update
(see :mod:`~scvi.external.cytoanvi._continual`); it only shares the cscanvi paper as a source.
"""

from __future__ import annotations

import torch


def mask_augment(
    x: torch.Tensor,
    mask_percentage: float = 0.5,
    nan_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Randomly zero a fixed fraction of features per cell.

    When ``nan_mask`` is provided (1 = observed, 0 = missing), only **observed** features are
    candidates for masking — per-cell missing backbone markers are never perturbed.

    Default 0.5 matches the paper's TTA (mask 50% of genes per perturbation).
    """
    if nan_mask is None:
        _, feature_dim = x.shape
        num_masked = int(mask_percentage * feature_dim)
        mask = torch.cat(
            [
                torch.ones(num_masked, dtype=torch.bool),
                torch.zeros(feature_dim - num_masked, dtype=torch.bool),
            ]
        )
        mask = mask[torch.randperm(mask.size(0))]
        mask = mask.unsqueeze(0).expand(x.shape[0], -1).to(x.device)
        return x * mask

    out = x.clone()
    batch_size, _ = x.shape
    for i in range(batch_size):
        observed = (nan_mask[i] > 0).nonzero(as_tuple=True)[0]
        n_obs = observed.numel()
        if n_obs == 0:
            continue
        num_masked = max(1, int(mask_percentage * n_obs))
        perm = observed[torch.randperm(n_obs, device=x.device)]
        out[i, perm[:num_masked]] = 0.0
    return out


def bregman_information_lse(zs: torch.Tensor, axis: int = 0, class_axis: int = -1) -> torch.Tensor:
    """Bregman Information of ``zs`` under the log-sum-exp generator.

    ``BI[Z] = E[LSE(Z)] - LSE(E[Z])``, estimated over the sample ``axis``. Taken from
    https://github.com/MLO-lab/Uncertainty_Estimates_via_BVD. Higher = more uncertain / novel.
    """
    e_of_lse = zs.logsumexp(axis=class_axis).mean(axis)
    lse_of_e = zs.mean(axis).unsqueeze(axis).logsumexp(axis=class_axis).squeeze(axis)
    return e_of_lse - lse_of_e


def compute_uncertainty_scores(
    inference_inputs: dict,
    module,
    tta_rep: int = 10,
    nan_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-cell Bregman-Information uncertainty over ``tta_rep`` mask-augmented latent draws."""
    input_x = inference_inputs["x"]
    with torch.no_grad():
        module.eval()
        all_zs = []
        for _ in range(tta_rep):
            aug_inputs = dict(inference_inputs)
            aug_inputs["x"] = mask_augment(input_x, nan_mask=nan_mask)
            all_zs.append(module.inference(**aug_inputs)["z"])
        zs_out = torch.stack(all_zs).detach().cpu()  # tta_rep x batch x n_latent
    return bregman_information_lse(zs_out)
