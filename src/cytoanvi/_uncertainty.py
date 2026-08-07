"""Experimental test-time-augmentation (TTA) utilities for CytoANVI.

The retained estimator has negative scientific validation and is not a supported novelty or OOD
capability. It remains here, behind an explicitly experimental model method, so historical results
can be reproduced while a separately calibrated replacement is developed.
"""

from __future__ import annotations

import numpy as np
import torch


def _random_scores(
    x: torch.Tensor,
    *,
    generator: torch.Generator | None,
    seed: int | None,
    cell_indices: torch.Tensor | np.ndarray | None,
    augmentation_index: int,
) -> torch.Tensor:
    """Draw row-independent scores, optionally stateless with respect to chunking."""
    if seed is not None and generator is not None:
        raise ValueError("Pass either seed or generator, not both.")
    if seed is None:
        return torch.rand(x.shape, dtype=torch.float32, device=x.device, generator=generator)

    if cell_indices is None:
        cell_indices = np.arange(x.shape[0], dtype=np.int64)
    elif isinstance(cell_indices, torch.Tensor):
        cell_indices = cell_indices.detach().cpu().numpy()
    cell_indices = np.asarray(cell_indices, dtype=np.int64)
    if cell_indices.ndim != 1 or len(cell_indices) != x.shape[0]:
        raise ValueError(
            "cell_indices must be one-dimensional with one entry per row; "
            f"got shape {cell_indices.shape} for {x.shape[0]} rows."
        )

    # Key a child RNG by (seed, cell position, augmentation). A cell's mask is therefore
    # independent of which minibatch contains it and of every other cell's mask. NumPy's PCG64
    # stream also gives the same mask on every Torch device before the scores are transferred.
    scores = np.empty(x.shape, dtype=np.float32)
    seed_word = int(seed) % (2**32)
    augmentation_word = int(augmentation_index) % (2**32)
    for row, cell_index in enumerate(cell_indices):
        sequence = np.random.SeedSequence(
            [seed_word, int(cell_index) % (2**32), augmentation_word]
        )
        scores[row] = np.random.default_rng(sequence).random(x.shape[1], dtype=np.float32)
    return torch.as_tensor(scores, device=x.device)


def mask_augment(
    x: torch.Tensor,
    mask_percentage: float = 0.5,
    nan_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    *,
    seed: int | None = None,
    cell_indices: torch.Tensor | np.ndarray | None = None,
    augmentation_index: int = 0,
) -> torch.Tensor:
    """Randomly zero a fixed fraction of features per cell.

    ``mask_percentage`` is the fraction of features zeroed in both branches.
    When ``nan_mask`` is provided (1 = observed, 0 = missing), only **observed** features are
    candidates for masking — per-cell missing backbone markers are never perturbed.

    Default 0.5 matches the paper's TTA (mask 50% of genes per perturbation). ``generator`` offers
    stateful reproducibility for direct unit use. ``seed`` together with ``cell_indices`` and
    ``augmentation_index`` instead defines a stateless per-cell draw that is exactly invariant to
    batch/chunk partitioning.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be two-dimensional; got shape {tuple(x.shape)}.")
    if not np.isfinite(mask_percentage) or not 0.0 < mask_percentage <= 1.0:
        raise ValueError("mask_percentage must be finite and in (0, 1].")
    if nan_mask is not None and tuple(nan_mask.shape) != tuple(x.shape):
        raise ValueError(
            f"nan_mask must match x shape {tuple(x.shape)}; got {tuple(nan_mask.shape)}."
        )

    observed = torch.ones_like(x, dtype=torch.bool) if nan_mask is None else nan_mask > 0
    # Assign independent random noise to observed entries; unobserved entries get -1 so they can
    # never be selected as top-k masking candidates.
    noise = _random_scores(
        x,
        generator=generator,
        seed=seed,
        cell_indices=cell_indices,
        augmentation_index=augmentation_index,
    )
    noise = noise.masked_fill(~observed, -1.0)
    n_obs = observed.sum(dim=1)
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


def experimental_get_uncertainty_threshold(
    uncertainty_ref: np.ndarray,
    specificity: float = 0.95,
) -> float:
    """Calibrate the experimental TTA score on a finite, non-empty reference array.

    This helper does not make TTA a supported novelty capability. It is retained for explicit
    experimental reproduction only.

    Parameters
    ----------
    uncertainty_ref
        Per-cell uncertainty scores for held-out **reference** cells (same type seen during
        training). Typically obtained via
        :meth:`~cytoanvi.CytoANVI.experimental_get_uncertainty` on a held-out split.
    specificity
        Desired specificity: fraction of reference cells correctly retained below the threshold.
        Default 0.95 (flag at most 5 % of reference cells as novel).

    Returns
    -------
    float
        Uncertainty threshold T.

    """
    arr = np.asarray(uncertainty_ref, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"uncertainty_ref must be 1-D; got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("uncertainty_ref must contain at least one calibration score.")
    if not np.isfinite(arr).all():
        raise ValueError("uncertainty_ref must contain only finite calibration scores.")
    if not (0.0 < specificity < 1.0):
        raise ValueError(f"specificity must be in (0, 1); got {specificity}")
    return float(np.quantile(arr, specificity))


def compute_uncertainty_scores(
    inference_inputs: dict,
    module,
    tta_rep: int = 10,
    nan_mask: torch.Tensor | None = None,
    mode: str = "latent",
    generator: torch.Generator | None = None,
    *,
    seed: int | None = None,
    cell_indices: torch.Tensor | np.ndarray | None = None,
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
    seed
        Stateless deterministic seed. With stable ``cell_indices``, results do not depend on how
        cells are partitioned into chunks. Mutually exclusive with ``generator``.
    cell_indices
        Stable positions in the evaluated cell order, one per input row.
    """
    if not isinstance(tta_rep, int) or isinstance(tta_rep, bool) or tta_rep <= 0:
        raise ValueError("tta_rep must be a positive integer.")
    if mode not in {"latent", "logit"}:
        raise ValueError("mode must be one of {'latent', 'logit'}.")
    input_x = inference_inputs["x"]
    all_draws = []
    for augmentation_index in range(tta_rep):
        aug_x = mask_augment(
            input_x,
            nan_mask=nan_mask,
            generator=generator,
            seed=seed,
            cell_indices=cell_indices,
            augmentation_index=augmentation_index,
        )
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
            # Use the encoder mean, matching the stated estimator and avoiding an unrelated
            # posterior sample that would defeat fixed-seed TTA reproducibility.
            draw = module.inference(**aug_inputs)["qz"].loc
        all_draws.append(draw)
    draws_out = torch.stack(all_draws).detach()  # (tta_rep, batch, dim) — stays on device
    return bregman_information_lse(draws_out)
