"""Direct unit tests for :mod:`cytoanvi._uncertainty`.

These functions were previously exercised only indirectly, through ``CytoANVI.get_uncertainty``
integration tests that assert shape and finiteness. Shape/finiteness cannot distinguish a correct
masking rule from a subtly wrong one, so the masking *semantics* — the fraction actually zeroed,
the floor, and the never-touch-unobserved guarantee — went unpinned. These tests drive raw tensors
directly (matching the style of ``test_cytoanvi_elbo_components.py``) and need no model or fixture.

Note on scope: ``get_uncertainty``'s TTA method is a documented negative result (F-013, AUROC
0.484, below chance). These tests pin what the code *does*; they are deliberately not an
endorsement of the method.
"""

import math

import pytest
import torch

from cytoanvi._uncertainty import (
    bregman_information_lse,
    experimental_get_uncertainty_threshold,
    mask_augment,
)


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _zeroed_columns(row: torch.Tensor) -> set[int]:
    return set(torch.nonzero(row == 0).flatten().tolist())


# ---------------------------------------------------------------------------
# Independent per-cell masking and chunk-invariant deterministic masks.
# ---------------------------------------------------------------------------


def test_mask_augment_none_branch_masks_independently_per_cell():
    x = torch.ones(6, 12)
    out = mask_augment(x, mask_percentage=0.3, nan_mask=None, generator=_gen())

    per_row = [_zeroed_columns(out[i]) for i in range(out.shape[0])]
    assert len({frozenset(s) for s in per_row}) > 1, (
        "expected independent per-row masks when nan_mask is absent"
    )


def test_mask_augment_nan_mask_branch_masks_independently_per_cell():
    x = torch.ones(6, 12)
    nan_mask = torch.ones_like(x)
    out = mask_augment(x, mask_percentage=0.5, nan_mask=nan_mask, generator=_gen())

    per_row = [_zeroed_columns(out[i]) for i in range(out.shape[0])]
    assert len({frozenset(s) for s in per_row}) > 1, (
        "expected per-row independent masks in the nan_mask branch"
    )


def test_fixed_seed_masks_are_exactly_chunk_invariant():
    x = torch.arange(8 * 11, dtype=torch.float32).reshape(8, 11) + 1.0
    positions = torch.arange(len(x))

    full = mask_augment(
        x,
        mask_percentage=0.5,
        seed=1729,
        cell_indices=positions,
        augmentation_index=3,
    )
    chunked = torch.cat(
        [
            mask_augment(
                x[start:stop],
                mask_percentage=0.5,
                seed=1729,
                cell_indices=positions[start:stop],
                augmentation_index=3,
            )
            for start, stop in ((0, 3), (3, 5), (5, 8))
        ]
    )

    assert torch.equal(full, chunked)


# ---------------------------------------------------------------------------
# mask_percentage contract, both branches.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("feature_dim", "mask_percentage", "expected_masked"),
    [
        (10, 0.5, 5),
        (10, 0.3, 3),
        (2, 0.3, 1),  # int(0.3 * 2) == 0 -> the max(1, ...) floor kicks in
        (1, 0.3, 1),  # single feature: floor still guarantees one masked entry
        (7, 0.5, 3),  # int(3.5) truncates toward zero, it does not round
    ],
)
def test_mask_augment_none_branch_count_and_floor(feature_dim, mask_percentage, expected_masked):
    """Exactly ``max(1, int(p * feature_dim))`` entries are zeroed, including the floor cases."""
    x = torch.ones(3, feature_dim)
    out = mask_augment(x, mask_percentage=mask_percentage, nan_mask=None, generator=_gen())

    counts = (out == 0).sum(dim=1)
    assert torch.equal(counts, torch.full((3,), expected_masked))


def test_mask_augment_nan_mask_branch_masks_floor_p_times_n_observed_per_row():
    """Per row: exactly ``max(1, floor(p * n_obs))`` entries zeroed, and 0 if nothing observed.

    Rows are built with observed counts [10, 7, 1, 0] at ``p=0.5``, giving expected masked counts
    [5, 3, 1, 0] — one plain case, one truncation case (3.5 -> 3), one floor case (0.5 -> 1), and
    the all-unobserved passthrough.
    """
    x = torch.ones(4, 10)
    nan_mask = torch.zeros(4, 10)
    nan_mask[0, :10] = 1
    nan_mask[1, :7] = 1
    nan_mask[2, :1] = 1
    # row 3 stays all-unobserved

    out = mask_augment(x, mask_percentage=0.5, nan_mask=nan_mask, generator=_gen())

    counts = (out == 0).sum(dim=1)
    assert torch.equal(counts, torch.tensor([5, 3, 1, 0]))


def test_mask_augment_nan_mask_branch_never_perturbs_unobserved_features():
    """Unobserved entries must be returned untouched, on every draw — not merely usually.

    Unobserved positions get noise ``-1`` so they sort last and can never enter the top-k masked
    slots. Checked across 20 seeds because a probabilistic leak would show up intermittently.
    """
    x = torch.arange(4 * 10, dtype=torch.float32).reshape(4, 10) + 1.0  # no zeros, so 0 == masked
    nan_mask = torch.ones(4, 10)
    nan_mask[:, 6:] = 0  # last four features unobserved for every row

    for seed in range(20):
        out = mask_augment(x, mask_percentage=0.5, nan_mask=nan_mask, generator=_gen(seed))
        assert torch.equal(out[:, 6:], x[:, 6:]), f"unobserved features perturbed at seed {seed}"


def test_mask_augment_nan_mask_branch_all_unobserved_row_passes_through():
    """A row with nothing observed gets ``k=0`` and must come back bit-identical.

    This pins the ``k=0 -> mask all-True -> pass through`` comment at ``_uncertainty.py:51`` as an
    actual assertion rather than a claim in a comment.
    """
    x = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5) + 1.0
    nan_mask = torch.ones(2, 5)
    nan_mask[1, :] = 0  # second row entirely unobserved

    out = mask_augment(x, mask_percentage=0.5, nan_mask=nan_mask, generator=_gen())

    assert torch.equal(out[1], x[1])
    assert (out[0] == 0).sum().item() == 2  # first row still masked normally: floor(0.5 * 5) == 2


# ---------------------------------------------------------------------------
# bregman_information_lse
# ---------------------------------------------------------------------------


def test_bregman_information_is_zero_when_all_draws_identical():
    """BI[Z] = E[LSE(Z)] - LSE(E[Z]) is exactly 0 when there is no variation across draws.

    Stronger than the existing non-negativity smoke check: it pins the value, so a sign error or
    a swapped-term regression is caught rather than merely staying non-negative.
    """
    base = torch.randn(6, 4)
    zs = base.unsqueeze(0).expand(5, 6, 4).clone()  # 5 identical draws

    bi = bregman_information_lse(zs)

    assert bi.shape == (6,)
    torch.testing.assert_close(bi, torch.zeros(6), atol=1e-6, rtol=0)


def test_bregman_information_matches_hand_computed_value():
    """Pins the exact ``dim=``/``axis=`` semantics against an independently computed value.

    The expected value is derived with plain ``math.log``/``math.exp`` rather than by re-deriving
    it through ``torch.logsumexp``, so the assertion does not depend on the same reduction the
    function under test uses. Two draws of one cell over two classes:

        E[LSE(Z)] = log(e^0 + e^5)                       (both draws give the same LSE)
        LSE(E[Z]) = log(e^2.5 + e^2.5) = 2.5 + log(2)
    """
    zs = torch.tensor([[[0.0, 5.0]], [[5.0, 0.0]]])  # (draws=2, cells=1, classes=2)

    expected = math.log(math.exp(0.0) + math.exp(5.0)) - (2.5 + math.log(2.0))
    bi = bregman_information_lse(zs)

    assert bi.shape == (1,)
    torch.testing.assert_close(bi, torch.tensor([expected]), atol=1e-6, rtol=0)


def test_bregman_information_is_nonnegative_for_varying_draws():
    """BI is a Jensen gap, so it is non-negative for any input by construction."""
    zs = torch.randn(8, 12, 5)

    bi = bregman_information_lse(zs)

    assert bi.shape == (12,)
    assert (bi >= -1e-6).all()


@pytest.mark.parametrize(
    "calibration",
    [
        [],
        [0.1, float("nan")],
        [0.1, float("inf")],
        [0.1, float("-inf")],
    ],
)
def test_experimental_threshold_rejects_empty_or_nonfinite_calibration(calibration):
    with pytest.raises(ValueError, match="at least one|finite"):
        experimental_get_uncertainty_threshold(calibration)


def test_experimental_threshold_accepts_finite_calibration():
    threshold = experimental_get_uncertainty_threshold([0.1, 0.2, 0.3], specificity=0.5)

    assert threshold == pytest.approx(0.2)
