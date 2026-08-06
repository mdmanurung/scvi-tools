"""Unit tests for CytoANVI continual-update utilities.

Covers properties that are NOT already tested in test_cytoanvi.py:
- I1: fisher_importances returns non-negative, finite, covering tensors.

The existing test_cytoanvi.py already covers:
- test_fisher_importances_raises_on_empty_adata
- test_continual_update_penalty_math
- test_cytoanvi_continual_update (full integration)
"""

from __future__ import annotations

import pytest
import torch

from conftest import (
    BATCH_KEY,
    LABELS_KEY,
    SAMPLE_KEY,
    SCALED_LAYER_KEY,
    UNLABELED,
    make_adata,
    setup_and_train,
)

from cytoanvi._continual import fisher_importances


# ---------------------------------------------------------------------------
# I1: Fisher importances are non-negative and finite
# ---------------------------------------------------------------------------


def test_fisher_importances_are_nonnegative_and_finite():
    """fisher_importances returns per-parameter CPU tensors that are non-negative and finite.

    Because importances are mean-squared gradients they must be >= 0 everywhere.
    Every trained parameter must appear in the output.
    """
    adata = make_adata(n_genes=20, batch_size=64)
    model = setup_and_train(adata, max_epochs=2)

    importances = fisher_importances(model, adata, max_cells=256, seed=0)

    # Must return a non-empty list of (name, tensor) pairs
    assert len(importances) > 0, "fisher_importances returned an empty list"

    trained_param_names = {name for name, _ in model.module.named_parameters()}
    importance_names = set()

    for name, imp in importances:
        # Each importance tensor must live on CPU
        assert imp.device.type == "cpu", f"importance tensor for '{name}' is not on CPU"
        # Must be finite
        assert torch.all(torch.isfinite(imp)), (
            f"importance tensor for '{name}' contains non-finite values"
        )
        # Must be non-negative (squared gradient)
        assert torch.all(imp >= 0.0), (
            f"importance tensor for '{name}' contains negative values (min={imp.min():.6g})"
        )
        importance_names.add(name)

    # Every trained parameter should have a corresponding importance entry
    missing = trained_param_names - importance_names
    assert len(missing) == 0, (
        f"fisher_importances is missing entries for {len(missing)} trained parameters: "
        f"{sorted(missing)[:5]}..."
    )


# ---------------------------------------------------------------------------
# I2: replay-buffer cycling. An off-by-one here would silently make Experience
# Replay rehearse the same batch forever, weakening the anti-forgetting guarantee
# that is the whole point of the feature. Existing integration tests only assert
# `is_trained` / prediction shape, neither of which can detect it.
# ---------------------------------------------------------------------------


def test_next_replay_batch_cycles_over_the_buffer():
    from cytoanvi._continual import ContinualUpdate

    batches = [{"x": torch.tensor([i])} for i in range(3)]
    cont = ContinualUpdate([], [], replay_batches=batches)

    got = [int(cont.next_replay_batch(i, "cpu")["x"].item()) for i in range(7)]

    # wraps modulo len(buffer) rather than clamping, restarting, or drifting
    assert got == [0, 1, 2, 0, 1, 2, 0]


@pytest.mark.parametrize("empty", [None, []])
def test_next_replay_batch_returns_none_for_empty_buffer(empty):
    """No replay buffer (e.g. after save/load) must yield None, not raise or index into nothing."""
    from cytoanvi._continual import ContinualUpdate

    cont = ContinualUpdate([], [], replay_batches=empty)

    assert cont.next_replay_batch(0, "cpu") is None
    assert cont.next_replay_batch(5, "cpu") is None


def test_next_replay_batch_moves_tensors_to_device():
    from cytoanvi._continual import ContinualUpdate

    cont = ContinualUpdate([], [], replay_batches=[{"x": torch.ones(2), "y": torch.zeros(2)}])

    out = cont.next_replay_batch(0, torch.device("cpu"))

    assert set(out) == {"x", "y"}
    assert all(v.device.type == "cpu" for v in out.values())


# ---------------------------------------------------------------------------
# I3: device caches must never be pickled, and penalty() must still work after a
# round-trip via the dict(self.old_params) fallback. Previously this invariant held
# only as a side effect of ordering in the save/load integration test.
# ---------------------------------------------------------------------------


class _PenaltyStub(torch.nn.Module):
    """Minimal module with one parameter — mirrors test_continual_update_penalty_math."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    @property
    def device(self):
        return torch.device("cpu")


def _make_continual():
    from cytoanvi._continual import ContinualUpdate

    return ContinualUpdate(
        old_params=[("w", torch.tensor([0.0, 0.0]))],
        importances=[("w", torch.tensor([1.0, 1.0]))],
        ctrl_importances=[("w", torch.tensor([2.0, 3.0]))],
        combine_type="product",
    )


def test_to_device_populates_caches_and_getstate_excludes_them():
    cont = _make_continual()
    assert not hasattr(cont, "_dev_old")

    cont.to_device(torch.device("cpu"))

    assert cont._dev_old["w"].tolist() == [0.0, 0.0]
    assert cont._dev_imps["w"].tolist() == [1.0, 1.0]
    assert cont._dev_ctrl["w"].tolist() == [2.0, 3.0]

    state = cont.__getstate__()
    leaked = sorted(k for k in state if k.startswith("_dev"))
    assert not leaked, f"device caches leaked into __getstate__: {leaked}"
    # the real state is still there
    assert {"old_params", "importances", "ctrl_importances", "combine_type"} <= set(state)


def test_pickle_roundtrip_drops_device_caches_and_preserves_penalty():
    """Round-tripping must exercise the non-cached penalty path and give the same number.

    Asserting only that pickling doesn't crash would pass even if the fallback in `penalty`
    (`dict(self.old_params)` when `_dev_old` is absent) were broken.
    """
    import pickle

    module = _PenaltyStub()
    cont = _make_continual()
    cont.to_device(torch.device("cpu"))
    before = cont.penalty(module).detach().item()

    revived = pickle.loads(pickle.dumps(cont))

    assert not hasattr(revived, "_dev_old")
    assert not hasattr(revived, "_dev_imps")
    assert not hasattr(revived, "_dev_ctrl")
    assert revived.penalty(module).detach().item() == before
    # product combine: w = [1,1]*[2,3] = [2,3]; theta-theta_ref = [1,2] -> 2*1 + 3*4 = 14
    assert before == 14.0
