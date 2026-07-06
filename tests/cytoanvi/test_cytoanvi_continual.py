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
