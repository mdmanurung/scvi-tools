"""Unit tests for CytoANVI continual-update utilities.

Covers properties that are NOT already tested in test_cytoanvi.py:
- I1: fisher_importances returns non-negative, finite, covering tensors.

The existing test_cytoanvi.py already covers:
- test_fisher_importances_raises_on_empty_adata
- test_continual_update_penalty_math
- test_cytoanvi_continual_update (full integration)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cytoanvi import CytoANVI
from cytoanvi._continual import fisher_importances
from scvi.data import synthetic_iid
from scvi.external import cytovi as cytovi_pp

SCALED_LAYER_KEY = "scaled"
BATCH_KEY = "batch"
LABELS_KEY = "labels"
SAMPLE_KEY = "sample_key"
UNLABELED = "label_0"


def _make_adata(n_genes: int = 20, n_batches: int = 2, n_labels: int = 5):
    """Minimal synthetic cytometry AnnData (mirrors test_cytoanvi._make_adata)."""
    adata = synthetic_iid(
        batch_size=64,
        n_genes=n_genes,
        n_proteins=0,
        n_regions=0,
        n_batches=n_batches,
        n_labels=n_labels,
        rna_dist="normal",
    )
    adata.obs[SAMPLE_KEY] = np.random.default_rng(42).choice(
        ["group_a", "group_b"], size=adata.shape[0]
    )
    adata.layers["raw"] = adata.X.copy()
    cytovi_pp.transform_arcsinh(adata)
    cytovi_pp.scale(adata)
    return adata


def _setup_and_train(adata, max_epochs: int = 2):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    model.train(max_epochs=max_epochs, accelerator="cpu")
    return model


# ---------------------------------------------------------------------------
# I1: Fisher importances are non-negative and finite
# ---------------------------------------------------------------------------

def test_fisher_importances_are_nonnegative_and_finite():
    """fisher_importances returns per-parameter CPU tensors that are non-negative and finite.

    Because importances are mean-squared gradients they must be >= 0 everywhere.
    Every trained parameter must appear in the output.
    """
    adata = _make_adata()
    model = _setup_and_train(adata, max_epochs=2)

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
