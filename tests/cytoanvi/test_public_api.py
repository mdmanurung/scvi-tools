"""Public API contract tests for the top-level CytoANVI package."""

from __future__ import annotations

import builtins
import importlib
import inspect
import sys

import numpy as np
import pytest
from anndata import AnnData

PUBLIC_EXPORTS = [
    "CytoANVI",
    "CytoANVAE",
    "hierarchy",
    "mapping_qc",
]

OPTIONAL_BACKENDS = {
    "annbatch",
    "cupy",
    "flowsom",
    "mapqc",
    "rapids_singlecell",
    "scHPL",
}


def test_top_level_public_exports_are_stable():
    import cytoanvi

    assert cytoanvi.__all__ == PUBLIC_EXPORTS


def test_top_level_imports_are_public_entrypoints():
    from cytoanvi import (
        CytoANVAE,
        CytoANVI,
        hierarchy,
        mapping_qc,
    )

    assert CytoANVI.__name__ == "CytoANVI"
    assert CytoANVAE.__name__ == "CytoANVAE"
    assert hierarchy.__name__ == "cytoanvi.hierarchy"
    assert mapping_qc.__name__ == "cytoanvi.mapping_qc"


def test_legacy_scvi_external_cytoanvi_import_is_intentionally_absent():
    with pytest.raises((ImportError, AttributeError)):
        exec("from scvi.external import CytoANVI", {})


def test_tta_threshold_helper_is_not_a_stable_top_level_export():
    import cytoanvi

    assert not hasattr(cytoanvi, "get_uncertainty_threshold")
    with pytest.raises(ImportError):
        exec("from cytoanvi import get_uncertainty_threshold", {})


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "setup_anndata",
            "(adata: 'AnnData', labels_key: 'str', unlabeled_category: 'str', "
            "layer: 'str | None' = None, batch_key: 'str | None' = None, "
            "sample_key: 'str | None' = None, categorical_covariate_keys: "
            "'list[str] | None' = None, continuous_covariate_keys: 'list[str] | None' = None, "
            "nan_layer: 'str | None' = None, **kwargs)",
        ),
        (
            "__init__",
            "(self, adata: 'AnnData', n_hidden: 'int' = 128, "
            "n_latent: 'int | None' = None, n_layers: 'int' = 1, "
            "dropout_rate: 'float' = 0.1, protein_likelihood: "
            "\"Literal['normal', 'beta']\" = 'normal', latent_distribution: "
            "\"Literal['normal', 'ln']\" = 'normal', encode_backbone_only: "
            "'bool | None' = None, encoder_marker_list: 'list | None' = None, "
            "linear_classifier: 'bool' = False, y_prior: "
            "\"Literal['uniform', 'empirical'] | torch.Tensor | None\" = 'uniform', "
            "class_weighting: "
            "\"Literal['none', 'inverse_frequency', 'sqrt_inverse_frequency'] | "
            "torch.Tensor | None\" = 'none', class_weight_clip: 'float' = 10.0, "
            "hierarchy_edges: 'dict[str, list[str]] | None' = None, "
            "reachability_matrix: 'np.ndarray | torch.Tensor | None' = None, **model_kwargs)",
        ),
        (
            "train",
            "(self, max_epochs: 'int | None' = 1000, "
            "n_samples_per_label: 'float | None' = None, lr: 'float' = 0.001, "
            "accelerator: 'str' = 'auto', devices: 'int | list[int] | str' = 'auto', "
            "train_size: 'float' = 0.9, validation_size: 'float | None' = None, "
            "batch_size: 'int' = 4096, early_stopping: 'bool' = True, "
            "check_val_every_n_epoch: 'int | None' = None, "
            "n_steps_kl_warmup: 'int | None' = None, "
            "n_epochs_kl_warmup: 'int | None' = 400, "
            "adversarial_classifier: 'bool | None' = None, plan_kwargs: 'dict | None' = None, "
            "early_stopping_patience: 'int | None' = 30, **kwargs)",
        ),
        (
            "from_cytovi_model",
            "(cytovi_model: 'CYTOVI', unlabeled_category: 'str', "
            "labels_key: 'str | None' = None, adata: 'AnnData | None' = None, "
            "**cytoanvi_kwargs)",
        ),
        (
            "prepare_query_anndata",
            "(adata: 'AnnData', reference_model: 'str | CytoANVI', "
            "return_reference_var_names: 'bool' = False, inplace: 'bool' = True) "
            "-> 'AnnData | pd.Index | None'",
        ),
        (
            "load_query_data_with_replay",
            "(adata: 'AnnData', reference_model: 'CytoANVI', replay_adata: 'AnnData', "
            "control_adata: 'AnnData', combine_type: 'str' = 'product', "
            "freeze_classifier: 'bool' = True, seed: 'int' = 0, **load_query_kwargs)",
        ),
        (
            "get_uncertainty",
            "(self, adata: 'AnnData | None' = None, indices=None, "
            "batch_size: 'int | None' = None, tta_rep: 'int' = 50, "
            "mode: 'str' = 'latent') -> 'np.ndarray'",
        ),
        (
            "experimental_get_uncertainty",
            "(self, adata: 'AnnData | None' = None, indices=None, "
            "batch_size: 'int | None' = None, tta_rep: 'int' = 50, "
            "mode: 'str' = 'latent', seed: 'int | None' = None) -> 'np.ndarray'",
        ),
        (
            "score_query_mapping",
            "(self, reference_adata: 'AnnData', query_adata: 'AnnData', *, "
            "sample_key: 'str', n_nhoods: 'int', k_min: 'int', k_max: 'int', "
            "**kwargs) -> 'AnnData'",
        ),
    ],
)
def test_cytoanvi_public_method_signatures_are_stable(name, expected):
    from cytoanvi import CytoANVI

    assert str(inspect.signature(getattr(CytoANVI, name))) == expected


def test_importing_public_modules_does_not_import_optional_backends(monkeypatch):
    original_import = builtins.__import__
    saved_modules = {
        module_name: module
        for module_name, module in sys.modules.items()
        if module_name == "cytoanvi" or module_name.startswith("cytoanvi.")
    }

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.partition(".")[0] in OPTIONAL_BACKENDS:
            raise AssertionError(f"optional backend {name!r} imported during public import")
        return original_import(name, globals, locals, fromlist, level)

    try:
        for module_name in list(sys.modules):
            if module_name == "cytoanvi" or module_name.startswith("cytoanvi."):
                sys.modules.pop(module_name)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        cytoanvi = importlib.import_module("cytoanvi")
        hierarchy = importlib.import_module("cytoanvi.hierarchy")
        mapping_qc = importlib.import_module("cytoanvi.mapping_qc")

        assert cytoanvi.hierarchy is hierarchy
        assert cytoanvi.mapping_qc is mapping_qc
    finally:
        for module_name in list(sys.modules):
            if module_name == "cytoanvi" or module_name.startswith("cytoanvi."):
                sys.modules.pop(module_name)
        sys.modules.update(saved_modules)


def test_hierarchy_workflow_requires_optional_schpl_extra():
    from cytoanvi import hierarchy

    latent = AnnData(X=np.array([[0.0, 1.0], [1.0, 0.0]]))
    latent.obs["batch"] = ["b0", "b1"]
    latent.obs["cell_type"] = ["a", "b"]

    with pytest.raises(ImportError, match=r"pip install cytoanvi\[cytoanvi-hierarchy\]"):
        hierarchy.learn_hierarchy(
            latent,
            batch_key="batch",
            batch_order=["b0", "b1"],
            cell_type_key="cell_type",
        )


def test_mapping_qc_workflow_requires_optional_mapqc_extra(monkeypatch):
    from cytoanvi import mapping_qc

    original_import = builtins.__import__

    def missing_mapqc(name, globals=None, locals=None, fromlist=(), level=0):
        if name.partition(".")[0] == "mapqc":
            raise ImportError("blocked mapqc for optional-extra test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_mapqc)

    joint = AnnData(X=np.array([[0.0, 1.0], [1.0, 0.0]]))
    with pytest.raises(ImportError, match=r"pip install cytoanvi\[cytoanvi-mapping-qc\]"):
        mapping_qc.run_mapqc_on_joint(
            joint,
            sample_key="sample",
            n_nhoods=2,
            k_min=1,
            k_max=2,
        )
