"""Parity tests for the private MrTotalVI registry adapter.

These tests assert that the streaming registry / tensor-emission code path in
`scvi.external.mrtotalvi._datamodule` produces *exactly* what the existing
in-memory `MrTotalVI.setup_anndata` + `AnnDataManager` + `AnnDataLoader` path
produces for the same data -- same registry structure, same tensor shapes,
dtypes, and values. No shortcuts (e.g. comparing only a subset of keys, or
comparing shuffled batches) -- see inline comments for why each check is
structured the way it is.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "src")

import scvi
from scvi.data import AnnDataManager
from scvi.dataloaders import AnnDataLoader
from scvi.external import MrTotalVI
from scvi.external.mrtotalvi._datamodule import (
    _InMemoryProteinMappedDataset,
)
from scvi.external.mrtotalvi._datamodule import (
    _MrTotalVIRegistryAdapter as MrTotalVIBatchDataModule,
)

N_DONORS = 4


def _make_adata(n_donors: int = N_DONORS, seed: int = 0) -> scvi.AnnData:
    rng = np.random.default_rng(seed)
    adata = scvi.data.synthetic_iid()
    n_cells = adata.n_obs
    adata.obs["sample"] = np.array([f"donor_{i % n_donors}" for i in range(n_cells)])
    adata.obs["cat1"] = rng.integers(0, 3, size=n_cells)
    adata.obs["cont1"] = rng.normal(size=n_cells)
    adata.uns["protein_names"] = np.asarray(
        [f"protein_{i}" for i in range(adata.obsm["protein_expression"].shape[1])],
        dtype=object,
    )
    return adata


def test_registry_adapter_is_not_publicly_exported():
    import scvi.external.mrtotalvi as public_api

    assert not hasattr(public_api, "MrTotalVIBatchDataModule")
    assert "MrTotalVIBatchDataModule" not in public_api.__all__


@pytest.fixture(scope="module")
def adata_basic():
    return _make_adata()


def _registered_reference_registry(adata, **setup_kwargs) -> dict:
    """setup_anndata on a private copy; returns the resulting registry dict."""
    adata = adata.copy()
    MrTotalVI.setup_anndata(adata, **setup_kwargs)
    mgr = MrTotalVI._get_most_recent_anndata_manager(adata)
    return mgr.registry


def _reference_loader(adata, batch_size, **setup_kwargs) -> AnnDataLoader:
    adata = adata.copy()
    MrTotalVI.setup_anndata(adata, **setup_kwargs)
    mgr = MrTotalVI._get_most_recent_anndata_manager(adata)
    return AnnDataLoader(mgr, shuffle=False, batch_size=batch_size)


def _deep_compare(a, b, path: str = "") -> list[str]:
    """Recursively compares two registry-shaped structures; returns mismatch messages."""
    mismatches = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            mismatches.append(f"key set mismatch at {path}: {set(a)} vs {set(b)}")
            return mismatches
        for k in a:
            mismatches.extend(_deep_compare(a[k], b[k], f"{path}.{k}"))
        return mismatches
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a2, b2 = np.asarray(a), np.asarray(b)
        if a2.shape != b2.shape or a2.dtype != b2.dtype or not np.array_equal(a2, b2):
            mismatches.append(
                f"array mismatch at {path}: shape/dtype/values differ "
                f"({a2.dtype}{a2.shape} vs {b2.dtype}{b2.shape})"
            )
        return mismatches
    if a != b:
        mismatches.append(f"value mismatch at {path}: {a!r} vs {b!r}")
    return mismatches


# ---------------------------------------------------------------------------
# Registry parity
# ---------------------------------------------------------------------------


def test_registry_matches_in_memory_default(adata_basic):
    """No labels_key, no covariates -- the default `_setup_and_train` shape.

    This exercises the `labels_key=None` dummy-column case, which the
    generic streaming-registry reference (`MappedCollectionDataModule`)
    handles differently (n_labels=0) than MrTotalVI's plain
    `CategoricalObsField` (n_labels=1, categorical_mapping=[0]).
    """
    ref = _registered_reference_registry(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        parallel=False,
    )
    mismatches = _deep_compare(
        ref["field_registries"], dm.registry["field_registries"], "field_registries"
    )
    assert mismatches == []
    assert dm.registry["model_name"] == ref["model_name"] == "MrTotalVI"
    assert dm.registry["setup_method_name"] == "setup_datamodule"


def test_registry_matches_in_memory_no_batch_key(adata_basic):
    """batch_key=None also hits the dummy-column path (mirrors labels_key=None)."""
    ref = _registered_reference_registry(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
    )
    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        parallel=False,
    )
    mismatches = _deep_compare(
        ref["field_registries"], dm.registry["field_registries"], "field_registries"
    )
    assert mismatches == []


def test_registry_matches_in_memory_with_labels_and_covariates(adata_basic):
    """labels_key set + categorical/continuous covariates.

    Exercises the `extra_categorical_covs` "mappings" (plural) key -- the
    generic reference's own `extra_categorical_covs` property uses the
    singular "mapping", which does not match the real
    `CategoricalJointField.MAPPINGS_KEY = "mappings"`.
    """
    ref = _registered_reference_registry(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        labels_key="labels",
        categorical_covariate_keys=["cat1"],
        continuous_covariate_keys=["cont1"],
    )
    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        labels_key="labels",
        categorical_covariate_keys=["cat1"],
        continuous_covariate_keys=["cont1"],
        parallel=False,
    )
    mismatches = _deep_compare(
        ref["field_registries"], dm.registry["field_registries"], "field_registries"
    )
    assert mismatches == []
    # extra_categorical_covs mapping dtype must be preserved (int64 for an
    # int-valued covariate column), not coerced to string/object.
    cat_mapping = dm.registry["field_registries"]["extra_categorical_covs"]["state_registry"][
        "mappings"
    ]["cat1"]
    assert cat_mapping.dtype == np.int64


def test_registry_matches_in_memory_protein_batch_mask(adata_basic):
    """Zeroing one batch's protein counts must produce a matching protein_batch_mask.

    Keys are the *encoded integer* batch codes stringified ("0", "1", ...),
    not the original batch label strings -- verified against ground truth,
    not assumed.
    """
    adata = adata_basic.copy()
    zero_mask = (adata.obs["batch"] == "batch_1").to_numpy()
    adata.obsm["protein_expression"][zero_mask] = 0

    ref = _registered_reference_registry(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    assert "protein_batch_mask" in ref["field_registries"]["proteins"]["state_registry"]

    dm = MrTotalVIBatchDataModule(
        adata,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        parallel=False,
    )
    mismatches = _deep_compare(
        ref["field_registries"]["proteins"],
        dm.registry["field_registries"]["proteins"],
        "proteins",
    )
    assert mismatches == []


def test_registry_matches_in_memory_protein_names_uns_key(adata_basic):
    ref = _registered_reference_registry(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        protein_names_uns_key="protein_names",
    )
    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        protein_names_uns_key="protein_names",
        parallel=False,
    )
    mismatches = _deep_compare(
        ref["field_registries"]["proteins"],
        dm.registry["field_registries"]["proteins"],
        "proteins",
    )
    assert mismatches == []


# ---------------------------------------------------------------------------
# Tensor parity
# ---------------------------------------------------------------------------


def test_tensors_match_in_memory_dataloader_default(adata_basic):
    """Unshuffled comparison of every emitted tensor: key set, dtype, shape, values.

    Both sides must be unshuffled, or this check is vacuous.
    """
    ref_loader = _reference_loader(
        adata_basic,
        batch_size=37,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    ref_batch = next(iter(ref_loader))

    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        parallel=False,
    )
    dm_batch = next(iter(dm.inference_dataloader(shuffle=False, batch_size=37)))

    assert set(ref_batch.keys()) == set(dm_batch.keys())
    for key, ref_tensor in ref_batch.items():
        dm_tensor = dm_batch[key]
        assert dm_tensor.dtype == ref_tensor.dtype, (
            f"{key}: dtype {dm_tensor.dtype} != {ref_tensor.dtype}"
        )
        assert dm_tensor.shape == ref_tensor.shape, (
            f"{key}: shape {dm_tensor.shape} != {ref_tensor.shape}"
        )
        assert torch.equal(dm_tensor, ref_tensor), f"{key}: values differ"


def test_tensors_match_in_memory_dataloader_with_covariates(adata_basic):
    ref_loader = _reference_loader(
        adata_basic,
        batch_size=41,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        labels_key="labels",
        categorical_covariate_keys=["cat1"],
        continuous_covariate_keys=["cont1"],
    )
    ref_batch = next(iter(ref_loader))

    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        labels_key="labels",
        categorical_covariate_keys=["cat1"],
        continuous_covariate_keys=["cont1"],
        parallel=False,
    )
    dm_batch = next(iter(dm.inference_dataloader(shuffle=False, batch_size=41)))

    assert set(ref_batch.keys()) == set(dm_batch.keys())
    for key, ref_tensor in ref_batch.items():
        dm_tensor = dm_batch[key]
        assert dm_tensor.dtype == ref_tensor.dtype
        assert torch.equal(dm_tensor, ref_tensor), f"{key}: values differ"


def test_tensors_cover_full_dataset_across_batches(adata_basic):
    """Full unshuffled pass (not just batch 0) matches, including the final
    (possibly ragged) batch and every ind_x value exactly once, in order.
    """
    ref_loader = _reference_loader(
        adata_basic,
        batch_size=64,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        parallel=False,
    )
    dm_loader = dm.inference_dataloader(shuffle=False, batch_size=64)

    ref_batches = list(ref_loader)
    dm_batches = list(dm_loader)
    assert len(ref_batches) == len(dm_batches)
    for ref_batch, dm_batch in zip(ref_batches, dm_batches, strict=True):
        for key in ref_batch:
            assert torch.equal(dm_batch[key], ref_batch[key]), f"{key}: values differ"

    ind_x = torch.cat([b["ind_x"] for b in dm_batches]).flatten()
    assert torch.equal(ind_x, torch.arange(adata_basic.n_obs, dtype=torch.int64))


# ---------------------------------------------------------------------------
# Save/load reconstruction (registry serializability)
# ---------------------------------------------------------------------------


def test_registry_round_trips_through_torch_save(adata_basic, tmp_path):
    """The registry must be a torch.save/torch.load-able plain structure --
    matching how `registry_` travels through `_save_load._load_saved_files`.
    """
    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        labels_key="labels",
        categorical_covariate_keys=["cat1"],
        continuous_covariate_keys=["cont1"],
        parallel=False,
    )
    reg = dm.registry
    path = tmp_path / "registry.pt"
    torch.save(reg, path)
    loaded = torch.load(path, weights_only=False)

    mismatches = _deep_compare(reg, loaded, "registry")
    assert mismatches == []


def test_summary_stats_reconstructable_from_registry(adata_basic):
    """`AnnDataManager._get_summary_stats_from_registry` must succeed on the
    datamodule's registry and produce the correct per-field counts,
    including n_proteins (from `f"n_{registry_key}"` on `ArrayLikeField`).
    """
    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        categorical_covariate_keys=["cat1"],
        continuous_covariate_keys=["cont1"],
        parallel=False,
    )
    stats = AnnDataManager._get_summary_stats_from_registry(dm.registry)
    assert stats["n_vars"] == dm.n_vars == 100
    assert stats["n_proteins"] == dm.n_proteins == 100
    assert stats["n_batch"] == dm.n_batch == 2
    assert stats["n_labels"] == dm.n_labels == 1  # no labels_key -> dummy single category
    assert stats["n_sample"] == dm.n_samples == N_DONORS
    assert stats["n_extra_categorical_covs"] == 1
    assert stats["n_extra_continuous_covs"] == 1


# ---------------------------------------------------------------------------
# Explicitly unsupported / gap-documenting behavior
# ---------------------------------------------------------------------------


def test_size_factor_key_raises_not_implemented(adata_basic):
    with pytest.raises(NotImplementedError, match="size_factor_key"):
        MrTotalVIBatchDataModule(
            adata_basic,
            protein_expression_obsm_key="protein_expression",
            sample_key="sample",
            size_factor_key="some_key",
        )


def test_panel_key_raises_not_implemented(adata_basic):
    with pytest.raises(NotImplementedError, match="panel_key"):
        MrTotalVIBatchDataModule(
            adata_basic,
            protein_expression_obsm_key="protein_expression",
            sample_key="sample",
            panel_key="panel",
        )


def test_mismatched_var_names_across_files_raises():
    adata1 = _make_adata(seed=1)
    adata2 = _make_adata(seed=2)
    adata2 = adata2[:, adata2.var_names[::-1]].copy()  # break var_names alignment

    with pytest.raises(ValueError, match="var_names"):
        MrTotalVIBatchDataModule(
            [adata1, adata2],
            protein_expression_obsm_key="protein_expression",
            protein_names_uns_key="protein_names",
            sample_key="sample",
            batch_key="batch",
            parallel=False,
        )


# ---------------------------------------------------------------------------
# Multi-file (list[AnnData]) collection support
# ---------------------------------------------------------------------------


def test_multi_file_collection_matches_concatenated_in_memory():
    """A 2-file `list[AnnData]` collection must equal a single AnnData built
    by concatenating those same two files, cell-order preserved.
    """
    adata1 = _make_adata(seed=10)
    adata2 = _make_adata(seed=11)
    import anndata as ad

    combined = ad.concat(
        [adata1, adata2], join="inner", merge="same", index_unique="-", keys=["a", "b"]
    )
    combined.uns["protein_names"] = adata1.uns["protein_names"].copy()

    ref = _registered_reference_registry(
        combined,
        protein_expression_obsm_key="protein_expression",
        protein_names_uns_key="protein_names",
        sample_key="sample",
        batch_key="batch",
    )
    dm = MrTotalVIBatchDataModule(
        [adata1, adata2],
        protein_expression_obsm_key="protein_expression",
        protein_names_uns_key="protein_names",
        sample_key="sample",
        batch_key="batch",
        parallel=False,
    )
    mismatches = _deep_compare(
        ref["field_registries"], dm.registry["field_registries"], "field_registries"
    )
    assert mismatches == []
    assert dm.n_obs == adata1.n_obs + adata2.n_obs


# ---------------------------------------------------------------------------
# Internal in-memory backend sanity (not lamindb -- see module "Known gaps")
# ---------------------------------------------------------------------------


def test_in_memory_backend_used_for_anndata_input(adata_basic):
    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        parallel=False,
    )
    assert isinstance(dm._dataset, _InMemoryProteinMappedDataset)


# ---------------------------------------------------------------------------
# DataFrame protein access and axis parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adata_protein_df():
    """`protein_expression` stored as a DataFrame -- a real `setup_anndata` input shape."""
    import pandas as pd

    adata = _make_adata()
    prot = np.asarray(adata.obsm["protein_expression"])
    adata.obsm["protein_expression"] = pd.DataFrame(
        prot,
        index=adata.obs_names,
        columns=[f"prot_{i}" for i in range(prot.shape[1])],
    )
    return adata


def test_dataframe_protein_obsm_uses_positional_rows(adata_protein_df):
    dm = MrTotalVIBatchDataModule(
        adata_protein_df,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        parallel=False,
    )
    batch = next(iter(dm.inference_dataloader(shuffle=False, batch_size=16)))
    expected = adata_protein_df.obsm["protein_expression"].iloc[:16].to_numpy()
    np.testing.assert_array_equal(
        batch[scvi.REGISTRY_KEYS.PROTEIN_EXP_KEY].numpy(),
        expected,
    )


def test_dataframe_protein_obsm_registry_preserves_columns(adata_protein_df):
    ref = _registered_reference_registry(
        adata_protein_df,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
    )
    ref_names = ref["field_registries"]["proteins"]["state_registry"]["column_names"]
    # sanity: the real path does source names from the DataFrame columns
    assert list(ref_names[:3]) == ["prot_0", "prot_1", "prot_2"]

    dm = MrTotalVIBatchDataModule(
        adata_protein_df,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        parallel=False,
    )
    dm_names = dm.registry["field_registries"]["proteins"]["state_registry"]["column_names"]

    assert list(dm_names) == list(ref_names)


# ---------------------------------------------------------------------------
# lamindb backend: error-handling branches, reachable without installing lamindb
#
# `@dependencies("lamindb")` gates on `importlib.import_module`, which consults
# `sys.modules` first -- so stubbing an empty module there unblocks the real
# function body. Only the ERROR paths are reachable this way; the success path
# would need a full fake `MappedCollection` and is deliberately not attempted
# (it would be an untested shadow-reimplementation of lamindb's API).
# ---------------------------------------------------------------------------


def test_is_lamindb_collection_duck_typing(adata_basic):
    """Pure predicate, no decorator involved -- unconditional direct coverage."""
    from scvi.external.mrtotalvi._datamodule import _is_lamindb_collection

    class FakeCollection:
        def mapped(self):  # pragma: no cover - presence is what is tested
            ...

    assert _is_lamindb_collection(FakeCollection()) is True
    assert _is_lamindb_collection(object()) is False  # no .mapped
    assert _is_lamindb_collection(adata_basic) is False  # AnnData excluded
    assert _is_lamindb_collection([adata_basic]) is False  # list excluded


def test_lamindb_dispatch_blocked_without_lamindb_installed():
    """A lamindb-shaped collection dispatches to the lamindb backend, then the decorator blocks.

    Proves the duck-typed dispatch works even though the backend itself cannot run here.
    """

    class FakeCollection:
        def mapped(self, **kwargs):  # pragma: no cover - never reached
            return "unreachable"

    with pytest.raises(ModuleNotFoundError, match="lamindb"):
        MrTotalVIBatchDataModule(
            FakeCollection(),
            protein_expression_obsm_key="protein_expression",
            sample_key="sample",
            parallel=False,
        )


def test_lamindb_backend_refuses_before_calling_mapped(monkeypatch):
    """With lamindb stubbed, the backend is refused before any `.mapped()` call."""
    import types

    monkeypatch.setitem(sys.modules, "lamindb", types.ModuleType("lamindb"))
    calls = []

    class FakeCollection:
        def mapped(self, **kwargs):
            calls.append(kwargs)
            return "should not be reached"

    with pytest.raises(NotImplementedError, match="authoritative protein feature axis"):
        MrTotalVIBatchDataModule(
            FakeCollection(),
            protein_expression_obsm_key="protein_expression",
            sample_key="sample",
            protein_names_uns_key="protein_names",
            parallel=False,
        )
    assert calls == [], "mapped() must not be called when the uns-key gap applies"


def test_lamindb_backend_refuses_even_without_uns_key(monkeypatch):
    """No inferred/default protein axis may make the lamindb backend acceptable."""
    import types

    monkeypatch.setitem(sys.modules, "lamindb", types.ModuleType("lamindb"))
    calls = []

    class FakeCollection:
        def mapped(self, **kwargs):
            calls.append(kwargs)
            return "should not be reached"

    with pytest.raises(NotImplementedError, match="authoritative protein feature axis"):
        MrTotalVIBatchDataModule(
            FakeCollection(),
            protein_expression_obsm_key="protein_expression",
            sample_key="sample",
            parallel=False,
        )
    assert calls == [], "mapped() must not be called by the quarantined backend"


# ---------------------------------------------------------------------------
# _InMemoryProteinMappedDataset: input shapes and helpers no fixture reaches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sparse_format", ["csr", "csc"])
def test_tensors_match_dense_for_sparse_input(sparse_format):
    """Sparse X and protein obsm must emit tensors identical to the dense path.

    Exercises the `.toarray()` branches in `__getitem__`/`full_protein_matrix`, which every
    existing fixture bypasses because `synthetic_iid` returns dense arrays.
    """
    from scipy.sparse import csc_matrix, csr_matrix

    to_sparse = csr_matrix if sparse_format == "csr" else csc_matrix

    dense = _make_adata()
    sparse = dense.copy()
    sparse.X = to_sparse(np.asarray(dense.X))
    sparse.obsm["protein_expression"] = to_sparse(np.asarray(dense.obsm["protein_expression"]))

    kwargs = {
        "protein_expression_obsm_key": "protein_expression",
        "sample_key": "sample",
        "batch_key": "batch",
        "parallel": False,
    }
    dm_dense = MrTotalVIBatchDataModule(dense, **kwargs)
    dm_sparse = MrTotalVIBatchDataModule(sparse, **kwargs)

    b_dense = next(iter(dm_dense.inference_dataloader(shuffle=False, batch_size=32)))
    b_sparse = next(iter(dm_sparse.inference_dataloader(shuffle=False, batch_size=32)))

    assert set(b_dense) == set(b_sparse)
    for key in b_dense:
        torch.testing.assert_close(b_sparse[key], b_dense[key])


def test_locate_resolves_file_and_local_index_at_every_boundary():
    """3 files, so both interior boundaries are checked -- the existing multi-file test uses 2."""
    a1, a2, a3 = _make_adata(seed=1), _make_adata(seed=2), _make_adata(seed=3)
    ds = _InMemoryProteinMappedDataset(
        [a1, a2, a3],
        obs_keys=["sample"],
        protein_expression_obsm_key="protein_expression",
        protein_names_uns_key="protein_names",
    )
    n = a1.n_obs

    assert ds._locate(0) == (0, 0)
    assert ds._locate(n - 1) == (0, n - 1)
    assert ds._locate(n) == (1, 0)
    assert ds._locate(2 * n - 1) == (1, n - 1)
    assert ds._locate(2 * n) == (2, 0)
    assert ds._locate(ds.n_obs - 1) == (2, n - 1)


def test_mismatched_protein_column_count_across_files_raises():
    a1, a2 = _make_adata(seed=1), _make_adata(seed=2)
    prot = np.asarray(a2.obsm["protein_expression"])
    a2.obsm["protein_expression"] = prot[:, : prot.shape[1] // 2]

    with pytest.raises(ValueError, match="same number of"):
        _InMemoryProteinMappedDataset(
            [a1, a2],
            obs_keys=["sample"],
            protein_expression_obsm_key="protein_expression",
            protein_names_uns_key="protein_names",
        )


def test_multifile_same_width_reordered_protein_axis_raises():
    a1, a2 = _make_adata(seed=1), _make_adata(seed=2)
    a2.uns["protein_names"] = a2.uns["protein_names"][::-1].copy()

    with pytest.raises(ValueError, match="exact order"):
        _InMemoryProteinMappedDataset(
            [a1, a2],
            obs_keys=["sample"],
            protein_expression_obsm_key="protein_expression",
            protein_names_uns_key="protein_names",
        )


def test_multifile_same_width_renamed_protein_axis_raises():
    a1, a2 = _make_adata(seed=1), _make_adata(seed=2)
    a2.uns["protein_names"] = a2.uns["protein_names"].copy()
    a2.uns["protein_names"][0] = "renamed"

    with pytest.raises(ValueError, match="exact order"):
        _InMemoryProteinMappedDataset(
            [a1, a2],
            obs_keys=["sample"],
            protein_expression_obsm_key="protein_expression",
            protein_names_uns_key="protein_names",
        )


@pytest.mark.parametrize("invalid_names", [["", "p1"], ["p0", "p0"]])
def test_multifile_empty_or_duplicate_protein_names_raise(invalid_names):
    a1, a2 = _make_adata(seed=1), _make_adata(seed=2)
    width = a1.obsm["protein_expression"].shape[1]
    names = [f"p{i}" for i in range(width)]
    names[:2] = invalid_names
    a1.uns["protein_names"] = np.asarray(names, dtype=object)
    a2.uns["protein_names"] = np.asarray(names, dtype=object)

    with pytest.raises(ValueError, match="non-empty|unique"):
        _InMemoryProteinMappedDataset(
            [a1, a2],
            obs_keys=["sample"],
            protein_expression_obsm_key="protein_expression",
            protein_names_uns_key="protein_names",
        )


def test_dataframe_and_uns_protein_axes_must_agree_exactly(adata_protein_df):
    adata = adata_protein_df.copy()
    adata.uns["protein_names"] = np.asarray(
        adata.obsm["protein_expression"].columns,
        dtype=object,
    ).copy()
    adata.uns["protein_names"][0] = "same_width_but_renamed"

    with pytest.raises(ValueError, match="agree exactly"):
        _InMemoryProteinMappedDataset(
            [adata],
            obs_keys=["sample"],
            protein_expression_obsm_key="protein_expression",
            protein_names_uns_key="protein_names",
        )


def test_training_and_validation_protein_axes_must_agree_exactly():
    train = _make_adata(seed=1)
    validation = _make_adata(seed=2)
    validation.uns["protein_names"] = validation.uns["protein_names"][::-1].copy()

    with pytest.raises(ValueError, match="Training and validation.*exact order"):
        MrTotalVIBatchDataModule(
            train,
            collection_val=validation,
            protein_expression_obsm_key="protein_expression",
            protein_names_uns_key="protein_names",
            sample_key="sample",
            batch_key="batch",
            parallel=False,
        )


def test_training_and_validation_require_authoritative_protein_axes():
    train = _make_adata(seed=1)
    validation = _make_adata(seed=2)
    train.uns.pop("protein_names")
    validation.uns.pop("protein_names")

    with pytest.raises(ValueError, match="require authoritative names"):
        MrTotalVIBatchDataModule(
            train,
            collection_val=validation,
            protein_expression_obsm_key="protein_expression",
            sample_key="sample",
            batch_key="batch",
            parallel=False,
        )


@pytest.mark.parametrize(
    "malformed_name",
    [None, ["nested"], ("tuple",)],
)
def test_list_like_or_non_string_protein_name_fails_cleanly(malformed_name):
    adata = _make_adata(seed=1)
    names = adata.uns["protein_names"].astype(object).tolist()
    names[0] = malformed_name
    adata.uns["protein_names"] = names

    with pytest.raises(ValueError, match="one-dimensional|non-empty strings"):
        _InMemoryProteinMappedDataset(
            [adata],
            obs_keys=["sample"],
            protein_expression_obsm_key="protein_expression",
            protein_names_uns_key="protein_names",
        )


def test_multifile_without_authoritative_protein_names_raises():
    a1, a2 = _make_adata(seed=1), _make_adata(seed=2)
    a1.uns.pop("protein_names")
    a2.uns.pop("protein_names")

    with pytest.raises(ValueError, match="require authoritative names"):
        _InMemoryProteinMappedDataset(
            [a1, a2],
            obs_keys=["sample"],
            protein_expression_obsm_key="protein_expression",
        )


def test_empty_collection_list_raises():
    with pytest.raises(ValueError, match="at least one AnnData"):
        _InMemoryProteinMappedDataset(
            [],
            obs_keys=["sample"],
            protein_expression_obsm_key="protein_expression",
        )


def test_full_protein_matrix_and_full_encoded_column_direct():
    """Direct calls -- both helpers are otherwise only reached transitively."""
    a1, a2 = _make_adata(seed=1), _make_adata(seed=2)
    ds = _InMemoryProteinMappedDataset(
        [a1, a2],
        obs_keys=["sample", "batch"],
        protein_expression_obsm_key="protein_expression",
        protein_names_uns_key="protein_names",
    )

    expected = np.concatenate(
        [
            np.asarray(a1.obsm["protein_expression"]),
            np.asarray(a2.obsm["protein_expression"]),
        ],
        axis=0,
    ).astype(np.float32)
    np.testing.assert_array_equal(ds.full_protein_matrix(), expected)

    codes = ds.full_encoded_column("sample")
    assert codes.dtype == np.int64
    assert codes.shape == (ds.n_obs,)
    assert set(codes.tolist()) == set(range(len(ds.encoders["sample"])))


def test_protein_batch_mask_absent_when_no_batch_is_fully_zero(adata_basic):
    """Asserts the `return None` path directly, not as a deep-compare side effect."""
    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        parallel=False,
    )
    state = dm.registry["field_registries"]["proteins"]["state_registry"]

    assert "protein_batch_mask" not in state


def test_continuous_covariates_bypass_the_categorical_encoder(adata_basic):
    """A continuous key must be passed through as a float, never categorically encoded."""
    dm = MrTotalVIBatchDataModule(
        adata_basic,
        protein_expression_obsm_key="protein_expression",
        sample_key="sample",
        batch_key="batch",
        continuous_covariate_keys=["cont1"],
        parallel=False,
    )

    assert "cont1" not in dm._dataset.encoders
    assert "cont1" not in dm._dataset.categories
    assert "cont1" in dm._dataset._continuous_keys
