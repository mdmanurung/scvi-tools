"""Tests for the optional CytoANVI AnnBatch benchmark backend."""

from __future__ import annotations

import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from benchmarks.common.training import NAN_LAYER, SCALED_LAYER
from benchmarks.cytoanvi import run as cyto_run
from benchmarks.cytoanvi.data import make_synthetic_panels
from cytoanvi import CytoANVI


def test_annbatch_missing_dependency_is_clear(monkeypatch, tmp_path):
    from benchmarks.common.annbatch import AnnBatchConfig, make_cytoanvi_annbatch_datamodule

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "annbatch":
            raise ModuleNotFoundError("No module named 'annbatch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    adata = ad.AnnData(
        X=np.ones((12, 4), dtype=np.float32),
        obs=pd.DataFrame(
            {"batch": ["b0"] * 6 + ["b1"] * 6, "labels": ["A", "B", "Unknown"] * 4}
        ),
    )
    CytoANVI.setup_anndata(
        adata,
        layer=None,
        batch_key="batch",
        labels_key="labels",
        unlabeled_category="Unknown",
    )
    model = CytoANVI(adata, n_latent=2)

    with pytest.raises(ImportError, match="Install the optional extra"):
        make_cytoanvi_annbatch_datamodule(
            model,
            AnnBatchConfig(enabled=True, cache_dir=tmp_path),
            batch_size=4,
        )


def test_cli_does_not_import_annbatch_for_default_runs(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("AnnBatch config should not be built for default CLI runs")

    monkeypatch.setattr(cyto_run, "AnnBatchConfig", fail_if_called, raising=False)

    args = types.SimpleNamespace(annbatch=False, annbatch_cache_dir=None)

    assert cyto_run._annbatch_config_from_args(args) is None


def test_cli_passes_annbatch_config_to_tasks(monkeypatch, tmp_path):
    seen = {}

    def record_task(name):
        def _record(*args, **kwargs):
            seen[name] = kwargs["annbatch_config"]
            return {"task": name}

        return _record

    monkeypatch.setattr(cyto_run.task_mod, "task_b1_label_transfer", record_task("b1"))
    monkeypatch.setattr(cyto_run.task_mod, "task_b2_integration", record_task("b2"))
    monkeypatch.setattr(
        cyto_run.task_mod, "task_b3_panel_divergent", lambda *a, **k: {"task": "b3"}
    )
    monkeypatch.setattr(cyto_run.task_mod, "task_b4_continual", record_task("b4"))
    monkeypatch.setattr(cyto_run.task_mod, "task_b5_novelty", record_task("b5"))
    monkeypatch.setattr(
        cyto_run.task_mod, "task_b6_lambda_sweep", lambda *a, **k: {"task": "b6"}
    )
    monkeypatch.setattr(cyto_run.task_mod, "task_b8_hce_label_transfer", record_task("b8"))
    monkeypatch.setattr(cyto_run.task_mod, "task_b9_mapqc", record_task("b9"))
    monkeypatch.setattr(cyto_run, "task_b7_multimodal_integration", record_task("b7"))

    args = types.SimpleNamespace(
        dataset="paired-rna-cytof",
        task="all",
        labels_key="labels",
        unlabeled="Unknown",
        batch_key="batch",
        sample_key=None,
        seed=0,
        max_epochs=1,
        batch_size=16,
        n_samples_per_label=None,
        reduce_lr_on_plateau=False,
        subsample_per_batch=10,
        holdout_sweep=False,
        holdout_type=None,
        ewc_lambdas=None,
        hierarchy_edges=None,
        mapqc_run=True,
        mapqc_n_nhoods=1,
        mapqc_k_min=2,
        mapqc_k_max=5,
        annbatch=True,
        annbatch_cache_dir=tmp_path,
        annbatch_chunk_size=8,
        annbatch_preload_nchunks=2,
    )

    p1 = ad.AnnData(
        X=np.ones((20, 4), dtype=np.float32),
        obs=pd.DataFrame({"labels": ["A"] * 20, "batch": ["b0"] * 20}),
    )
    p2 = p1.copy()

    cyto_run._run_tasks(args, p1, p2, "Unknown", seed=0)

    assert set(seen) == {"b1", "b2", "b4", "b5", "b7", "b8", "b9"}
    assert all(cfg.enabled for cfg in seen.values())
    assert {cfg.cache_dir for cfg in seen.values()} == {tmp_path}


def test_cli_passes_annbatch_reuse_cache_config(tmp_path):
    args = types.SimpleNamespace(
        annbatch=True,
        annbatch_cache_dir=tmp_path,
        annbatch_chunk_size=8,
        annbatch_preload_nchunks=2,
        annbatch_cache_mode="reuse",
        annbatch_cache_key="roider/full seed:0",
    )

    cfg = cyto_run._annbatch_config_from_args(args)

    assert cfg.enabled
    assert cfg.cache_dir == tmp_path
    assert cfg.cache_mode == "reuse"
    assert cfg.cache_key == "roider/full seed:0"


def test_annbatch_reuse_cache_key_is_stable_and_sanitized(tmp_path):
    from benchmarks.common.annbatch import AnnBatchConfig, AnnBatchSemiSupervisedDataModule

    adata = ad.AnnData(
        X=np.ones((12, 4), dtype=np.float32),
        obs=pd.DataFrame(
            {"batch": ["b0"] * 6 + ["b1"] * 6, "labels": ["A", "B", "Unknown"] * 4}
        ),
    )
    CytoANVI.setup_anndata(
        adata,
        layer=None,
        batch_key="batch",
        labels_key="labels",
        unlabeled_category="Unknown",
    )
    model = CytoANVI(adata, n_latent=2)
    cfg = AnnBatchConfig(
        enabled=True,
        cache_dir=tmp_path,
        cache_mode="reuse",
        cache_key="roider/full seed:0",
    )

    kwargs = dict(
        adata_manager=model.adata_manager,
        config=cfg,
        batch_size=4,
        dataset_collection_cls=object,
        loader_cls=object,
    )
    dm1 = AnnBatchSemiSupervisedDataModule(**kwargs)
    dm2 = AnnBatchSemiSupervisedDataModule(**kwargs)

    assert dm1._cache_root == dm2._cache_root
    assert dm1._cache_root == tmp_path / "roider_full_seed_0"


@pytest.mark.optional
def test_annbatch_adapter_matches_standard_loader_batch_shapes(tmp_path):
    pytest.importorskip("annbatch")

    from benchmarks.common.annbatch import AnnBatchConfig, make_cytoanvi_annbatch_datamodule
    from scvi.dataloaders import SemiSupervisedDataSplitter

    adata, _, _ = make_synthetic_panels()
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER,
        batch_key="batch",
        labels_key="labels",
        unlabeled_category="label_0",
        nan_layer=NAN_LAYER,
    )
    model = CytoANVI(adata, n_latent=2)
    standard = SemiSupervisedDataSplitter(
        adata_manager=model.adata_manager,
        train_size=0.9,
        batch_size=32,
    )
    standard.setup()

    annbatch_dm = make_cytoanvi_annbatch_datamodule(
        model,
        AnnBatchConfig(
            enabled=True,
            cache_dir=tmp_path,
            chunk_size=16,
            preload_nchunks=2,
        ),
        batch_size=32,
    )
    annbatch_dm.setup()

    standard_full, standard_labelled = next(iter(standard.train_dataloader()))
    ann_full, ann_labelled = next(iter(annbatch_dm.train_dataloader()))

    assert set(ann_full) == set(standard_full)
    assert set(ann_labelled) == set(standard_labelled)
    for key, value in standard_full.items():
        assert ann_full[key].shape[1:] == value.shape[1:]
        assert ann_full[key].dtype == value.dtype


@pytest.mark.optional
def test_annbatch_semisupervised_batch_preserves_required_tensors(tmp_path):
    pytest.importorskip("annbatch")

    from benchmarks.common.annbatch import AnnBatchConfig, make_cytoanvi_annbatch_datamodule

    adata, _, _ = make_synthetic_panels()
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER,
        batch_key="batch",
        labels_key="labels",
        unlabeled_category="label_0",
        nan_layer=NAN_LAYER,
    )
    model = CytoANVI(adata, n_latent=2)
    datamodule = make_cytoanvi_annbatch_datamodule(
        model,
        AnnBatchConfig(enabled=True, cache_dir=tmp_path, chunk_size=16, preload_nchunks=2),
        batch_size=32,
    )
    datamodule.setup()

    batch = next(iter(datamodule.train_dataloader()))

    assert isinstance(batch, tuple)
    assert len(batch) == 2
    full, labelled = batch
    for tensors in (full, labelled):
        assert {"X", "batch", "labels", "nan_layer"} <= set(tensors)
        assert tensors["X"].shape[0] <= 32
        assert tensors["batch"].shape == (tensors["X"].shape[0], 1)
        assert tensors["labels"].shape == (tensors["X"].shape[0], 1)
        assert tensors["nan_layer"].shape == tensors["X"].shape
