"""mapping_qc helpers tested with mocked mapqc (no mapqc install required)."""

import numpy as np
import pytest

from scvi.data import synthetic_iid
from scvi.external import CytoANVI
from scvi.external import cytovi as cytovi_pp
from scvi.external.cytoanvi import mapping_qc

SCALED_LAYER_KEY = "scaled"
BATCH_KEY = "batch"
LABELS_KEY = "labels"
SAMPLE_KEY = "sample_key"
UNLABELED = "label_0"
N_EPOCHS = 2


def _make_adata(n_batches=2, n_labels=5):
    adata = synthetic_iid(
        batch_size=256,
        n_genes=30,
        n_proteins=0,
        n_regions=0,
        n_batches=n_batches,
        n_labels=n_labels,
        rna_dist="normal",
    )
    adata.obs[SAMPLE_KEY] = np.random.choice(["group_a", "group_b"], size=adata.shape[0])
    adata.layers["raw"] = adata.X.copy()
    cytovi_pp.transform_arcsinh(adata)
    cytovi_pp.scale(adata)
    return adata


def _setup_and_train(adata):
    CytoANVI.setup_anndata(
        adata,
        layer=SCALED_LAYER_KEY,
        batch_key=BATCH_KEY,
        labels_key=LABELS_KEY,
        unlabeled_category=UNLABELED,
        sample_key=SAMPLE_KEY,
    )
    model = CytoANVI(adata, n_latent=10)
    model.train(max_epochs=N_EPOCHS, accelerator="cpu")
    return model


def _assign_mapqc_samples(adata):
    adata = adata.copy()
    batches = sorted(adata.obs[BATCH_KEY].astype(str).unique())
    is_ref = adata.obs[BATCH_KEY].astype(str) == batches[0]
    adata.obs["mapqc_sample"] = ""
    ref_idx = np.where(is_ref.to_numpy())[0]
    query_idx = np.where((~is_ref).to_numpy())[0]
    ref_samples = ["r0", "r1", "r2", "r3"]
    query_samples = ["q0", "q1"]
    adata.obs.iloc[ref_idx, adata.obs.columns.get_loc("mapqc_sample")] = np.take(
        ref_samples, np.arange(len(ref_idx)) % len(ref_samples)
    )
    adata.obs.iloc[query_idx, adata.obs.columns.get_loc("mapqc_sample")] = np.take(
        query_samples, np.arange(len(query_idx)) % len(query_samples)
    )
    return adata, is_ref.to_numpy()


def test_build_mapqc_anndata_shapes():
    adata = _make_adata()
    work, is_ref = _assign_mapqc_samples(adata)
    model = _setup_and_train(work.copy())

    joint = mapping_qc.build_mapqc_anndata(
        model,
        work[is_ref].copy(),
        work[~is_ref].copy(),
        sample_key="mapqc_sample",
    )

    assert joint.n_obs == work.n_obs
    assert mapping_qc.DEFAULT_EMB_KEY in joint.obsm
    assert set(joint.obs[mapping_qc.DEFAULT_REF_Q_KEY].unique()) == {
        mapping_qc.REF_CAT,
        mapping_qc.QUERY_CAT,
    }


def test_build_mapqc_anndata_raises_on_few_ref_samples():
    adata = _make_adata()
    model = _setup_and_train(adata)
    adata.obs["mapqc_sample"] = "only_one"
    with pytest.raises(ValueError, match="at least 3 reference samples"):
        mapping_qc.build_mapqc_anndata(model, adata, adata, sample_key="mapqc_sample")


def test_run_mapqc_on_cytoanvi_mock(monkeypatch):
    adata = _make_adata()
    work, is_ref = _assign_mapqc_samples(adata)
    ref = work[is_ref].copy()
    query = work[~is_ref].copy()
    model = _setup_and_train(work.copy())

    def fake_run_mapqc(joint, **kwargs):
        query_mask = joint.obs[mapping_qc.DEFAULT_REF_Q_KEY] == mapping_qc.QUERY_CAT
        joint.obs["mapqc_score"] = np.nan
        joint.obs.loc[query_mask, "mapqc_score"] = 1.5
        joint.obs["mapqc_filtering"] = None
        joint.obs.loc[query_mask, "mapqc_filtering"] = "pass"
        joint.obs["mapqc_nhood_filtering"] = None
        joint.obs["mapqc_nhood_number"] = np.nan
        joint.obs["mapqc_k"] = np.nan
        joint.uns["mapqc_params"] = {"n_nhoods": kwargs["n_nhoods"]}
        return joint

    monkeypatch.setattr(mapping_qc, "_require_mapqc", lambda: None)
    monkeypatch.setattr(mapping_qc, "run_mapqc_on_joint", fake_run_mapqc)

    joint = mapping_qc.run_mapqc_on_cytoanvi(
        model,
        ref,
        query,
        sample_key="mapqc_sample",
        n_nhoods=3,
        k_min=10,
        k_max=20,
    )

    query_scores = joint.obs.loc[
        joint.obs[mapping_qc.DEFAULT_REF_Q_KEY] == mapping_qc.QUERY_CAT, "mapqc_score"
    ]
    assert query_scores.notna().all()


def test_score_query_mapping_delegates(monkeypatch):
    adata = _make_adata()
    work, is_ref = _assign_mapqc_samples(adata)
    ref = work[is_ref].copy()
    query = work[~is_ref].copy()
    model = _setup_and_train(work.copy())
    sentinel = object()

    def fake_run(*args, **kwargs):
        return sentinel

    monkeypatch.setattr(mapping_qc, "run_mapqc_on_cytoanvi", fake_run)
    assert model.score_query_mapping(
        ref,
        query,
        sample_key="mapqc_sample",
        n_nhoods=3,
        k_min=10,
        k_max=20,
    ) is sentinel


def test_task_b9_plumbing_only():
    from benchmarks.cytoanvi import data, tasks

    _, p1, _ = data.make_synthetic_panels()
    result = tasks.task_b9_mapqc(
        p1,
        unlabeled_category="label_0",
        labels_key="labels",
        batch_key="batch",
        seed=0,
        max_epochs=2,
        run_mapqc=False,
    )
    assert result["status"] == "plumbing_only"
    assert result["joint_n_obs"] == p1.n_obs
    assert "query_label_transfer" in result


def test_task_b9_mapqc_requires_extra_before_training(monkeypatch):
    from benchmarks.cytoanvi import data, tasks

    _, p1, _ = data.make_synthetic_panels()

    def _missing_mapqc():
        raise ImportError(mapping_qc._MAPQC_INSTALL_MSG)

    def _unexpected_train(*args, **kwargs):
        raise AssertionError("train_cytoanvi should not run before mapQC dependency validation")

    monkeypatch.setattr(mapping_qc, "_require_mapqc", _missing_mapqc)
    monkeypatch.setattr(tasks, "train_cytoanvi", _unexpected_train)

    with pytest.raises(ImportError, match="cytoanvi-mapping-qc"):
        tasks.task_b9_mapqc(
            p1,
            unlabeled_category="label_0",
            labels_key="labels",
            batch_key="batch",
            seed=0,
            max_epochs=2,
            run_mapqc=True,
        )


def test_run_mapqc_raises_without_extra(monkeypatch):
    adata = _make_adata()
    work, is_ref = _assign_mapqc_samples(adata)
    ref = work[is_ref].copy()
    query = work[~is_ref].copy()
    model = _setup_and_train(work.copy())
    joint = mapping_qc.build_mapqc_anndata(model, ref, query, sample_key="mapqc_sample")

    def _missing_mapqc():
        raise ImportError(mapping_qc._MAPQC_INSTALL_MSG)

    monkeypatch.setattr(mapping_qc, "_require_mapqc", _missing_mapqc)
    with pytest.raises(ImportError, match="cytoanvi-mapping-qc"):
        mapping_qc.run_mapqc_on_joint(
            joint,
            sample_key="mapqc_sample",
            n_nhoods=3,
            k_min=10,
            k_max=20,
        )
