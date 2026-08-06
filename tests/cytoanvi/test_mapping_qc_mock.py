"""mapping_qc helpers tested with mocked mapqc (no mapqc install required)."""

import numpy as np
import pytest
from conftest import (
    BATCH_KEY,
    LABELS_KEY,
    N_EPOCHS,
    SAMPLE_KEY,
    SCALED_LAYER_KEY,
    UNLABELED,
    make_adata,
    setup_and_train,
)

from cytoanvi import mapping_qc


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
    adata = make_adata()
    work, is_ref = _assign_mapqc_samples(adata)
    model = setup_and_train(work.copy())

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


def test_build_mapqc_anndata_handles_label_category_absent_from_reference():
    """F-039: query has a label category the model's registry never saw.

    Reproduces the real Roider B9 failure mode (``ValueError: Category 31 not found in source
    registry. Cannot transfer setup without extend_categories=True``) at unit-test scale: the
    model is trained on a reference subset that excludes one label category entirely, and
    ``query`` (a disjoint AnnData, structurally like the post-scArches query model's own
    ``.adata`` differs from both ``reference_adata``/``query_adata`` in the real B9 pipeline)
    carries cells of that category. Before the fix this raised at
    ``model.get_latent_representation`` inside ``build_mapqc_anndata``.
    """
    adata = make_adata()
    work, is_ref = _assign_mapqc_samples(adata)
    novel_label = "label_4"
    assert novel_label in work.obs[LABELS_KEY].astype(str).unique()

    # Keep the existing batch-based ref/query split, but additionally strip the novel label
    # out of the reference side so the trained model's registry never observes it.
    ref_mask = is_ref & (work.obs[LABELS_KEY].astype(str) != novel_label).to_numpy()
    query_mask = ~ref_mask
    ref = work[ref_mask].copy()
    query = work[query_mask].copy()
    assert novel_label not in ref.obs[LABELS_KEY].astype(str).unique()
    assert novel_label in query.obs[LABELS_KEY].astype(str).unique()

    model = setup_and_train(ref.copy())

    # The pre-fix code path (`self.adata_manager.transfer_fields(adata)` with no
    # `extend_categories`) must not be reached — build_mapqc_anndata should register `ref`/
    # `query` explicitly instead of falling back to it. Fail loudly if it is.
    def _unsafe_transfer_fields(*args, **kwargs):
        raise AssertionError(
            "build_mapqc_anndata must not fall back to the unkwarg'd "
            "adata_manager.transfer_fields path"
        )

    model.adata_manager.transfer_fields = _unsafe_transfer_fields

    joint = mapping_qc.build_mapqc_anndata(
        model,
        ref,
        query,
        sample_key="mapqc_sample",
    )

    assert joint.n_obs == ref.n_obs + query.n_obs
    assert mapping_qc.DEFAULT_EMB_KEY in joint.obsm
    assert joint.obsm[mapping_qc.DEFAULT_EMB_KEY].shape[0] == joint.n_obs

    # The true label column must survive untouched — build_mapqc_anndata must only neutralize
    # out-of-registry categories on a throwaway registration copy, never on the returned data.
    query_rows = joint.obs[mapping_qc.DEFAULT_REF_Q_KEY] == mapping_qc.QUERY_CAT
    assert (joint.obs.loc[query_rows, LABELS_KEY].astype(str) == novel_label).any()


def test_build_mapqc_anndata_handles_label_only_category_absent_from_reference():
    """F-039, isolated to the labels field specifically.

    The previous test's ref/query split is batch-partitioned (via ``_assign_mapqc_samples``), so
    its pre-fix failure is actually raised by the *batch* field (``ref`` never sees ``batch_1``),
    not the labels field — both are fixed by the same change, but that test alone doesn't prove
    the labels-specific half of the fix (remapping an out-of-registry label to
    ``unlabeled_category`` on a scratch copy), which is the half that ``extend_categories=True``
    cannot rescue (``LabelsWithUnlabeledObsField.transfer_field`` hard-codes
    ``extend_categories=False``).

    Here ``ref``/``query`` share **both** original batches (split by index parity instead), so
    the only out-of-registry category anywhere is the excluded label. Verified against the real
    production traceback (job 25211799,
    ``.scratch/cytoanvi-benchmark/slurm/out/cytoanvi_b9_roider_full_25211799.log``): the
    ``ValueError`` there is raised from ``scvi/data/fields/_scanvi.py``
    (``LabelsWithUnlabeledObsField.transfer_field`` -> ``CategoricalObsField.transfer_field``),
    confirming the real B9 failure was the labels field, matching this test's construction.
    """
    adata = make_adata()
    novel_label = "label_4"
    labels = adata.obs[LABELS_KEY].astype(str)
    assert novel_label in labels.unique()

    is_novel = (labels == novel_label).to_numpy()
    idx = np.arange(adata.n_obs)
    ref_mask = (idx % 2 == 0) & ~is_novel
    query_mask = ~ref_mask
    ref = adata[ref_mask].copy()
    query = adata[query_mask].copy()
    assert novel_label not in ref.obs[LABELS_KEY].astype(str).unique()
    assert novel_label in query.obs[LABELS_KEY].astype(str).unique()
    # Both sides must see both batches, so the batch field never sees a novel category either —
    # isolating the labels-field bypass from the (already-covered) batch-field one.
    assert set(ref.obs[BATCH_KEY].astype(str)) == set(adata.obs[BATCH_KEY].astype(str))
    assert set(query.obs[BATCH_KEY].astype(str)) == set(adata.obs[BATCH_KEY].astype(str))

    ref.obs["mapqc_sample"] = np.take(["r0", "r1", "r2", "r3"], np.arange(ref.n_obs) % 4)
    query.obs["mapqc_sample"] = np.take(["q0", "q1"], np.arange(query.n_obs) % 2)

    model = setup_and_train(ref.copy())

    def _unsafe_transfer_fields(*args, **kwargs):
        raise AssertionError(
            "build_mapqc_anndata must not fall back to the unkwarg'd "
            "adata_manager.transfer_fields path"
        )

    model.adata_manager.transfer_fields = _unsafe_transfer_fields

    joint = mapping_qc.build_mapqc_anndata(model, ref, query, sample_key="mapqc_sample")

    assert joint.n_obs == ref.n_obs + query.n_obs
    assert joint.obsm[mapping_qc.DEFAULT_EMB_KEY].shape[0] == joint.n_obs
    query_rows = joint.obs[mapping_qc.DEFAULT_REF_Q_KEY] == mapping_qc.QUERY_CAT
    assert (joint.obs.loc[query_rows, LABELS_KEY].astype(str) == novel_label).any()


def test_build_mapqc_anndata_raises_on_few_ref_samples():
    adata = make_adata()
    model = setup_and_train(adata)
    adata.obs["mapqc_sample"] = "only_one"
    with pytest.raises(ValueError, match="at least 3 reference samples"):
        mapping_qc.build_mapqc_anndata(model, adata, adata, sample_key="mapqc_sample")


def test_query_control_mapqc_rate_validates_ref_query_column():
    adata = make_adata()
    adata.obs["mapqc_score"] = 1.0
    adata.obs["mapqc_filtering"] = "pass"
    adata.obs["status"] = "control"

    with pytest.raises(ValueError, match=mapping_qc.DEFAULT_REF_Q_KEY):
        mapping_qc.query_control_mapqc_rate(
            adata,
            control_value="control",
            case_control_key="status",
        )


def test_run_mapqc_on_cytoanvi_mock(monkeypatch):
    adata = make_adata()
    work, is_ref = _assign_mapqc_samples(adata)
    ref = work[is_ref].copy()
    query = work[~is_ref].copy()
    model = setup_and_train(work.copy())

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
    adata = make_adata()
    work, is_ref = _assign_mapqc_samples(adata)
    ref = work[is_ref].copy()
    query = work[~is_ref].copy()
    model = setup_and_train(work.copy())
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


def test_task_b9_mapqc_returns_blocked_artifact_before_training(monkeypatch):
    from benchmarks.cytoanvi import data, tasks

    _, p1, _ = data.make_synthetic_panels()

    def _missing_mapqc():
        raise ImportError(mapping_qc._MAPQC_INSTALL_MSG)

    def _unexpected_train(*args, **kwargs):
        raise AssertionError("train_cytoanvi should not run before mapQC dependency validation")

    monkeypatch.setattr(mapping_qc, "_require_mapqc", _missing_mapqc)
    monkeypatch.setattr(tasks, "train_cytoanvi", _unexpected_train)

    result = tasks.task_b9_mapqc(
        p1,
        unlabeled_category="label_0",
        labels_key="labels",
        batch_key="batch",
        seed=0,
        max_epochs=2,
        run_mapqc=True,
    )

    assert result["status"] == "blocked"
    assert "cytoanvi-mapping-qc" in result["blocked_reason"]


def test_run_mapqc_raises_without_extra(monkeypatch):
    adata = make_adata()
    work, is_ref = _assign_mapqc_samples(adata)
    ref = work[is_ref].copy()
    query = work[~is_ref].copy()
    model = setup_and_train(work.copy())
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


def test_patched_get_per_cell_filtering_info_guards_empty_mode():
    """mapqc 0.1.1 crashes on ``mode().iloc[0]`` when no cell has a filter reason (empty mode).

    Reproduces the B9 blocker: all neighbourhoods pass (filter_info all None) so pandas
    ``mode()`` returns a 0-row frame. The upstream ``.iloc[0]`` raises IndexError; our guarded
    reimplementation must not crash and must still label scored/unsampled cells correctly.
    """
    import pandas as pd

    f = mapping_qc._patched_get_per_cell_filtering_info
    n_nhoods, n_cells = 4, 6
    scores = np.array([0.5, np.nan, 1.2, np.nan, np.nan, np.nan])
    mask = np.zeros((n_nhoods, n_cells), dtype=int)
    mask[0, [0, 1]] = 1
    mask[1, [2, 3]] = 1
    mask[2, [4]] = 1  # cell 5 in zero neighbourhoods

    # empty-mode case (all filter_info None) — the exact upstream crash
    out = f(scores, mask, pd.DataFrame({"filter_info": [None] * n_nhoods}))
    assert out[0] == "pass" and out[2] == "pass"
    assert out[5] == "not sampled"

    # non-empty case — most-prevalent reason still filled for sampled-but-unscored cells
    out2 = f(scores, mask, pd.DataFrame({"filter_info": ["low_n", None, "low_n", "high_var"]}))
    assert out2[1] == "low_n"
    assert out2[0] == "pass" and out2[5] == "not sampled"


def test_patch_mapqc_empty_mode_noop_without_mapqc():
    """The patch installer must be a safe no-op when mapqc is not importable."""
    # Should not raise regardless of whether mapqc is installed.
    mapping_qc._patch_mapqc_empty_mode()
