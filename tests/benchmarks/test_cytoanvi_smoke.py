"""Smoke test for the CytoANVI benchmark CLI (synthetic, no download)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.cytoanvi import metrics
from benchmarks.cytoanvi import run as cyto_run
from benchmarks.cytoanvi import tasks
import cytoanvi
from scvi.model.base import SemisupervisedTrainingMixin

REPO = Path(__file__).resolve().parents[2]


def test_b1_diagnostics_detect_prediction_collapse():
    y_train = np.asarray(["A"] * 10 + ["B"] * 3 + ["C"])
    y_true = np.asarray(["A"] * 8 + ["B"] * 2 + ["C"])
    y_pred = np.asarray(["A"] * len(y_true))

    out = metrics.label_transfer_diagnostics(y_train, y_true, y_pred, rare_max_count=2)

    assert out["n_predicted_labels"] == 1
    assert out["predicted_label_coverage"] == 1 / 3
    assert out["majority_prediction_fraction"] == 1.0
    assert out["collapse_warning"] is True
    assert out["rare_labels"] == ["B", "C"]
    assert "rare_macro_f1" in out


def test_balanced_recipe_resolves_training_config():
    args = type(
        "Args",
        (),
        {
            "cytoanvi_recipe": "balanced",
            "class_weighting": None,
            "class_weight_clip": 10.0,
            "y_prior": None,
            "classification_ratio": None,
            "reduce_lr_on_plateau": False,
        },
    )()

    config = cyto_run._cytoanvi_training_config_from_args(args)

    assert config["y_prior"] == "empirical"
    assert config["class_weighting"] == "sqrt_inverse_frequency"
    assert config["class_weight_clip"] == 10.0
    assert config["reduce_lr_on_plateau"] is True


def test_publication_recipe_resolves_training_config():
    args = type(
        "Args",
        (),
        {
            "cytoanvi_recipe": "publication",
            "class_weighting": None,
            "class_weight_clip": 10.0,
            "y_prior": None,
            "classification_ratio": None,
            "reduce_lr_on_plateau": False,
            "learning_rate": None,
            "gradient_clip_val": None,
        },
    )()

    config = cyto_run._cytoanvi_training_config_from_args(args)

    assert config["y_prior"] == "empirical"
    assert config["class_weighting"] == "sqrt_inverse_frequency"
    assert config["class_weight_clip"] == 10.0
    assert config["reduce_lr_on_plateau"] is True
    assert config["learning_rate"] == 5e-4
    assert config["gradient_clip_val"] == 1.0


def test_train_cytoanvi_forwards_learning_rate_and_gradient_clip(monkeypatch):
    from benchmarks.common import training as common_training

    labels = np.asarray(["Unknown"] * 4 + ["A"] * 8 + ["B"] * 8)
    adata = ad.AnnData(
        X=np.ones((len(labels), 4), dtype=np.float32),
        obs=pd.DataFrame({"labels": labels, "batch": ["b0"] * len(labels)}),
    )
    captured = {}

    class FakeModule:
        def eval(self):
            captured["eval"] = True

    class FakeCytoANVI:
        @staticmethod
        def setup_anndata(*args, **kwargs):
            captured["setup"] = kwargs

        def __init__(self, *args, **kwargs):
            captured["init"] = kwargs
            self.module = FakeModule()

        def train(self, max_epochs=None, **kwargs):
            captured["max_epochs"] = max_epochs
            captured["train"] = kwargs

    monkeypatch.setattr(cytoanvi, "CytoANVI", FakeCytoANVI)

    common_training.train_cytoanvi(
        adata,
        labels_key="labels",
        unlabeled_category="Unknown",
        batch_key="batch",
        max_epochs=3,
        learning_rate=5e-4,
        gradient_clip_val=0.5,
    )

    assert captured["max_epochs"] == 3
    assert captured["train"]["plan_kwargs"]["lr"] == 5e-4
    assert captured["train"]["gradient_clip_val"] == 0.5
    assert captured["eval"] is True


def test_run_b9_forwards_publication_training_config(monkeypatch):
    labels = np.asarray(["Unknown"] * 4 + ["A"] * 8 + ["B"] * 8)
    adata = ad.AnnData(
        X=np.ones((len(labels), 4), dtype=np.float32),
        obs=pd.DataFrame({"labels": labels, "batch": ["b0"] * len(labels)}),
    )
    captured = {}

    def fake_b9(*args, cytoanvi_training_config=None, **kwargs):
        captured["training"] = cytoanvi_training_config
        return {"task": "b9_mapqc", "status": "ok"}

    monkeypatch.setattr(cyto_run.task_mod, "task_b9_mapqc", fake_b9)

    args = type(
        "Args",
        (),
        {
            "task": "b9",
            "dataset": "synthetic",
            "labels_key": "labels",
            "batch_key": "batch",
            "sample_key": None,
            "max_epochs": 3,
            "batch_size": 32,
            "subsample_per_batch": 200,
            "n_samples_per_label": None,
            "annbatch": False,
            "cytoanvi_recipe": "publication",
            "class_weighting": None,
            "class_weight_clip": 10.0,
            "y_prior": None,
            "classification_ratio": None,
            "reduce_lr_on_plateau": False,
            "learning_rate": None,
            "gradient_clip_val": None,
            "case_control_key": None,
            "control_values": None,
            "case_values": None,
            "mapqc_run": False,
            "mapqc_n_nhoods": 3,
            "mapqc_k_min": 5,
            "mapqc_k_max": 15,
        },
    )()

    cyto_run._run_tasks(args, adata, None, "Unknown", seed=0)

    assert captured["training"]["y_prior"] == "empirical"
    assert captured["training"]["class_weighting"] == "sqrt_inverse_frequency"
    assert captured["training"]["learning_rate"] == 5e-4


def test_b9_query_surgery_uses_trainer_safe_publication_knobs(monkeypatch):
    labels = np.asarray(["A", "B"] * 40)
    adata = ad.AnnData(
        X=np.ones((len(labels), 4), dtype=np.float32),
        obs=pd.DataFrame(
            {"labels": labels, "batch": ["b0"] * len(labels)},
            index=[f"cell_{i}" for i in range(len(labels))],
        ),
    )
    captured = {}

    def fake_assign(work, batch_key, seed):
        is_ref = np.zeros(work.n_obs, dtype=bool)
        is_ref[: work.n_obs // 2] = True
        return work.copy(), is_ref

    def fake_train_cytoanvi(ref_adata, **kwargs):
        captured["ref_train"] = kwargs
        return object(), ref_adata.copy()

    class FakeQueryModel:
        def __init__(self, n_obs):
            self.n_obs = n_obs

        def train(self, max_epochs=None, **kwargs):
            captured["query_epochs"] = max_epochs
            captured["query_train"] = kwargs

        def predict(self):
            return np.repeat("A", self.n_obs)

    class FakeCytoANVI:
        @staticmethod
        def load_query_data(query_train, ref_model):
            captured["query_labels"] = np.asarray(query_train.obs["labels"].astype(str))
            return FakeQueryModel(query_train.n_obs)

    def fake_build_mapqc_anndata(query_model, ref_adata, query_adata, sample_key):
        obs = pd.DataFrame(index=[f"joint_{i}" for i in range(ref_adata.n_obs + query_adata.n_obs)])
        return ad.AnnData(
            X=np.ones((ref_adata.n_obs + query_adata.n_obs, 2), dtype=np.float32),
            obs=obs,
        )

    fake_mapping_qc = SimpleNamespace(
        DEFAULT_EMB_KEY="X_benchmark",
        build_mapqc_anndata=fake_build_mapqc_anndata,
    )

    monkeypatch.setattr(tasks, "_assign_mapqc_pseudo_samples", fake_assign)
    monkeypatch.setattr(tasks, "train_cytoanvi", fake_train_cytoanvi)
    monkeypatch.setattr(cytoanvi, "CytoANVI", FakeCytoANVI)
    monkeypatch.setattr(cytoanvi, "mapping_qc", fake_mapping_qc)

    result = tasks.task_b9_mapqc(
        adata,
        labels_key="labels",
        unlabeled_category="Unknown",
        batch_key="batch",
        max_epochs=100,
        run_mapqc=False,
        cytoanvi_training_config={
            "learning_rate": 5e-4,
            "gradient_clip_val": 0.5,
            "reduce_lr_on_plateau": True,
            "classification_ratio": 20.0,
        },
    )

    assert result["status"] == "plumbing_only"
    assert captured["ref_train"]["learning_rate"] == 5e-4
    assert captured["ref_train"]["gradient_clip_val"] == 0.5
    assert captured["query_epochs"] == 50
    assert captured["query_train"]["lr"] == 5e-4
    assert captured["query_train"]["gradient_clip_val"] == 0.5
    assert captured["query_train"]["plan_kwargs"]["reduce_lr_on_plateau"] is True
    assert captured["query_train"]["plan_kwargs"]["classification_ratio"] == 20.0
    assert captured["query_train"]["plan_kwargs"]["weight_decay"] == 0.0
    assert set(captured["query_labels"]) == {"Unknown"}


@pytest.mark.slow
def test_cytoanvi_benchmark_synthetic_smoke():
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.cytoanvi.run",
        "--dataset",
        "synthetic",
        "--task",
        "b1",
        "--max-epochs",
        "2",
        "--subsample-per-batch",
        "200",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO / 'src'}:{REPO}"
    env_root = Path(sys.executable).resolve().parents[1]
    env["LD_LIBRARY_PATH"] = f"{env_root / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "macro_f1" in proc.stdout


def test_b4_uses_replay_latent_drift_for_unseen_query_batches(monkeypatch):
    obs = pd.DataFrame(
        {
            "batch": ["ref_batch"] * 100 + ["query_batch"] * 100,
            "labels": ["label_1"] * 50 + ["label_2"] * 50 + ["label_1"] * 50 + ["label_2"] * 50,
        },
        index=[f"cell_{i}" for i in range(200)],
    )
    adata = ad.AnnData(X=np.ones((200, 4), dtype=np.float32), obs=obs)

    class FakeModel:
        def __init__(self, *, role, prediction="label_1"):
            self.role = role
            self.prediction = prediction

        def train(self, *args, **kwargs):
            return None

        def predict(self):
            return np.asarray([self.prediction] * self.n_obs)

        def get_latent_representation(self, latent_adata):
            batches = set(latent_adata.obs["batch"].astype(str))
            if self.role == "reference" and "query_batch" in batches:
                raise AssertionError("reference model should not score unseen query batches")
            offset = {"reference": 0.0, "plain": 1.0, "continual": 0.5}[self.role]
            return np.full((latent_adata.n_obs, 2), offset, dtype=np.float32)

    def fake_train_cytoanvi(ref_adata, **kwargs):
        model = FakeModel(role="reference")
        model.n_obs = ref_adata.n_obs
        return model, ref_adata.copy()

    class FakeCytoANVI:
        @staticmethod
        def select_replay_by_uncertainty(ref_model, ref_adata, fraction):
            return ref_adata[:20].copy()

        @staticmethod
        def load_query_data(query_adata, ref_model):
            model = FakeModel(role="plain")
            model.n_obs = query_adata.n_obs
            return model

        @staticmethod
        def load_query_data_with_replay(query_adata, ref_model, replay_adata, control_adata):
            model = FakeModel(role="continual")
            model.n_obs = query_adata.n_obs
            return model

    monkeypatch.setattr(tasks, "train_cytoanvi", fake_train_cytoanvi)
    monkeypatch.setattr(cytoanvi, "CytoANVI", FakeCytoANVI)

    result = tasks.task_b4_continual(
        adata,
        query_batch_values=["query_batch"],
        max_epochs=2,
        control_frac=0.2,
        replay_frac=0.2,
    )

    assert "replay_latent_drift" in result["plain_surgery"]
    assert "replay_latent_drift" in result["continual_update"]
    assert "control_latent_drift" not in result["plain_surgery"]
    assert result["plain_surgery"]["replay_latent_drift"] > result["continual_update"][
        "replay_latent_drift"
    ]


def test_b6_does_not_recommend_lambda_when_replay_drift_ties(monkeypatch):
    # B6 now calls _b4_setup once before looping over λ values; both helpers must be patched
    # so the test can pass object() as adata without hitting real adata.obs access.
    def fake_setup(adata, **kwargs):
        return {
            "ref_model": None,
            "ref_adata": None,
            "query_adata": None,
            "true_query": np.asarray([]),
            "control": None,
            "replay": None,
            "query_batch_values": ["q"],
            "train_extra": {},
            "_fallback_split": False,
        }

    def fake_b4(*args, ewc_importance=1.0, _setup=None, **kwargs):
        return {
            "continual_update": {
                "replay_latent_drift": 0.0,
                "query_label_transfer": {"macro_f1": 0.8 + float(ewc_importance) / 100.0},
            }
        }

    monkeypatch.setattr(tasks, "_b4_setup", fake_setup)
    monkeypatch.setattr(tasks, "task_b4_continual", fake_b4)

    result = tasks.task_b6_lambda_sweep(
        object(),
        lambdas=[0.0, 1.0, 10.0],
        max_epochs=2,
    )

    assert result["recommendation_status"] == "no_recommendation"
    assert "recommended_lambda" not in result
    assert set(result["per_lambda"]) == {"0.0", "1.0", "10.0"}


def test_b1_optional_baseline_errors_do_not_abort(monkeypatch):
    labels = np.asarray(["A", "B"] * 8)
    obs = pd.DataFrame(
        {
            "batch": ["b0"] * len(labels),
            "labels": labels,
        },
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype=np.float32), obs=obs)

    class FakeModel:
        def __init__(self, pred):
            self._pred = pred

        def predict(self):
            return self._pred

    def fake_train_cytoanvi(work, labels_key, unlabeled_category, **kwargs):
        pred = np.asarray(work.obs[labels_key].astype(str))
        pred[pred == unlabeled_category] = "A"
        return FakeModel(pred), work.copy()

    def fake_unlabeled_predictions(work, labels_key, unlabeled_category, **kwargs):
        mask = np.asarray(work.obs[labels_key].astype(str)) == unlabeled_category
        return np.full(mask.sum(), "A", dtype=object), np.ones((work.n_obs, 2)), mask

    def fake_missing_optional(*args, **kwargs):
        raise ImportError("missing optional")

    monkeypatch.setattr(tasks, "train_cytoanvi", fake_train_cytoanvi)
    monkeypatch.setattr(tasks, "cytovi_latent_and_knn", fake_unlabeled_predictions)
    monkeypatch.setattr(tasks, "raw_marker_knn", fake_unlabeled_predictions)
    monkeypatch.setattr(tasks, "harmony_latent_and_knn", fake_missing_optional)
    monkeypatch.setattr(tasks, "xgboost_classifier", fake_missing_optional)
    monkeypatch.setattr(tasks, "rapids_graph_knn", fake_missing_optional)
    monkeypatch.setattr(tasks, "flowsom_knn", fake_missing_optional)

    result = tasks.task_b1_label_transfer(
        adata,
        unlabeled_category="Unknown",
        holdout_frac=0.25,
        max_epochs=1,
    )

    assert "macro_f1" in result["cytovi_knn"]
    assert "macro_f1" in result["raw_marker_knn"]
    assert result["harmony_knn"] == {"error": "missing optional"}
    assert result["xgboost"] == {"error": "missing optional"}
    assert result["rapids_graph"] == {"error": "missing optional"}
    assert "phenograph" not in result
    assert result["flowsom"] == {"error": "missing optional"}


def test_b1_optional_baseline_metric_conversion_errors_do_not_abort(monkeypatch):
    labels = np.asarray(["A", "B"] * 8)
    obs = pd.DataFrame(
        {
            "batch": ["b0"] * len(labels),
            "labels": labels,
        },
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype=np.float32), obs=obs)

    class FakeModel:
        def predict(self):
            return labels.copy()

    def fake_train_cytoanvi(work, labels_key, unlabeled_category, **kwargs):
        return FakeModel(), work.copy()

    def fake_unlabeled_predictions(work, labels_key, unlabeled_category, **kwargs):
        mask = np.asarray(work.obs[labels_key].astype(str)) == unlabeled_category
        return np.full(mask.sum(), "A", dtype=object), np.ones((work.n_obs, 2)), mask

    def fake_bad_predictions(work, labels_key, unlabeled_category, **kwargs):
        mask = np.asarray(work.obs[labels_key].astype(str)) == unlabeled_category
        return np.asarray(["A", "B"]), np.ones((work.n_obs, 2)), mask

    monkeypatch.setattr(tasks, "train_cytoanvi", fake_train_cytoanvi)
    monkeypatch.setattr(tasks, "cytovi_latent_and_knn", fake_unlabeled_predictions)
    monkeypatch.setattr(tasks, "raw_marker_knn", fake_unlabeled_predictions)
    monkeypatch.setattr(tasks, "harmony_latent_and_knn", fake_bad_predictions)
    monkeypatch.setattr(tasks, "xgboost_classifier", fake_unlabeled_predictions)
    monkeypatch.setattr(tasks, "rapids_graph_knn", fake_unlabeled_predictions)
    monkeypatch.setattr(tasks, "flowsom_knn", fake_unlabeled_predictions)

    result = tasks.task_b1_label_transfer(
        adata,
        unlabeled_category="Unknown",
        holdout_frac=0.25,
        max_epochs=1,
    )

    assert result["harmony_knn"]["error"]
    assert "macro_f1" in result["xgboost"]


def test_b1_baseline_selection_can_skip_optional_baselines(monkeypatch):
    labels = np.asarray(["A", "B"] * 8)
    obs = pd.DataFrame(
        {"batch": ["b0"] * len(labels), "labels": labels},
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype=np.float32), obs=obs)

    class FakeModel:
        def predict(self):
            pred = labels.copy()
            pred[pred == "Unknown"] = "A"
            return pred

    def fake_train_cytoanvi(work, labels_key, unlabeled_category, **kwargs):
        return FakeModel(), work.copy()

    def fake_unlabeled_predictions(work, labels_key, unlabeled_category, **kwargs):
        mask = np.asarray(work.obs[labels_key].astype(str)) == unlabeled_category
        return np.full(mask.sum(), "A", dtype=object), np.ones((work.n_obs, 2)), mask

    def fail_if_called(*args, **kwargs):
        raise AssertionError("optional baseline should not run")

    monkeypatch.setattr(tasks, "train_cytoanvi", fake_train_cytoanvi)
    monkeypatch.setattr(tasks, "cytovi_latent_and_knn", fake_unlabeled_predictions)
    monkeypatch.setattr(tasks, "raw_marker_knn", fake_unlabeled_predictions)
    monkeypatch.setattr(tasks, "harmony_latent_and_knn", fail_if_called)
    monkeypatch.setattr(tasks, "xgboost_classifier", fail_if_called)
    monkeypatch.setattr(tasks, "rapids_graph_knn", fail_if_called)
    monkeypatch.setattr(tasks, "flowsom_knn", fail_if_called)

    result = tasks.task_b1_label_transfer(
        adata,
        unlabeled_category="Unknown",
        holdout_frac=0.25,
        max_epochs=1,
        b1_baselines="none",
    )

    assert "macro_f1" in result["cytovi_knn"]
    assert "macro_f1" in result["raw_marker_knn"]
    assert result["harmony_knn"] == {"skipped": "not requested"}
    assert result["xgboost"] == {"skipped": "not requested"}
    assert result["rapids_graph"] == {"skipped": "not requested"}
    assert result["flowsom"] == {"skipped": "not requested"}


def test_b1_baseline_selection_runs_only_requested_optional_baseline(monkeypatch):
    labels = np.asarray(["A", "B"] * 8)
    obs = pd.DataFrame(
        {"batch": ["b0"] * len(labels), "labels": labels},
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype=np.float32), obs=obs)

    class FakeModel:
        def predict(self):
            return labels.copy()

    def fake_train_cytoanvi(work, labels_key, unlabeled_category, **kwargs):
        return FakeModel(), work.copy()

    def fake_unlabeled_predictions(work, labels_key, unlabeled_category, **kwargs):
        mask = np.asarray(work.obs[labels_key].astype(str)) == unlabeled_category
        return np.full(mask.sum(), "A", dtype=object), np.ones((work.n_obs, 2)), mask

    def fail_if_called(*args, **kwargs):
        raise AssertionError("unrequested optional baseline should not run")

    monkeypatch.setattr(tasks, "train_cytoanvi", fake_train_cytoanvi)
    monkeypatch.setattr(tasks, "cytovi_latent_and_knn", fake_unlabeled_predictions)
    monkeypatch.setattr(tasks, "raw_marker_knn", fake_unlabeled_predictions)
    monkeypatch.setattr(tasks, "harmony_latent_and_knn", fail_if_called)
    monkeypatch.setattr(tasks, "xgboost_classifier", fail_if_called)
    monkeypatch.setattr(tasks, "rapids_graph_knn", fake_unlabeled_predictions)
    monkeypatch.setattr(tasks, "flowsom_knn", fail_if_called)

    result = tasks.task_b1_label_transfer(
        adata,
        unlabeled_category="Unknown",
        holdout_frac=0.25,
        max_epochs=1,
        b1_baselines="rapids-graph",
    )

    assert "macro_f1" in result["rapids_graph"]
    assert result["harmony_knn"] == {"skipped": "not requested"}
    assert result["xgboost"] == {"skipped": "not requested"}
    assert result["flowsom"] == {"skipped": "not requested"}


def test_b1_forwards_training_accelerator_and_devices(monkeypatch):
    labels = np.asarray(["A", "B"] * 8)
    obs = pd.DataFrame(
        {"batch": ["b0"] * len(labels), "labels": labels},
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype=np.float32), obs=obs)
    seen = {}

    class FakeModel:
        def predict(self):
            return labels.copy()

    def fake_train_cytoanvi(work, labels_key, unlabeled_category, **kwargs):
        seen["cytoanvi"] = kwargs
        return FakeModel(), work.copy()

    def fake_cytovi_latent_and_knn(work, labels_key, unlabeled_category, **kwargs):
        seen["cytovi"] = kwargs
        mask = np.asarray(work.obs[labels_key].astype(str)) == unlabeled_category
        return np.full(mask.sum(), "A", dtype=object), np.ones((work.n_obs, 2)), mask

    def fake_unlabeled_predictions(work, labels_key, unlabeled_category, **kwargs):
        mask = np.asarray(work.obs[labels_key].astype(str)) == unlabeled_category
        return np.full(mask.sum(), "A", dtype=object), np.ones((work.n_obs, 2)), mask

    monkeypatch.setattr(tasks, "train_cytoanvi", fake_train_cytoanvi)
    monkeypatch.setattr(tasks, "cytovi_latent_and_knn", fake_cytovi_latent_and_knn)
    monkeypatch.setattr(tasks, "raw_marker_knn", fake_unlabeled_predictions)

    tasks.task_b1_label_transfer(
        adata,
        unlabeled_category="Unknown",
        holdout_frac=0.25,
        max_epochs=1,
        b1_baselines="none",
        accelerator="cpu",
        devices="1",
    )

    assert seen["cytoanvi"]["accelerator"] == "cpu"
    assert seen["cytoanvi"]["devices"] == "1"
    assert seen["cytovi"]["accelerator"] == "cpu"
    assert seen["cytovi"]["devices"] == "1"


def test_b5_holdout_sweep_forwards_mode_nan_layer_and_training_config(monkeypatch):
    labels = np.asarray(["A", "B"] * 8)
    obs = pd.DataFrame(
        {"batch": ["b0"] * len(labels), "labels": labels},
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype=np.float32), obs=obs)
    seen = {}

    def fake_b5(*args, holdout_type=None, nan_layer=None, b5_mode=None, cytoanvi_training_config=None, **kwargs):
        seen[holdout_type] = {
            "nan_layer": nan_layer,
            "b5_mode": b5_mode,
            "training": cytoanvi_training_config,
        }
        return {"task": "b5_novelty", "auroc": 0.6, "n_novel": 8}

    monkeypatch.setattr(tasks, "task_b5_novelty", fake_b5)

    result = tasks.task_b5_holdout_sweep(
        adata,
        nan_layer="_nan_mask",
        b5_mode="inductive",
        cytoanvi_training_config={"class_weighting": "sqrt_inverse_frequency"},
    )

    assert result["task"] == "b5_holdout_sweep"
    assert set(seen) == {"A", "B"}
    assert all(v["nan_layer"] == "_nan_mask" for v in seen.values())
    assert all(v["b5_mode"] == "inductive" for v in seen.values())
    assert all(v["training"]["class_weighting"] == "sqrt_inverse_frequency" for v in seen.values())


def test_b5_inductive_trains_only_seen_labeled_cells(monkeypatch):
    labels = np.asarray(["Unknown"] * 6 + ["A"] * 10 + ["B"] * 10 + ["C"] * 8)
    obs = pd.DataFrame(
        {"batch": ["b0"] * len(labels), "labels": labels},
        index=[f"cell_{i}" for i in range(len(labels))],
    )
    adata = ad.AnnData(X=np.ones((len(labels), 4), dtype=np.float32), obs=obs)
    train_labels = []

    class FakeModel:
        def get_uncertainty(self, adata, mode="latent", batch_size=None):
            scores = np.linspace(0.0, 1.0, adata.n_obs)
            return scores if mode == "latent" else scores[::-1]

    def fake_train_cytoanvi(work, labels_key, unlabeled_category, **kwargs):
        train_labels.extend(np.asarray(work.obs[labels_key].astype(str)))
        return FakeModel(), work.copy()

    monkeypatch.setattr(tasks, "train_cytoanvi", fake_train_cytoanvi)

    result = tasks.task_b5_novelty(
        adata,
        unlabeled_category="Unknown",
        holdout_type="C",
        b5_mode="inductive",
        max_epochs=1,
    )

    assert set(train_labels) == {"A", "B"}
    assert result["b5_evaluation_mode"] == "inductive_calibrated"
    assert result["n_train_seen"] == 16
    assert result["n_calibration_seen"] == 4
    assert result["n_eval"] == 12


def test_b4_real_case_control_split_metadata(monkeypatch):
    obs = pd.DataFrame(
        {
            "batch": ["b0"] * 200,
            "labels": ["label_1"] * 100 + ["label_2"] * 100,
            "status": ["control"] * 100 + ["case"] * 100,
        },
        index=[f"cell_{i}" for i in range(200)],
    )
    adata = ad.AnnData(X=np.ones((200, 4), dtype=np.float32), obs=obs)

    class FakeModel:
        def get_latent_representation(self, adata):
            return np.zeros((adata.n_obs, 2), dtype=np.float32)

    def fake_train_cytoanvi(ref_adata, **kwargs):
        return FakeModel(), ref_adata.copy()

    class FakeCytoANVI:
        @staticmethod
        def select_replay_by_uncertainty(ref_model, ref_adata, fraction):
            return ref_adata[:20].copy()

    monkeypatch.setattr(tasks, "train_cytoanvi", fake_train_cytoanvi)
    monkeypatch.setattr(cytoanvi, "CytoANVI", FakeCytoANVI)

    setup = tasks._b4_setup(
        adata,
        case_control_key="status",
        control_values=["control"],
        case_values=["case"],
    )

    assert setup["case_control_mode"] == "real"
    assert setup["case_control_key"] == "status"
    assert setup["_fallback_split"] is False
    assert setup["ref_adata"].n_obs == 100
    assert setup["query_adata"].n_obs == 100


def test_cytoanvi_train_routes_semisupervised_options(monkeypatch):
    captured = {}

    def fake_super_train(self, **kwargs):
        captured.update(kwargs)
        return "trained"

    monkeypatch.setattr(SemisupervisedTrainingMixin, "train", fake_super_train)
    model = object.__new__(cytoanvi.CytoANVI)
    model.module = type("Module", (), {"continual": None})()

    result = cytoanvi.CytoANVI.train(
        model,
        max_epochs=3,
        n_samples_per_label=5,
        lr=0.02,
        n_steps_kl_warmup=9,
        n_epochs_kl_warmup=None,
        plan_kwargs={"reduce_lr_on_plateau": True},
        gradient_clip_val=1.0,
    )

    assert result == "trained"
    assert captured["n_samples_per_label"] == 5
    assert captured["plan_kwargs"]["lr"] == 0.02
    assert captured["plan_kwargs"]["n_steps_kl_warmup"] == 9
    assert captured["plan_kwargs"]["n_epochs_kl_warmup"] is None
    assert captured["plan_kwargs"]["reduce_lr_on_plateau"] is True
    assert captured["gradient_clip_val"] == 1.0
