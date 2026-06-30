"""Smoke test for the CytoANVI benchmark CLI (synthetic, no download)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from benchmarks.cytoanvi import tasks
import cytoanvi

REPO = Path(__file__).resolve().parents[2]


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
