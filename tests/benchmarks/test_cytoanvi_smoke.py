"""Smoke test for the CytoANVI benchmark CLI (synthetic, no download)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

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
