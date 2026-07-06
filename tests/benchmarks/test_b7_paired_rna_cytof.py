"""Smoke test for B7 paired RNA+CyTOF multimodal benchmark."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_b7_paired_rna_cytof_smoke():
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.cytoanvi.run",
        "--dataset",
        "paired-rna-cytof",
        "--task",
        "b7",
        "--max-epochs",
        "2",
        "--unlabeled",
        "Unknown",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO / 'src'}:{REPO}"
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "b7_multimodal_integration" in proc.stdout or "rna_macro_f1_paired" in proc.stdout
