# ruff: noqa
"""A/B speed profile for CytoANVI training: is it matmul-bound or launch/overhead-bound?

Trains CytoANVI for a FIXED number of epochs (early_stopping off) on a roider-full subsample under
four configs and times each. Relative times reveal the mechanism and hence which lever wins:
  baseline            : fp32 (matmul_precision 'highest'), finite-checks ON
  +tf32               : matmul_precision 'high' (TF32 on Ampere+/Ada)
  +tf32+nochecks      : also disable per-step finite-check CUDA syncs on the ~n_labels*batch tensors
  +tf32+nochecks+compile : also torch.compile the module (kernel fusion; wins if launch-bound)
"""

from __future__ import annotations

import time

import numpy as np
import torch
from benchmarks.cytoanvi import data

import cytoanvi._module as cm
from cytoanvi import CytoANVI

N_CELLS = 100_000
BATCH = 16384
EPOCHS = 150  # fixed, early stopping OFF, so every config runs the same #steps


def _prep():
    _m, p1, _p2 = data.load_roider_full()  # cached r=1.0 labels
    rng = np.random.default_rng(0)
    idx = rng.choice(p1.n_obs, size=min(N_CELLS, p1.n_obs), replace=False)
    sub = p1[idx].copy()
    CytoANVI.setup_anndata(
        sub,
        labels_key="cell_type",
        unlabeled_category="Unknown",
        batch_key="batch",
        layer="scaled",
    )
    return sub


def _train_once(sub, *, matmul, checks, compile_module):
    torch.set_float32_matmul_precision(matmul)
    cm._FINITE_CHECKS_ENABLED = checks
    model = CytoANVI(sub)
    n_labels = int(model.module.n_labels)
    if compile_module:
        try:
            model.module = torch.compile(model.module)
        except Exception as e:  # noqa: BLE001
            return None, n_labels, f"compile-failed:{type(e).__name__}"
    t = time.perf_counter()
    try:
        model.train(max_epochs=EPOCHS, batch_size=BATCH, early_stopping=False, accelerator="gpu")
    except Exception as e:  # noqa: BLE001
        return None, n_labels, f"train-failed:{type(e).__name__}:{e}"
    return time.perf_counter() - t, n_labels, "ok"


def main():
    sub = _prep()
    print(f"[SPEED] subsample n_obs={sub.n_obs} epochs={EPOCHS} batch={BATCH}", flush=True)
    configs = [
        ("baseline", dict(matmul="highest", checks=True, compile_module=False)),
        ("tf32", dict(matmul="high", checks=True, compile_module=False)),
        ("tf32+nochecks", dict(matmul="high", checks=False, compile_module=False)),
        ("tf32+nochecks+compile", dict(matmul="high", checks=False, compile_module=True)),
    ]
    for name, kw in configs:
        secs, n_labels, status = _train_once(sub, **kw)
        s = f"{secs:.1f}s" if secs is not None else "n/a"
        print(f"[SPEED] config={name:22s} n_labels={n_labels} train={s} ({status})", flush=True)
    print("[SPEED] DONE", flush=True)


if __name__ == "__main__":
    main()
