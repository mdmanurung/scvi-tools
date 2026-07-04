"""Isolate the B5 per-cluster cost drivers: training vs TTA uncertainty vs CytoVI baseline.

Loads cached roider-full (r=1.0 Leiden; no recompute), randomly subsamples to a target cell count
(avoids the inductive >=5-per-label constraint — we only need representative per-phase costs), and
times each expensive operation the B5 inductive path performs per held-out type.
"""

from __future__ import annotations

import time

import numpy as np

from benchmarks.cytoanvi import baselines, data
from benchmarks.common.training import train_cytoanvi

N_CELLS = 130_000
BATCH = 16384
TTA_REP = 50
EPOCHS = 1000  # early stopping applies, as in production


def _t(msg, since):
    print(f"[TTA-PROFILE] {msg}: {time.perf_counter() - since:.1f}s", flush=True)
    return time.perf_counter()


def main() -> None:
    t = time.perf_counter()
    _merged, p1, _p2 = data.load_roider_full()  # cached r=1.0 labels, no leiden recompute
    t = _t(f"load_roider_full (p1 n_obs={p1.n_obs})", t)

    rng = np.random.default_rng(0)
    idx = rng.choice(p1.n_obs, size=min(N_CELLS, p1.n_obs), replace=False)
    sub = p1[idx].copy()
    # split like inductive B5: ~80% seen/train, ~20% eval
    n = sub.n_obs
    perm = rng.permutation(n)
    train_adata = sub[perm[: int(0.8 * n)]].copy()
    eval_adata = sub[perm[int(0.8 * n) :]].copy()
    t = _t(f"subsample+split (train={train_adata.n_obs} eval={eval_adata.n_obs})", t)

    model, _ = train_cytoanvi(
        train_adata,
        labels_key="cell_type",
        unlabeled_category="Unknown",  # not present -> all cells labeled
        batch_key="batch",
        max_epochs=EPOCHS,
        batch_size=BATCH,
    )
    t = _t(f"cytoanvi_train ({EPOCHS} max epochs, early-stop)", t)

    _unc_latent = model.get_uncertainty(eval_adata, mode="latent", tta_rep=TTA_REP, batch_size=BATCH)
    t = _t(f"latent_tta (tta_rep={TTA_REP}, n_eval={eval_adata.n_obs})", t)

    _unc_logit = model.get_uncertainty(eval_adata, mode="logit", tta_rep=TTA_REP, batch_size=BATCH)
    t = _t(f"logit_tta (tta_rep={TTA_REP})", t)

    _score = baselines.cytovi_novelty_score(
        train_adata,
        eval_adata,
        batch_key="batch",
        max_epochs=EPOCHS,
        batch_size=BATCH,
    )
    t = _t("cytovi_baseline (train + kNN-distance score)", t)
    print("[TTA-PROFILE] DONE", flush=True)


if __name__ == "__main__":
    main()
