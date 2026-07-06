"""Calibrate Leiden resolution -> cluster count on roider-full panel 1.

Recomputes Leiden at several resolutions (via the real load_roider_full path, so each is written to
the resolution-keyed cache and the coarse B3/B5 runs reuse it) and prints the cluster count, so we
can pick the resolution that yields ~12 classes — the training-speed lever (L-041: cost scales with
n_labels) that is also more interpretable than the 47 clusters at r=1.0.
"""

from __future__ import annotations

from benchmarks.cytoanvi import data

RESOLUTIONS = [0.2, 0.3, 0.4, 0.5, 0.6]


def main() -> None:
    for res in RESOLUTIONS:
        _merged, p1, _p2 = data.load_roider_full(leiden_resolution=res, leiden_refresh=True)
        n = int(p1.obs["cell_type"].nunique())
        print(f"[LEIDEN-CAL] resolution={res} n_clusters={n}", flush=True)
    print("[LEIDEN-CAL] DONE", flush=True)


if __name__ == "__main__":
    main()
