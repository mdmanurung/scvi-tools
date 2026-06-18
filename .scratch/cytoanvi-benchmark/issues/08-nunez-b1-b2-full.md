# 08 — Run B1 + B2 on full Nuñez (scib, epochs=1000)

Status: ready-for-agent
Blocked-by: cytovi-benchmark/01

### 2026-06-18 — B8 HCE benchmark harness

- **Task B8** added to `benchmarks/cytoanvi/tasks.py` (flat CE vs HCE + hierarchical predict).
- Smoke: `--dataset synthetic --task b8 --max-epochs 50`
- Real runs: pass `--hierarchy-edges` JSON with coarse+fine observed labels (issue 12).

## Task

On **full** Nuñez batch replicate (B-D2):

- **B1:** 5-fold stratified holdout (20% labels → unlabeled); CytoANVI `predict` vs CytoVI k-NN
- **B2:** scib-metrics on both latents (`benchmarks/common/scib.py`)
- `max_epochs=1000`, seeds 0, 1, 2

```bash
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset nunez --task b1 --max-epochs 1000 --seed 0 \
  --labels-key <col> --batch-key batch --out .scratch/cytoanvi-benchmark/results/nunez_b1_s0.json
```

## Acceptance

- B1: macro-F1 mean ± SD over 3 seeds; pass if ≥ baseline +0.03
- B2: scib aggregates for CytoANVI and CytoVI; bio within ±0.02, batch ≥ baseline

### 2026-06-17 — e1000 vignette Nuñez (in flight)

- **Labels:** `data/nunez_annotated.h5ad` (11 tutorial PBMC types via `annotate_nunez.py`).
  `load_nunez()` prefers this file; no `--leiden-resolution` needed.
- **Running:** `run_e1000.sh` → `nunez_r005_e1000_b1_multiseed.json` (started before output rename;
  uses annotated h5ad regardless of filename). Future runs → `nunez_annotated_e1000_b*.json`.
- **Smoke (epochs=100, Leiden r=0.05):** `nunez_r005_seed0_summary.json` — superseded for publication.

### 2026-06-17 — e1000 Roider vignette complete

Rolling summary: `results/e1000/roider_e1000_partial_summary.json`. B1 Δ **+0.12**; B2 batch still
slightly below CytoVI; B3 concordance **0.877**.

### 2026-06-17 — e1000 in flight (Roider)

Vignette Roider **B1+B2 @ 1000 epochs** complete; **B3** running. Rolling summary:
``results/e1000/roider_e1000_partial_summary.json``. B1 macro-F1: CytoANVI **0.908±0.008** vs
k-NN **0.787±0.039** (Δ **+0.12**). B2 bio: CytoANVI **0.737** vs CytoVI **0.628**; batch:
CytoANVI **0.792** vs CytoVI **0.798** (batch still slightly worse).

### 2026-06-17

- cytovi-benchmark/03 (scib infra) and issue 05 (readfcs) are done
- Vignette Nuñez FCS still blocked on Figshare egress from HPC; use
  `python -m benchmarks.common.fetch_data --fetch` from a networked shell

Supersedes issue 02 (vignette Roider B1/B2) for primary integration/transfer validation — Nuñez is
the paper's clean fully-labelled batch-replicate setting.
