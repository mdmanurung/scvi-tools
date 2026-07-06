---
date: 2026-06-18
topic: cytoanvi
branch: feat/cytoanvi
status: in-progress
summary: CytoANVI hardened for release — 30+ synthetic tests, vignette benchmarks pass B1/B3/B5, B4/B6 continual harness added, docs/tutorial fixed.
---

# Lab notes — cytoanvi (2026-06-18)

**Takeaway:** Core CytoANVI implementation strengthened across code robustness, benchmarks, docs,
and tests. Vignette smoke benchmarks show strong label-transfer gains; full-cohort `max_epochs=1000`
runs remain the publication gate.

## What changed (2026-06-18 review implementation)

### Code
- TTA uncertainty respects per-cell `nan_layer` (only masks observed backbone features).
- `n_labels == 0` guard; `get_latent_representation` / `get_uncertainty` force eval mode.
- `encoder_marker_mask_` persisted on save/load → `prepare_query_anndata` works from saved paths.
- `CytoANVI.select_replay_by_uncertainty()` for cscanvi-style replay selection.
- `train()` warns when continual update lacks replay buffer after save/load.
- Fisher importances subsample to 10k cells + log progress.

### Benchmarks
- **B4** continual vs plain surgery (pseudo batch split).
- **B6** λ (`ewc_importance`) sweep.
- `--require-annotated-nunez` flag for manual tutorial labels.

### Docs / tutorial
- Fixed continual-update API example (`control_adata`, `plan_kwargs`).
- Fixed `merge_batches` cross-ref; added `classification_ratio` and B2 tradeoff notes.
- New `docs/tutorials/notebooks/cytometry/CytoANVI_tutorial.md`.

### Tests
- 30+ unit tests (inherited CytoVI smoke, continual resume, path-based prep, example script).
- Benchmark synthetic smoke test (`tests/benchmarks/test_cytoanvi_smoke.py`).

## Prior vignette results (epochs=100, not publication-grade)

| Task | Result |
|------|--------|
| B1 | CytoANVI +0.115 macro-F1 vs CytoVI k-NN (Roider, 3 seeds) |
| B2 | Better bio (+0.10), worse batch (−0.04) |
| B3 | p2 concordance 0.86 *(⚠️ smoke; full-cohort: 0.671±0.008 — gate not met)* |
| B5 | Several holdout types AUROC > 0.70 *(⚠️ smoke best-type framing; full-cohort mean_auroc 0.484±0.019 — NEGATIVE)* |

## Still open

- Full Roider/Nuñez cohorts at `max_epochs=1000` with scib B2.
- Real case/control axis for biological validation of B4 (current split is plumbing-only).
- Tune and document CytoVI-specific default λ after B6 sweep on full data.
- Push branch + upstream PR.
