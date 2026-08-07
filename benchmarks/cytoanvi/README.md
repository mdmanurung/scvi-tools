# CytoANVI benchmark harness

Real-data benchmarking for CytoANVI against CytoVI's own workflow, on the two datasets the CytoVI
vignettes use. Plan: `notes/2026-06-10-cytoanvi-benchmark-plan.md`.

These are retrospective exploratory diagnostics, not sealed 0.2.0 scientific evidence or promotion.
The [usage-readiness matrix](../../docs/usage_readiness.md) is authoritative; its cohorts, margins,
independent review, installed artifact, and promotion gates remain blocked.

## Historical full-cohort diagnostics (Roider, max_epochs=1000, 3 seeds; unsealed)

| Task | Metric | CytoANVI | Baseline | Notes |
|------|--------|----------|----------|-------|
| B1 | Roider macro-F1 | **0.9317±0.0022** | CytoVI+kNN 0.8928±0.0034 | Retrospective Δ+0.0388; the P2 margin is unfrozen |
| B1 | Nuñez macro-F1 | **0.9751±0.0003** | CytoVI+kNN 0.9749 | parity |
| B3 | p1 holdout macro-F1 | **0.828±0.015** | — | Exploratory only; no independent labels or frozen margin |
| B3 | p2 inter-method agreement | 0.671±0.008 | CytoVI-kNN (concordance, not accuracy) | No ground-truth labels; not a promotion gate |
| B5 | novelty mean AUROC | historical mean below 0.5 (**NEGATIVE**) | CytoVI kNN-OOD comparator | TTA uncertainty below chance; stable API is no-go |
| B8 | HCE vs flat-CE holdout F1 | **+0.6–1.8 pp** | flat CE | hierarchy helps on coarse labels |

## Layout
- `data.py` — Figshare download + loaders for D1 (Roider `.h5ad`) and D2 (Nuñez `.fcs`), plus a
  synthetic loader for the smoke test.
- `metrics.py` — dependency-light metrics (scanpy + sklearn): label-transfer F1, kNN batch-mixing
  (iLISI-like), ARI/NMI/silhouette bio-conservation, novelty AUROC, concordance.
- `baselines.py` — CytoVI latent + k-NN label transfer plus optional FlowSOM and RAPIDS graph baselines.
- `tasks.py` — B1 (label transfer), B2 (integration), B3 (panel-divergent map), B5 (novelty).
- `paired_rna_cytof.py` — B7 paired scRNA+CyTOF integration (scennep + CytoANVI, Plan A).
- `run.py` — CLI.

## Smoke test (no download — proves the plumbing)
```bash
PYTHONPATH=src:. python \
  -m benchmarks.cytoanvi.run --dataset synthetic --task all --max-epochs 3 \
  --subsample-per-batch 200
```

Requires `scib-metrics` (and `python-igraph` + `leidenalg` for some scib metrics).

### B7 — paired RNA + CyTOF (scennep + CytoANVI)

```bash
PYTHONPATH=src:. python \
  -m benchmarks.cytoanvi.run --dataset paired-rna-cytof --task b7 --max-epochs 30
```

Primary metric: **RNA macro-F1** on paired-sample RNA cells after label transfer via `predict()`.
Secondary: scib batch mixing on `X_CytoANVI` (`batch_key=modality`).

B7 ignores global `--batch-key`, `--labels-key`, and `--sample-key` (Plan A columns are fixed).

Vignette: `python vignettes/rna_cytof_cocluster.py --smoke` (writes `.scratch/paired_cytoanvi/merged.h5ad`).

## Real data

Use repo-root `data/` as the canonical local cache. `benchmarks/cytoanvi/data/` is an optional
vignette-only fallback and remains gitignored except for metadata files.

| Canonical file | Figshare id / source | dataset |
|----------------|-----------------------|---------|
| `data/Roider_et_al_BNHL_panel1.h5ad` | 56891468 | D1 vignette panel 1 |
| `data/Roider_et_al_BNHL_panel2.h5ad` | 56891471 | D1 vignette panel 2 |
| `data/Nunez_PBMCs_batch1.fcs` | 55982654 | D2 batch 1 |
| `data/Nunez_PBMCs_batch2.fcs` | 55982657 | D2 batch 2 |
| `data/nunez_annotated.h5ad` | derived by `annotate_nunez.py --inductive` | D2 publication input |
| `data/roider_full/merged.h5ad` | derived from Roider raw archive | full-cohort B3/B5 input |

The loader resolves Nuñez files from both `--data-dir` and repo-root `data/`. Full Roider runs use
repo-root `data/roider_full/` by default.

**Nuñez labels (D2):** FCS files have no cell types. For the 11 PBMC subsets from the CytoVI
tutorial, generate the inductive annotation once and reuse it:

```bash
PYTHONPATH=src:. python \
  -m benchmarks.cytoanvi.annotate_nunez \
  --data-dir data \
  --out data/nunez_annotated.h5ad \
  --max-epochs 1000 \
  --inductive \
  --metadata-out data/nunez_annotated.json
```

`load_nunez()` prefers `data/nunez_annotated.h5ad` when present (skips FCS + proxy Leiden).
Checkpoint: `.scratch/cytoanvi-benchmark/nunez_cytovi_ckpt` for fast re-runs.

```bash
mkdir -p data
curl -L -o data/Roider_et_al_BNHL_panel1.h5ad "https://figshare.com/ndownloader/files/56891468"
curl -L -o data/Roider_et_al_BNHL_panel2.h5ad "https://figshare.com/ndownloader/files/56891471"
curl -L -o data/Nunez_PBMCs_batch1.fcs "https://figshare.com/ndownloader/files/55982654"
curl -L -o data/Nunez_PBMCs_batch2.fcs "https://figshare.com/ndownloader/files/55982657"
```

Then inspect to verify obs-column names:
```bash
PYTHONPATH=src:. python -m benchmarks.cytoanvi.run --dataset roider --data-dir data --inspect
```

Fetch or validate vignette files:
```bash
PYTHONPATH=src:. python -m benchmarks.common.fetch_data --data-dir data --fetch
PYTHONPATH=src:. python -m benchmarks.common.fetch_data --data-dir data --validate-only
PYTHONPATH=src:. python -m benchmarks.common.fetch_data --list-full-cohort
```
and run with the discovered keys:
```bash
PYTHONPATH=src:. python -m benchmarks.cytoanvi.run --dataset roider --task all \
    --data-dir data \
    --labels-key <cell_type_col> --batch-key <batch_col> --sample-key <patient_col> \
    --unlabeled Unknown --out roider_results.json
```

Strict publication aggregation must use the manifest:

```bash
PYTHONPATH=src:. python benchmarks/common/aggregate_results.py \
  --manifest .scratch/cytoanvi-benchmark/publication_manifest.json \
  --output .scratch/cytoanvi-benchmark/results/final_summary.json
```

The command fails until every required manifest artifact is present and marked `complete`.

### Dependency notes
- **`scib-metrics`** required for B2 (and Track A A2). Install in the benchmark env.
- **D2 (Nuñez `.fcs`)** — `readfcs`/`flowio` (pulled in by cyCombinePy or cytovi).
- **B1 optional baselines** — `cytoanvi[cytoanvi-baselines]` enables FlowSOM; `cytoanvi[rapids]` enables the RAPIDS SingleCell graph baseline.
  Use `--b1-baselines fast`, `none`, `rapids-graph`, or a comma-separated set to avoid slow optional baselines in smoke runs.
- **Track A cyCombine baseline:** [cyCombinePy](https://github.com/mdmanurung/cyCombinePy) — batch correction only, not imputation.
- Metrics use **scib-metrics** aggregates (`batch_correction`, `bio_conservation`, `total`).

## Tasks → CytoANVI features

This table describes historical benchmark reproduction. B4/B5 do not advertise supported
uncertainty or replay-selection APIs: the stable entry points fail closed in 0.2.0, and the
explicitly experimental TTA implementation is retained only to reproduce the negative evidence.
See [`docs/usage_readiness.md`](../../docs/usage_readiness.md).

| Task | Measures | API exercised | Baseline |
|------|----------|---------------|----------|
| B1 | label-transfer accuracy / macro-F1 on held-out labels | `predict` | CytoVI + kNN |
| B2 | batch mixing vs bio conservation of the latent | `get_latent_representation` | CytoVI latent |
| B3 | panel-1 → panel-2 mapping (panel-aware prep + surgery) | `prepare_query_anndata`, `load_query_data`, `predict` | CytoVI kNN (concordance) |
| B4 | historical continual vs plain surgery (pseudo batch split) | explicit experimental TTA plus manual replay construction | plain `load_query_data` |
| B5 | historical held-out-type diagnostic | `experimental_get_uncertainty` (negative-evidence reproduction only) | — |
| B6 | λ (`ewc_importance`) sweep | continual plan kwargs | — |
| B7 | paired scRNA+CyTOF label transfer (RNA macro-F1 on shared samples) | `prepare_paired_cytoanvi`, `predict` | — |
| B8 | flat CE vs HCE on held-out labels | `set_hierarchy`, `predict`, `predict_hierarchical` | flat CE on same holdout |
| B9 | mapQC on query controls after surgery | `score_query_mapping` / `mapping_qc` | low control `mapqc_score > 2` rate |

Use `--require-annotated-nunez` for Nuñez runs that must use manual tutorial labels (not Leiden proxy).

B8 smoke:
```bash
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset synthetic --task b8 --max-epochs 50 --seed 0
```
Pass `--hierarchy-edges path/to/edges.json` for real-data ontologies (parent→children dict).

B9 smoke (plumbing on synthetic; full mapQC on real data):
```bash
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset synthetic --task b9 --max-epochs 50 --seed 0
# Force mapQC on synthetic (may fail — for debugging only):
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.cytoanvi.run \
  --dataset synthetic --task b9 --max-epochs 50 --mapqc-run
```
Real cohort: omit `--mapqc-run` on non-synthetic datasets (mapQC enabled automatically).
