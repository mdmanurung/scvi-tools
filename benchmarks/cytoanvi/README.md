# CytoANVI benchmark harness

Real-data benchmarking for CytoANVI against CytoVI's own workflow, on the two datasets the CytoVI
vignettes use. Plan: `notes/2026-06-10-cytoanvi-benchmark-plan.md`.

## Layout
- `data.py` — Figshare download + loaders for D1 (Roider `.h5ad`) and D2 (Nuñez `.fcs`), plus a
  synthetic loader for the smoke test.
- `metrics.py` — dependency-light metrics (scanpy + sklearn): label-transfer F1, kNN batch-mixing
  (iLISI-like), ARI/NMI/silhouette bio-conservation, novelty AUROC, concordance.
- `baselines.py` — CytoVI latent + k-NN label transfer (the vignette's method).
- `tasks.py` — B1 (label transfer), B2 (integration), B3 (panel-divergent map), B5 (novelty).
- `paired_rna_cytof.py` — B7 paired scRNA+CyTOF integration (scennep + CytoANVI, Plan A).
- `run.py` — CLI.

## Smoke test (no download — proves the plumbing)
```bash
ENV=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test
PYTHONPATH=src:. LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python \
  -m benchmarks.cytoanvi.run --dataset synthetic --task all --max-epochs 3 \
  --subsample-per-batch 200
```

Requires `scib-metrics` (and `python-igraph` + `leidenalg` for some scib metrics).

### B7 — paired RNA + CyTOF (scennep + CytoANVI)

```bash
PYTHONPATH=src:. LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python \
  -m benchmarks.cytoanvi.run --dataset paired-rna-cytof --task b7 --max-epochs 30
```

Primary metric: **RNA macro-F1** on paired-sample RNA cells after label transfer via `predict()`.
Secondary: scib batch mixing on `X_CytoANVI` (`batch_key=modality`).

B7 ignores global `--batch-key`, `--labels-key`, and `--sample-key` (Plan A columns are fixed).

Vignette: `python vignettes/rna_cytof_cocluster.py --smoke` (writes `.scratch/paired_cytoanvi/merged.h5ad`).

## Real data

**The sandbox cannot download the Figshare files (HTTP 202 / blocked egress).** Fetch them into
`benchmarks/cytoanvi/data/` from a networked shell (e.g. `! curl ...` in the Claude session):

| file | Figshare id | dataset |
|------|-------------|---------|
| `roider_p1.h5ad` | 56891468 | D1 panel 1 (labelled) |
| `roider_p2.h5ad` | 56891471 | D1 panel 2 (unlabelled) |
| `Nunez_PBMCs_batch1.fcs` | 55982654 | D2 batch 1 |
| `Nunez_PBMCs_batch2.fcs` | 55982657 | D2 batch 2 |

Place Nuñez files in **`data/`** at the repo root (preferred) or `benchmarks/cytoanvi/data/`.
The loader resolves both automatically.

**Nuñez labels (D2):** FCS files have no cell types. For the 11 PBMC subsets from the CytoVI
tutorial, generate once and reuse:

```bash
PYTHONPATH=src:. LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python \
  -m benchmarks.cytoanvi.annotate_nunez \
  --data-dir data --out data/nunez_annotated.h5ad --max-epochs 100
```

`load_nunez()` prefers `data/nunez_annotated.h5ad` when present (skips FCS + proxy Leiden).
Checkpoint: `.scratch/cytoanvi-benchmark/nunez_cytovi_ckpt` for fast re-runs.

```bash
cd benchmarks/cytoanvi/data
for id in 56891468 56891471; do curl -L -o $id.h5ad "https://figshare.com/ndownloader/files/$id"; done
# rename to roider_p1.h5ad / roider_p2.h5ad
```

Then **inspect** to discover the real obs-column names (label/batch/sample keys), since they're not
known until the data is in hand:
```bash
... python -m benchmarks.cytoanvi.run --dataset roider --inspect
```

Fetch or validate vignette files:
```bash
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.common.fetch_data --fetch
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.common.fetch_data --validate-only
PYTHONPATH=src:. $ENV/bin/python -m benchmarks.common.fetch_data --list-full-cohort
```
and run with the discovered keys:
```bash
... python -m benchmarks.cytoanvi.run --dataset roider --task all \
    --labels-key <cell_type_col> --batch-key <batch_col> --sample-key <patient_col> \
    --unlabeled Unknown --out roider_results.json
```

### Dependency notes
- **`scib-metrics`** required for B2 (and Track A A2). Install in the benchmark env.
- **D2 (Nuñez `.fcs`)** — `readfcs`/`flowio` (pulled in by cyCombinePy or cytovi).
- **Track A cyCombine baseline:** [cyCombinePy](https://github.com/mdmanurung/cyCombinePy) — batch correction only, not imputation.
- Metrics use **scib-metrics** aggregates (`batch_correction`, `bio_conservation`, `total`).

## Tasks → CytoANVI features
| Task | Measures | API exercised | Baseline |
|------|----------|---------------|----------|
| B1 | label-transfer accuracy / macro-F1 on held-out labels | `predict` | CytoVI + kNN |
| B2 | batch mixing vs bio conservation of the latent | `get_latent_representation` | CytoVI latent |
| B3 | panel-1 → panel-2 mapping (panel-aware prep + surgery) | `prepare_query_anndata`, `load_query_data`, `predict` | CytoVI kNN (concordance) |
| B4 | continual vs plain surgery (pseudo batch split) | `load_query_data_with_replay`, `select_replay_by_uncertainty` | plain `load_query_data` |
| B5 | flags a held-out (novel) cell type | `get_uncertainty` | — |
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
