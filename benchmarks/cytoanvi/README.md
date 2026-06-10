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
- `run.py` — CLI.

## Smoke test (no download — proves the plumbing)
```bash
ENV=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test
PYTHONPATH=src:. LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python \
    -m benchmarks.cytoanvi.run --dataset synthetic --task all --max-epochs 3
```

## Real data

**The sandbox cannot download the Figshare files (HTTP 202 / blocked egress).** Fetch them into
`benchmarks/cytoanvi/data/` from a networked shell (e.g. `! curl ...` in the Claude session):

| file | Figshare id | dataset |
|------|-------------|---------|
| `roider_p1.h5ad` | 56891468 | D1 panel 1 (labelled) |
| `roider_p2.h5ad` | 56891471 | D1 panel 2 (unlabelled) |
| `Nunez_PBMCs_batch1.fcs` | 55982654 | D2 batch 1 |
| `Nunez_PBMCs_batch2.fcs` | 55982657 | D2 batch 2 |

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
and run with the discovered keys:
```bash
... python -m benchmarks.cytoanvi.run --dataset roider --task all \
    --labels-key <cell_type_col> --batch-key <batch_col> --sample-key <patient_col> \
    --unlabeled Unknown --out roider_results.json
```

### Dependency notes
- **D2 (Nuñez `.fcs`) needs an FCS reader** (`readfcs`/`flowio`) — not currently in the `scvi-test`
  env. Install it there, or skip D2 (D1 needs no FCS reader).
- Metrics are deliberately scib-metrics-free (scanpy Leiden + sklearn). `bio_conservation` needs
  `python-igraph` + `leidenalg` for the Leiden flavor; install if missing.

## Tasks → CytoANVI features (B4 continual deferred — needs a case/control axis)
| Task | Measures | API exercised | Baseline |
|------|----------|---------------|----------|
| B1 | label-transfer accuracy / macro-F1 on held-out labels | `predict` | CytoVI + kNN |
| B2 | batch mixing vs bio conservation of the latent | `get_latent_representation` | CytoVI latent |
| B3 | panel-1 → panel-2 mapping (panel-aware prep + surgery) | `prepare_query_anndata`, `load_query_data`, `predict` | CytoVI kNN (concordance) |
| B5 | flags a held-out (novel) cell type | `get_uncertainty` | — |
