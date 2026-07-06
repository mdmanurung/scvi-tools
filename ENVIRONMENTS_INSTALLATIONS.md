# Environments and Installations

Use this file as a short environment index. The full CytoANVI reproduction workflow lives in
`REPRODUCE.md`.

## Python environment

**Supported Python**: 3.12-3.14 (see `pyproject.toml`)

**Release-candidate environment**:

```bash
conda env create -f environment-lock.yml
conda activate cytoanvi-release
pip install -e ".[dev,docs,cytoanvi-hierarchy,cytoanvi-mapping-qc,cytoanvi-annbatch,cytoanvi-baselines]"
```

## Key dependencies

| Package | Purpose |
|---------|---------|
| `anndata` | Single-cell data container |
| `scanpy` | Single-cell analysis utilities |
| `torch` | Deep learning backend |
| `lightning` | Training loop |
| `scib-metrics` | Integration benchmarking (B2 task) |
| `readfcs` / `flowio` | FCS input for Nuñez regeneration |
| `scHPL` | Optional CytoANVI hierarchy/treeArches workflows |
| `mapqc` | Optional query-mapping QC |
| `flowsom` | Optional benchmark baseline |

## GPU requirements

Large CytoANVI benchmarks require a CUDA GPU. Full Roider runs were sized for an A100 40 GB class
GPU with `--batch-size 8192`. Small synthetic tutorials can run on CPU.

Recommended cache/library settings on shared filesystems:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$USER"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba-$USER"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
```

## Data paths

| Dataset | Path |
|---------|------|
| Nuñez FCS files | `data/Nunez_PBMCs_batch1.fcs`, `data/Nunez_PBMCs_batch2.fcs` |
| Nuñez inductive annotation | `data/nunez_annotated.h5ad` |
| Roider BNHL vignette panels | `data/Roider_et_al_BNHL_panel1.h5ad`, `data/Roider_et_al_BNHL_panel2.h5ad` |
| Roider full cohort | `data/roider_full/` |

Checksums and publication notes are tracked in `benchmarks/cytoanvi/data/data_manifest.tsv`.

## Benchmark outputs

Benchmark results are saved to `.scratch/cytoanvi-benchmark/results/` as JSON files. Publication
summaries must be generated with:

```bash
PYTHONPATH=src:. python benchmarks/common/aggregate_results.py \
  --manifest .scratch/cytoanvi-benchmark/publication_manifest.json \
  --output .scratch/cytoanvi-benchmark/results/final_summary.json
```

Recursive aggregation is exploratory only.
