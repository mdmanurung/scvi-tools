# CytoANVI Release Reproduction Guide

This guide describes the release-candidate reproduction path for CytoANVI. It is intentionally
strict: publication summaries must be generated from `.scratch/cytoanvi-benchmark/publication_manifest.json`,
not by recursively aggregating every JSON in a scratch directory.

## Current publication gate

The branch is a release candidate, not a final publication release, until these required Roider
full-cohort artifacts are complete and pass manifest aggregation:

- `.scratch/cytoanvi-benchmark/results/roider_full_b3_s0.json`
- `.scratch/cytoanvi-benchmark/results/roider_full_b3_s1.json`
- `.scratch/cytoanvi-benchmark/results/roider_full_b3_s2.json`
- `.scratch/cytoanvi-benchmark/results/roider_full_b5_sweep_s0.json`

As of 2026-07-03, SLURM jobs `25140597` (B3) and `25140598` (B5) were still running. Do not use
older `roider_b3_*`, `roider_b5_*`, or recursive `final_summary.json` files as publication
evidence.

## Environment

Use Python 3.13 on Linux for the benchmark reproduction environment. The local validation
environment used PyTorch CUDA wheels and an A100 40 GB GPU for full Roider runs.

```bash
conda env create -f environment-lock.yml
conda activate cytoanvi-release
pip install -e ".[dev,docs,cytoanvi-hierarchy,cytoanvi-mapping-qc,cytoanvi-annbatch,cytoanvi-baselines]"
```

On this HPC filesystem, importing PyTorch/Scanpy may need writable caches and conda libraries:

```bash
export CONDA_PREFIX="${CONDA_PREFIX:-$(python -c 'import sys; print(sys.prefix)')}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-$USER"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba-$USER"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
```

Check the public imports:

```bash
python - <<'PY'
import scvi
import cytoanvi
from cytoanvi import CytoANVI
print(scvi.__version__)
print(cytoanvi.__all__)
print(CytoANVI.__name__)
PY
```

## Data layout

The canonical local data cache is repo-root `data/`, which is ignored by git. Lightweight metadata
lives in `benchmarks/cytoanvi/data/README.md` and `benchmarks/cytoanvi/data/data_manifest.tsv`.

Validate small vignette assets:

```bash
PYTHONPATH=src:. python -m benchmarks.common.fetch_data --data-dir data --validate-only
```

If Nuñez annotations need regeneration:

```bash
PYTHONPATH=src:. python -m benchmarks.cytoanvi.annotate_nunez \
  --data-dir data \
  --out data/nunez_annotated.h5ad \
  --max-epochs 1000 \
  --inductive \
  --metadata-out data/nunez_annotated.json
```

The runner expects `data/nunez_annotated.h5ad` when `--require-annotated-nunez` is set. Keep
`data/nunez_annotated_leaky_v1.h5ad` as provenance only; do not use it for publication artifacts.

For Roider full-cohort reproduction, prepare `data/roider_full/merged.h5ad` from the raw Roider
archive, then verify the files and checksums listed in
`benchmarks/cytoanvi/data/data_manifest.tsv`. Derived Nuñez and Roider files must be archived or
regenerated deterministically before final journal submission.

## Required benchmark commands

Nuñez B1/B2/B8 required artifacts are already represented in the publication manifest. To rerun
them:

```bash
PYTHONPATH=src:. python -m benchmarks.cytoanvi.run \
  --dataset nunez \
  --task b1 \
  --seeds 0,1,2 \
  --data-dir data \
  --labels-key cell_type \
  --batch-key batch \
  --require-annotated-nunez \
  --cytoanvi-recipe publication \
  --out .scratch/cytoanvi-benchmark/results/e1000/nunez_inductive_b1_multiseed.json

for seed in 0 1 2; do
  PYTHONPATH=src:. python -m benchmarks.cytoanvi.run \
    --dataset nunez \
    --task b2 \
    --seed "$seed" \
    --data-dir data \
    --labels-key cell_type \
    --batch-key batch \
    --require-annotated-nunez \
    --cytoanvi-recipe publication \
    --out ".scratch/cytoanvi-benchmark/results/nunez_b2_s${seed}.json"
done

for seed in 0 1 2; do
  PYTHONPATH=src:. python -m benchmarks.cytoanvi.run \
    --dataset nunez \
    --task b8 \
    --seed "$seed" \
    --data-dir data \
    --labels-key cell_type \
    --batch-key batch \
    --require-annotated-nunez \
    --hierarchy-edges benchmarks/cytoanvi/hierarchy_nunez_tutorial.json \
    --cytoanvi-recipe publication \
    --out ".scratch/cytoanvi-benchmark/results/nunez_b8_s${seed}.json"
done
```

Roider full-cohort B3/B5 should be run through the SLURM scripts in
`.scratch/cytoanvi-benchmark/slurm/` or equivalent scheduler jobs with the same CLI options:

```bash
PYTHONPATH=src:. python -m benchmarks.cytoanvi.run \
  --dataset roider-full \
  --task b3 \
  --seeds 0,1,2 \
  --labels-key cell_type \
  --batch-key panel_batch \
  --sample-key PatientID \
  --batch-size 8192 \
  --cytoanvi-recipe publication

PYTHONPATH=src:. python -m benchmarks.cytoanvi.run \
  --dataset roider-full \
  --task b5 \
  --seed 0 \
  --labels-key cell_type \
  --batch-key panel_batch \
  --sample-key PatientID \
  --batch-size 8192 \
  --holdout-sweep \
  --b5-mode inductive \
  --cytoanvi-recipe publication
```

Write outputs exactly to the manifest paths listed in
`.scratch/cytoanvi-benchmark/publication_manifest.json`.

## Publication aggregation

Run strict aggregation only after every required manifest artifact is present and has
`"status": "complete"`:

```bash
PYTHONPATH=src:. python benchmarks/common/aggregate_results.py \
  --manifest .scratch/cytoanvi-benchmark/publication_manifest.json \
  --output .scratch/cytoanvi-benchmark/results/final_summary.json
```

Expected release-ready properties:

- `aggregation_mode == "publication_manifest"`
- all required artifacts exist
- no required artifact has `running`, `failed`, `blocked`, `deferred`, or `superseded` status
- positional JSON inputs are rejected in manifest mode
- unknown recursive JSON files are rejected when `--input` is supplied

## Release checks

Run these checks before tagging a release candidate:

```bash
python -m pytest tests/benchmarks/test_aggregate_results.py tests/cytoanvi/test_public_api.py -q
python -m pytest tests/cytoanvi -q
python -m build --sdist --wheel
python -m twine check --strict dist/*
```

The final release still requires a clean release branch, public data/archive DOI, and independent
fresh-environment reproduction.
