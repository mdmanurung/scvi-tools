# CytoANVI benchmark data cache

This directory is an optional local cache for small vignette assets. The canonical release
benchmark layout uses the repo-root `data/` directory because the full Roider cohort and derived
Nuñez annotations are shared with other benchmark scripts.

Do not commit raw data, derived `.h5ad` files, model checkpoints, or benchmark outputs here. Only
this README and `data_manifest.tsv` are intended to be tracked.

## Canonical local layout

| Purpose | Canonical path |
|---------|----------------|
| Nuñez FCS input | `data/Nunez_PBMCs_batch1.fcs`, `data/Nunez_PBMCs_batch2.fcs` |
| Nuñez inductive annotation | `data/nunez_annotated_inductive.h5ad` |
| Roider vignette panels | `data/Roider_et_al_BNHL_panel1.h5ad`, `data/Roider_et_al_BNHL_panel2.h5ad` |
| Roider full cohort cache | `data/roider_full/merged.h5ad` |
| Roider full labels/metadata | `data/roider_full/panel1_cell_type_leiden_r1.0.parquet`, `data/roider_full/patient_entity.json` |

The loader checks both the requested `--data-dir` and the repo-root `data/` directory for Nuñez
files. Full Roider runs read from repo-root `data/roider_full/` by default.

## Vignette-only fallback

For small local smoke runs, this directory may contain:

```text
benchmarks/cytoanvi/data/Roider_et_al_BNHL_panel1.h5ad
benchmarks/cytoanvi/data/Roider_et_al_BNHL_panel2.h5ad
benchmarks/cytoanvi/data/roider_p1.h5ad
benchmarks/cytoanvi/data/roider_p2.h5ad
benchmarks/cytoanvi/data/Nunez_PBMCs_batch1.fcs
benchmarks/cytoanvi/data/Nunez_PBMCs_batch2.fcs
```

The `roider_p*.h5ad` names are retained as legacy aliases; new downloads should use the
`Roider_et_al_BNHL_panel*.h5ad` names.

Those files are not publication evidence unless they are listed in the publication manifest and
marked complete.
