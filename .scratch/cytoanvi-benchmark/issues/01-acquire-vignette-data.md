# 01 — Acquire CytoVI vignette datasets [SMOKE ONLY]

Status: wontfix
Blocks: —

> **Superseded** for PR benchmarks by full-cohort data acquisition:
> `.scratch/cytovi-benchmark/issues/01-nunez-full-fcs-and-labels.md` and
> `02-roider-full-cohort.md`. Keep vignette files only for `--smoke` / CI.

## Task

Fetch the CytoVI vignette datasets into `benchmarks/cytoanvi/data/` (gitignored). The dev
environment cannot download them — Figshare returns HTTP 202 ("file generating") with 0 bytes,
even outside the sandbox.

| file | Figshare id | needed for |
|------|-------------|-----------|
| `roider_p1.h5ad` | 56891468 | D1 panel 1 (labelled) — **priority** |
| `roider_p2.h5ad` | 56891471 | D1 panel 2 (unlabelled) |
| `Nunez_PBMCs_batch1.fcs` | 55982654 | D2 (optional, needs issue 05) |
| `Nunez_PBMCs_batch2.fcs` | 55982657 | D2 (optional, needs issue 05) |

## How

From a networked shell (e.g. `! curl ...` in the Claude session, or an HPC login node):
```bash
cd benchmarks/cytoanvi/data
curl -L -o roider_p1.h5ad "https://figshare.com/ndownloader/files/56891468"
curl -L -o roider_p2.h5ad "https://figshare.com/ndownloader/files/56891471"
```
If `ndownloader` 202s, download from the Figshare article page in a browser instead.

## Acceptance

- `benchmarks/cytoanvi/data/roider_p1.h5ad` and `roider_p2.h5ad` exist and are non-empty.
- `... python -m benchmarks.cytoanvi.run --dataset roider --inspect` prints obs columns / layers
  (used to set `--labels-key` / `--batch-key` / `--sample-key` for the run issues).

## Comments

### 2026-06-17

Vignette files acquired at `data/Roider_et_al_BNHL_panel{1,2}.h5ad`, symlinked into
`benchmarks/cytoanvi/data/`. Keys: `--labels-key cell_type --batch-key batch --sample-key PatientID
--unlabeled Unknown`. Smoke B1–B5 complete — see `results/roider_multiseed_summary.json`.
