# 05 — Install an FCS reader to enable D2 (Nuñez)

Status: ready-for-human
Blocks: D2 tasks (optional)

## Task

D2 (Nuñez PBMC) ships as `.fcs`. `cytovi.read_fcs` needs an FCS reader (`readfcs` / `flowio`),
which is **not** installed in the `scvi-test` conda env. D1 (Roider `.h5ad`) needs none, so D2 is
optional — do this only if we want the single-panel, fully-labelled benchmark.

## How

Decide whether to mutate the shared `scvi-test` env (previously avoided) or use a separate env:
```bash
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test/bin/pip install readfcs
```

## Acceptance

- `python -c "import readfcs"` succeeds in the chosen env.
- `... python -m benchmarks.cytoanvi.run --dataset nunez --inspect` loads the FCS files.

## Comments
