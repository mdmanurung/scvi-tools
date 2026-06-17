# 02 — Run B1 (label transfer) + B2 (integration) on D1 panel 1 [SUPERSEDED]

Status: wontfix
Blocked-by: 01

> Superseded by issue 08 (full Nuñez, scib, epochs=1000). Vignette Roider B1/B2 no longer used for PR claims.

## Task

Run B1 and B2 on the labelled Roider panel 1, CytoANVI vs the CytoVI + k-NN baseline.

```bash
ENV=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test
PYTHONPATH=src:. LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python \
  -m benchmarks.cytoanvi.run --dataset roider --task b1 \
  --labels-key <col> --batch-key <col> --sample-key <col> --unlabeled <value> \
  --out .scratch/cytoanvi-benchmark/results/roider_b1.json
# repeat with --task b2; run ≥3 seeds (--seed 0,1,2)
```
Column names come from issue 01's `--inspect`.

## Acceptance

- B1: CytoANVI `macro_f1` reported vs `cytovi_knn`, ≥3 seeds, mean ± sd. Target: ≥ baseline +0.03.
- B2: `batch_mixing_norm` and `ari`/`nmi`/`label_silhouette` for both latents. Target: CytoANVI
  within ±0.02 silhouette of CytoVI at equal-or-better batch mixing.
- Results JSON committed under `.scratch/cytoanvi-benchmark/results/`; summary added to the PRD.

## Comments

### 2026-06-17 — vignette smoke completed (historical)

Ran on tutorial subsamples, `max_epochs=100`, seeds 0–2. B1 pass (Δ macro-F1 +0.115 mean).
B2 ambiguous (better silhouette, worse batch mixing). Results:
`results/roider_multiseed_summary.json`, `roider_seed{0,1,2}_b{1,2}.json`.
Superseded for PR claims by issue 08 (full Nuñez).
