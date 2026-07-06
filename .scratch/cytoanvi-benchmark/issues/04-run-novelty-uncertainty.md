# 04 — Run B5 (novelty / get_uncertainty) on D1 [SUPERSEDED]

Status: wontfix
Blocked-by: 01

> Superseded by issue 09 (full Roider, `--holdout-sweep`, epochs=1000).

## Task

Hold one cell type out of the reference entirely, map a query containing it, and check that
`get_uncertainty` (test-time-augmentation Bregman Information) scores the held-out type higher.

```bash
... python -m benchmarks.cytoanvi.run --dataset roider --task b5 \
  --labels-key <col> --batch-key <col> --unlabeled <value> \
  --out .scratch/cytoanvi-benchmark/results/roider_b5.json
```
Optionally sweep `--holdout-type` across the larger populations (extend run.py to pass it through).

## Acceptance

- Held-out-type novelty `auroc` > 0.7 for at least one held-out population.
- Sanity: uncertainty distribution of held-out cells visibly shifted vs seen cells.
- Results JSON committed; summary in PRD.

## Comments

### 2026-06-17 — vignette holdout sweep completed (historical)

13 cell types swept (`results/b5_sweep/`). Three pass AUROC >0.7: **Tfh 0.875**, **Treg CD69+ 0.779**,
**Ttox EM3 0.742**. Summary in `roider_multiseed_summary.json`. Superseded for PR by issue 09.
