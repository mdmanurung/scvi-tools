#!/usr/bin/env bash
# Phase 3b REDESIGN (2026-07-04): B5 novelty sweep, roider-full, SEED 1.
# Redesign vs the timed-out 47-cluster sweep (L-037/L-038):
#   * --b5-max-holdout-types 11  -> hold out only the 11 most populous Leiden clusters
#       (major populations; feasible in 48h AND more interpretable than 47 sub-clusters).
#   * --b5-cytovi-baseline        -> unsupervised CytoVI OOD baseline AUROC per type, so B5
#       is a COMPARISON (CytoANVI uncertainty vs CytoVI kNN-distance), not a bare number.
#   * --b5-no-logit               -> skip the 2nd TTA pass (headline uses latent) to cut cost.
#   * --b5-checkpoint             -> incremental per-type write, so a timeout yields partial results.
# Est: 11 types x ~3h/type (CytoANVI train + latent TTA + CytoVI train + scoring) ~= 33-46h; the
# checkpoint is the safety net if wall-time is exceeded.
#
#SBATCH --job-name=cytoanvi_p3b_b5redesign_s1
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "=== B5 REDESIGN roider-full sweep seed 1 (11 types, CytoVI baseline, batch_size=8192) ==="
echo "Start: $(date)"

$PY -m benchmarks.cytoanvi.run \
  --dataset roider-full --task b5 \
  --labels-key cell_type \
  --max-epochs 1000 --seed 1 \
  --batch-size 8192 \
  --cytoanvi-recipe publication \
  --b5-mode inductive \
  --holdout-sweep \
  --b5-max-holdout-types 11 \
  --b5-cytovi-baseline \
  --b5-no-logit \
  --b5-checkpoint "$RESULTS/roider_full_b5_sweep_s1_partial.json" \
  --out "$RESULTS/roider_full_b5_sweep_s1.json"

echo "Phase 3b REDESIGN (B5 seed 1) complete at $(date)"
