#!/usr/bin/env bash
# Phase 3b COARSE (2026-07-04): B5 novelty sweep, roider-full, SEED 2, COARSE Leiden.
# Speed lever from L-041: CytoANVI training is launch-bound by the n_labels-way ELBO
# marginalization; fewer label classes shrink it linearly. Coarse Leiden (r=RES -> ~12 clusters,
# vs 47 at r=1.0) makes training ~3-4x faster AND the labels more interpretable (addresses the
# "47 arbitrary sub-clusters" critique). Sweeps ALL coarse clusters (no max-holdout-types cap).
# Keeps the CytoVI OOD baseline comparison + incremental checkpoint. batch_size 16384.
# NOTE: RES is set from the calibration job 25145242 (resolution giving ~12 clusters).
#
#SBATCH --job-name=cytoanvi_p3b_b5coarse_s2
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

RES=__RES__  # <-- calibrated coarse resolution (~12 clusters); set from job 25145242

echo "=== B5 COARSE roider-full sweep seed 2 (leiden r=$RES, CytoVI baseline, batch 16384) ==="
echo "Start: $(date)"

$PY -m benchmarks.cytoanvi.run \
  --dataset roider-full --task b5 \
  --labels-key cell_type \
  --leiden-resolution "$RES" \
  --max-epochs 1000 --seed 2 \
  --batch-size 16384 \
  --cytoanvi-recipe publication \
  --b5-mode inductive \
  --holdout-sweep \
  --b5-cytovi-baseline \
  --b5-checkpoint "$RESULTS/roider_full_b5coarse_s2_partial.json" \
  --out "$RESULTS/roider_full_b5coarse_s2.json"

echo "Phase 3b COARSE (B5 seed 2) complete at $(date)"
