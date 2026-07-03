#!/usr/bin/env bash
# Phase 3b (Issue 09): B5 novelty AUROC holdout sweep, roider-full, SEED 1.
# Generates: roider_full_b5_sweep_s1.json
# Multi-seed completion of BLOCKER (needs compute): B5 roider-full at 3 seeds.
# Companion to seed-0 job 25140598 and seed-2 script.
#
#SBATCH --job-name=cytoanvi_p3b_roider_b5sweep_s1
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "=== B5 Roider calibrated holdout sweep seed 1 (batch_size=8192) ==="
echo "Start: $(date)"

$PY -m benchmarks.cytoanvi.run \
  --dataset roider-full --task b5 \
  --labels-key cell_type \
  --max-epochs 1000 --seed 1 \
  --batch-size 8192 \
  --cytoanvi-recipe publication \
  --b5-mode inductive \
  --holdout-sweep \
  --out "$RESULTS/roider_full_b5_sweep_s1.json"

echo "Phase 3b (B5 sweep seed 1) complete at $(date)"
