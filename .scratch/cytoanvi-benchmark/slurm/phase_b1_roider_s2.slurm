#!/usr/bin/env bash
# B1 label transfer, roider-full cohort, seed 2.
# Generates: roider_full_b1_s2.json
#
#SBATCH --job-name=cytoanvi_b1_roider_s2
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "=== B1 Roider full-cohort label transfer seed 2 (batch_size=16384) ==="
echo "Start: $(date)"

$PY -m benchmarks.cytoanvi.run \
  --dataset roider-full --task b1 \
  --labels-key cell_type \
  --max-epochs 1000 --seed 2 \
  --batch-size 16384 \
  --cytoanvi-recipe publication \
  --out "$RESULTS/roider_full_b1_s2.json"

echo "B1 Roider full-cohort seed 2 complete at $(date)"
