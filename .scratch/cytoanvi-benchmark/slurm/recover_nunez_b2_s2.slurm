#!/usr/bin/env bash
# One-off recovery for the missing Nuñez B2 seed 2 artifact.
# Generates: nunez_b2_s2.json
#SBATCH --job-name=cytoanvi_recover_nunez_b2_s2
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "=== Recovering B2 Nunez seed 2 ==="
$PY -m benchmarks.cytoanvi.run \
  --dataset nunez --task b2 \
  --labels-key cell_type \
  --max-epochs 1000 --seed 2 \
  --out "$RESULTS/nunez_b2_s2.json"

echo "Nuñez B2 seed 2 recovery complete at $(date)"
