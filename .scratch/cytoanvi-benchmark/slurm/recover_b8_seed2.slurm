#!/usr/bin/env bash
# Recovery: B8 seed 2 only (seeds 0+1 complete from job 25108052; seed 2 HCE killed at epoch 352)
#SBATCH --job-name=cytoanvi_b8_s2_recover
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

HIER=$ROOT/benchmarks/cytoanvi/hierarchy_nunez_tutorial.json

echo "=== B8 Nunez HCE seed 2 (recovery) ==="
$PY -m benchmarks.cytoanvi.run \
  --dataset nunez --task b8 \
  --labels-key cell_type \
  --max-epochs 1000 --seed 2 \
  --hierarchy-edges "$HIER" \
  --out "$RESULTS/nunez_b8_s2.json"

echo "B8 seed 2 recovery complete at $(date)"
