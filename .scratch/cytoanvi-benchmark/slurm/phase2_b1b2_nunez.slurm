#!/usr/bin/env bash
# Phase 2 (Issue 08): B1 label-transfer + B2 scib-metrics on full Nunez cohort.
# 3 seeds x 2 tasks = 6 JSON files under results/nunez_b{1,2}_s{0,1,2}.json
#SBATCH --job-name=cytoanvi_p2_nunez_b1b2
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

for SEED in 0 1 2; do
  echo "=== B1 Nunez seed $SEED ==="
  $PY -m benchmarks.cytoanvi.run \
    --dataset nunez --task b1 \
    --labels-key cell_type \
    --max-epochs 1000 --seed "$SEED" \
    --out "$RESULTS/nunez_b1_s${SEED}.json"

  echo "=== B2 Nunez seed $SEED ==="
  $PY -m benchmarks.cytoanvi.run \
    --dataset nunez --task b2 \
    --labels-key cell_type \
    --max-epochs 1000 --seed "$SEED" \
    --out "$RESULTS/nunez_b2_s${SEED}.json"
done

echo "Phase 2 complete at $(date)"
