#!/usr/bin/env bash
# Phase 4 (Issue 10): manual B4/B6 continual-update plumbing smoke on full Roider.
# Not publication evidence until a real rLN reference/replay and FL/MCL query design exists.
# Run manually only; submit_all.sh intentionally excludes this optional/deferred phase.
# Generates: roider_full_b4_s0.json + roider_full_b6_sweep_s0.json
#SBATCH --job-name=cytoanvi_p4_b4b6_continual
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "=== B4 Roider continual update (seed 0) ==="
$PY -m benchmarks.cytoanvi.run \
  --dataset roider-full --task b4 \
  --labels-key cell_type \
  --max-epochs 1000 --seed 0 \
  --out "$RESULTS/roider_full_b4_s0.json"

echo "=== B6 Roider EWC lambda sweep (0,1,10,100,1000) ==="
$PY -m benchmarks.cytoanvi.run \
  --dataset roider-full --task b6 \
  --labels-key cell_type \
  --max-epochs 1000 --seed 0 \
  --ewc-lambdas "0,1,10,100,1000" \
  --out "$RESULTS/roider_full_b6_sweep_s0.json"

echo "Phase 4 complete at $(date)"
