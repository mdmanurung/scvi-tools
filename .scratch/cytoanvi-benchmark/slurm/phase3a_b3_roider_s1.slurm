#!/usr/bin/env bash
# Phase 3a RERUN (2026-07-04): B3 panel-divergent mapping, roider-full, SEED 1.
# Supersedes the timed-out 3-seed serial job 25140597 (L-037): that packed 3 seeds x ~4
# trainings on the 1.24M-cell merged cohort into a 14h wall-limit and died at seed 1.
# Fix: ONE seed per job (3 run in parallel -> wall-clock ~= one seed, not the serial sum),
# batch_size=16384 (training is ~90% of the cost, L-039; fewer steps/epoch = faster), 48h limit.
# Generates: roider_full_b3_s1.json
#
#SBATCH --job-name=cytoanvi_p3a_b3_s1
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "=== B3 Roider panel-divergent seed 1 (batch_size=16384) ==="
echo "Start: $(date)"

$PY -m benchmarks.cytoanvi.run \
  --dataset roider-full --task b3 \
  --labels-key cell_type \
  --max-epochs 1000 --seed 1 \
  --batch-size 16384 \
  --cytoanvi-recipe publication \
  --out "$RESULTS/roider_full_b3_s1.json"

echo "Phase 3a RERUN (B3 seed 1) complete at $(date)"
