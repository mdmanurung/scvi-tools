#!/usr/bin/env bash
# Phase 3b (Issue 09): B5 novelty AUROC holdout sweep, roider-full, seed 0.
# Generates: roider_full_b5_sweep_s0.json
#
# Wall time sizing: Leiden cluster count confirmed = 47 (smoke test job 25129287, 2026-06-30).
# B5 uses panel-1 only (620k cells), so each cluster = one full CytoANVI training on 620k cells
#   at batch_size=8192: ~76 steps/epoch × 23ms/step × 1000 epochs ≈ 29 min/cluster (estimated).
# 47 clusters × 29 min = ~23h + 1h data prep = ~24h; safely within 48h.
# Actual epoch timing from smoke test: see smoke_b3_roider.slurm log (20-epoch timing uses
#   merged 1.24M cells for B3; B5 uses 620k so B5/cluster ≈ B3-epoch-time × 500 × 0.5).
# If actual timing pushes toward 46h, fall back to --leiden-resolution 0.5 (~20 clusters).
#
# BEFORE SUBMITTING:
#   1. Confirm no NaN divergence in smoke test (check cytoanvi_smoke_b3_roider_25129287.log).
#   2. Optionally adjust --time downward from 48h once smoke epoch time confirmed.
#
#SBATCH --job-name=cytoanvi_p3b_roider_b5sweep
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
# Default 48h; resize downward after reviewing smoke test.
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "=== B5 Roider holdout sweep seed 0 (batch_size=8192) ==="
echo "Start: $(date)"

$PY -m benchmarks.cytoanvi.run \
  --dataset roider-full --task b5 \
  --labels-key cell_type \
  --max-epochs 1000 --seed 0 \
  --batch-size 8192 \
  --holdout-sweep \
  --out "$RESULTS/roider_full_b5_sweep_s0.json"

echo "Phase 3b (B5 sweep) complete at $(date)"
