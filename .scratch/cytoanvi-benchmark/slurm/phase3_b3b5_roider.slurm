#!/usr/bin/env bash
# Phase 3a (Issue 09): B3 panel-divergent mapping, roider-full, 3 seeds.
# Generates: roider_full_b3_s{0,1,2}.json
#
# B5 holdout sweep is split into phase3b_b5sweep_roider.slurm to avoid a single
# 48h job that tries to run (B3×3 + clusters×per-train) — the sweep alone is
# clusters×~50 min and dominates wall time.  Size phase3b from smoke_b3_roider.slurm.
#
# batch_size=8192: ~64× fewer steps/epoch vs default 128, avoids NaN on 1.24M cells,
# confirmed by the scverse forum thread.  Epoch time confirmed by smoke test.
#
#SBATCH --job-name=cytoanvi_p3a_roider_b3
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
# Wall budget: 3 seeds × (~2h data-prep/Leiden + ~1h train) = ~9h; 14h adds margin.
# Adjust upward if smoke test shows >~3 min/epoch at batch_size=8192.
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

for SEED in 0 1 2; do
  echo "=== B3 Roider seed $SEED (batch_size=8192) ==="
  $PY -m benchmarks.cytoanvi.run \
    --dataset roider-full --task b3 \
    --labels-key cell_type \
    --max-epochs 1000 --seed "$SEED" \
    --batch-size 8192 \
    --cytoanvi-recipe publication \
    --out "$RESULTS/roider_full_b3_s${SEED}.json"
done

echo "Phase 3a (B3) complete at $(date)"
echo "Next: submit phase3b_b5sweep_roider.slurm after reviewing smoke test epoch times."
