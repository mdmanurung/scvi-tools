#!/usr/bin/env bash
# Phase 3a COARSE (2026-07-04): B3 panel-divergent mapping, roider-full, SEED 2, COARSE Leiden.
# Speed lever from L-041: fewer label classes shrink the n_labels-way ELBO marginalization that
# dominates CytoANVI training. Coarse Leiden (r=RES -> ~12 clusters vs 47 at r=1.0) -> ~3-4x
# faster training AND more interpretable labels. batch_size 16384, one seed per job (parallel).
# NOTE: RES is set from the calibration job 25145242 (resolution giving ~12 clusters).
#
#SBATCH --job-name=cytoanvi_p3a_b3coarse_s2
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

echo "=== B3 COARSE roider-full panel-divergent seed 2 (leiden r=$RES, batch 16384) ==="
echo "Start: $(date)"

$PY -m benchmarks.cytoanvi.run \
  --dataset roider-full --task b3 \
  --labels-key cell_type \
  --leiden-resolution "$RES" \
  --max-epochs 1000 --seed 2 \
  --batch-size 16384 \
  --cytoanvi-recipe publication \
  --out "$RESULTS/roider_full_b3coarse_s2.json"

echo "Phase 3a COARSE (B3 seed 2) complete at $(date)"
