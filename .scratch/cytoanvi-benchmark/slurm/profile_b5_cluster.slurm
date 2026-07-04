#!/usr/bin/env bash
# One-off PROFILING job: run a SINGLE B5 holdout on a subsampled roider-full (6 patients ~ 60k
# cells) to settle where the ~3h/cluster wall-clock goes. The instrumented inductive path prints
# [B5-TIMING] lines per phase: data_prep / cytoanvi_train / latent_tta / logit_tta / cytovi_baseline.
# compute_logit is ON here (default) so the logit pass we drop in production is also measured.
# Proportions transfer to full scale for the compute-bound phases; this is a diagnostic, not a result.
#
#SBATCH --job-name=cytoanvi_b5_profile
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "=== B5 single-cluster PROFILE (roider-full, 6 patients, batch_size=16384) ==="
echo "Start: $(date)"

# Coarse Leiden (r=0.3 -> few large clusters) + 14 patients so every label clears the inductive
# >=5-seen-per-label floor; phase PROPORTIONS (train/TTA/cytovi/data) still transfer to full scale.
$PY -m benchmarks.cytoanvi.run \
  --dataset roider-full --task b5 \
  --labels-key cell_type --batch-key batch \
  --roider-max-patients 14 \
  --leiden-resolution 0.3 \
  --b5-mode inductive --b5-cytovi-baseline \
  --max-epochs 1000 --seed 0 \
  --batch-size 16384 \
  --cytoanvi-recipe publication \
  --out "$RESULTS/roider_b5_profile.json"

echo "=== per-phase timing summary ==="
grep "B5-TIMING" "$SLURM_OUT/cytoanvi_b5_profile_${SLURM_JOB_ID}.log" || true
echo "Profile complete at $(date)"
