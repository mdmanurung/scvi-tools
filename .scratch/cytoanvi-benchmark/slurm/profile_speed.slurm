#!/usr/bin/env bash
# A/B speed profile for CytoANVI training (baseline / +TF32 / +no-checks / +compile).
# Target a fast Ampere+/Ada card (L40S / A100-MIG) so TF32 & compile can actually show up.
#SBATCH --job-name=cytoanvi_speed_profile
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "Start: $(date)"
$PY .scratch/cytoanvi-benchmark/profile_speed.py
echo "Done: $(date)"
