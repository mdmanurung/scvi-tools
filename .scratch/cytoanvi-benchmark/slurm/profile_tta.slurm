#!/usr/bin/env bash
# Micro-benchmark: isolate B5 per-cluster cost drivers (train vs latent-TTA vs logit-TTA vs CytoVI).
#SBATCH --job-name=cytoanvi_tta_profile
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "Start: $(date)"
$PY .scratch/cytoanvi-benchmark/profile_tta.py
echo "Done: $(date)"
