#!/usr/bin/env bash
# Calibrate Leiden resolution -> cluster count on roider-full (CPU only; caches each resolution).
#SBATCH --job-name=cytoanvi_leiden_cal
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --partition=medium
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "Start: $(date)"
$PY .scratch/cytoanvi-benchmark/calibrate_leiden.py
echo "Done: $(date)"
