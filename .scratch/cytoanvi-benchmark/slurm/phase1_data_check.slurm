#!/usr/bin/env bash
# Phase 1: validate that required input data files are present and loadable.
# Generates: validation/data_check.txt
#SBATCH --job-name=cytoanvi_p1_data
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

LOG=$RESULTS/validation/data_check.txt
mkdir -p "$(dirname "$LOG")"

{
  echo "=== data validation $(date) ==="
  $PY -m benchmarks.common.fetch_data --validate-only
  echo "--- full-cohort listing ---"
  $PY -m benchmarks.common.fetch_data --list-full-cohort
  echo "=== DONE ==="
} 2>&1 | tee "$LOG"
