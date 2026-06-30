#!/usr/bin/env bash
# Phase 0 (full suite): run targeted CytoANVI pytest suite on a GPU node.
# Generates: validation/pytest_cytoanvi_full.log
#SBATCH --job-name=cytoanvi_p0_pytest
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

LOG=$RESULTS/validation/pytest_cytoanvi_full.log
mkdir -p "$(dirname "$LOG")"

$PY -m pytest \
  tests/cytoanvi/test_cytoanvi.py \
  tests/cytoanvi/test_hce.py \
  tests/cytoanvi/test_mapping_qc_mock.py \
  tests/benchmarks/ \
  -q --tb=short 2>&1 | tee "$LOG"
