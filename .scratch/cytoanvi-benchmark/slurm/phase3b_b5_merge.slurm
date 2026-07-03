#!/usr/bin/env bash
# Auto-merge B5 roider-full 3-seed sweep once seeds 0/1/2 finish.
# Dependency: runs after jobs 25140598 (s0), 25144240 (s1), 25144241 (s2) all FINISH
# (afterany — robust to a single-seed failure; the merge script reports any missing seed
# and aggregates whatever landed). Produces roider_full_b5_sweep_multiseed.json with the
# PRIMARY mean_auroc ± std headline. CPU-only, seconds of work.
#
#SBATCH --job-name=cytoanvi_p3b_b5_merge
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --partition=short
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --dependency=afterany:25140598:25144240:25144241
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "=== B5 roider-full 3-seed merge ==="
echo "Start: $(date)"
echo "Seed files present:"
ls -la "$RESULTS"/roider_full_b5_sweep_s{0,1,2}.json 2>&1 || true

$PY .scratch/cytoanvi-benchmark/aggregate_b5_multiseed.py \
  --out results/roider_full_b5_sweep_multiseed.json \
  results/roider_full_b5_sweep_s0.json \
  results/roider_full_b5_sweep_s1.json \
  results/roider_full_b5_sweep_s2.json

echo ""
echo "Merge complete at $(date)"
echo "Output: $RESULTS/roider_full_b5_sweep_multiseed.json"
echo "NOTE: full-manifest publication aggregation still blocked on B3 (held/timed-out)."
