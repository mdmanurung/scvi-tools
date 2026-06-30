#!/usr/bin/env bash
# Phase 7: aggregate manifest-listed publication result JSONs and print go/no-go summary.
# Run after Phases 2–6 are complete.
# Generates: results/final_summary.json
#SBATCH --job-name=cytoanvi_p7_aggregate
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

MANIFEST="$ROOT/.scratch/cytoanvi-benchmark/publication_manifest.json"

echo "=== Aggregating results at $(date) ==="
$PY -m benchmarks.common.aggregate_results \
  --manifest "$MANIFEST" \
  --output "$RESULTS/final_summary.json" 2>&1

echo "=== Summary ==="
cat "$RESULTS/final_summary.json"
echo "Phase 7 complete at $(date)"
