#!/usr/bin/env bash
# Phase 5 (Issue 12): B8 flat CE vs HCE label transfer on real Nunez data.
# Requires benchmarks/cytoanvi/hierarchy_nunez_tutorial.json (already exists).
# Generates: nunez_b8_s{0,1,2}.json
#SBATCH --job-name=cytoanvi_p5_b8_hce
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

HIER=$ROOT/benchmarks/cytoanvi/hierarchy_nunez_tutorial.json

for SEED in 0 1 2; do
  echo "=== B8 Nunez HCE seed $SEED ==="
  $PY -m benchmarks.cytoanvi.run \
    --dataset nunez --task b8 \
    --labels-key cell_type \
    --max-epochs 1000 --seed "$SEED" \
    --hierarchy-edges "$HIER" \
    --out "$RESULTS/nunez_b8_s${SEED}.json"
done

echo "Phase 5 complete at $(date)"
