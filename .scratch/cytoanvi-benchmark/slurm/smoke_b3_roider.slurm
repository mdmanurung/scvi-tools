#!/usr/bin/env bash
# Smoke test: roider-full B3 seed 0, batch_size=8192, 20 epochs.
# Purpose (step 3 of recovery plan):
#   (a) Measure steady-state epoch time at batch_size=8192 (skip epoch 1 compile warmup).
#   (b) Confirm no NaN divergence early.
#   (c) Print Leiden cluster count -> sizes the B5 holdout-sweep job wall time.
# Does NOT write a publication artifact; output goes only to the SLURM log.
#
#SBATCH --job-name=cytoanvi_smoke_b3_roider
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

echo "=== Smoke test: roider-full Leiden cluster count + B3 seed 0 epoch timing ==="
echo "Start: $(date)"

# Print Leiden cluster count from the p1 panel (cell_type = Leiden labels, res=1.0).
$PY - <<'PYEOF'
import sys
sys.path.insert(0, 'src')
import numpy as np
from benchmarks.cytoanvi import data as d
print("Loading roider-full (Leiden res=1.0)...", flush=True)
merged, p1, p2 = d.load_roider_full(
    "benchmarks/cytoanvi/data",
    leiden_labels=True,
    leiden_resolution=1.0,
)
labels = np.asarray(p1.obs["cell_type"].astype(str))
types = sorted(t for t in set(labels))
print(f"Leiden clusters at res=1.0: {len(types)}")
print(f"Types: {types}")
print(f"p1 shape: {p1.shape}, p2 shape: {p2.shape if p2 is not None else 'None'}", flush=True)
PYEOF

echo "=== B3 seed 0, batch_size=8192, 20 epochs (timing smoke) ==="
$PY -m benchmarks.cytoanvi.run \
  --dataset roider-full --task b3 \
  --labels-key cell_type \
  --max-epochs 20 --seed 0 \
  --batch-size 8192

echo "=== Smoke test complete at $(date) ==="
echo "Check the log above for:"
echo "  - Leiden cluster count (=> B5-sweep wall time = clusters × ~50 min)"
echo "  - Epoch times at epochs 2-10 (skip epoch 1 compile warmup)"
echo "  - No NaN in loss"
