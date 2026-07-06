#!/usr/bin/env bash
# Phase 6 (Issue 13): B9 mapQC query-to-reference mapping QC on Nunez real data.
# Requires mapqc to already be installed in the benchmark environment.
# Generates: nunez_b9_s0.json
#SBATCH --job-name=cytoanvi_p6_b9_mapqc
#SBATCH --output=/exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/out/%x_%j.log
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-long
#SBATCH --exclude=res-hpc-gpu10
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
set -euo pipefail
source /exports/para-lipg-hpc/mdmanurung/scvi-tools/.scratch/cytoanvi-benchmark/slurm/_env.sh

# Do not install dependencies inside the benchmark queue. If mapqc is missing,
# record B9 as blocked and let aggregation proceed without this optional phase.
$PY -c "import mapqc" 2>/dev/null || {
  echo "B9 blocked: mapqc is not installed in this environment."
  echo "Install scvi-tools[cytoanvi-mapping-qc] or mapqc before resubmitting phase 6."
  $PY - <<'PY'
import json
import os

out = os.path.join(os.environ["RESULTS"], "nunez_b9_s0.json")
payload = {
    "dataset": "nunez",
    "seed": 0,
    "b9": {
        "task": "b9_mapqc",
        "status": "blocked",
        "blocked_reason": "mapqc is not installed in the benchmark environment.",
        "run_mapqc": True,
    },
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2)
print(f"wrote {out}")
PY
  exit 0
}

echo "=== B9 Nunez mapQC seed 0 ==="
$PY -m benchmarks.cytoanvi.run \
  --dataset nunez --task b9 \
  --labels-key cell_type \
  --max-epochs 1000 --seed 0 \
  --mapqc-run \
  --out "$RESULTS/nunez_b9_s0.json"

echo "Phase 6 complete at $(date)"
