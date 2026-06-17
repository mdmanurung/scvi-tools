#!/usr/bin/env bash
# Publication-grade benchmark batch: max_epochs=1000
set -euo pipefail

ENV=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test
ROOT=/exports/para-lipg-hpc/mdmanurung/scvi-tools
OUT="$ROOT/.scratch/cytoanvi-benchmark/results/e1000"
AOUT="$ROOT/.scratch/cytovi-benchmark/results/e1000"
LOG="$ROOT/.scratch/cytoanvi-benchmark/results/e1000_batch.log"

cd "$ROOT"
export PYTHONPATH=src:. LD_LIBRARY_PATH="$ENV/lib"
mkdir -p "$OUT" "$AOUT"

run() {
  local name=$1
  shift
  echo "" | tee -a "$LOG"
  echo "=== $(date -Is) START $name ===" | tee -a "$LOG"
  if "$@" >> "$LOG" 2>&1; then
    echo "=== $(date -Is) OK   $name ===" | tee -a "$LOG"
  else
    echo "=== $(date -Is) FAIL $name (exit $?) ===" | tee -a "$LOG"
    return 1
  fi
}

echo "=== e1000 batch started $(date -Is) ===" | tee "$LOG"

# --- Track B: Roider (vignette) ---
run roider_b1 \
  "$ENV/bin/python" -m benchmarks.cytoanvi.run \
  --dataset roider --task b1 --seeds 0,1,2 --max-epochs 1000 \
  --labels-key cell_type --batch-key batch --sample-key PatientID \
  --unlabeled Unknown \
  --out "$OUT/roider_e1000_b1_multiseed.json"

run roider_b2 \
  "$ENV/bin/python" -m benchmarks.cytoanvi.run \
  --dataset roider --task b2 --seeds 0,1,2 --max-epochs 1000 \
  --labels-key cell_type --batch-key batch --sample-key PatientID \
  --unlabeled Unknown --subsample-per-batch 10000 \
  --out "$OUT/roider_e1000_b2_multiseed.json"

run roider_b3 \
  "$ENV/bin/python" -m benchmarks.cytoanvi.run \
  --dataset roider --task b3 --seeds 0,1,2 --max-epochs 1000 \
  --labels-key cell_type --batch-key batch --sample-key PatientID \
  --unlabeled Unknown \
  --out "$OUT/roider_e1000_b3_multiseed.json"

# --- Track B: Nuñez (tutorial labels via data/nunez_annotated.h5ad; generate with annotate_nunez) ---
run nunez_b1 \
  "$ENV/bin/python" -m benchmarks.cytoanvi.run \
  --dataset nunez --task b1 --seeds 0,1,2 --max-epochs 1000 \
  --labels-key cell_type --batch-key batch --unlabeled Unknown \
  --max-cells 100000 \
  --out "$OUT/nunez_annotated_e1000_b1_multiseed.json"

run nunez_b2 \
  "$ENV/bin/python" -m benchmarks.cytoanvi.run \
  --dataset nunez --task b2 --seeds 0,1,2 --max-epochs 1000 \
  --labels-key cell_type --batch-key batch --unlabeled Unknown \
  --max-cells 100000 --subsample-per-batch 10000 \
  --out "$OUT/nunez_annotated_e1000_b2_multiseed.json"

# --- Track A: A2 batch integration (Nuñez, 9 conditions) ---
run a2_nunez \
  "$ENV/bin/python" -m benchmarks.cytovi.run \
  --dataset nunez --task a2 --seed 0 --max-epochs 1000 \
  --labels-key cell_type --batch-key batch \
  --max-cells 100000 \
  --preproc-schemes minmax,zscore,rank \
  --out "$AOUT/a2_nunez_annotated_e1000_seed0.json"

# --- Track B: Roider B5 holdout sweep (13 cell types) ---
if [[ -f "$OUT/roider_e1000_b5_sweep.json" ]]; then
  echo "=== skip roider_b5_sweep (output exists) ===" | tee -a "$LOG"
else
  run roider_b5_sweep \
    "$ENV/bin/python" -m benchmarks.cytoanvi.run \
    --dataset roider --task b5 --seed 0 --max-epochs 1000 \
    --labels-key cell_type --batch-key batch --sample-key PatientID \
    --unlabeled Unknown --holdout-sweep \
    --out "$OUT/roider_e1000_b5_sweep.json"
fi

# --- Track A: A3 marker imputation sweep (Nuñez, all markers LOO) ---
if [[ -f "$AOUT/a3_nunez_r005_e1000_seed0.json" ]]; then
  echo "=== skip a3_nunez (output exists) ===" | tee -a "$LOG"
elif [[ -f "$AOUT/a3_nunez_annotated_e1000_seed0.json" ]]; then
  echo "=== skip a3_nunez (output exists) ===" | tee -a "$LOG"
else
  run a3_nunez \
    "$ENV/bin/python" -m benchmarks.cytovi.run \
    --dataset nunez --task a3 --seed 0 --max-epochs 1000 \
    --batch-key batch \
    --max-cells 50000 \
    --n-posterior-samples 50 \
    --out "$AOUT/a3_nunez_annotated_e1000_seed0.json"
fi

echo "=== e1000 batch finished $(date -Is) ===" | tee -a "$LOG"
