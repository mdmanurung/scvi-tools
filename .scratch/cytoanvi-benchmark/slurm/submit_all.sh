#!/usr/bin/env bash
# submit_all.sh — submit all CytoANVI benchmark phases to SLURM with correct dependencies.
#
# Usage (from repo root):
#   bash .scratch/cytoanvi-benchmark/slurm/submit_all.sh
#
# To start from a specific phase (e.g. skip Phase 0/1 if already done):
#   bash .scratch/cytoanvi-benchmark/slurm/submit_all.sh --start-phase 2
#
# Risk burn-down order: 0 -> 1 -> 2 -> 3 -> 5 -> 6 -> 7
# Phase 4 is manual plumbing only and is excluded from default publication aggregation.
# Phase 7 depends on 2+3+5+6 all succeeding.

set -euo pipefail
SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
START_PHASE=${1:-0}

submit_if() {
  local phase="$1"
  local script="$2"
  local dep_flag="${3:-}"
  if [[ "$phase" -ge "$START_PHASE" ]]; then
    local id
    if [[ -n "$dep_flag" ]]; then
      id=$(sbatch "$dep_flag" "$script" | awk '{print $4}')
    else
      id=$(sbatch "$script" | awk '{print $4}')
    fi
    echo "Submitted phase $phase job $id ($script)" >&2
    echo "$id"
  else
    echo "0"  # dummy id when skipped
  fi
}

echo "=== CytoANVI benchmark submission $(date) ==="

# Phase 0 — env gate pytest (no deps)
JOB0=$(submit_if 0 "$SLURM_DIR/phase0_pytest.slurm")

# Phase 1 — data check (no deps, can run in parallel with phase 0)
JOB1=$(submit_if 1 "$SLURM_DIR/phase1_data_check.slurm")

# Phase 2 — B1+B2 Nunez (depends on P1 data check)
DEP2="--dependency=afterok:${JOB1}"
[[ "$JOB1" == "0" ]] && DEP2=""
JOB2=$(submit_if 2 "$SLURM_DIR/phase2_b1b2_nunez.slurm" "$DEP2")

# Phase 3 — B3+B5 Roider (depends on P1)
DEP3="--dependency=afterok:${JOB1}"
[[ "$JOB1" == "0" ]] && DEP3=""
JOB3=$(submit_if 3 "$SLURM_DIR/phase3_b3b5_roider.slurm" "$DEP3")

# Phase 5 — B8 HCE (depends on P2 for Nunez labels)
DEP5="--dependency=afterok:${JOB2}"
[[ "$JOB2" == "0" ]] && DEP5=""
JOB5=$(submit_if 5 "$SLURM_DIR/phase5_b8_hce.slurm" "$DEP5")

# Phase 6 — B9 mapQC (depends on P2)
DEP6="--dependency=afterok:${JOB2}"
[[ "$JOB2" == "0" ]] && DEP6=""
JOB6=$(submit_if 6 "$SLURM_DIR/phase6_b9_mapqc.slurm" "$DEP6")

JOB4="manual"

# Phase 7 — aggregate (depends on all benchmark phases)
ALL_BENCH="${JOB2}:${JOB3}:${JOB5}:${JOB6}"
ALL_BENCH=$(echo "$ALL_BENCH" | tr ':' '\n' | grep -v '^0$' | tr '\n' ':' | sed 's/:$//')
if [[ -n "$ALL_BENCH" ]]; then
  DEP7="--dependency=afterok:${ALL_BENCH}"
else
  DEP7=""
fi
JOB7=$(submit_if 7 "$SLURM_DIR/phase7_aggregate.slurm" "$DEP7")

echo ""
echo "=== Job chain submitted ==="
echo "  P0 pytest      : $JOB0"
echo "  P1 data check  : $JOB1"
echo "  P2 B1+B2 Nunez : $JOB2"
echo "  P3 B3+B5 Roider: $JOB3"
echo "  P5 B8 HCE      : $JOB5"
echo "  P6 B9 mapQC    : $JOB6"
echo "  P4 B4+B6 cont. : manual only; not submitted"
echo "  P7 aggregate   : $JOB7"
echo ""
echo "Monitor with: squeue -u $USER --format='%.10i %.20j %.8T %.12M %.12l'"
echo "Logs: .scratch/cytoanvi-benchmark/slurm/out/"
echo "Results: .scratch/cytoanvi-benchmark/results/"
