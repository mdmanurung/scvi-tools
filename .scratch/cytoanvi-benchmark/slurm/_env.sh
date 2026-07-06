#!/usr/bin/env bash
# Shared environment for CytoANVI benchmark SLURM jobs.
# Source this at the top of every job script: source "$(dirname "$0")/_env.sh"

ENV=/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/scvi-test
ROOT=/exports/para-lipg-hpc/mdmanurung/scvi-tools
RESULTS=$ROOT/.scratch/cytoanvi-benchmark/results
SLURM_OUT=$ROOT/.scratch/cytoanvi-benchmark/slurm/out

export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ROOT/src:$ROOT
export PYTHONUNBUFFERED=1
export ROOT RESULTS SLURM_OUT

PY=$ENV/bin/python

mkdir -p "$RESULTS" "$SLURM_OUT"
cd "$ROOT"
