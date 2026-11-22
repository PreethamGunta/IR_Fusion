#!/usr/bin/env bash
# One-shot protocol run inside the current allocation.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
PY="${PY:-python}"
: "${SLURM_JOB_ID:?run inside a live salloc allocation (see README)}"
srun --jobid="$SLURM_JOB_ID" --overlap --ntasks=1 --cpus-per-task="${CPUS:-16}" \
  "$PY" -m src.run_protocol "$@"
