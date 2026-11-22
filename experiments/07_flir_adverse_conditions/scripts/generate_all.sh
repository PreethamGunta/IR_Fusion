#!/usr/bin/env bash
# All eight conditions at one severity (test + train). Run inside a live salloc.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
PY="${PY:-python}"
SEV="${1:-medium}"
: "${SLURM_JOB_ID:?run inside a live salloc allocation (see README)}"
srun --jobid="$SLURM_JOB_ID" --overlap --ntasks=1 --cpus-per-task="${CPUS:-16}" \
  "$PY" -m src.generate --condition all --severity "$SEV" \
  --splits test train --workers "${WORKERS:-16}"
