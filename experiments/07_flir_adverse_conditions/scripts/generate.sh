#!/usr/bin/env bash
# One condition/severity through the generator. Run inside a live salloc.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
PY="${PY:-python}"
COND="${1:?usage: generate.sh <condition> [severity] [splits...]}"
SEV="${2:-medium}"
shift 2 || true
SPLITS=("$@")
if [ "${#SPLITS[@]}" -eq 0 ]; then SPLITS=(test train); fi
: "${SLURM_JOB_ID:?run inside a live salloc allocation (see README)}"
srun --jobid="$SLURM_JOB_ID" --overlap --ntasks=1 --cpus-per-task="${CPUS:-16}" \
  "$PY" -m src.generate --condition "$COND" --severity "$SEV" \
  --splits "${SPLITS[@]}" --workers "${WORKERS:-16}"
