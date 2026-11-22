#!/usr/bin/env bash
# Allocation-aware supervisor: runs the protocol in whichever gateir allocation
# is currently RUNNING, sleeps through the gap, and resumes on the next one.
# Launch it in a dedicated tmux session from the login node:
#
#   tmux new-session -d -s exp07_run \
#     'PY=/path/to/gateir/python bash experiments/07_flir_adverse_conditions/scripts/supervise.sh'
#
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
PY="${PY:-python}"
LOG="logs/supervisor.log"
mkdir -p logs results

echo "[supervisor] start $(date -u) python=$PY" >> "$LOG"
while true; do
  if [ -f results/PIPELINE_DONE ]; then
    echo "[supervisor] PIPELINE_DONE $(date -u)" >> "$LOG"
    exit 0
  fi
  if [ -f results/PIPELINE_STOPPED ]; then
    echo "[supervisor] PIPELINE_STOPPED: $(cat results/PIPELINE_STOPPED) $(date -u)" >> "$LOG"
    exit 1
  fi

  JOBID="$(squeue -u "$USER" -h -t RUNNING -o '%i %j' 2>/dev/null | awk '$2 ~ /^gateir/ {print $1; exit}')"
  if [ -z "$JOBID" ]; then
    echo "[supervisor] no running gateir allocation; waiting $(date -u)" >> "$LOG"
    sleep 60
    continue
  fi

  echo "[supervisor] launching protocol on job $JOBID $(date -u)" >> "$LOG"
  srun --jobid="$JOBID" --overlap --ntasks=1 --cpus-per-task=16 \
    "$PY" -m src.run_protocol >> "$LOG" 2>&1
  RC=$?
  echo "[supervisor] protocol exited $RC $(date -u)" >> "$LOG"

  if [ "$RC" -eq 42 ]; then
    # Out of allocation time; the same job is about to end. Wait for the next.
    sleep 300
  else
    sleep 30
  fi
done
