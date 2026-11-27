#!/usr/bin/env bash
# Stage hand-off: when the GAT batch finishes, launch the matched IR-only
# control and start a monitor bound to it. Runs on the login node; the only
# heavy work is the `srun`, which executes inside allocation 10187.
#
# Rules.md pre-flight for the launched control is documented in
# scripts/run_control.py (goal, duplicate check, resource profile,
# comparability).
set -u

EXP="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home2/home/preetham_g/.conda/envs/gateir/bin/python"
ALLOC="10187"
BATCH="$EXP/logs/clean_batch.log"
CTRL_LOG="$EXP/logs/control_batch.log"
CHAIN_LOG="$EXP/logs/chain.log"
DEADLINE=$(( $(date +%s) + 10 * 3600 ))   # give up after 10 h

log() { echo "[$(date -u +%H:%M:%S)] $*" >> "$CHAIN_LOG"; }

log "chain armed: waiting for the GAT batch to complete"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if grep -q "CLEAN RUN COMPLETE" "$BATCH" 2>/dev/null; then
    if ! squeue -h -j "$ALLOC" >/dev/null 2>&1 || [ -z "$(squeue -h -j "$ALLOC" 2>/dev/null)" ]; then
      log "allocation $ALLOC is gone; control not launched"
      exit 1
    fi
    log "GAT batch complete; launching e00_ir_unified"
    srun --jobid="$ALLOC" --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=32 \
      bash -lc "cd $EXP && PYTHONPATH=src:. $PY scripts/run_control.py" \
      >> "$CTRL_LOG" 2>&1 &
    SRUN_PID=$!
    sleep 20
    if kill -0 "$SRUN_PID" 2>/dev/null; then
      log "control step running (pid $SRUN_PID)"
    else
      log "control step exited early; check $CTRL_LOG"
    fi
    MON_LOG="$CTRL_LOG" MON_RUNGS="e00_ir_unified" \
      MON_ERR_PREFIX="control_" MON_ALLOC="$ALLOC" \
      MON_COMPLETE="CONTROL RUN COMPLETE" \
      nohup "$PY" "$EXP/scripts/monitor_batch.py" \
      >> "$EXP/logs/chain_monitor.log" 2>&1 &
    log "control monitor started (log: logs/chain_monitor.log)"
    wait "$SRUN_PID" || true
    log "control step finished; chain exiting"
    exit 0
  fi
  sleep 60
done

log "chain deadline reached without seeing the GAT completion marker"
exit 2
