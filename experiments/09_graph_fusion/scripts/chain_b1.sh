#!/usr/bin/env bash
# Stage hand-off: when the matched IR-only control finishes, launch B1
# (fuse-free warm-up + ramp) and start a monitor bound to it.
#
# Ordering note: run_b1.py runs a 2-epoch smoke first (exercises alpha=0 then
# alpha=1) and aborts the probe if the smoke produces no usable validation, so
# this chain cannot silently burn the window on untested code.
set -u

EXP="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home2/home/preetham_g/.conda/envs/gateir/bin/python"
ALLOC="10187"
CTRL_LOG="$EXP/logs/control_batch.log"
B1_LOG="$EXP/logs/b1_batch.log"
CHAIN_LOG="$EXP/logs/chain.log"
DEADLINE=$(( $(date +%s) + 9 * 3600 ))

log() { echo "[$(date -u +%H:%M:%S)] $*" >> "$CHAIN_LOG"; }

log "chain-b1 armed: waiting for the control to complete"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if grep -q "CONTROL RUN COMPLETE failed=\[\]" "$CTRL_LOG" 2>/dev/null; then
    if [ -z "$(squeue -h -j "$ALLOC" 2>/dev/null)" ]; then
      log "allocation $ALLOC is gone; B1 not launched"
      exit 1
    fi
    log "control complete; launching B1 (smoke + 20-ep probe)"
    CTRL_AP=$("$PY" "$EXP/scripts/best_ap.py" e00_ir_unified 2>/dev/null || echo 0)
    log "control best val AP50-95 = $CTRL_AP"
    # Catastrophic gate: if the control cannot even match half the IR-only
    # screening reference (0.5625), the unified pipeline is the defect and a
    # fused run under it is uninterpretable -- skip B1 and leave the window for
    # the pipeline fix. A merely degraded control (0.30-0.50) still proceeds:
    # the deficit is common to both arms, so the B1-vs-control comparison
    # remains internally valid (documented in PREFLIGHT_B1.md §5).
    if ! awk -v a="$CTRL_AP" 'BEGIN{exit !(a >= 0.30)}'; then
      log "control catastrophic (<0.30): B1 SKIPPED, fix the pipeline first"
      exit 1
    fi
    srun --jobid="$ALLOC" --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=32 \
      bash -lc "cd $EXP && PYTHONPATH=src:. $PY scripts/run_b1.py" \
      >> "$B1_LOG" 2>&1 &
    SRUN_PID=$!
    sleep 30
    if kill -0 "$SRUN_PID" 2>/dev/null; then
      log "B1 step running (pid $SRUN_PID)"
    else
      log "B1 step exited early; check $B1_LOG"
    fi
    MON_LOG="$B1_LOG" MON_RUNGS="b1_smoke,b1_warmup_probe" \
      MON_ERR_PREFIX="clean_" MON_ALLOC="$ALLOC" \
      MON_COMPLETE="B1 RUN COMPLETE" \
      nohup "$PY" "$EXP/scripts/monitor_batch.py" \
      >> "$EXP/logs/chain_monitor.log" 2>&1 &
    log "B1 monitor started"
    wait "$SRUN_PID" || true
    log "B1 step finished; chain exiting"
    exit 0
  fi
  sleep 60
done

log "chain-b1 deadline reached without the control completion marker"
exit 2
