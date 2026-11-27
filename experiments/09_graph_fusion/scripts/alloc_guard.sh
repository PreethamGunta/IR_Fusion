#!/usr/bin/env bash
# Allocation boundary guard. Waits for the current allocation to end, then
# verifies the pre-requested replacement is RUNNING; if it was cancelled (the
# 10230/10275 failure mode), requests a fresh allocation from a LOGIN-level
# tmux session (the existing salloc_session pane sits inside an allocation
# shell, so a new session is used to keep the client outside job cgroups).
#
# Usage: bash scripts/alloc_guard.sh CURRENT_JOBID PENDING_JOBID
# Exits (and wakes the session) once a replacement allocation is RUNNING or
# the fallback request is queued, or on timeout.
set -uo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUR="${1:?current jobid}"
NEXT="${2:-10327}"
LOG="$EXP/logs/alloc_guard.log"

state() { squeue -j "$1" -h -o "%T" 2>/dev/null; }

echo "[allocguard] $(date -u) watching $CUR (replacement $NEXT)" >> "$LOG"
# 1. wait for the current allocation to end
i=0
while [ $i -lt 1500 ]; do
  st=$(state "$CUR")
  [ "$st" != "RUNNING" ] && break
  i=$((i+1)); sleep 60
done
echo "[allocguard] $(date -u) $CUR state='${st:-gone}' — boundary reached" >> "$LOG"

# 2. is any allocation of ours running? (the replacement should auto-start)
i=0
while [ $i -lt 20 ]; do
  run=$(squeue -u "$USER" -h -t RUNNING -o "%i %T" 2>/dev/null | awk '{print $1}' | head -1)
  [ -n "$run" ] && { echo "[allocguard] $(date -u) allocation RUNNING: $run — OK" | tee -a "$LOG"; exit 0; }
  nst=$(state "$NEXT")
  if [ "$nst" = "CANCELLED" ] || { [ $i -ge 5 ] && [ -z "$nst" ]; }; then
    echo "[allocguard] $(date -u) replacement $NEXT state='${nst:-gone}' — requesting fresh allocation" | tee -a "$LOG"
    sess="salloc_guard_$(date +%H%M%S)"
    tmux new-session -d -s "$sess" \
      "salloc --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=24:00:00 -J girkd"
    echo "[allocguard] $(date -u) fresh salloc requested in tmux session $sess" | tee -a "$LOG"
    i=0
  fi
  i=$((i+1)); sleep 60
done

run=$(squeue -u "$USER" -h -t RUNNING -o "%i %T" 2>/dev/null | head -1)
echo "[allocguard] $(date -u) TIMEOUT: no RUNNING allocation after 20+ min (running='${run:-none}')" | tee -a "$LOG"
[ -n "$run" ] && exit 0 || exit 1
