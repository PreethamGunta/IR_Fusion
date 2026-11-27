#!/usr/bin/env bash
# GPU watchdog v2 — exits (wakes the session) on ANY of:
#   A. no $USER GPU process AND no training AND no launcher for 5 min
#   B. our GPU process(es) present but our GPU's utilization is 0% for 10 min
#      (the 2026-11-26 miss: 34 GB allocated, 0% util, OOM-killed workers)
#   C. more than one concurrent srun training step for $USER (duplicate-run
#      bug: a 32-CPU allocation + srun without --ntasks=1 ran 4 copies)
#   D. launcher alive but nothing on the GPU for 15 min (stuck start)
# Usage: bash scripts/gpu_idle_monitor.sh [grace_minutes]
set -uo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRACE_MIN="${1:-6}"
LOG="$EXP/logs/gpu_monitor.log"
mkdir -p "$EXP/logs"
sleep $((GRACE_MIN * 60))

ours_procs() {
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null |
  while IFS=, read -r u p; do
    u=${u// /}; p=${p// /}
    [ -z "$p" ] && continue
    [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$USER" ] && echo "$u $p"
  done
}

gpu_util_for() {  # $1 = gpu uuid
  nvidia-smi --query-gpu=uuid,utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
  awk -F', ' -v u="$1" '$1==u {print $2; exit}'
}

idle=0; zeroutil=0; dup=0
while true; do
  mapfile -t ours < <(ours_procs)
  nours=${#ours[@]}
  train=$(pgrep -fc "train_unified[_]v2.py" 2>/dev/null || true); train=${train:-0}
  launcher=$(pgrep -fc "run_v2_arm.sh" 2>/dev/null || true); launcher=${launcher:-0}
  ndup=$(pgrep -fc "srun .*train_unified[_]v2" 2>/dev/null || true); ndup=${ndup:-0}

  util=0
  if [ "$nours" -gt 0 ]; then
    for pair in "${ours[@]}"; do
      u=${pair%% *}
      v=$(gpu_util_for "$u"); v=${v:-0}
      [ "$v" -gt "$util" ] && util=$v
    done
  fi

  # no GPU work, no training, no launcher
  if [ "$nours" -eq 0 ] && [ "$train" -eq 0 ] && [ "$launcher" -eq 0 ]; then
    idle=$((idle + 1))
  elif [ "$nours" -eq 0 ] && [ "$train" -eq 0 ] && [ "$launcher" -gt 0 ]; then
    idle=$((idle + 1))   # launcher but no GPU work (startup / stuck)
  else
    idle=0
  fi

  if [ "$nours" -gt 0 ] && [ "$util" -eq 0 ]; then zeroutil=$((zeroutil + 1)); else zeroutil=0; fi
  if [ "$ndup" -gt 1 ]; then dup=$((dup + 1)); else dup=0; fi

  echo "[gpuidle] $(date -u) ours=$nours util=$util train=$train launcher=$launcher steps=$ndup idle_min=$idle zeroutil_min=$zeroutil dup_min=$dup" >> "$LOG"

  reason=""
  if { [ "$idle" -ge 5 ] && [ "$launcher" -eq 0 ]; } || [ "$idle" -ge 15 ]; then
    reason="no GPU work for ${idle} min (ours=$nours train=$train launcher=$launcher)"
  elif [ "$zeroutil" -ge 10 ]; then
    reason="our GPU processes hold memory but utilization has been 0% for ${zeroutil} min"
  elif [ "$dup" -ge 3 ] && [ "$ndup" -ge 4 ]; then
    # one legitimate step shows as 2 srun processes (client + wrapper);
    # >=4 means two or more real steps.
    reason="DUPLICATE training steps persisted for ${dup} min ($ndup srun procs ≈ $((ndup / 2)) steps) — check --ntasks=1"
  fi

  if [ -n "$reason" ]; then
    {
      echo "[gpuidle] ALERT $(date -u): $reason"
      echo "--- GPU state ---"
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
      echo "--- our GPU processes ---"
      for pair in "${ours[@]}"; do echo "  $pair"; done
      echo "--- latest run dirs ---"
      ls -dt "$EXP"/runs_g2/*/ "$EXP"/runs/*/ 2>/dev/null | head -4
      echo "--- tail of newest train log ---"
      newest=$(ls -t "$EXP"/logs/v2_*.log 2>/dev/null | head -1)
      [ -n "$newest" ] && echo "($newest)" && tail -c 400 "$newest" | tr '\r' '\n' | tail -3
    } | tee -a "$LOG"
    exit 1
  fi
  sleep 60
done
