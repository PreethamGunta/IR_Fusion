#!/usr/bin/env bash
# Append a compact status snapshot every 15 minutes while the 100-epoch runs
# are pending or running.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p logs

while true; do
  echo "===== $(date) ====="
  squeue -u "$USER" -o '%i %j %T %M %R' 2>/dev/null | head -8
  squeue -s -j "${JOBID:-9743}" -o '%i %T %M' 2>/dev/null | head -4
  for d in b1_yolo12s_100 g1_unet_yolo12s_100 b1_cyclediff75_yolo12s_100 g1_unet_cd75_yolo12s_100; do
    ep="none"
    [ -f "runs/$d/results.csv" ] && ep="$(tail -1 "runs/$d/results.csv" | cut -d, -f1)"
    done_mark=""
    [ -f "runs/$d/DONE" ] && done_mark=" DONE"
    echo "$d: epoch ${ep}${done_mark}"
  done
  ls results/*100*.json 2>/dev/null
  tail -c 400 logs/supervisor.log 2>/dev/null | tr '\r' '\n' | grep -aE '=== |TRAIN FAILED|Terminated|epochs completed' | tail -1
  echo
  sleep 900
done
