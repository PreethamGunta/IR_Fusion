#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# Wait until the user's other process has been gone for >20 continuous minutes,
# then run the UNIRI shortlist inside the live salloc allocation.
# Checks every minute; never cancels the allocation.
set -uo pipefail

ROOT="$ROOT"
cd "$ROOT"
mkdir -p uniri_gate_experiments/logs uniri_gate_experiments/results

FREE_SINCE=""
NEED=1200   # 20 minutes
POLL=60

echo "=== watcher start $(date) ==="
while true; do
  if pgrep -f 'python .*si_ablation_eval\.py' >/dev/null; then
    FREE_SINCE=""
    echo "busy $(date)"
  else
    now=$(date +%s)
    if [ -z "$FREE_SINCE" ]; then
      FREE_SINCE=$now
      echo "free since $(date); need ${NEED}s"
    else
      elapsed=$((now - FREE_SINCE))
      echo "free for ${elapsed}s"
      if [ "$elapsed" -ge "$NEED" ]; then
        echo "free for >20 min $(date)"
        break
      fi
    fi
  fi
  sleep "$POLL"
done

JOBID="$(squeue -u "$USER" -h -o '%i %j' | awk '$2=="gateir"{print $1; exit}')"
if [ -z "$JOBID" ]; then
  echo "BLOCKER: no running 'gateir' allocation; exiting"
  exit 1
fi

echo "launching on job $JOBID $(date)"
srun --jobid="$JOBID" --overlap --cpus-per-task=16 --gres=gpu:1 \
  bash uniri_gate_experiments/scripts/run_shortlist.sh \
  > uniri_gate_experiments/logs/run_shortlist.log 2>&1
echo "=== watcher done $(date) ==="
