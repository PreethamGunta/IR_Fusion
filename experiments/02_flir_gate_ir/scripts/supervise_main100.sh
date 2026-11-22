#!/usr/bin/env bash
# Keep the 100-epoch FLIR comparison running inside the current salloc.
#
# Every arm is resumable (ultralytics --resume from weights/last.pt) and writes
# runs/<name>/DONE on success, so a crash only costs the current few epochs.
# Re-running the chain is therefore safe and cheap.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
JOB="${JOBID:-${SLURM_JOB_ID:-9743}}"
ARMS=(
  b1_yolo12s_100
  g1_unet_yolo12s_100
  b1_cyclediff75_yolo12s_100
  g1_unet_cd75_yolo12s_100
)

all_done () {
  local a
  for a in "${ARMS[@]}"; do
    [ -f "runs/$a/DONE" ] || return 1
  done
  return 0
}

echo "=== [supervisor] job $JOB start $(date) ==="
for attempt in $(seq 1 30); do
  if all_done; then
    echo "=== [supervisor] all arms DONE $(date) ==="
    break
  fi
  echo "=== [supervisor] attempt $attempt $(date) ==="
  srun --jobid="$JOB" --overlap --cpus-per-task=16 --gres=gpu:1 \
    bash scripts/run_flir_main_100_all.sh
  rc=$?
  echo "=== [supervisor] chain exited rc=$rc $(date) ==="
  if all_done; then
    echo "=== [supervisor] all arms DONE $(date) ==="
    break
  fi
  # if the allocation is gone there is nothing to supervise
  if ! squeue -j "$JOB" -h -o '%T' 2>/dev/null | grep -q RUNNING; then
    echo "=== [supervisor] job $JOB no longer RUNNING, stopping $(date) ==="
    break
  fi
  sleep 30
done
echo "=== [supervisor] finish $(date) ==="
