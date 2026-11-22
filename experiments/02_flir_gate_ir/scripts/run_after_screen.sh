#!/usr/bin/env bash
# Wait for the CycleGAN screen + YOLOv12s main comparison to finish, then run
# Phase 2 (CycleDiff) and Phase 3 (B0/B2/B3) in the same allocation.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
PY="${PY:-python}"

echo "=== [after] waiting for results/screen_summary.json $(date) ==="
for _ in $(seq 1 480); do
  [ -f results/screen_summary.json ] && break
  sleep 60
done
if [ ! -f results/screen_summary.json ]; then
  echo "=== [after] no screen summary; aborting ==="
  exit 1
fi

echo "=== [after] waiting for gate_screen to finish $(date) ==="
for _ in $(seq 1 480); do
  tmux has-session -t gate_screen 2>/dev/null || break
  sleep 60
done

echo "=== [after] starting Phase 2 $(date) ==="
bash scripts/run_phase2_direct.sh || echo "PHASE2 FAILED $(date)"
echo "=== [after] starting Phase 3 $(date) ==="
bash scripts/run_phase3_direct.sh || echo "PHASE3 FAILED $(date)"
$PY -m src.make_report || true
touch results/.after_screen_done
echo "=== [after] done $(date) ==="
