#!/usr/bin/env bash
# Night chain B: warm-start screens -> best fine-tune full run.
# Rebuilt after chain A died from editing a running script (never edit a
# running shell script; bash resumes at a stale byte offset).
#
# Order:
#   1. e30ws_screen_10ep    E30  + warm start (lr0 0.002), 10 ep
#   2. e30r2ws_screen_10ep  E30r2 + warm start, 10 ep
#   3. full 50-ep fine-tune of the better screen (promotion bar: >= 0.60,
#      since a warm-started arm starts near E00's 0.63)
# E41 is deferred: batch-8 (OOM-halved) makes a 100-ep run ~26 h.
set -euo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python
JOBID="${1:?usage: chain_g2b.sh JOBID}"
PROJECT=runs_g2
LOG="$EXP/logs/chain_g2b.log"
exec >> "$LOG" 2>&1
echo "[chain2b] $(date -u) START jobid=$JOBID"

best_of() {
  "$PY" - "$EXP/$PROJECT/$1/results.csv" <<'EOF'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
    print(f"{max(float(r['metrics/mAP50-95(B)']) for r in rows):.5f}")
except Exception:
    print("0.0")
EOF
}

wait_done() {
  local i=0
  while [ $i -lt "$2" ]; do
    [ -f "$EXP/$PROJECT/$1/DONE" ] && return 0
    i=$((i + 1)); sleep 60
  done
  echo "[chain2b] $(date -u) TIMEOUT waiting for $1 DONE"; return 1
}

run_arm() {  # name fuse epochs extra...
  local name="$1" fuse="$2" eps="$3"; shift 3
  PROJECT="$PROJECT" bash "$EXP/scripts/run_v2_arm.sh" "$JOBID" "$name" "$fuse" "$eps" "$@" || {
    echo "[chain2b] $(date -u) FAILED launching $name"; exit 1; }
  echo "[chain2b] $(date -u) $name best=$(best_of "$name")"
}

# 1) warm-start screen, E30
run_arm e30ws_screen_10ep E30 10 --d 128 --lr0 0.002 --lrf 0.1 --patience 10 \
  --close-mosaic 0 --fuse-warmup 2 --fuse-ramp 2 \
  --init-ir runs/e00_ir/weights/best.pt --init-rgb runs/e01_rgb/weights/best.pt
ws1=$(best_of e30ws_screen_10ep)

# 2) warm-start screen, E30 r=2 (same budget)
run_arm e30r2ws_screen_10ep E30 10 --d 128 --fuse-r 2 --lr0 0.002 --lrf 0.1 \
  --patience 10 --close-mosaic 0 --fuse-warmup 2 --fuse-ramp 2 \
  --init-ir runs/e00_ir/weights/best.pt --init-rgb runs/e01_rgb/weights/best.pt
ws2=$(best_of e30r2ws_screen_10ep)

echo "[chain2b] $(date -u) screens: e30ws=$ws1 e30r2ws=$ws2"
# 3) promote the better (bar 0.60 — a warm-started arm starts near E00's 0.63)
pick=$("$PY" -c "print('e30ws' if float('$ws1') >= float('$ws2') else 'e30r2ws')")
best=$("$PY" -c "print(max(float('$ws1'), float('$ws2')))")
echo "[chain2b] $(date -u) pick=$pick best=$best (bar 0.60)"
"$PY" -c "import sys; sys.exit(0 if float('$best') >= 0.60 else 1)" || {
  echo "[chain2b] $(date -u) ABORT: warm-start screens below 0.60 — needs review"; exit 1; }

case "$pick" in
  e30ws)
    run_arm e30ws_v2_50ep E30 50 --d 128 --lr0 0.002 --lrf 0.1 --patience 20 \
      --close-mosaic 5 --fuse-warmup 2 --fuse-ramp 2 \
      --init-ir runs/e00_ir/weights/best.pt --init-rgb runs/e01_rgb/weights/best.pt ;;
  e30r2ws)
    run_arm e30r2ws_v2_50ep E30 50 --d 128 --fuse-r 2 --lr0 0.002 --lrf 0.1 \
      --patience 20 --close-mosaic 5 --fuse-warmup 2 --fuse-ramp 2 \
      --init-ir runs/e00_ir/weights/best.pt --init-rgb runs/e01_rgb/weights/best.pt ;;
esac
echo "[chain2b] $(date -u) CHAIN COMPLETE: winner=$pick"
