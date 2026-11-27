#!/usr/bin/env bash
# Graph-v2 chain: warm-start fidelity gate -> 10-epoch screens -> full run for
# the most promising arm. Runs sequentially (solo-GPU rule) inside the live
# allocation. Every step is guarded; failures abort loudly.
#
# Usage: bash scripts/chain_g2_screens.sh JOBID
#
# Promotion rule (calibrated on the flagship screens, same recipe):
#   E30 screen best 0.54067 (ep8), E10 screen best 0.56665 (ep10);
#   E30 final 0.64838 from a 0.54067 screen, E10 final 0.65380 from 0.56665.
#   -> a 10-ep screen >= 0.550 is "promising"; best-of-chain is promoted even
#      below that if it clears the E30 screen (0.5407).
# Full-budget settings depend on the arm:
#   COCO-init arms (E41, r=2): E00 recipe, 100 ep, close_mosaic 10
#   warm-start arms: fine-tune, 50 ep, lr0 0.002, lrf 0.1, patience 20,
#                    close_mosaic 5, fuse warm-up 2 / ramp 2
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python
JOBID="${1:?usage: chain_g2_screens.sh JOBID}"
PROJECT=runs_g2
LOG="$EXP/logs/chain_g2.log"
exec >> "$LOG" 2>&1
echo "[chain2] $(date -u) START jobid=$JOBID"

best_of() {  # $1 = run name (in $PROJECT)
  "$PY" - "$EXP/$PROJECT/$1/results.csv" <<'EOF'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
    print(f"{max(float(r['metrics/mAP50-95(B)']) for r in rows):.5f}")
except Exception:
    print("0.0")
EOF
}

wait_done() {  # $1 = run name, $2 = timeout minutes
  local i=0
  while [ $i -lt "$2" ]; do
    [ -f "$EXP/$PROJECT/$1/DONE" ] && return 0
    i=$((i + 1)); sleep 60
  done
  echo "[chain2] $(date -u) TIMEOUT waiting for $1 DONE"
  return 1
}

launch() {  # name fuse epochs extra...
  local name="$1" fuse="$2" eps="$3"; shift 3
  PROJECT="$PROJECT" bash "$EXP/scripts/run_v2_arm.sh" "$JOBID" "$name" "$fuse" "$eps" "$@" || {
    echo "[chain2] $(date -u) FAILED launching $name"; exit 1; }
  wait_done "$name" 90 || { echo "[chain2] $(date -u) $name DONE missing"; exit 1; }
  echo "[chain2] $(date -u) $name best=$(best_of "$name")"
}

# ---- 0. warm-start smoke must have passed its fidelity gate -----------------
if ! wait_done e30ws_smoke_3ep 60; then
  echo "[chain2] $(date -u) ABORT: warm-start smoke did not finish"; exit 1
fi
smoke=$(best_of e30ws_smoke_3ep)
echo "[chain2] $(date -u) warm-start smoke best=$smoke (fidelity gate: >=0.60)"
"$PY" -c "import sys; sys.exit(0 if float('$smoke') >= 0.60 else 1)" || {
  echo "[chain2] $(date -u) ABORT: warm start did not reproduce E00-level val " \
       "(best=$smoke < 0.60); the technique is not behaving as designed"; exit 1; }

# ---- 1. screens (10 ep each) -----------------------------------------------
SCREENS=(e41_screen_10ep e30r2_screen_10ep e30ws_screen_10ep e35ws_screen_10ep)
declare -A BEST
launch e41_screen_10ep   E41 10 --d 128 --close-mosaic 0
BEST[e41_screen_10ep]=$(best_of e41_screen_10ep)
if "$PY" -c "import sys; sys.exit(0 if float('${BEST[e41_screen_10ep]}') >= 0.550 else 1)"; then
  winner=e41; echo "[chain2] $(date -u) PROMOTE e41 (${BEST[e41_screen_10ep]})"; fi

if [ -z "${winner:-}" ]; then
  launch e30r2_screen_10ep E30 10 --d 128 --close-mosaic 0 --fuse-r 2
  BEST[e30r2_screen_10ep]=$(best_of e30r2_screen_10ep)
  if "$PY" -c "import sys; sys.exit(0 if float('${BEST[e30r2_screen_10ep]}') >= 0.550 else 1)"; then
    winner=e30r2; echo "[chain2] $(date -u) PROMOTE e30r2 (${BEST[e30r2_screen_10ep]})"; fi
fi

if [ -z "${winner:-}" ]; then
  launch e30ws_screen_10ep E30 10 --d 128 --lr0 0.002 --lrf 0.1 --patience 10 \
    --close-mosaic 0 --fuse-warmup 2 --fuse-ramp 2 \
    --init-ir runs/e00_ir/weights/best.pt --init-rgb runs/e01_rgb/weights/best.pt
  BEST[e30ws_screen_10ep]=$(best_of e30ws_screen_10ep)
  if "$PY" -c "import sys; sys.exit(0 if float('${BEST[e30ws_screen_10ep]}') >= 0.550 else 1)"; then
    winner=e30ws; echo "[chain2] $(date -u) PROMOTE e30ws (${BEST[e30ws_screen_10ep]})"; fi
fi

if [ -z "${winner:-}" ]; then
  launch e35ws_screen_10ep E35 10 --d 128 --lr0 0.002 --lrf 0.1 --patience 10 \
    --close-mosaic 0 --fuse-warmup 2 --fuse-ramp 2 \
    --init-ir runs/e00_ir/weights/best.pt --init-rgb runs/e01_rgb/weights/best.pt
  BEST[e35ws_screen_10ep]=$(best_of e35ws_screen_10ep)
  if "$PY" -c "import sys; sys.exit(0 if float('${BEST[e35ws_screen_10ep]}') >= 0.550 else 1)"; then
    winner=e35ws; echo "[chain2] $(date -u) PROMOTE e35ws (${BEST[e35ws_screen_10ep]})"; fi
fi

if [ -z "${winner:-}" ]; then
  # no screen crossed 0.550: promote the best screen if it beats E30's screen
  best_name=""; best_val=0
  for n in e41_screen_10ep e30r2_screen_10ep e30ws_screen_10ep e35ws_screen_10ep; do
    v=$(best_of "$n"); echo "[chain2] screen $n best=$v"
    "$PY" -c "import sys; sys.exit(0 if float('$v') > float('$best_val') else 1)" && {
      best_val=$v; best_name=$n; }
  done
  echo "[chain2] $(date -u) no screen >=0.550; best=$best_name ($best_val)"
  if "$PY" -c "import sys; sys.exit(0 if float('$best_val') >= 0.5407 else 1)"; then
    case "$best_name" in
      e41_screen_10ep) winner=e41 ;;
      e30r2_screen_10ep) winner=e30r2 ;;
      e30ws_screen_10ep) winner=e30ws ;;
      e35ws_screen_10ep) winner=e35ws ;;
    esac
    echo "[chain2] $(date -u) PROMOTE best screen anyway: $winner"
  else
    echo "[chain2] $(date -u) ABORT: nothing beats the E30 screen; needs design review"
    exit 1
  fi
fi

# ---- 2. full run for the winner -------------------------------------------
echo "[chain2] $(date -u) FULL RUN: $winner"
case "$winner" in
  e41)
    launch e41_v2_100ep E41 100 --d 128 --close-mosaic 10 ;;
  e30r2)
    launch e30r2_v2_100ep E30 100 --d 128 --close-mosaic 10 --fuse-r 2 ;;
  e30ws)
    launch e30ws_v2_50ep E30 50 --d 128 --lr0 0.002 --lrf 0.1 --patience 20 \
      --close-mosaic 5 --fuse-warmup 2 --fuse-ramp 2 \
      --init-ir runs/e00_ir/weights/best.pt --init-rgb runs/e01_rgb/weights/best.pt ;;
  e35ws)
    launch e35ws_v2_50ep E35 50 --d 128 --lr0 0.002 --lrf 0.1 --patience 20 \
      --close-mosaic 5 --fuse-warmup 2 --fuse-ramp 2 \
      --init-ir runs/e00_ir/weights/best.pt --init-rgb runs/e01_rgb/weights/best.pt ;;
esac
echo "[chain2] $(date -u) CHAIN COMPLETE: winner=$winner full-run done"
