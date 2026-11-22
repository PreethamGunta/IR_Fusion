#!/usr/bin/env bash
# Train the IR, RGB and early-fusion (6-channel) teachers on the paired split.
# Resumable: a run with a DONE marker is skipped; failures are logged and the
# script continues so one bad arm cannot block the rest.
set -uo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$EXP/src"
LOGS="$EXP/logs"; mkdir -p "$LOGS"
cd "$SRC" || exit 1
export PYTHONPATH="$SRC"
PY=python
PRET="$EXP/../../models/pretrained"

run_one () {
  local name="$1"; shift
  if [ -f "$EXP/runs/$name/DONE" ]; then echo "[$(date)] skip $name"; return 0; fi
  if [ -f "$EXP/runs/$name/weights/last.pt" ]; then set -- "$@" --resume; fi
  echo "[$(date)] start $name"
  if "$@" 2>&1 | tee -a "$LOGS/$name.log"; then
    touch "$EXP/runs/$name/DONE"
    echo "[$(date)] done $name"
  else
    echo "[$(date)] FAILED $name" | tee -a "$LOGS/$name.log"
  fi
}

run_one ir_teacher $PY -m graphkd.training.train \
  --name ir_teacher --model "$PRET/yolo12s.pt" --data "$EXP/configs/ir.yaml" \
  --epochs 100 --patience 30 --batch 16 --workers 8 --exist-ok

run_one rgb_teacher $PY -m graphkd.training.train \
  --name rgb_teacher --model "$PRET/yolo12s.pt" --data "$EXP/configs/rgb.yaml" \
  --epochs 100 --patience 30 --batch 16 --workers 8 --exist-ok

run_one fused_teacher $PY -m graphkd.training.train \
  --name fused_teacher --model yolo12s.yaml --pretrained "$PRET/yolo12s.pt" \
  --fused --data "$EXP/configs/paired.yaml" --epochs 100 --patience 30 \
  --batch 16 --workers 8 --no-val --exist-ok

echo "[$(date)] teacher phase finished"
