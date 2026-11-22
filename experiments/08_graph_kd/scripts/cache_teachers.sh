#!/usr/bin/env bash
# Cache teacher prediction graphs for every training image.
set -uo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$EXP/src"
LOGS="$EXP/logs"; mkdir -p "$LOGS"
cd "$SRC" || exit 1
export PYTHONPATH="$SRC"
PY=python

cache_one () {
  local name="$1"; shift
  if [ -f "$EXP/data/cache/$name/meta.json" ]; then echo "[$(date)] cache $name exists"; return 0; fi
  echo "[$(date)] cache $name"
  "$@" 2>&1 | tee -a "$LOGS/cache_$name.log"
}

W="$EXP/runs/ir_teacher/weights/best.pt"
[ -f "$W" ] || W="$EXP/runs/ir_teacher/weights/last.pt"
cache_one ir $PY -m graphkd.graph.cache_teacher --weights "$W" \
  --data "$EXP/configs/ir.yaml" --out "$EXP/data/cache/ir" --name ir --k 40

W="$EXP/runs/rgb_teacher/weights/best.pt"
[ -f "$W" ] || W="$EXP/runs/rgb_teacher/weights/last.pt"
cache_one rgb $PY -m graphkd.graph.cache_teacher --weights "$W" \
  --data "$EXP/configs/rgb.yaml" --out "$EXP/data/cache/rgb" --name rgb --k 40

# The fused teacher has no val split; always use the final checkpoint.
W="$EXP/runs/fused_teacher/weights/last.pt"
cache_one fused $PY -m graphkd.graph.cache_teacher --weights "$W" --paired \
  --data "$EXP/configs/paired.yaml" --out "$EXP/data/cache/fused" --name fused --k 40

echo "[$(date)] cache phase finished"
