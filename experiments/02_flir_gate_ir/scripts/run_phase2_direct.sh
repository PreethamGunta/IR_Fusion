#!/usr/bin/env bash
# Phase 2 — CycleDiff-75 with the winning CycleGAN filter (YOLOv8s, 60 epochs).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
FLIR="$ROOT/data/flir_adas_v2"
WEIGHTS="${WEIGHTS:-$ROOT/models/pretrained/yolov8s.pt}"
WORKERS="${WORKERS:-12}"
EPOCHS="${EPOCHS_CYCLEDIFF:-40}"
COND="cyclediff75"

mkdir -p configs data results runs logs

WINNER="$($PY -c "import json;print(json.load(open('results/screen_summary.json'))['best_filter'])" 2>/dev/null || true)"
if [ -z "$WINNER" ] || [ "$WINNER" = "None" ]; then echo "FATAL: no winner"; exit 1; fi
echo "=== [phase2] winner=$WINNER ==="

FT="$FLIR/synthetic/cyclediff_75fog/fog/images/train"
FS="$FLIR/synthetic/cyclediff_75fog/fog/images/test"
BASE="data/filtered/$WINNER/$COND"

if [ ! -d "$BASE/train" ] || [ "$(ls "$BASE/train" 2>/dev/null | wc -l)" -lt 5000 ]; then
  rm -rf "data/filtered/$WINNER"
  if [ "$WINNER" = "unet" ]; then
    $PY -m src.apply_unet --src "$FT" --dst "$BASE/train" --batch 8
    $PY -m src.apply_unet --src "$FS" --dst "$BASE/test" --batch 8
  else
    $PY -m src.filters --src "$FT" --dst "$BASE/train" --filter "$WINNER" --workers "$WORKERS"
    $PY -m src.filters --src "$FS" --dst "$BASE/test" --filter "$WINNER" --workers "$WORKERS"
  fi
fi

$PY -m src.build_direct init
$PY -m src.build_direct b1 --condition "$COND" --name b1_cyclediff75_direct
$PY -m src.build_direct g1 --filter "$WINNER" --condition "$COND" --name "g1_${WINNER}_cyclediff75_direct"

train_arm () {
  local dataset="$1" name="$2"
  if [ -f "runs/$name/DONE" ]; then echo "  [skip] $name"; return 0; fi
  echo "=== [phase2] train $name ==="
  $PY -m src.train_detector --dataset "$dataset" --name "$name" --epochs "$EPOCHS" \
    --batch 16 --imgsz 640 --workers "$WORKERS" --weights "$WEIGHTS" --cache ram \
    2>&1 | tee "logs/train_${name}.log" || { echo "TRAIN FAILED: $name"; return 0; }
  touch "runs/$name/DONE"
}

eval_arm () {
  local name="$1" weights="$2" data="$3"
  [ -f "$weights" ] || { echo "  [skip eval] no weights $weights"; return 0; }
  echo "=== [phase2] eval $name ==="
  $PY -m src.eval_detector --weights "$weights" --data "configs/$data" \
    --name "$name" --force 2>&1 | tee -a logs/eval_phase2.log || echo "EVAL FAILED: $name"
}

train_arm b1_cyclediff75_direct "b1_cyclediff75_yolov8s_${EPOCHS}"
train_arm "g1_${WINNER}_cyclediff75_direct" "g1_${WINNER}_cyclediff75_yolov8s_${EPOCHS}"

wb="runs/b1_cyclediff75_yolov8s_${EPOCHS}/weights/best.pt"
wg="runs/g1_${WINNER}_cyclediff75_yolov8s_${EPOCHS}/weights/best.pt"

eval_arm "b1_cyclediff75_yolov8s_${EPOCHS}_clear_vtt" "$wb" "b1_cyclediff75_direct_eval_clear.yaml"
eval_arm "b1_cyclediff75_yolov8s_${EPOCHS}_fog_vtt"   "$wb" "b1_cyclediff75_direct_eval_fog.yaml"
eval_arm "g1_${WINNER}_cyclediff75_yolov8s_${EPOCHS}_clear_vtt" "$wg" "g1_${WINNER}_cyclediff75_direct_eval_clear.yaml"
eval_arm "g1_${WINNER}_cyclediff75_yolov8s_${EPOCHS}_fog_vtt"   "$wg" "g1_${WINNER}_cyclediff75_direct_eval_fog.yaml"

rm -rf "data/filtered/$WINNER"
$PY -m src.summarize_results || true
$PY -m src.make_report || true
echo "=== Phase 2 done $(date) ==="
