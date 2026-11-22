#!/usr/bin/env bash
# CycleGAN main numbers, filter-at-a-time to stay inside the home space quota.
#
# Screen: YOLOv8s, 60 epochs (fast enough for all five filters).
# Main:   YOLOv12s, 40 epochs (compute-limited), B1 vs winning filter.
# For each filter: generate filtered fog -> direct YAMLs -> train -> evaluate
# -> DELETE the filtered images. Peak disk use is one filter (~2.5 GB).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
FLIR="$ROOT/data/flir_adas_v2"
WEIGHTS_SCREEN="${WEIGHTS_SCREEN:-$ROOT/models/pretrained/yolov8s.pt}"
WEIGHTS_FINAL="${WEIGHTS_FINAL:-$ROOT/models/pretrained/yolo12s.pt}"
WORKERS="${WORKERS:-12}"
EPOCHS_SCREEN="${EPOCHS_SCREEN:-60}"
EPOCHS_FINAL="${EPOCHS_FINAL:-40}"
COND="${COND:-cyclegan}"
FILTERS="${FILTERS:-gamma clahe dcp unet gamma_clahe}"

mkdir -p configs data results runs logs

generate_filter () {
  local f="$1"
  local base="data/filtered/$f/$COND"
  if [ -d "$base/train" ] && [ "$(ls "$base/train" 2>/dev/null | wc -l)" -gt 5000 ] \
     && [ -d "$base/test" ] && [ "$(ls "$base/test" 2>/dev/null | wc -l)" -gt 3000 ]; then
    echo "  [skip] filtered $f exists"
    return 0
  fi
  rm -rf "data/filtered/$f"
  if [ "$COND" = "cyclegan" ]; then
    FT="$FLIR/yolo/train_3a/images/train_adv"
    FS="$FLIR/synthetic/cyclegan/fog/images/test"
  else
    FT="$FLIR/synthetic/cyclediff_75fog/fog/images/train"
    FS="$FLIR/synthetic/cyclediff_75fog/fog/images/test"
  fi
  if [ "$f" = "unet" ]; then
    $PY -m src.apply_unet --src "$FT" --dst "$base/train" --batch 8
    $PY -m src.apply_unet --src "$FS" --dst "$base/test" --batch 8
  else
    $PY -m src.filters --src "$FT" --dst "$base/train" --filter "$f" --workers "$WORKERS"
    $PY -m src.filters --src "$FS" --dst "$base/test" --filter "$f" --workers "$WORKERS"
  fi
}

train_arm () {
  local dataset="$1" name="$2" epochs="$3" weights="$4"
  if [ -f "runs/$name/DONE" ]; then echo "  [skip] $name"; return 0; fi
  echo "=== train $name ($epochs epochs, $(basename "$weights")) ==="
  $PY -m src.train_detector --dataset "$dataset" --name "$name" --epochs "$epochs" \
    --batch 16 --imgsz 640 --workers "$WORKERS" --weights "$weights" --cache ram \
    2>&1 | tee "logs/train_${name}.log" || { echo "TRAIN FAILED: $name"; return 0; }
  touch "runs/$name/DONE"
}

eval_arm () {
  local name="$1" weights="$2" data="$3"
  [ -f "$weights" ] || { echo "  [skip eval] no weights $weights"; return 0; }
  echo "=== eval $name ==="
  $PY -m src.eval_detector --weights "$weights" --data "configs/$data" \
    --name "$name" --force 2>&1 | tee -a logs/eval_screen.log || echo "EVAL FAILED: $name"
}

# --- shared matched VTT clear test tree -------------------------------------
$PY -m src.build_direct init

# --- B1 baseline (screen) ---------------------------------------------------
$PY -m src.build_direct b1 --condition "$COND" --name b1_direct
train_arm b1_direct "b1_yolov8s_${EPOCHS_SCREEN}" "$EPOCHS_SCREEN" "$WEIGHTS_SCREEN"
eval_arm "b1_yolov8s_${EPOCHS_SCREEN}_clear_vtt" "runs/b1_yolov8s_${EPOCHS_SCREEN}/weights/best.pt" b1_direct_eval_clear.yaml
eval_arm "b1_yolov8s_${EPOCHS_SCREEN}_fog_vtt"   "runs/b1_yolov8s_${EPOCHS_SCREEN}/weights/best.pt" b1_direct_eval_fog.yaml

# --- per-filter screen ------------------------------------------------------
for f in $FILTERS; do
  generate_filter "$f" || { echo "FILTER FAILED: $f"; continue; }
  $PY -m src.build_direct g1 --filter "$f" --condition "$COND" --name "g1_${f}_direct" \
    || { echo "BUILD FAILED: $f"; rm -rf "data/filtered/$f"; continue; }
  train_arm "g1_${f}_direct" "g1_${f}_yolov8s_${EPOCHS_SCREEN}" "$EPOCHS_SCREEN" "$WEIGHTS_SCREEN"
  eval_arm "g1_${f}_clear_vtt" "runs/g1_${f}_yolov8s_${EPOCHS_SCREEN}/weights/best.pt" "g1_${f}_direct_eval_clear.yaml"
  eval_arm "g1_${f}_fog_vtt"   "runs/g1_${f}_yolov8s_${EPOCHS_SCREEN}/weights/best.pt" "g1_${f}_direct_eval_fog.yaml"
  rm -rf "data/filtered/$f"
  echo "  [cleanup] removed filtered $f"
done

$PY -m src.summarize_screen || true

# --- main comparison: B1 vs winner, YOLOv12s --------------------------------
WINNER="$($PY -c "import json;print(json.load(open('results/screen_summary.json'))['best_filter'])" 2>/dev/null || true)"
if [ -n "$WINNER" ] && [ "$WINNER" != "None" ]; then
  echo "=== main comparison, winner: $WINNER ==="
  generate_filter "$WINNER"
  $PY -m src.build_direct g1 --filter "$WINNER" --condition "$COND" --name "g1_${WINNER}_direct"
  train_arm b1_direct "b1_yolo12s_${EPOCHS_FINAL}" "$EPOCHS_FINAL" "$WEIGHTS_FINAL"
  train_arm "g1_${WINNER}_direct" "g1_${WINNER}_yolo12s_${EPOCHS_FINAL}" "$EPOCHS_FINAL" "$WEIGHTS_FINAL"
  eval_arm "b1_yolo12s_${EPOCHS_FINAL}_clear_vtt" "runs/b1_yolo12s_${EPOCHS_FINAL}/weights/best.pt" b1_direct_eval_clear.yaml
  eval_arm "b1_yolo12s_${EPOCHS_FINAL}_fog_vtt"   "runs/b1_yolo12s_${EPOCHS_FINAL}/weights/best.pt" b1_direct_eval_fog.yaml
  eval_arm "g1_${WINNER}_${EPOCHS_FINAL}_clear_vtt" "runs/g1_${WINNER}_yolo12s_${EPOCHS_FINAL}/weights/best.pt" "g1_${WINNER}_direct_eval_clear.yaml"
  eval_arm "g1_${WINNER}_${EPOCHS_FINAL}_fog_vtt"   "runs/g1_${WINNER}_yolo12s_${EPOCHS_FINAL}/weights/best.pt" "g1_${WINNER}_direct_eval_fog.yaml"
  rm -rf "data/filtered/$WINNER"
else
  echo "NO WINNER — skipping main comparison"
fi

$PY -m src.summarize_results || true
$PY -m src.make_report || true
echo "=== screen finished $(date) ==="
