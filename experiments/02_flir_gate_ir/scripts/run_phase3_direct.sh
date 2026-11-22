#!/usr/bin/env bash
# Phase 3 — ablations, prioritized for the remaining allocation.
# Order: B3 (always-on, winner) -> B0 (clear-only) -> B2 (fog-only) ->
#        B3 CycleDiff / B2 CycleDiff if time.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
FLIR="$ROOT/data/flir_adas_v2"
WEIGHTS="${WEIGHTS:-$ROOT/models/pretrained/yolov8s.pt}"
WORKERS="${WORKERS:-12}"
EPOCHS="${EPOCHS_ABL:-40}"
B3_FILTERS="${B3_FILTERS:-}"

mkdir -p configs data results runs logs

WINNER="$($PY -c "import json;print(json.load(open('results/screen_summary.json'))['best_filter'])" 2>/dev/null || true)"
if [ -z "$WINNER" ] || [ "$WINNER" = "None" ]; then echo "FATAL: no winner"; exit 1; fi
[ -n "$B3_FILTERS" ] || B3_FILTERS="$WINNER"
echo "=== [phase3] winner=$WINNER B3=$B3_FILTERS epochs=$EPOCHS ==="

train_arm () {
  local dataset="$1" name="$2"
  if [ -f "runs/$name/DONE" ]; then echo "  [skip] $name"; return 0; fi
  local RESUME=""
  [ -f "runs/$name/weights/last.pt" ] && RESUME="--resume"
  echo "=== [phase3] train $name ${RESUME:+(resume)} ==="
  $PY -m src.train_detector --dataset "$dataset" --name "$name" --epochs "$EPOCHS" \
    --batch 16 --imgsz 640 --workers "$WORKERS" --weights "$WEIGHTS" --cache ram $RESUME \
    2>&1 | tee "logs/train_${name}.log" || { echo "TRAIN FAILED: $name"; return 0; }
  touch "runs/$name/DONE"
}

eval_arm () {
  local name="$1" weights="$2" data="$3"
  [ -f "$weights" ] || { echo "  [skip eval] no weights $weights"; return 0; }
  echo "=== [phase3] eval $name ==="
  $PY -m src.eval_detector --weights "$weights" --data "configs/$data" \
    --name "$name" --force 2>&1 | tee -a logs/eval_phase3.log || echo "EVAL FAILED: $name"
}

generate_images () {
  local f="$1" condition="$2" split="$3" src="$4"
  local dst="data/filtered/$f/$condition/$split"
  if [ -f "$dst/.done" ]; then return 0; fi
  echo "=== [phase3] filtered $f/$condition/$split ==="
  rm -rf "$dst"
  if [ "$f" = "unet" ]; then
    $PY -m src.apply_unet --src "$src" --dst "$dst" --batch 8
  else
    $PY -m src.filters --src "$src" --dst "$dst" --filter "$f" --workers "$WORKERS"
  fi
  touch "$dst/.done"
}

# --- B3: always-on filter (winner) — PRIORITY ------------------------------
for f in $B3_FILTERS; do
  generate_images "$f" clear train "$FLIR/yolo/train_3a/images/train_clear"
  generate_images "$f" clear val "$FLIR/yolo/train_3a/images/val"
  generate_images "$f" clear test "$FLIR/FLIR_ADAS_v2/test/clear"
  generate_images "$f" cyclegan train "$FLIR/yolo/train_3a/images/train_adv"
  generate_images "$f" cyclegan test "$FLIR/synthetic/cyclegan/fog/images/test"
  $PY -m src.build_direct b3 --filter "$f" --condition cyclegan --name "b3_${f}_direct"
  train_arm "b3_${f}_direct" "b3_${f}_yolov8s_${EPOCHS}"
  w="runs/b3_${f}_yolov8s_${EPOCHS}/weights/best.pt"
  eval_arm "b3_${f}_${EPOCHS}_clear_vtt" "$w" "b3_${f}_direct_eval_clear.yaml"
  eval_arm "b3_${f}_${EPOCHS}_fog_vtt" "$w" "b3_${f}_direct_eval_fog.yaml"
  rm -rf "data/filtered/$f/cyclegan"
done

# --- B0: clear only ---------------------------------------------------------
$PY -m src.build_direct b0 --condition cyclegan --name b0_direct
$PY -m src.build_direct b0 --condition cyclediff75 --name b0_cyclediff75_direct
train_arm b0_direct "b0_yolov8s_${EPOCHS}"
w0="runs/b0_yolov8s_${EPOCHS}/weights/best.pt"
eval_arm "b0_${EPOCHS}_clear_vtt" "$w0" "b0_direct_eval_clear.yaml"
eval_arm "b0_${EPOCHS}_fog_vtt" "$w0" "b0_direct_eval_fog.yaml"
eval_arm "b0_${EPOCHS}_cyclediff75_fog_vtt" "$w0" "b0_cyclediff75_direct_eval_fog.yaml"

# --- B2: fog only, unfiltered ----------------------------------------------
$PY -m src.build_direct b2 --condition cyclegan --name b2_direct
$PY -m src.build_direct b2 --condition cyclediff75 --name b2_cyclediff75_direct
train_arm b2_direct "b2_yolov8s_${EPOCHS}"
eval_arm "b2_${EPOCHS}_fog_vtt" "runs/b2_yolov8s_${EPOCHS}/weights/best.pt" "b2_direct_eval_fog.yaml"

# --- CycleDiff extras if time ----------------------------------------------
train_arm b2_cyclediff75_direct "b2_cyclediff75_yolov8s_${EPOCHS}"
eval_arm "b2_cyclediff75_${EPOCHS}_fog_vtt" "runs/b2_cyclediff75_yolov8s_${EPOCHS}/weights/best.pt" "b2_cyclediff75_direct_eval_fog.yaml"

f="$WINNER"
generate_images "$f" cyclediff75 train "$FLIR/synthetic/cyclediff_75fog/fog/images/train"
generate_images "$f" cyclediff75 test "$FLIR/synthetic/cyclediff_75fog/fog/images/test"
$PY -m src.build_direct b3 --filter "$f" --condition cyclediff75 --name "b3_${f}_cyclediff75_direct"
train_arm "b3_${f}_cyclediff75_direct" "b3_${f}_cyclediff75_yolov8s_${EPOCHS}"
w="runs/b3_${f}_cyclediff75_yolov8s_${EPOCHS}/weights/best.pt"
eval_arm "b3_${f}_cyclediff75_${EPOCHS}_clear_vtt" "$w" "b3_${f}_cyclediff75_direct_eval_clear.yaml"
eval_arm "b3_${f}_cyclediff75_${EPOCHS}_fog_vtt" "$w" "b3_${f}_cyclediff75_direct_eval_fog.yaml"

$PY -m src.summarize_results || true
$PY -m src.make_report || true
echo "=== Phase 3 done $(date) ==="
