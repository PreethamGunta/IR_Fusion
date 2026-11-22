#!/usr/bin/env bash
# Phase 3 — B0 / B2 / B3 ablations for the methods already run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
FLIR="$ROOT/data/flir_adas_v2"
WORKERS="${WORKERS:-16}"
EPOCHS="${EPOCHS_ABL:-60}"
WEIGHTS="${WEIGHTS:-$ROOT/models/pretrained/yolo12s.pt}"
B3_FILTERS="${B3_FILTERS:-gamma clahe dcp unet gamma_clahe}"

mkdir -p results runs logs

WINNER="$($PY -c "import json; print(json.load(open('results/screen_summary.json'))['best_filter'])")"
echo "=== [phase3] winner=$WINNER B3 filters=$B3_FILTERS ==="

filter_clear () {
  local f="$1"
  local src dst
  for split in train val test; do
    dst="data/filtered/$f/clear/$split"
    if [ -f "$dst/.done" ]; then continue; fi
    case "$split" in
      train) src="$FLIR/yolo/train_3a/images/train_clear" ;;
      val)   src="$FLIR/yolo/train_3a/images/val" ;;
      test)  src="$FLIR/FLIR_ADAS_v2/test/clear" ;;
    esac
    echo "=== [phase3] filtered clear $split for $f ==="
    rm -rf "$dst"
    if [ "$f" = "unet" ]; then
      $PY -m src.apply_unet --src "$src" --dst "$dst" --batch 8
    else
      $PY -m src.filters --src "$src" --dst "$dst" --filter "$f" --workers "$WORKERS"
    fi
    touch "$dst/.done"
  done
}

for f in $B3_FILTERS; do filter_clear "$f"; done

$PY -m src.build_ablations --filter "$WINNER" --filters $B3_FILTERS

train_arm () {
  local dataset="$1" name="$2"
  if [ -f "runs/$name/DONE" ]; then echo "  [skip] $name"; return; fi
  echo "=== [phase3] train $name ==="
  $PY -m src.train_detector --dataset "$dataset" --name "$name" --epochs "$EPOCHS" \
    --batch 16 --imgsz 640 --workers "$WORKERS" --weights "$WEIGHTS" \
    2>&1 | tee "logs/train_${name}.log" || { echo "TRAIN FAILED: $name"; return 0; }
  touch "runs/$name/DONE"
}

eval_arm () {
  local name="$1" weights="$2" data="$3"
  echo "=== [phase3] eval $name ==="
  $PY -m src.eval_detector --weights "$weights" --data "configs/$data" \
    --name "$name" --force 2>&1 | tee -a logs/eval_phase3.log || { echo "EVAL FAILED: $name"; return 0; }
}

# --- B0: clear only ---------------------------------------------------------
train_arm b0 "b0_yolo12s_${EPOCHS}"
w0="runs/b0_yolo12s_${EPOCHS}/weights/best.pt"
eval_arm "b0_${EPOCHS}_clear_vtt" "$w0" "b0_eval_clear.yaml"
eval_arm "b0_${EPOCHS}_fog_vtt" "$w0" "b0_eval_fog.yaml"
eval_arm "b0_${EPOCHS}_cyclediff75_fog_vtt" "$w0" "b0_cyclediff75_eval_fog.yaml"

# --- B2: fog only, unfiltered ----------------------------------------------
train_arm b2 "b2_yolo12s_${EPOCHS}"
w2="runs/b2_yolo12s_${EPOCHS}/weights/best.pt"
eval_arm "b2_${EPOCHS}_fog_vtt" "$w2" "b2_eval_fog.yaml"

train_arm b2_cyclediff75 "b2_cyclediff75_yolo12s_${EPOCHS}"
w2c="runs/b2_cyclediff75_yolo12s_${EPOCHS}/weights/best.pt"
eval_arm "b2_cyclediff75_${EPOCHS}_fog_vtt" "$w2c" "b2_cyclediff75_eval_fog.yaml"

# --- B3: always-on filter ---------------------------------------------------
for f in $B3_FILTERS; do
  if [ ! -f "configs/b3_${f}_train.yaml" ]; then echo "  [skip] b3_${f} (no config)"; continue; fi
  train_arm "b3_${f}" "b3_${f}_yolo12s_${EPOCHS}"
  w="runs/b3_${f}_yolo12s_${EPOCHS}/weights/best.pt"
  eval_arm "b3_${f}_${EPOCHS}_clear_vtt" "$w" "b3_${f}_eval_clear.yaml"
  eval_arm "b3_${f}_${EPOCHS}_fog_vtt" "$w" "b3_${f}_eval_fog.yaml"
done

if [ -f "configs/b3_${WINNER}_cyclediff75_train.yaml" ]; then
  train_arm "b3_${WINNER}_cyclediff75" "b3_${WINNER}_cyclediff75_yolo12s_${EPOCHS}"
  w="runs/b3_${WINNER}_cyclediff75_yolo12s_${EPOCHS}/weights/best.pt"
  eval_arm "b3_${WINNER}_cyclediff75_${EPOCHS}_clear_vtt" "$w" "b3_${WINNER}_cyclediff75_eval_clear.yaml"
  eval_arm "b3_${WINNER}_cyclediff75_${EPOCHS}_fog_vtt" "$w" "b3_${WINNER}_cyclediff75_eval_fog.yaml"
fi

$PY -m src.summarize_results
echo "=== Phase 3 done $(date) ==="
