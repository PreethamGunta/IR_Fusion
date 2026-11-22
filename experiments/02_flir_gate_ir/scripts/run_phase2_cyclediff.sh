#!/usr/bin/env bash
# Phase 2 — CycleDiff-75 with the winning CycleGAN filter.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
FLIR="$ROOT/data/flir_adas_v2"
WORKERS="${WORKERS:-16}"
EPOCHS="${EPOCHS_CYCLEDIFF:-60}"
WEIGHTS="${WEIGHTS:-$ROOT/models/pretrained/yolo12s.pt}"

mkdir -p data/filtered results runs logs

WINNER="$($PY -c "import json; print(json.load(open('results/screen_summary.json'))['best_filter'])")"
if [ -z "$WINNER" ]; then echo "FATAL: no winner in results/screen_summary.json"; exit 1; fi
echo "=== [phase2] winner filter: $WINNER ==="

FOG_TRAIN="$FLIR/synthetic/cyclediff_75fog/fog/images/train"
FOG_TEST="$FLIR/synthetic/cyclediff_75fog/fog/images/test"

filter_dir () {
  local src="$1" dst="$2"
  if [ -f "$dst/.done" ]; then echo "  [skip] $dst"; return; fi
  rm -rf "$dst"
  if [ "$WINNER" = "unet" ]; then
    $PY -m src.apply_unet --src "$src" --dst "$dst" --batch 8
  else
    $PY -m src.filters --src "$src" --dst "$dst" --filter "$WINNER" --workers "$WORKERS"
  fi
  touch "$dst/.done"
}

filter_dir "$FOG_TRAIN" "data/filtered/$WINNER/cyclediff75/train"
filter_dir "$FOG_TEST" "data/filtered/$WINNER/cyclediff75/test"

$PY -m src.build_cyclediff --filter "$WINNER"
$PY -m src.gate --condition cyclediff75 --workers "$WORKERS"
$PY -m src.route --condition cyclediff75 --filter "$WINNER" --name "gated_learned_${WINNER}_cyclediff75"
$PY -m src.route --condition cyclediff75 --filter "$WINNER" --name "gated_oracle_${WINNER}_cyclediff75" --oracle

train_arm () {
  local dataset="$1" name="$2"
  if [ -f "runs/$name/DONE" ]; then echo "  [skip] $name"; return; fi
  echo "=== [phase2] train $name ==="
  $PY -m src.train_detector --dataset "$dataset" --name "$name" --epochs "$EPOCHS" \
    --batch 16 --imgsz 640 --workers "$WORKERS" --weights "$WEIGHTS" \
    2>&1 | tee "logs/train_${name}.log" || { echo "TRAIN FAILED: $name"; return 0; }
  touch "runs/$name/DONE"
}

eval_arm () {
  local name="$1" weights="$2" data="$3"
  echo "=== [phase2] eval $name ==="
  $PY -m src.eval_detector --weights "$weights" --data "configs/$data" \
    --name "$name" --force 2>&1 | tee -a logs/eval_phase2.log || { echo "EVAL FAILED: $name"; return 0; }
}

train_arm "b1_cyclediff75" "b1_cyclediff75_yolo12s_${EPOCHS}"
train_arm "g1_${WINNER}_cyclediff75" "g1_${WINNER}_cyclediff75_yolo12s_${EPOCHS}"

wb="runs/b1_cyclediff75_yolo12s_${EPOCHS}/weights/best.pt"
wg="runs/g1_${WINNER}_cyclediff75_yolo12s_${EPOCHS}/weights/best.pt"

eval_arm "b1_cyclediff75_${EPOCHS}_clear_vtt" "$wb" "b1_cyclediff75_eval_clear.yaml"
eval_arm "b1_cyclediff75_${EPOCHS}_fog_vtt"   "$wb" "b1_cyclediff75_eval_fog.yaml"

eval_arm "g1_${WINNER}_cyclediff75_${EPOCHS}_clear_vtt"   "$wg" "g1_${WINNER}_cyclediff75_eval_clear.yaml"
eval_arm "g1_${WINNER}_cyclediff75_${EPOCHS}_fog_vtt"     "$wg" "g1_${WINNER}_cyclediff75_eval_fog.yaml"
eval_arm "g1_${WINNER}_cyclediff75_${EPOCHS}_gated_clear" "$wg" "gated_learned_${WINNER}_cyclediff75_eval_clear.yaml"
eval_arm "g1_${WINNER}_cyclediff75_${EPOCHS}_gated_fog"   "$wg" "gated_learned_${WINNER}_cyclediff75_eval_fog.yaml"
eval_arm "g1_${WINNER}_cyclediff75_${EPOCHS}_oracle_clear" "$wg" "gated_oracle_${WINNER}_cyclediff75_eval_clear.yaml"
eval_arm "g1_${WINNER}_cyclediff75_${EPOCHS}_oracle_fog"   "$wg" "gated_oracle_${WINNER}_cyclediff75_eval_fog.yaml"

$PY -m src.summarize_results
echo "=== Phase 2 done $(date) ==="
