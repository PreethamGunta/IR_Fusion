#!/usr/bin/env bash
# Retrain the U-Net on CycleDiff-75 pairs and test whether target-generator
# training fixes the cross-generator failure (CycleGAN-trained U-Net hurt).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
FLIR="$ROOT/data/flir_adas_v2"
WEIGHTS="${WEIGHTS:-$ROOT/models/pretrained/yolov8s.pt}"
WORKERS="${WORKERS:-12}"
EPOCHS="${EPOCHS_CDUNET:-40}"
UNET_OUT="results/unet_dehazer_cyclediff75.pt"
FILTER_NAME="unet_cd75"

mkdir -p configs data results runs logs
$PY -m src.build_direct init

if [ ! -f "$UNET_OUT" ]; then
  echo "=== [cdunet] training U-Net on CycleDiff-75 pairs ==="
  $PY -m src.train_unet --condition cyclediff75 --epochs 30 --batch 16 --crop 256 \
    --workers "$WORKERS" --out "$UNET_OUT" 2>&1 | tee logs/train_unet_cyclediff75.log \
    || { echo "UNET TRAIN FAILED"; exit 1; }
fi

BASE="data/filtered/$FILTER_NAME/cyclediff75"
if [ ! -f "$BASE/test/.done" ] || [ ! -f "$BASE/train/.done" ]; then
  echo "=== [cdunet] applying CycleDiff-trained U-Net ==="
  rm -rf "data/filtered/$FILTER_NAME"
  $PY -m src.apply_unet --weights "$UNET_OUT" \
    --src "$FLIR/synthetic/cyclediff_75fog/fog/images/train" --dst "$BASE/train" --batch 8
  $PY -m src.apply_unet --weights "$UNET_OUT" \
    --src "$FLIR/synthetic/cyclediff_75fog/fog/images/test" --dst "$BASE/test" --batch 8
  touch "$BASE/train/.done" "$BASE/test/.done"
fi

$PY -m src.build_direct g1 --filter "$FILTER_NAME" --condition cyclediff75 --name g1_unet_cd75_direct

train_arm () {
  local dataset="$1" name="$2"
  if [ -f "runs/$name/DONE" ]; then echo "  [skip] $name"; return 0; fi
  echo "=== [cdunet] train $name ==="
  $PY -m src.train_detector --dataset "$dataset" --name "$name" --epochs "$EPOCHS" \
    --batch 16 --imgsz 640 --workers "$WORKERS" --weights "$WEIGHTS" --cache ram \
    2>&1 | tee "logs/train_${name}.log" || { echo "TRAIN FAILED: $name"; return 0; }
  touch "runs/$name/DONE"
}

eval_arm () {
  local name="$1" weights="$2" data="$3"
  [ -f "$weights" ] || { echo "  [skip eval] no weights $weights"; return 0; }
  echo "=== [cdunet] eval $name ==="
  $PY -m src.eval_detector --weights "$weights" --data "configs/$data" \
    --name "$name" --force 2>&1 | tee -a logs/eval_cdunet.log || echo "EVAL FAILED: $name"
}

train_arm g1_unet_cd75_direct "g1_unet_cd75_yolov8s_${EPOCHS}"
eval_arm "g1_unet_cd75_${EPOCHS}_clear_vtt" "runs/g1_unet_cd75_yolov8s_${EPOCHS}/weights/best.pt" g1_unet_cd75_direct_eval_clear.yaml
eval_arm "g1_unet_cd75_${EPOCHS}_fog_vtt"   "runs/g1_unet_cd75_yolov8s_${EPOCHS}/weights/best.pt" g1_unet_cd75_direct_eval_fog.yaml

rm -rf "data/filtered/$FILTER_NAME"
$PY -m src.make_report || true
echo "=== cdunet done $(date) ==="
