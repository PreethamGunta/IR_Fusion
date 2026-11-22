#!/usr/bin/env bash
# Proper 100-epoch YOLOv12s B1 vs G1 on the CycleDiff-75 synthetic images.
# G1 uses the U-Net retrained on CycleDiff-75 pairs (target-generator filter).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
FLIR="$ROOT/data/flir_adas_v2"
WEIGHTS="${WEIGHTS:-$ROOT/models/pretrained/yolo12s.pt}"
WORKERS="${WORKERS:-12}"
EPOCHS="${EPOCHS_MAIN100:-100}"
UNET="results/unet_dehazer_cyclediff75.pt"
FILTER="unet_cd75"
COND="cyclediff75"

mkdir -p configs data results runs logs
$PY -m src.build_direct init

if [ ! -f "$UNET" ]; then
  echo "=== [main100-cd] training CycleDiff U-Net ==="
  $PY -m src.train_unet --condition "$COND" --epochs 30 --batch 16 --crop 256 \
    --workers "$WORKERS" --out "$UNET" 2>&1 | tee logs/train_unet_cyclediff75.log \
    || { echo "UNET TRAIN FAILED"; exit 1; }
fi

BASE="data/filtered/$FILTER/$COND"
if [ ! -f "$BASE/train/.done" ] || [ ! -f "$BASE/test/.done" ]; then
  echo "=== [main100-cd] applying CycleDiff U-Net ==="
  rm -rf "data/filtered/$FILTER"
  $PY -m src.apply_unet --weights "$UNET" \
    --src "$FLIR/synthetic/cyclediff_75fog/fog/images/train" --dst "$BASE/train" --batch 8
  $PY -m src.apply_unet --weights "$UNET" \
    --src "$FLIR/synthetic/cyclediff_75fog/fog/images/test" --dst "$BASE/test" --batch 8
  touch "$BASE/train/.done" "$BASE/test/.done"
fi

$PY -m src.build_direct b1 --condition "$COND" --name b1_cyclediff75_direct
$PY -m src.build_direct g1 --filter "$FILTER" --condition "$COND" --name g1_unet_cd75_direct

train_arm () {
  local dataset="$1" name="$2"
  if [ -f "runs/$name/DONE" ]; then echo "  [skip] $name"; return 0; fi
  local RESUME=""
  [ -f "runs/$name/weights/last.pt" ] && RESUME="--resume"
  echo "=== [main100-cd] train $name ($EPOCHS epochs) ${RESUME:+(resume)} ==="
  $PY -m src.train_detector --dataset "$dataset" --name "$name" --epochs "$EPOCHS" \
    --batch 16 --imgsz 640 --workers "$WORKERS" --weights "$WEIGHTS" --cache ram $RESUME \
    2>&1 | tee "logs/train_${name}.log" || { echo "TRAIN FAILED: $name"; return 0; }
  touch "runs/$name/DONE"
}

eval_arm () {
  local name="$1" weights="$2" data="$3"
  [ -f "$weights" ] || { echo "  [skip eval] no weights $weights"; return 0; }
  echo "=== [main100-cd] eval $name ==="
  $PY -m src.eval_detector --weights "$weights" --data "configs/$data" \
    --name "$name" --force 2>&1 | tee -a logs/eval_main100_cyclediff.log || echo "EVAL FAILED: $name"
}

train_arm b1_cyclediff75_direct "b1_cyclediff75_yolo12s_${EPOCHS}"
train_arm g1_unet_cd75_direct "g1_unet_cd75_yolo12s_${EPOCHS}"

eval_arm "b1_cyclediff75_yolo12s_${EPOCHS}_clear_vtt" "runs/b1_cyclediff75_yolo12s_${EPOCHS}/weights/best.pt" b1_cyclediff75_direct_eval_clear.yaml
eval_arm "b1_cyclediff75_yolo12s_${EPOCHS}_fog_vtt"   "runs/b1_cyclediff75_yolo12s_${EPOCHS}/weights/best.pt" b1_cyclediff75_direct_eval_fog.yaml
eval_arm "g1_unet_cd75_${EPOCHS}_clear_vtt" "runs/g1_unet_cd75_yolo12s_${EPOCHS}/weights/best.pt" g1_unet_cd75_direct_eval_clear.yaml
eval_arm "g1_unet_cd75_${EPOCHS}_fog_vtt"   "runs/g1_unet_cd75_yolo12s_${EPOCHS}/weights/best.pt" g1_unet_cd75_direct_eval_fog.yaml

rm -rf "data/filtered/$FILTER"
$PY -m src.summarize_results || true
$PY -m src.make_report || true
echo "=== main100-cyclediff done $(date) ==="
