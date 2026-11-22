#!/usr/bin/env bash
# 100-epoch YOLOv12s confirmation of the main comparison (B1 vs G1-U-Net).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
FLIR="$ROOT/data/flir_adas_v2"
WEIGHTS="${WEIGHTS:-$ROOT/models/pretrained/yolo12s.pt}"
WORKERS="${WORKERS:-12}"
EPOCHS="${EPOCHS_MAIN100:-100}"

mkdir -p configs data results runs logs
$PY -m src.build_direct init

BASE="data/filtered/unet/cyclegan"
if [ ! -f "$BASE/test/.done" ] || [ ! -f "$BASE/train/.done" ]; then
  echo "=== [main100] generating U-Net filtered CycleGAN fog ==="
  rm -rf "data/filtered/unet"
  $PY -m src.apply_unet --src "$FLIR/yolo/train_3a/images/train_adv" --dst "$BASE/train" --batch 8
  $PY -m src.apply_unet --src "$FLIR/synthetic/cyclegan/fog/images/test" --dst "$BASE/test" --batch 8
  touch "$BASE/train/.done" "$BASE/test/.done"
fi

$PY -m src.build_direct b1 --condition cyclegan --name b1_direct
$PY -m src.build_direct g1 --filter unet --condition cyclegan --name g1_unet_direct

train_arm () {
  local dataset="$1" name="$2"
  if [ -f "runs/$name/DONE" ]; then echo "  [skip] $name"; return 0; fi
  local RESUME=""
  [ -f "runs/$name/weights/last.pt" ] && RESUME="--resume"
  echo "=== [main100] train $name ($EPOCHS epochs) ${RESUME:+(resume)} ==="
  $PY -m src.train_detector --dataset "$dataset" --name "$name" --epochs "$EPOCHS" \
    --batch 16 --imgsz 640 --workers "$WORKERS" --weights "$WEIGHTS" --cache ram $RESUME \
    2>&1 | tee "logs/train_${name}.log" || { echo "TRAIN FAILED: $name"; return 0; }
  touch "runs/$name/DONE"
}

eval_arm () {
  local name="$1" weights="$2" data="$3"
  [ -f "$weights" ] || { echo "  [skip eval] no weights $weights"; return 0; }
  echo "=== [main100] eval $name ==="
  $PY -m src.eval_detector --weights "$weights" --data "configs/$data" \
    --name "$name" --force 2>&1 | tee -a logs/eval_main100.log || echo "EVAL FAILED: $name"
}

train_arm b1_direct "b1_yolo12s_${EPOCHS}"
train_arm g1_unet_direct "g1_unet_yolo12s_${EPOCHS}"

eval_arm "b1_yolo12s_${EPOCHS}_clear_vtt" "runs/b1_yolo12s_${EPOCHS}/weights/best.pt" b1_direct_eval_clear.yaml
eval_arm "b1_yolo12s_${EPOCHS}_fog_vtt"   "runs/b1_yolo12s_${EPOCHS}/weights/best.pt" b1_direct_eval_fog.yaml
eval_arm "g1_unet_${EPOCHS}_clear_vtt"    "runs/g1_unet_yolo12s_${EPOCHS}/weights/best.pt" g1_unet_direct_eval_clear.yaml
eval_arm "g1_unet_${EPOCHS}_fog_vtt"      "runs/g1_unet_yolo12s_${EPOCHS}/weights/best.pt" g1_unet_direct_eval_fog.yaml

rm -rf "data/filtered/unet"
$PY -m src.make_report || true
echo "=== main100 done $(date) ==="
