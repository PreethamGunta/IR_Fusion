#!/usr/bin/env bash
# Phase 1 — CycleGAN main numbers with every filter.
#
# Stage order:
#   prep   : build filtered images (gamma/clahe/dcp/unet/gamma_clahe), train U-Net,
#            build G1 datasets, materialise gated test sets
#   screen : train YOLOv12s at 60 epochs for B1 and every G1 filter, evaluate
#   final  : retrain B1 and the winning filter at 100 epochs, evaluate
#
# Usage:
#   STAGE=prep   srun --jobid=9639 --overlap bash scripts/run_phase1_cyclegan.sh
#   STAGE=screen ...
#   STAGE=final  ...
#   STAGE=all    ...
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
FLIR="$ROOT/data/flir_adas_v2"
COND="${COND:-cyclegan}"
FILTERS="${FILTERS:-gamma clahe dcp unet gamma_clahe}"
WORKERS="${WORKERS:-16}"
EPOCHS_SCREEN="${EPOCHS_SCREEN:-60}"
EPOCHS_FINAL="${EPOCHS_FINAL:-100}"
WEIGHTS="${WEIGHTS:-$ROOT/models/pretrained/yolo12s.pt}"
STAGE="${STAGE:-all}"
# Clear-frame filtering is only needed for the always-on (B3) ablation. The G1
# main numbers use unfiltered clear + filtered fog, so skip it during Phase 1.
SKIP_CLEAR="${SKIP_CLEAR:-1}"

mkdir -p data/filtered results runs logs

filter_set () {
  local f="$1" src="$2" dst="$3"
  if [ -f "$dst/.done" ]; then echo "  [skip] $dst"; return; fi
  rm -rf "$dst"
  $PY -m src.filters --src "$src" --dst "$dst" --filter "$f" --workers "$WORKERS"
  touch "$dst/.done"
}

apply_unet () {
  local src="$1" dst="$2"
  if [ -f "$dst/.done" ]; then echo "  [skip] $dst"; return; fi
  rm -rf "$dst"
  $PY -m src.apply_unet --src "$src" --dst "$dst" --batch 8
  touch "$dst/.done"
}

prep_filters () {
  echo "=== [prep] classical filters: $FILTERS ==="
  for f in $FILTERS; do
    [ "$f" = "unet" ] && continue
    echo "--- filter $f"
    if [ "$SKIP_CLEAR" = "0" ]; then
      filter_set "$f" "$FLIR/yolo/train_3a/images/train_clear" "data/filtered/$f/clear/train"
      filter_set "$f" "$FLIR/yolo/train_3a/images/val" "data/filtered/$f/clear/val"
      filter_set "$f" "$FLIR/FLIR_ADAS_v2/test/clear" "data/filtered/$f/clear/test"
    fi
    if [ "$COND" = "cyclegan" ]; then
      FOG_TRAIN="$FLIR/yolo/train_3a/images/train_adv"
      FOG_TEST="$FLIR/synthetic/cyclegan/fog/images/test"
    else
      FOG_TRAIN="$FLIR/synthetic/cyclediff_75fog/fog/images/train"
      FOG_TEST="$FLIR/synthetic/cyclediff_75fog/fog/images/test"
    fi
    filter_set "$f" "$FOG_TRAIN" "data/filtered/$f/$COND/train"
    filter_set "$f" "$FOG_TEST" "data/filtered/$f/$COND/test"
  done

  echo "=== [prep] U-Net dehazer ==="
  if [ ! -f results/unet_dehazer.pt ]; then
    $PY -m src.train_unet --condition "$COND" --epochs 30 --batch 16 --crop 256 --workers "$WORKERS" \
      2>&1 | tee logs/train_unet.log
  else
    echo "  [skip] results/unet_dehazer.pt exists"
  fi

  echo "=== [prep] applying U-Net ==="
  if [ "$SKIP_CLEAR" = "0" ]; then
    apply_unet "$FLIR/yolo/train_3a/images/train_clear" "data/filtered/unet/clear/train"
    apply_unet "$FLIR/yolo/train_3a/images/val" "data/filtered/unet/clear/val"
    apply_unet "$FLIR/FLIR_ADAS_v2/test/clear" "data/filtered/unet/clear/test"
  fi
  if [ "$COND" = "cyclegan" ]; then
    FOG_TRAIN="$FLIR/yolo/train_3a/images/train_adv"
    FOG_TEST="$FLIR/synthetic/cyclegan/fog/images/test"
  else
    FOG_TRAIN="$FLIR/synthetic/cyclediff_75fog/fog/images/train"
    FOG_TEST="$FLIR/synthetic/cyclediff_75fog/fog/images/test"
  fi
  apply_unet "$FOG_TRAIN" "data/filtered/unet/$COND/train"
  apply_unet "$FOG_TEST" "data/filtered/unet/$COND/test"

  echo "=== [prep] datasets + routing ==="
  $PY -m src.build_screen --condition "$COND" --filters $FILTERS
  for f in $FILTERS; do
    $PY -m src.route --condition "$COND" --filter "$f" --name "gated_learned_$f"
    $PY -m src.route --condition "$COND" --filter "$f" --name "gated_oracle_$f" --oracle
  done
}

train_arm () {
  local dataset="$1" name="$2" epochs="$3"
  if [ -f "runs/$name/DONE" ]; then echo "  [skip] $name"; return; fi
  echo "=== train $name ($epochs epochs) ==="
  $PY -m src.train_detector \
    --dataset "$dataset" --name "$name" --epochs "$epochs" \
    --batch 16 --imgsz 640 --workers "$WORKERS" --weights "$WEIGHTS" \
    2>&1 | tee "logs/train_${name}.log" || { echo "TRAIN FAILED: $name"; return 0; }
  touch "runs/$name/DONE"
}

eval_arm () {
  local name="$1" weights="$2" data="$3"
  echo "=== eval $name ==="
  $PY -m src.eval_detector --weights "$weights" --data "configs/$data" \
    --name "$name" --force 2>&1 | tee -a "logs/eval_phase1.log" || { echo "EVAL FAILED: $name"; return 0; }
}

screen () {
  echo "=== [screen] training B1 + G1 filters at $EPOCHS_SCREEN epochs ==="
  train_arm b1 b1_yolo12s_60 "$EPOCHS_SCREEN"
  for f in $FILTERS; do
    train_arm "g1_$f" "g1_${f}_yolo12s_60" "$EPOCHS_SCREEN"
  done

  echo "=== [screen] evaluating ==="
  eval_arm b1_yolo12s_60_clear_vtt "runs/b1_yolo12s_60/weights/best.pt" "b1_eval_clear.yaml"
  eval_arm b1_yolo12s_60_fog_vtt   "runs/b1_yolo12s_60/weights/best.pt" "b1_eval_fog.yaml"
  for f in $FILTERS; do
    w="runs/g1_${f}_yolo12s_60/weights/best.pt"
    eval_arm "g1_${f}_clear_vtt"      "$w" "g1_${f}_eval_clear.yaml"
    eval_arm "g1_${f}_fog_vtt"        "$w" "g1_${f}_eval_fog.yaml"
    eval_arm "g1_${f}_gated_clear"    "$w" "gated_learned_${f}_eval_clear.yaml"
    eval_arm "g1_${f}_gated_fog"      "$w" "gated_learned_${f}_eval_fog.yaml"
    eval_arm "g1_${f}_oracle_clear"   "$w" "gated_oracle_${f}_eval_clear.yaml"
    eval_arm "g1_${f}_oracle_fog"     "$w" "gated_oracle_${f}_eval_fog.yaml"
  done
  $PY -m src.summarize_screen
}

final () {
  WINNER="$($PY -c "import json; print(json.load(open('results/screen_summary.json'))['best_filter'])")"
  echo "=== [final] winner = $WINNER ==="
  train_arm b1 b1_yolo12s_100 "$EPOCHS_FINAL"
  train_arm "g1_$WINNER" "g1_${WINNER}_yolo12s_100" "$EPOCHS_FINAL"

  eval_arm b1_yolo12s_100_clear_vtt "runs/b1_yolo12s_100/weights/best.pt" "b1_eval_clear.yaml"
  eval_arm b1_yolo12s_100_fog_vtt   "runs/b1_yolo12s_100/weights/best.pt" "b1_eval_fog.yaml"
  w="runs/g1_${WINNER}_yolo12s_100/weights/best.pt"
  eval_arm "g1_${WINNER}_100_clear_vtt"    "$w" "g1_${WINNER}_eval_clear.yaml"
  eval_arm "g1_${WINNER}_100_fog_vtt"      "$w" "g1_${WINNER}_eval_fog.yaml"
  eval_arm "g1_${WINNER}_100_gated_clear"  "$w" "gated_learned_${WINNER}_eval_clear.yaml"
  eval_arm "g1_${WINNER}_100_gated_fog"    "$w" "gated_learned_${WINNER}_eval_fog.yaml"
  eval_arm "g1_${WINNER}_100_oracle_clear" "$w" "gated_oracle_${WINNER}_eval_clear.yaml"
  eval_arm "g1_${WINNER}_100_oracle_fog"   "$w" "gated_oracle_${WINNER}_eval_fog.yaml"
  $PY -m src.summarize_results
}

case "$STAGE" in
  prep)   prep_filters ;;
  screen) screen ;;
  final)  final ;;
  all)
    ( prep_filters ) || echo "PREP FAILED"
    ( screen ) || echo "SCREEN FAILED"
    ( final ) || echo "FINAL FAILED"
    ;;
  *) echo "unknown STAGE: $STAGE" >&2; exit 2 ;;
esac

echo "=== Phase 1 stage '$STAGE' done $(date) ==="
