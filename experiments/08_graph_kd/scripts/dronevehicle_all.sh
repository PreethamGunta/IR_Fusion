#!/usr/bin/env bash
# DroneVehicle: full cross-modal KD matrix. Run after the FLIR core phase.
# Teachers are trained to convergence (100 epochs max, patience 30); the fused
# teacher receives paired flip/RGB jitter augmentation.  Everything is
# resumable via DONE markers and last.pt, so a second allocation can continue.
set -uo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$EXP/src"; LOGS="$EXP/logs"; mkdir -p "$LOGS"
cd "$SRC" || exit 1
export PYTHONPATH="$SRC"
PY=python
PRET="$EXP/../../models/pretrained"
DV="$EXP/../../data/dronevehicle"
IR_YAML="$EXP/configs/dronevehicle_ir.yaml"
RGB_YAML="$EXP/configs/dronevehicle_rgb.yaml"
PAIR_YAML="$EXP/configs/dronevehicle_paired.yaml"
CACHE="${DV_CACHE:-/tmp/.cache_8f2c91/dronevehicle_cache}"

run_one () {
  local name="$1"; shift
  if [ -f "$EXP/runs/$name/DONE" ]; then echo "[$(date)] skip $name"; return 0; fi
  if [ -f "$EXP/runs/$name/weights/last.pt" ]; then set -- "$@" --resume; fi
  echo "[$(date)] start $name"
  if "$@" 2>&1 | tee -a "$LOGS/$name.log"; then touch "$EXP/runs/$name/DONE"; else
    echo "[$(date)] FAILED $name" | tee -a "$LOGS/$name.log"; fi
}

if [ ! -f "$DV/manifest.json" ]; then
  echo "[$(date)] building DroneVehicle"
  $PY -m graphkd.data.build_dronevehicle || { echo "dronevehicle build failed"; exit 1; }
fi
bash "$EXP/scripts/make_dv_configs.sh"

# ---------------- teachers ----------------
run_one dv_ir_teacher $PY -m graphkd.training.train --name dv_ir_teacher \
  --model "$PRET/yolo12s.pt" --data "$IR_YAML" --epochs 100 --patience 30 \
  --batch 16 --workers 8 --exist-ok
run_one dv_rgb_teacher $PY -m graphkd.training.train --name dv_rgb_teacher \
  --model "$PRET/yolo12s.pt" --data "$RGB_YAML" --epochs 100 --patience 30 \
  --batch 16 --workers 8 --exist-ok
run_one dv_fused_teacher $PY -m graphkd.training.train --name dv_fused_teacher \
  --model yolo12s.yaml --pretrained "$PRET/yolo12s.pt" --fused \
  --data "$PAIR_YAML" --paired-yaml "$PAIR_YAML" --rgb-yaml "$RGB_YAML" --paired-rot90 \
  --epochs 100 --patience 30 --batch 16 --workers 8 --exist-ok

# ---------------- teacher caches ----------------
cache_one () {
  local name="$1"; shift
  [ -f "$CACHE/$name/meta.json" ] && { echo "cache $name exists"; return 0; }
  echo "[$(date)] cache $name"
  "$@" 2>&1 | tee -a "$LOGS/cache_dv_$name.log"
}
W="$EXP/runs/dv_ir_teacher/weights/best.pt"; [ -f "$W" ] || W="$EXP/runs/dv_ir_teacher/weights/last.pt"
cache_one ir $PY -m graphkd.graph.cache_teacher --weights "$W" --data "$IR_YAML" \
  --out "$CACHE/ir" --name dv_ir --k 40
W="$EXP/runs/dv_rgb_teacher/weights/best.pt"; [ -f "$W" ] || W="$EXP/runs/dv_rgb_teacher/weights/last.pt"
cache_one rgb $PY -m graphkd.graph.cache_teacher --weights "$W" --data "$RGB_YAML" \
  --out "$CACHE/rgb" --name dv_rgb --k 40
W="$EXP/runs/dv_fused_teacher/weights/last.pt"
cache_one fused $PY -m graphkd.graph.cache_teacher --weights "$W" --paired \
  --paired-yaml "$PAIR_YAML" --rgb-yaml "$RGB_YAML" --out "$CACHE/fused" --name dv_fused --k 40

# ---------------- students ----------------
ARMS="${ARMS:-a0_gt a2_ir_control a1_pred a2_node a3_node_edge a5_fgw a7_reliability \
b_consensus a4_sinkhorn d_gt_anchor e_condense f_decomposed c_three_teacher a6_partial}"
for arm in $ARMS; do
  if [ "$arm" = "a0_gt" ]; then
    run_one dv_$arm $PY -m graphkd.training.train --name dv_$arm \
      --model "$PRET/yolo12n.pt" --data "$IR_YAML" --no-aug \
      --epochs 100 --patience 30 --batch 16 --workers 8 --exist-ok
  else
    run_one dv_$arm $PY -m graphkd.training.train --name dv_$arm \
      --model "$PRET/yolo12n.pt" --data "$IR_YAML" \
      --kd-config "$EXP/configs/kd_dronevehicle/$arm.json" \
      --epochs 100 --patience 30 --batch 16 --workers 8 --exist-ok
  fi
done

# ---------------- evaluation ----------------
eval_one () {
  local name="$1"; local data="$2"; local extra="${3:-}"
  local w="$EXP/runs/$name/weights/best.pt"; [ -f "$w" ] || w="$EXP/runs/$name/weights/last.pt"
  [ -f "$w" ] || return 0
  [ -f "$EXP/results/${name}_test.json" ] && return 0
  $PY -m graphkd.training.evaluate --weights "$w" --name "$name" --data "$data" \
    --split test $extra 2>&1 | tee -a "$LOGS/eval_$name.log"
}
eval_one dv_ir_teacher "$IR_YAML"
eval_one dv_rgb_teacher "$RGB_YAML"
if [ -f "$EXP/runs/dv_fused_teacher/weights/last.pt" ]; then
  $PY -m graphkd.training.evaluate --weights "$EXP/runs/dv_fused_teacher/weights/last.pt" \
    --name dv_fused_teacher --data "$PAIR_YAML" --split test --paired \
    --paired-yaml "$PAIR_YAML" --rgb-yaml "$RGB_YAML" 2>&1 | tee -a "$LOGS/eval_dv_fused_teacher.log"
fi
for arm in $ARMS; do eval_one dv_$arm "$IR_YAML"; done
echo "[$(date)] DroneVehicle phase finished"
