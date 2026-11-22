#!/usr/bin/env bash
# Evaluate teachers and all student arms on the locked test split.
set -uo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$EXP/src"
LOGS="$EXP/logs"; mkdir -p "$LOGS"
cd "$SRC" || exit 1
export PYTHONPATH="$SRC"
PY=python

eval_one () {
  local name="$1"; local data="$2"; local extra="${3:-}"
  local w="$EXP/runs/$name/weights/best.pt"
  [ -f "$w" ] || w="$EXP/runs/$name/weights/last.pt"
  [ -f "$w" ] || { echo "no weights for $name"; return 0; }
  if [ -f "$EXP/results/${name}_test.json" ]; then echo "[$(date)] eval $name exists"; return 0; fi
  echo "[$(date)] eval $name"
  $PY -m graphkd.training.evaluate --weights "$w" --name "$name" \
    --data "$data" --split test $extra 2>&1 | tee -a "$LOGS/eval_$name.log"
}

eval_one ir_teacher "$EXP/configs/ir.yaml"
eval_one rgb_teacher "$EXP/configs/rgb.yaml"
# fused teacher: no val, always score the final checkpoint
if [ -f "$EXP/runs/fused_teacher/weights/last.pt" ] && [ ! -f "$EXP/results/fused_teacher_test.json" ]; then
  $PY -m graphkd.training.evaluate --weights "$EXP/runs/fused_teacher/weights/last.pt"     --name fused_teacher --data "$EXP/configs/paired.yaml" --split test --paired     2>&1 | tee -a "$LOGS/eval_fused_teacher.log"
fi
for arm in a0_gt a2_ir_control a1_pred_ir a3_node_edge_ir a5_fgw_ir a7_reliability_ir \
           b_consensus a4_sinkhorn_ir d_gt_anchor_ir a2_node a1_pred a4_sinkhorn \
           e_condense f_decomposed c_three_teacher a6_partial; do
  eval_one "$arm" "$EXP/configs/ir.yaml"
done
$PY -m graphkd.training.summarize > "$EXP/results/SUMMARY.stdout" 2>&1 || true
echo "[$(date)] eval phase finished"
