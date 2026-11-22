#!/usr/bin/env bash
# Train the GT-only student and all graph-KD arms, in priority order.
# Every arm is resumable (DONE marker) and failures do not stop the queue.
set -uo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$EXP/src"
LOGS="$EXP/logs"; mkdir -p "$LOGS"
cd "$SRC" || exit 1
export PYTHONPATH="$SRC"
PY=python
PRET="$EXP/../../models/pretrained"

for t in ir rgb fused; do
  if [ ! -f "$EXP/data/cache/$t/meta.json" ]; then
    echo "missing teacher cache $t; run scripts/cache_teachers.sh first"; exit 1
  fi
done

ARMS="${ARMS:-a0_gt a2_ir_control a1_pred_ir a3_node_edge_ir a5_fgw_ir a7_reliability_ir b_consensus a4_sinkhorn_ir d_gt_anchor_ir a2_node a1_pred a4_sinkhorn e_condense f_decomposed c_three_teacher a6_partial}"

run_one () {
  local name="$1"; shift
  if [ -f "$EXP/runs/$name/DONE" ]; then echo "[$(date)] skip $name"; return 0; fi
  if [ -f "$EXP/runs/$name/weights/last.pt" ]; then set -- "$@" --resume; fi
  echo "[$(date)] start $name"
  if "$@" 2>&1 | tee -a "$LOGS/$name.log"; then
    touch "$EXP/runs/$name/DONE"
    echo "[$(date)] done $name"
  else
    echo "[$(date)] FAILED $name" | tee -a "$LOGS/$name.log"
  fi
}

for arm in $ARMS; do
  if [ "$arm" = "a0_gt" ]; then
    run_one "$arm" $PY -m graphkd.training.train --name "$arm" \
      --model "$PRET/yolo12n.pt" --data "$EXP/configs/ir.yaml" --no-aug \
      --epochs 100 --patience 30 --batch 16 --workers 8 --exist-ok
  else
    run_one "$arm" $PY -m graphkd.training.train --name "$arm" \
      --model "$PRET/yolo12n.pt" --data "$EXP/configs/ir.yaml" \
      --kd-config "$EXP/configs/kd/$arm.json" \
      --epochs 100 --patience 30 --batch 16 --workers 8 --exist-ok
  fi
done
echo "[$(date)] student phase finished"
