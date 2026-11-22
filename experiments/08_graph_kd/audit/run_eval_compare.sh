#!/bin/bash
# Audit: compare custom COCO evaluator vs stock Ultralytics validator on the
# same weights and split. CPU-only; does not touch runs/ or existing results.
set -u
cd /home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd/src || exit 1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python
export PYTHONPATH=.
AUD=/home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd/audit
W=/home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd/runs/a0_gt/weights/best.pt

for split in val test; do
  echo "=== custom evaluate $split $(date -u +%H:%M:%S) ==="
  $PY -m graphkd.training.evaluate --weights "$W" --name "audit_a0_custom_$split" \
      --data ../configs/ir.yaml --split "$split" --batch 4 2>&1 | tail -40
  echo "=== stock validator $split $(date -u +%H:%M:%S) ==="
  $PY - "$split" "$AUD" <<'PY' 2>&1 | tail -20
import sys
from ultralytics import YOLO
split, aud = sys.argv[1], sys.argv[2]
m = YOLO('/home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd/runs/a0_gt/weights/best.pt')
r = m.val(data='/home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd/configs/ir.yaml',
          split=split, device='cpu', batch=4, workers=4, imgsz=640, plots=False,
          project=aud, name=f'stock_a0_{split}', exist_ok=True)
print(f"STOCK {split} map={r.box.map:.6f} map50={r.box.map50:.6f} map75={r.box.map75:.6f}")
PY
done
echo "=== done $(date -u +%H:%M:%S) ==="
