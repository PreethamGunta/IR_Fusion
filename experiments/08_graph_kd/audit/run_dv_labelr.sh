#!/bin/bash
# Audit-only: labelr val numbers and stock-validator calibration for DV.
set -u
cd /home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd || exit 1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src
PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python
AUD=$PWD/audit
$PY audit/make_dv_labelr_val.py
for n in dv_ir_teacher dv_a0_gt; do
  echo "=== $n labelr-val $(date -u +%H:%M:%S)"
  $PY -m graphkd.training.evaluate --weights runs/$n/weights/best.pt \
      --name audit_${n}_labelrval --data audit/dv_ir_labelr_val.yaml --split val --batch 4 2>&1 \
      | grep -a -E '"AP"|"AP50"|"AP75"|n_images|n_annotations|Error|Traceback' | tail -8
done
echo "=== stock dv_ir_teacher val $(date -u +%H:%M:%S) ==="
$PY - <<'PY' 2>&1 | tail -10
from ultralytics import YOLO
m = YOLO('/home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd/runs/dv_ir_teacher/weights/best.pt')
r = m.val(data='/home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd/configs/dronevehicle_ir.yaml',
          split='val', device='cpu', batch=4, workers=4, imgsz=640, plots=False,
          project='/home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd/audit',
          name='stock_dv_ir_teacher_val', exist_ok=True)
print(f"STOCK dv_ir_teacher val map={r.box.map:.6f} map50={r.box.map50:.6f} map75={r.box.map75:.6f}")
PY
echo "=== done $(date -u +%H:%M:%S) ==="
