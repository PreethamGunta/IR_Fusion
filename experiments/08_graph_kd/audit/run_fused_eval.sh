#!/bin/bash
# Audit: FLIR fused teacher checkpoint comparison, CPU-only.
set -u
cd /home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd/src || exit 1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python
export PYTHONPATH=.
EXP=/home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd

$PY - <<'PY'
import torch
for tag in ('best','last'):
    p=f'/home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd/runs/fused_teacher/weights/{tag}.pt'
    c=torch.load(p, map_location='cpu', weights_only=False)
    print(tag, 'epoch', c.get('epoch'), 'best_fitness', c.get('best_fitness'),
          'date', c.get('date'), 'version', c.get('version'))
PY

for tag in best last; do
  for split in val test; do
    echo "=== fused $tag $split $(date -u +%H:%M:%S) ==="
    $PY -m graphkd.training.evaluate \
        --weights "$EXP/runs/fused_teacher/weights/$tag.pt" \
        --name "audit_fused_${tag}_${split}" \
        --data "$EXP/configs/paired.yaml" --split "$split" --paired \
        --paired-yaml "$EXP/configs/paired.yaml" --rgb-yaml "$EXP/configs/rgb.yaml" \
        --batch 4 2>&1 | grep -a -E '"AP"|"AP50"|"AP75"|n_images|n_annotations|Error|Traceback' | tail -12
  done
done
echo "=== done $(date -u +%H:%M:%S) ==="
