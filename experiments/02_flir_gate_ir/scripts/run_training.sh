#!/usr/bin/env bash
# Train the detector arms sequentially inside the live SLURM allocation.
# Run with:  srun --jobid=9639 --overlap bash experiments/02_flir_gate_ir/scripts/run_training.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
PY="${PY:-python}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
ARMS="${ARMS:-b0 g1 b3 b2}"

mkdir -p runs logs

for arm in $ARMS; do
  echo "=== training $arm ($(date)) ==="
  $PY -m src.train_detector \
    --dataset "$arm" \
    --name "${arm}_yolov8s" \
    --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ" --workers "$WORKERS" \
    2>&1 | tee "logs/train_${arm}.log"
done

echo "=== training done ($(date)) ==="
