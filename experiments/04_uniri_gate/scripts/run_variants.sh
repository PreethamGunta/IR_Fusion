#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# Build and train the UNIRI preprocessing variants one at a time (bounded disk).
set -uo pipefail

cd $ROOT
PY="${PY:-python}"
VARIANTS="${VARIANTS:-always soft hard oracle}"
mkdir -p uniri_gate_experiments/logs uniri_gate_experiments/results

first=1
for v in $VARIANTS; do
  name="v12s_${v}"
  if [ -f "experiments/05_detector_bench/results/${name}.json" ]; then
    echo "=== skip ${name} (result exists) ==="
    continue
  fi
  echo "=== $(date) building variant ${v} ==="
  if [ "$first" = "1" ]; then
    $PY -m uniri_gate_experiments.src.build_variants --variants "$v" --workers 8 \
      2>&1 | tee "uniri_gate_experiments/logs/build_${v}.log" || { echo "BUILD FAILED: $v"; continue; }
    first=0
  else
    $PY -m uniri_gate_experiments.src.build_variants --variants "$v" --workers 8 --skip-gate \
      2>&1 | tee "uniri_gate_experiments/logs/build_${v}.log" || { echo "BUILD FAILED: $v"; continue; }
  fi

  echo "=== $(date) training ${name} ==="
  $PY bench/train_detector.py \
    --data "data/UNIRI-TID/yolo_${v}/uniri_${v}.yaml" \
    --name "$name" --epochs 100 --cache ram \
    2>&1 | tee "uniri_gate_experiments/logs/${name}.log" || echo "TRAIN FAILED: ${v}"

  for f in "experiments/05_detector_bench/results/${name}.json" \
           "experiments/05_detector_bench/results/eval_${name}_test.json" \
           "experiments/05_detector_bench/results/eval_${name}_val.json"; do
    [ -f "$f" ] && cp "$f" uniri_gate_experiments/results/
  done
  rm -rf "data/UNIRI-TID/yolo_${v}"
  echo "=== $(date) cleaned variant ${v} ==="
done
echo "=== variants done $(date) ==="
