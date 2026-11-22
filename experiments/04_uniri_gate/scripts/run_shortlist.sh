#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# UNIRI-TID shortlist: weather-aug YOLOv12s, RT-DETR-l rerun, then the
# preprocessing variants if that script exists.
set -uo pipefail

cd $ROOT
PY="${PY:-python}"
mkdir -p uniri_gate_experiments/logs uniri_gate_experiments/results

run_one () {
  local name="$1" model="$2" extra="$3"
  if [ -f "experiments/05_detector_bench/results/${name}.json" ]; then
    echo "=== skip $name (result exists) ==="
    return 0
  fi
  echo "=== $(date) training $name ($model) ==="
  $PY bench/train_detector.py --model "$model" --name "$name" --epochs 100 --cache ram $extra \
    2>&1 | tee "uniri_gate_experiments/logs/${name}.log" || echo "TRAIN FAILED: $name"
  for f in "experiments/05_detector_bench/results/${name}.json" \
           "experiments/05_detector_bench/results/eval_${name}_test.json" \
           "experiments/05_detector_bench/results/eval_${name}_val.json"; do
    [ -f "$f" ] && cp "$f" uniri_gate_experiments/results/
  done
}

# 1. strongest cheap no-restoration control
run_one v12s_weather weights/yolo12s.pt "--weather-aug"

# 2. preprocessing variants (always-on / soft router / hard gate / oracle)
if [ -f uniri_gate_experiments/scripts/run_variants.sh ]; then
  bash uniri_gate_experiments/scripts/run_variants.sh || echo "VARIANTS FAILED"
else
  echo "=== no variant script yet; skipping ==="
fi

# 3. valid DETR-family baseline (previous run was invalid: 29 epochs)
run_one rtdetrl_rerun weights/rtdetr-l.pt ""

echo "=== shortlist done $(date) ==="
