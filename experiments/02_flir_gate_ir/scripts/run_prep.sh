#!/usr/bin/env bash
# Prepare Stage A (gate) + Stage B (filter) inputs for the FLIR experiment.
# Relative paths only: the repo root is discovered from git, everything else
# is under experiments/02_flir_gate_ir/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."                                  # experiments/02_flir_gate_ir/
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
FLIR="$ROOT/data/flir_adas_v2"
FILTER="${FILTER:-gamma_clahe}"
COND="${COND:-cyclegan}"
WORKERS="${WORKERS:-8}"

mkdir -p data/filtered results logs

echo "=== [1/3] filtering clear images ($FILTER) ==="
$PY -m src.filters --src "$FLIR/yolo/train_3a/images/train_clear" \
  --dst "data/filtered/$FILTER/clear/train" --filter "$FILTER" --workers "$WORKERS"
$PY -m src.filters --src "$FLIR/yolo/train_3a/images/val" \
  --dst "data/filtered/$FILTER/clear/val" --filter "$FILTER" --workers "$WORKERS"
$PY -m src.filters --src "$FLIR/FLIR_ADAS_v2/test/clear" \
  --dst "data/filtered/$FILTER/clear/test" --filter "$FILTER" --workers "$WORKERS"

echo "=== [2/3] filtering fog images ($COND) ==="
if [ "$COND" = "cyclegan" ]; then
  FOG_TRAIN="$FLIR/yolo/train_3a/images/train_adv"
  FOG_TEST="$FLIR/synthetic/cyclegan/fog/images/test"
else
  FOG_TRAIN="$FLIR/synthetic/cyclediff_75fog/fog/images/train"
  FOG_TEST="$FLIR/synthetic/cyclediff_75fog/fog/images/test"
fi
$PY -m src.filters --src "$FOG_TRAIN" \
  --dst "data/filtered/$FILTER/$COND/train" --filter "$FILTER" --workers "$WORKERS"
$PY -m src.filters --src "$FOG_TEST" \
  --dst "data/filtered/$FILTER/$COND/test" --filter "$FILTER" --workers "$WORKERS"

echo "=== [3/3] building datasets + training the gate ==="
$PY -m src.build_datasets --condition "$COND" --filter "$FILTER"
$PY -m src.gate --condition "$COND" --workers "$WORKERS"
$PY -m src.route --condition "$COND" --filter "$FILTER" --name gated_learned
$PY -m src.route --condition "$COND" --filter "$FILTER" --name gated_oracle --oracle

echo "=== prep done ==="
