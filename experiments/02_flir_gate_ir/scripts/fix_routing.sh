#!/usr/bin/env bash
# Ensure every filter has a filtered CLEAR test set (needed when the learned gate
# misroutes a clear frame to the filter), then re-materialise the gated test sets.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
FLIR="$ROOT/data/flir_adas_v2"
COND="${COND:-cyclegan}"
FILTERS="${FILTERS:-gamma clahe dcp unet gamma_clahe}"
WORKERS="${WORKERS:-8}"

for f in $FILTERS; do
  dst="data/filtered/$f/clear/test"
  if [ ! -f "$dst/.done" ]; then
    echo "=== [fix] filtered clear test for $f ==="
    rm -rf "$dst"
    if [ "$f" = "unet" ]; then
      $PY -m src.apply_unet --src "$FLIR/FLIR_ADAS_v2/test/clear" --dst "$dst" --batch 8
    else
      $PY -m src.filters --src "$FLIR/FLIR_ADAS_v2/test/clear" --dst "$dst" --filter "$f" --workers "$WORKERS"
    fi
    touch "$dst/.done"
  fi
  echo "=== [fix] re-routing $f ==="
  $PY -m src.route --condition "$COND" --filter "$f" --name "gated_learned_$f"
  $PY -m src.route --condition "$COND" --filter "$f" --name "gated_oracle_$f" --oracle
done

echo "=== routing fixed $(date) ==="
