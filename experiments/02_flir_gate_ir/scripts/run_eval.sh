#!/usr/bin/env bash
# Evaluate every arm on the matched VTT clear/fog test sets.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
COND="${COND:-cyclegan}"
FILTER="${FILTER:-gamma_clahe}"

mkdir -p results logs

eval_one () {
  local name="$1" weights="$2" data="$3"
  echo "=== eval $name ==="
  $PY -m src.eval_detector --weights "$weights" --data "configs/$data" --name "$name" --force
}

# --- baselines ---------------------------------------------------------------
# Existing mixed-trained 3A baseline, re-scored on the matched VTT clear set.
eval_one "b1_cyclegan_clear_vtt" \
  "$ROOT/experiments/01_flir_weather_translation/runs/3a_cyclegan_fog-3/weights/best.pt" \
  "b1_eval_clear.yaml"
eval_one "b1_cyclegan_fog_vtt" \
  "$ROOT/experiments/01_flir_weather_translation/runs/3a_cyclegan_fog-3/weights/best.pt" \
  "b1_eval_fog.yaml"

# --- new arms ----------------------------------------------------------------
for pair in \
  "b0_clear_vtt b0_yolov8s b0_eval_clear.yaml" \
  "b0_fog_vtt b0_yolov8s b0_eval_fog.yaml" \
  "b2_fog_vtt b2_yolov8s b2_eval_fog.yaml" \
  "b3_clear_vtt b3_yolov8s b3_eval_clear.yaml" \
  "b3_fog_vtt b3_yolov8s b3_eval_fog.yaml" \
  "g1_clear_vtt g1_yolov8s g1_eval_clear.yaml" \
  "g1_fog_filtered g1_yolov8s g1_eval_fog.yaml" \
  "g1_gated_clear g1_yolov8s gated_learned_eval_clear.yaml" \
  "g1_gated_fog g1_yolov8s gated_learned_eval_fog.yaml" \
  "g1_oracle_clear g1_yolov8s gated_oracle_eval_clear.yaml" \
  "g1_oracle_fog g1_yolov8s gated_oracle_eval_fog.yaml" \
; do
  set -- $pair
  name="$1"; arm="$2"; cfg="$3"
  w="runs/$arm/weights/best.pt"
  if [ -f "$w" ]; then eval_one "$name" "$w" "$cfg"; else echo "SKIP $name (no weights: $w)"; fi
done

$PY -m src.summarize
