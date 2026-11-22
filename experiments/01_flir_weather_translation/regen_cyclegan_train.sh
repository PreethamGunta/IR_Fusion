#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir
cd $ROOT/experiments/01_flir_weather_translation/method3_cyclegan

for cond in fog rain snow; do
    echo "=== Regenerating cyclegan $cond train ==="
    python translate.py --config "$cond" --split train --out "$ROOT/data/flir_adas_v2/synthetic/cyclegan/$cond/train" 2>&1 | tail -5
    echo "Done: $cond train $(ls $ROOT/data/flir_adas_v2/synthetic/cyclegan/$cond/train/ | wc -l) images"
done
echo "All cyclegan train translations complete."
