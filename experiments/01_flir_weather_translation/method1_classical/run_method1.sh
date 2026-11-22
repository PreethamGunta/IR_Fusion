#!/bin/bash
# Run Method 1 classical weather simulation for all conditions/severities.
# Recovered from the recover-session-12 worktree; paths updated to the release.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
cd "$ROOT"

echo "=== Method 1: Classical Weather Simulation ==="

for weather in fog rain snow; do
    for severity in light medium heavy; do
        echo "--- Generating $weather $severity ---"
        python experiments/01_flir_weather_translation/method1_classical/apply_weather_classical.py \
            --weather "$weather" --severity "$severity" --splits train val
    done
done

echo "=== Method 1 generation complete ==="
ls -la data/flir_adas_v2/FLIR_ADAS_v2/images_thermal_*_*/ | head -30
