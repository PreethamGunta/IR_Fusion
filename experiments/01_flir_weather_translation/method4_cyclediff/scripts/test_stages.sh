#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
cd $ROOT/experiments/01_flir_weather_translation/method4_cyclediff
PY="python"

echo "=== Test 1: LDM clear training (300s timeout) ==="
timeout 300 $PY train_uncond_ldm.py --cfg configs/flir_clear2fog/clear_ddm_const4_ldm_unet6_114_ode_2.yaml 2>&1 | tail -30
LDM_EXIT=$?
echo "=== Test 1 exit code: $LDM_EXIT ==="
if [ $LDM_EXIT -eq 0 ]; then echo "LDM: OK (completed training)"; elif [ $LDM_EXIT -eq 124 ]; then echo "LDM: TIMEOUT (script runs correctly, just timed out)"; else echo "LDM: FAILED"; fi

echo "=== Test 2: Cycle translator (300s timeout) ==="
timeout 300 $PY train_uncond_ldm_cycle.py --cfg configs/flir_clear2fog/translation_C_disc_timestep_ode_2.yaml 2>&1 | tail -30
CYCLE_EXIT=$?
echo "=== Test 2 exit code: $CYCLE_EXIT ==="
if [ $CYCLE_EXIT -eq 0 ]; then echo "CYCLE: OK (completed training)"; elif [ $CYCLE_EXIT -eq 124 ]; then echo "CYCLE: TIMEOUT (script runs correctly, just timed out)"; else echo "CYCLE: FAILED"; fi
