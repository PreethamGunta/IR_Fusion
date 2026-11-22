#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
set -euo pipefail

# ============================================================
# CycleDiff: resume training (100k->105k) + translate train & test
# ============================================================
REPO="$ROOT"
CDIR="$REPO/experiments/01_flir_weather_translation/method4_cyclediff"
LOGDIR="$CDIR/logs"
mkdir -p "$LOGDIR"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir
cd "$CDIR"

TRAIN_CFG="configs/flir_clear2fog/translation_C_disc_timestep_ode_2_retrain.yaml"
TEST_CFG="configs/flir_clear2fog/translation_C_disc_timestep_ode_2_retrain.yaml"
TRAIN_IMG_CFG="configs/flir_clear2fog/translation_C_disc_timestep_ode_2_train.yaml"

echo "=== [1/3] Training CycleDiff 100k->105k (started $(date)) ==="
accelerate launch --num_processes=1 train_uncond_ldm_cycle.py --cfg "$TRAIN_CFG" \
    2>&1 | tee "$LOGDIR/train_100k_105k.log"
TRAIN_RC=${PIPESTATUS[0]}
if [ $TRAIN_RC -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_RC at $(date)" >&2
    exit 1
fi
echo "=== Training completed at $(date) ==="

# Verify final_model.pt exists and is at 105000
python3 -c "
import torch
d = torch.load('results/train_cycle_C_disc_G_blocks_10_clear2fog_v2/final_model.pt', map_location='cpu')
assert d['step'] == 105000, f'Expected step 105000, got {d[\"step\"]}'
print(f'final_model.pt verified: step {d[\"step\"]}')
"

echo "=== [2/3] Translating test images (3749) (started $(date)) ==="
accelerate launch --num_processes=1 translation_uncond_ldm_cycle.py --cfg "$TEST_CFG" \
    2>&1 | tee "$LOGDIR/translate_test.log"
TEST_RC=${PIPESTATUS[0]}
if [ $TEST_RC -ne 0 ]; then
    echo "ERROR: Test translation failed with exit code $TEST_RC at $(date)" >&2
    exit 1
fi
N_TEST=$(ls sample_results/clear2fog_model_10_timestep100_v2/*.png 2>/dev/null | wc -l)
echo "Test images translated: $N_TEST"
if [ "$N_TEST" -lt 3700 ]; then
    echo "ERROR: Expected ~3749 test images, got $N_TEST" >&2
    exit 1
fi
echo "=== Test translation completed at $(date) ==="

echo "=== [3/3] Translating train images (started $(date)) ==="
accelerate launch --num_processes=1 translation_uncond_ldm_cycle.py --cfg "$TRAIN_IMG_CFG" \
    2>&1 | tee "$LOGDIR/translate_train.log"
TRAIN_IMG_RC=${PIPESTATUS[0]}
if [ $TRAIN_IMG_RC -ne 0 ]; then
    echo "ERROR: Train translation failed with exit code $TRAIN_IMG_RC at $(date)" >&2
    exit 1
fi
N_TRAIN=$(ls $ROOT/data/flir_adas_v2/synthetic/cyclediff/fog/train/*.png 2>/dev/null | wc -l)
echo "Train images translated: $N_TRAIN"
if [ "$N_TRAIN" -lt 10000 ]; then
    echo "ERROR: Expected ~11886 train images, got $N_TRAIN" >&2
    exit 1
fi
echo "=== All done at $(date) ==="
