#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# train_cyclegan_rain.sh - CycleGAN training for rain with done-check/resume logic

set -euo pipefail

# Configuration
CONDITION="rain"
EPOCHS=50
BATCH_SIZE=1
LR=2e-4
LOG_DIR="experiments/01_flir_weather_translation/method3_cyclegan/logs/${CONDITION}"
CKPT_DIR="${LOG_DIR}"

# Check if training is already complete (final epoch checkpoint exists)
FINAL_CKPT="${CKPT_DIR}/checkpoint_epoch_${EPOCHS}.pt"
BEST_CKPT="${CKPT_DIR}/best_model.pt"

if [[ -f "${FINAL_CKPT}" ]]; then
    echo "[$(date)] ${CONDITION}: Training already complete (found ${FINAL_CKPT}). Skipping."
    exit 0
fi

# Determine resume checkpoint
RESUME_CKPT=""
if [[ -f "${BEST_CKPT}" ]]; then
    RESUME_CKPT="--resume ${BEST_CKPT}"
    echo "[$(date)] ${CONDITION}: Resuming from best model ${BEST_CKPT}"
elif [[ -f "${CKPT_DIR}/checkpoint_epoch_latest.pt" ]]; then
    RESUME_CKPT="--resume ${CKPT_DIR}/checkpoint_epoch_latest.pt"
    echo "[$(date)] ${CONDITION}: Resuming from latest checkpoint"
else
    echo "[$(date)] ${CONDITION}: Starting fresh training"
fi

# Run training
cd $ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir

echo "[$(date)] ${CONDITION}: Starting training for ${EPOCHS} epochs"

python experiments/01_flir_weather_translation/method3_cyclegan/train_cyclegan.py \
    --config ${CONDITION} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    ${RESUME_CKPT}

# Check if training completed successfully
if [[ -f "${FINAL_CKPT}" ]]; then
    echo "[$(date)] ${CONDITION}: Training completed successfully"
    exit 0
else
    echo "[$(date)] ${CONDITION}: Training may not have completed (no final checkpoint found)"
    exit 1
fi