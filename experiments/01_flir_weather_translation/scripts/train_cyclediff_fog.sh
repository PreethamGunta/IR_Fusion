#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# train_cyclediff_fog.sh - CycleDiff training for fog with done-check/resume logic

set -euo pipefail

# Configuration
CONDITION="fog"
LOG_DIR="experiments/01_flir_weather_translation/method4_cyclediff/logs/${CONDITION}"
RESULTS_DIR="experiments/01_flir_weather_translation/method4_cyclediff/results"

# Check if CycleDiff training is already complete (final model exists)
FINAL_CYCLE_CKPT="${RESULTS_DIR}/train_cycle_C_disc_G_blocks_12_clear2${CONDITION}/model-12.pt"

if [[ -f "${FINAL_CYCLE_CKPT}" ]]; then
    echo "[$(date)] ${CONDITION}: CycleDiff training already complete (found ${FINAL_CYCLE_CKPT}). Skipping."
    exit 0
fi

cd $ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir

echo "[$(date)] ${CONDITION}: Starting CycleDiff training"

# Stage 1: VAE training (clear + fog)
for DOMAIN in clear fog; do
    VAE_CKPT="results/ae_kl_256x256_d4_${DOMAIN}/model-10.pt"
    if [[ -f "${VAE_CKPT}" ]]; then
        echo "[$(date)] ${CONDITION}: VAE ${DOMAIN} already trained (found ${VAE_CKPT}). Skipping."
    else
        echo "[$(date)] ${CONDITION}: Training VAE ${DOMAIN}"
        accelerate launch experiments/01_flir_weather_translation/method4_cyclediff/train_vae.py \
            --cfg configs/flir_clear2${CONDITION}/${DOMAIN}_ae_kl_256x256_d4.yaml
    fi
done

# Stage 2: LDM training (clear + fog)
for DOMAIN in clear fog; do
    LDM_CKPT="results/ddm_const_uncond_unet_ldm_${DOMAIN}/model-24.pt"
    if [[ -f "${LDM_CKPT}" ]]; then
        echo "[$(date)] ${CONDITION}: LDM ${DOMAIN} already trained (found ${LDM_CKPT}). Skipping."
    else
        echo "[$(date)] ${CONDITION}: Training LDM ${DOMAIN}"
        accelerate launch experiments/01_flir_weather_translation/method4_cyclediff/train_uncond_ldm.py \
            --cfg configs/flir_clear2${CONDITION}/${DOMAIN}_ddm_const4_ldm_unet6_114_ode_2.yaml
    fi
done

# Stage 3: Cycle translator
CYCLE_CKPT="results/train_cycle_C_disc_G_blocks_12_clear2${CONDITION}/model-12.pt"
if [[ -f "${CYCLE_CKPT}" ]]; then
    echo "[$(date)] ${CONDITION}: Cycle translator already trained (found ${CYCLE_CKPT}). Skipping."
else
    echo "[$(date)] ${CONDITION}: Training Cycle translator"
    accelerate launch experiments/01_flir_weather_translation/method4_cyclediff/train_uncond_ldm_cycle.py \
        --cfg configs/flir_clear2${CONDITION}/translation_C_disc_timestep_ode_2.yaml
fi

# Final check
if [[ -f "${FINAL_CYCLE_CKPT}" ]]; then
    echo "[$(date)] ${CONDITION}: CycleDiff training completed successfully"
    exit 0
else
    echo "[$(date)] ${CONDITION}: Training may not have completed"
    exit 1
fi