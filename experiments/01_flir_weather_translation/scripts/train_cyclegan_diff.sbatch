#!/bin/bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=train_cyclegan_diff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --output=experiments/01_flir_weather_translation/logs/train_cyclegan_diff_%j.out
#SBATCH --error=experiments/01_flir_weather_translation/logs/train_cyclegan_diff_%j.err

# Master training script for FLIR ADAS v2
# Runs CycleGAN (fog, rain, snow) then CycleDiff (fog, rain, snow)
# Each script checks if training is complete and skips/resumes accordingly

set -euo pipefail

cd $ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir

mkdir -p experiments/01_flir_weather_translation/logs
mkdir -p experiments/01_flir_weather_translation/method3_cyclegan/logs/fog
mkdir -p experiments/01_flir_weather_translation/method3_cyclegan/logs/rain
mkdir -p experiments/01_flir_weather_translation/method3_cyclegan/logs/snow
mkdir -p experiments/01_flir_weather_translation/method4_cyclediff/logs/fog
mkdir -p experiments/01_flir_weather_translation/method4_cyclediff/logs/rain
mkdir -p experiments/01_flir_weather_translation/method4_cyclediff/logs/snow

echo "==========================================="
echo "FLIR ADAS v2 - CycleGAN + CycleDiff Training"
echo "Started at $(date)"
echo "==========================================="

# ============================================
# PHASE 1: CycleGAN (Method 3)
# ============================================
echo ""
echo "==========================================="
echo "PHASE 1: CycleGAN Training (Method 3)"
echo "==========================================="

# Train CycleGAN for fog
echo ""
echo ">>> Training CycleGAN for FOG..."
bash experiments/01_flir_weather_translation/scripts/train_cyclegan_fog.sh
FOG_EXIT=$?
if [[ $FOG_EXIT -ne 0 ]]; then
    echo "ERROR: CycleGAN fog training failed with exit code $FOG_EXIT"
    exit 1
fi
echo "CycleGAN fog completed successfully."

# Train CycleGAN for rain
echo ""
echo ">>> Training CycleGAN for RAIN..."
bash experiments/01_flir_weather_translation/scripts/train_cyclegan_rain.sh
RAIN_EXIT=$?
if [[ $RAIN_EXIT -ne 0 ]]; then
    echo "ERROR: CycleGAN rain training failed with exit code $RAIN_EXIT"
    exit 1
fi
echo "CycleGAN rain completed successfully."

# Train CycleGAN for snow
echo ""
echo ">>> Training CycleGAN for SNOW..."
bash experiments/01_flir_weather_translation/scripts/train_cyclegan_snow.sh
SNOW_EXIT=$?
if [[ $SNOW_EXIT -ne 0 ]]; then
    echo "ERROR: CycleGAN snow training failed with exit code $SNOW_EXIT"
    exit 1
fi
echo "CycleGAN snow completed successfully."

echo ""
echo "==========================================="
echo "ALL CYCLEGAN TRAINING COMPLETED"
echo "==========================================="

# ============================================
# PHASE 2: CycleDiff (Method 4)
# ============================================
echo ""
echo "==========================================="
echo "PHASE 2: CycleDiff Training (Method 4)"
echo "==========================================="

# Train CycleDiff for fog
echo ""
echo ">>> Training CycleDiff for FOG..."
bash experiments/01_flir_weather_translation/scripts/train_cyclediff_fog.sh
FOG_DIFF_EXIT=$?
if [[ $FOG_DIFF_EXIT -ne 0 ]]; then
    echo "ERROR: CycleDiff fog training failed with exit code $FOG_DIFF_EXIT"
    exit 1
fi
echo "CycleDiff fog completed successfully."

# Train CycleDiff for rain
echo ""
echo ">>> Training CycleDiff for RAIN..."
bash experiments/01_flir_weather_translation/scripts/train_cyclediff_rain.sh
RAIN_DIFF_EXIT=$?
if [[ $RAIN_DIFF_EXIT -ne 0 ]]; then
    echo "ERROR: CycleDiff rain training failed with exit code $RAIN_DIFF_EXIT"
    exit 1
fi
echo "CycleDiff rain completed successfully."

# Train CycleDiff for snow
echo ""
echo ">>> Training CycleDiff for SNOW..."
bash experiments/01_flir_weather_translation/scripts/train_cyclediff_snow.sh
SNOW_DIFF_EXIT=$?
if [[ $SNOW_DIFF_EXIT -ne 0 ]]; then
    echo "ERROR: CycleDiff snow training failed with exit code $SNOW_DIFF_EXIT"
    exit 1
fi
echo "CycleDiff snow completed successfully."

echo ""
echo "==========================================="
echo "ALL CYCLEDIFF TRAINING COMPLETED"
echo "==========================================="

echo ""
echo "==========================================="
echo "ALL TRAINING COMPLETED SUCCESSFULLY"
echo "Finished at $(date)"
echo "==========================================="

# Summary
echo ""
echo "Training Summary:"
echo "  CycleGAN Fog:   $([ -f experiments/01_flir_weather_translation/method3_cyclegan/logs/fog/checkpoint_epoch_50.pt ] && echo "DONE" || echo "INCOMPLETE")"
echo "  CycleGAN Rain:  $([ -f experiments/01_flir_weather_translation/method3_cyclegan/logs/rain/checkpoint_epoch_50.pt ] && echo "DONE" || echo "INCOMPLETE")"
echo "  CycleGAN Snow:  $([ -f experiments/01_flir_weather_translation/method3_cyclegan/logs/snow/checkpoint_epoch_50.pt ] && echo "DONE" || echo "INCOMPLETE")"
echo "  CycleDiff Fog:  $([ -f experiments/01_flir_weather_translation/method4_cyclediff/results/train_cycle_C_disc_G_blocks_12_clear2fog/model-12.pt ] && echo "DONE" || echo "INCOMPLETE")"
echo "  CycleDiff Rain: $([ -f experiments/01_flir_weather_translation/method4_cyclediff/results/train_cycle_C_disc_G_blocks_12_clear2rain/model-12.pt ] && echo "DONE" || echo "INCOMPLETE")"
echo "  CycleDiff Snow: $([ -f experiments/01_flir_weather_translation/method4_cyclediff/results/train_cycle_C_disc_G_blocks_12_clear2snow/model-12.pt ] && echo "DONE" || echo "INCOMPLETE")"