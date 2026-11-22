#!/bin/bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=cyclegan_fog
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --output=logs/fog_%j.out
#SBATCH --error=logs/fog_%j.err

cd $ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir

echo "=== CycleGAN Fog Training Started at $(date) ==="

CKPT_DIR="experiments/01_flir_weather_translation/method3_cyclegan/logs/fog"
RESUME_CKPT=""
if [ -f "$CKPT_DIR/best_model.pt" ]; then
    RESUME_CKPT="--resume $CKPT_DIR/best_model.pt"
elif [ -f "$CKPT_DIR/checkpoint_epoch_latest.pt" ]; then
    RESUME_CKPT="--resume $CKPT_DIR/checkpoint_epoch_latest.pt"
fi

python experiments/01_flir_weather_translation/method3_cyclegan/train_cyclegan.py \
    --config fog \
    --epochs 50 \
    --batch-size 1 \
    --lr 2e-4 \
    $RESUME_CKPT

echo "=== Fog training completed at $(date) ==="