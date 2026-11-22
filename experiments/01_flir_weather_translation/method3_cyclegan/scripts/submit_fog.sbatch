#!/bin/bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=cyclediff_fog
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

echo "Starting CycleGAN fog training at $(date)"
python experiments/01_flir_weather_translation/method3_cyclegan/train_cyclegan.py --config fog --epochs 50 --batch-size 1 --lr 2e-4
echo "Completed at $(date)"
