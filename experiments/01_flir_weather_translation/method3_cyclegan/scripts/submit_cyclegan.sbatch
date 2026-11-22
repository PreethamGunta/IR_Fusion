#!/bin/bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=cyclegan_flir_v2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=experiments/01_flir_weather_translation/method3_cyclegan/logs/cyclegan_%j.out
#SBATCH --error=experiments/01_flir_weather_translation/method3_cyclegan/logs/cyclegan_%j.err

set -uo pipefail
cd $ROOT

source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir

mkdir -p experiments/01_flir_weather_translation/method3_cyclegan/logs

echo "=== job $SLURM_JOB_ID on $(hostname), attempt ${SLURM_RESTART_COUNT:-0}, $(date) ==="
echo "=== gpu=${CUDA_VISIBLE_DEVICES:-none} cpus=$SLURM_CPUS_PER_TASK ==="

bash experiments/01_flir_weather_translation/method3_cyclegan/scripts/train_cyclegan.sh

echo "=== finished $(date) ==="
