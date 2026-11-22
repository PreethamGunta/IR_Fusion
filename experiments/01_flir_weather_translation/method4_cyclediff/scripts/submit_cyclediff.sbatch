#!/bin/bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=cyclediff_flir
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --output=$ROOT/experiments/01_flir_weather_translation/method4_cyclediff/logs/cyclediff_%j.out
#SBATCH --error=$ROOT/experiments/01_flir_weather_translation/method4_cyclediff/logs/cyclediff_%j.err
#SBATCH --open-mode=append

cd $ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir

mkdir -p experiments/01_flir_weather_translation/method4_cyclediff/logs

export PY="python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd experiments/01_flir_weather_translation/method4_cyclediff

echo "=== CycleDiff Training Started at $(date) ==="
echo "GPUs: $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Use the fixed orchestrator script (handles resume via checkpoint milestones)
# Single GPU per QOS limits — cycle training uses the single-GPU script
bash scripts/train_cyclediff.sh

echo "=== All CycleDiff training completed at $(date) ==="
