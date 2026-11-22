#!/bin/bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=cyclediff_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --output=logs/all_%j.out
#SBATCH --error=logs/all_%j.err

cd $ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir

echo "Starting CycleGAN all conditions training at $(date)"

for cond in fog rain snow; do
    echo "=== Training $cond at $(date) ==="
    python experiments/01_flir_weather_translation/method3_cyclegan/train_cyclegan.py --config $cond --epochs 50 --batch-size 1 --lr 2e-4
    if [ $? -ne 0 ]; then
        echo "ERROR: $cond training failed"
        exit 1
    fi
    echo "$cond completed at $(date)"
done

echo "All conditions completed at $(date)"
