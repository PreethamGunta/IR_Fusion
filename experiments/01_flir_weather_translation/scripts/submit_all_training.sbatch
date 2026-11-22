#!/bin/bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=flir_all_training
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --output=logs/training_master_%j.out
#SBATCH --error=logs/training_master_%j.err

cd $ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir

echo "=== Master Training Job Started at $(date) ==="

# Submit CycleGAN job (will run all 3 conditions sequentially)
echo "Submitting CycleGAN job..."
CYCLEGAN_JOB=$(sbatch --parsable experiments/01_flir_weather_translation/method3_cyclegan/scripts/submit_cyclegan.sbatch)
echo "CycleGAN job submitted: $CYCLEGAN_JOB"

# Submit CycleDiff job (depends on CycleGAN completing first, or can run in parallel)
# Using --dependency=afterok to run after CycleGAN completes
echo "Submitting CycleDiff job (depends on CycleGAN)..."
CYCLEDIFF_JOB=$(sbatch --dependency=afterok:$CYCLEGAN_JOB --parsable experiments/01_flir_weather_translation/method4_cyclediff/scripts/submit_cyclediff.sbatch)
echo "CycleDiff job submitted: $CYCLEDIFF_JOB"

echo "Both jobs submitted. CycleGAN: $CYCLEGAN_JOB, CycleDiff: $CYCLEDIFF_JOB"
echo "Monitor with: squeue -u \$USER"
echo "Logs in experiments/01_flir_weather_translation/method3_cyclegan/logs/ and experiments/01_flir_weather_translation/method4_cyclediff/logs/"
