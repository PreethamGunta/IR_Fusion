#!/bin/bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=cyclediff_train_translate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --requeue
#SBATCH --output=experiments/01_flir_weather_translation/method4_cyclediff/logs/train_translate_%j.out
#SBATCH --error=experiments/01_flir_weather_translation/method4_cyclediff/logs/train_translate_%j.err

cd $ROOT
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir

echo "=== Started at $(date) ==="
bash experiments/01_flir_weather_translation/method4_cyclediff/run_train_and_translate.sh
RC=$?
echo "=== Finished at $(date) with exit code $RC ==="
exit $RC
