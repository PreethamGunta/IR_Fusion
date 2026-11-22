#!/bin/bash
#SBATCH --job-name=fog_vae
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=fog_vae_%j.log
#SBATCH --export=ALL

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
cd "$ROOT"
export PY="${PY:-python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash experiments/01_flir_weather_translation/method4_cyclediff/scripts/train_cyclediff.sh fog
