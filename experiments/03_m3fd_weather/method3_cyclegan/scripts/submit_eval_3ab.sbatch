#!/usr/bin/env bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=gateir-cgan-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=experiments/03_m3fd_weather/method3_cyclegan/logs/eval_%j.out
#
# Method 3 CycleGAN — Protocol 3A / 3B evaluation.
#
# What this job does (details in run_eval_3ab.sh):
#   1. Translate test/val/train splits through thermal_fog and thermal_rain generators
#   2. Write dataset YAMLs with correct YOLO directory structure
#   3. 3A: evaluate existing clear baseline on CycleGAN fog/rain test sets (no retraining)
#   4. 3B fog: train YOLOv12s on clear+CycleGAN_fog_train, eval on fog test
#   5. 3B rain: same for rain
#   6. Print gap comparison against Method 2 reference numbers
#
# Why 6 h (not 24 h):
#   3A eval has no training. 3B training is ~3-4 h each, but fog and rain are
#   run sequentially; total is ~7-8 h. If this does not fit, the job will
#   requeue and resume (train_m3fd.py resumes from last.pt automatically).
#   Consider --time=12:00:00 if you want both fog and rain 3B in one shot.
#
# Submit:
#   sbatch experiments/03_m3fd_weather/method3_cyclegan/scripts/submit_eval_3ab.sbatch
#
# Monitor:
#   squeue -u preetham_g
#   tail -f experiments/03_m3fd_weather/method3_cyclegan/logs/eval_<jobid>.out
#
# After this job completes, read the gap summary at the bottom of the log.
# The critical number is the 3B fog gap. Bar: >= 0.10 to declare Method 3
# successful. Method 2 depth fog got 3A=0.154, 3B=-0.002.

set -u
cd $ROOT || exit 1

mkdir -p experiments/03_m3fd_weather/method3_cyclegan/logs

echo "=== job $SLURM_JOB_ID on $(hostname), attempt ${SLURM_RESTART_COUNT:-0}, $(date) ==="
echo "=== cpus=$SLURM_CPUS_PER_TASK mem=$SLURM_MEM_PER_NODE gpu=${CUDA_VISIBLE_DEVICES:-none} ==="

bash experiments/03_m3fd_weather/method3_cyclegan/scripts/run_eval_3ab.sh

echo "=== finished $(date) ==="
