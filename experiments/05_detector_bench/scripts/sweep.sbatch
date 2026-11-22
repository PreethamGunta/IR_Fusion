#!/usr/bin/env bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=gateir-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=bench/logs/sweep_%j.out
#
# Batch submission of the detector sweep.
#
# Why sbatch rather than salloc+tmux: job 7832 was lost to a NODE_FAIL at
# 05:57 with three runs still to go, and an interactive allocation cannot come
# back from that. With --requeue Slurm restarts this job on a healthy node, and
# because train_detector.py resumes from last.pt and overnight.sh skips runs
# that already have a result JSON, the restart continues the sweep instead of
# repeating it.
#
# 32 CPUs (up from 16) because training is CPU-bound on JPEG decode, not
# GPU-bound. Worker count affects prefetch only -- not sample order,
# augmentation or optimisation -- so runs already completed at 16 CPUs stay
# directly comparable with runs finished here.

set -u
cd $ROOT || exit 1

echo "=== job $SLURM_JOB_ID on $(hostname), attempt ${SLURM_RESTART_COUNT:-0}, $(date) ==="
echo "=== cpus=$SLURM_CPUS_PER_TASK mem=$SLURM_MEM_PER_NODE gpu=$CUDA_VISIBLE_DEVICES ==="
echo "=== already complete: $(ls bench/results/[a-z]*.json 2>/dev/null | wc -l) runs ==="

bash bench/scripts/overnight.sh

echo "=== finished $(date) ==="
