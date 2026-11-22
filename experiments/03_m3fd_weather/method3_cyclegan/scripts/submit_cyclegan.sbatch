#!/usr/bin/env bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=gateir-cgan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=experiments/03_m3fd_weather/method3_cyclegan/logs/cgan_%j.out

# Method 3 CycleGAN training: thermal_fog, thermal_rain, rgb_fog, rgb_rain.
#
# Submit:
#   sbatch experiments/03_m3fd_weather/method3_cyclegan/scripts/submit_cyclegan.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f experiments/03_m3fd_weather/method3_cyclegan/logs/cgan_<jobid>.out
#
# Why 24h and not the usual 6h:
# docs/memory.md:55 advises --time=6:00:00 because shorter jobs backfill
# sooner. That advice is right for a single ~3h detector run, and wrong here.
# Four resnet-9 CycleGANs at 512px on one RTX 6000 Ada do not fit in 6h by a
# wide margin, and a 6h job would be requeued three or four times to get the
# same work done. The cost of asking for 24h is queue position, not
# correctness; the cost of asking for 6h is that we spend it four times over.
#
# --requeue is not optional. Nodes on this cluster fail outright
# (bench/scripts/sweep.sbatch:13-19 lost a sweep that way), and four
# multi-hour training runs is exactly the shape of job that will hit one.
# train_cyclegan.py checkpoints every 100 iterations and run_cyclegan.sh skips
# any run that already wrote a DONE marker, so a requeue continues rather than
# restarting. It is safe to resubmit this same script as many times as needed.
#
# --cpus-per-task=8 rather than 16/32: image decode for 512px crops off
# 1024x768 PNGs is not the bottleneck that it is for the 1280x960 UNIRI
# frames, and the GPU is the constraint here. Bump it and the only effect is
# more idle dataloader workers.

set -u
cd $ROOT || exit 1

# Must be exported before the first cuBLAS handle is created, so it belongs
# here rather than in the Python. Inert unless --deterministic is passed;
# setting it unconditionally costs nothing and means debugging a resumed run
# does not require editing this file too.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p experiments/03_m3fd_weather/method3_cyclegan/logs

echo "=== job $SLURM_JOB_ID on $(hostname), attempt ${SLURM_RESTART_COUNT:-0}, $(date) ==="
echo "=== cpus=$SLURM_CPUS_PER_TASK mem=$SLURM_MEM_PER_NODE gpu=${CUDA_VISIBLE_DEVICES:-none} ==="
echo "=== already finished: $(ls -1 experiments/03_m3fd_weather/method3_cyclegan/runs/*/DONE 2>/dev/null | wc -l) / 4 runs ==="

bash experiments/03_m3fd_weather/method3_cyclegan/scripts/run_cyclegan.sh

echo "=== finished $(date) ==="
