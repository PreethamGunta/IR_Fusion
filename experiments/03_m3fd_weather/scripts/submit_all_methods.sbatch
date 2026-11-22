#!/bin/bash
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
#SBATCH --job-name=m3fd_all_methods
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=experiments/03_m3fd_weather/logs/all_methods_%j.log

# Unified M3FD pipeline: split -> CycleGAN -> clear baseline -> Methods 1/2/3.
#
# --open-mode=append  a requeue must not truncate the log. Without it the second
#                    attempt overwrote the first, destroying the only record of
#                    what the first attempt did -- and with --requeue the first
#                    attempt is exactly the one you need to read.
# --partition=gpu     matches every other job script here; omitting it relied on
#                    the default partition happening to have GPUs.
# --requeue           nodes fail. Safe here because run_all_methods.sh gates each
#                    stage on a split-hash marker, so a requeue resumes instead of
#                    restarting from the split.
#
# Runtime budget: ~9h CycleGAN (2 x ~4.2h) + ~0.5h clear + ~0.5h Method 1
# + ~2h Method 2 + ~3h Method 3 = ~15h. 23h leaves room for one requeue.
# A shorter --time gets backfilled faster but cannot hold the CycleGAN pair plus
# the YOLO runs, and `--time` is the wall clock, not the work.
#
# Submit with:
#   sbatch experiments/03_m3fd_weather/scripts/submit_all_methods.sbatch
#
# Monitor with:
#   squeue -u $USER
#   tail -f experiments/03_m3fd_weather/logs/all_methods_<jobid>.log
#   cat  experiments/03_m3fd_weather/logs/state/*.done

set -euo pipefail
cd $ROOT
mkdir -p experiments/03_m3fd_weather/logs experiments/03_m3fd_weather/logs/state

# Prefer the pinned conda env's interpreter directly over `conda activate`: the
# latter depends on shell init that #SBATCH does not guarantee, and silently
# falling back to the system python would change torch/ultralytics versions
# mid-experiment.
PY="python"
if [ ! -x "$PY" ]; then
    echo "FATAL: $PY not found. Build the env with bench/setup_env.sh"
    exit 1
fi
export PY

echo "Job $SLURM_JOB_ID on $(hostname), $(date)"
echo "Node: $(scontrol show hostnames "$SLURM_JOB_HOST" 2>/dev/null || hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo ""

bash experiments/03_m3fd_weather/scripts/run_all_methods.sh

echo ""
echo "Job $SLURM_JOB_ID completed at $(date)"