#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# This script runs inside salloc_session once allocation is granted.
# It sends the srun pipeline command to srun_training tmux session.
cd $ROOT

echo "=== SLURM allocation granted at $(date) ==="
echo "SALLOC_JOB_ID=$SLURM_JOB_ID"
echo "Starting srun pipeline in srun_training tmux..."

tmux send-keys -t srun_training "cd $ROOT && srun --gpu=1 --mem=64G --time=20:00:00 bash experiments/01_flir_weather_translation/run_yolo_full.sh 2>&1 | tee experiments/01_flir_weather_translation/logs/run_yolo_full_$(date +%s).log" Enter

echo "Pipeline started in srun_training tmux session"
