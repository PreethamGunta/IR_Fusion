#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# poll_salloc.sh — Check if salloc allocation is granted, then keep srun job alive.
# Called by cron every hour.

REPO="$ROOT"
LOG="$REPO/experiments/01_flir_weather_translation/logs/salloc_poll.log"
mkdir -p "$REPO/experiments/01_flir_weather_translation/logs"

# Check if the salloc tmux session exists
if ! tmux has-session -t salloc_session 2>/dev/null; then
    echo "[$(date)] salloc_session does not exist. Creating new one." >> "$LOG"
    tmux new-session -d -s salloc_session 'salloc --gres=gpu:1 --time=24:00:00 --partition=gpu'
    exit 0
fi

# Capture the tmux pane to see if we have an allocation
PANE_CONTENT=$(tmux capture-pane -t salloc_session -p 2>/dev/null)

# Check for salloc job ID in the pane
if echo "$PANE_CONTENT" | grep -q "salloc: Granted job"; then
    JOBID=$(echo "$PANE_CONTENT" | grep "salloc: Granted job" | head -1 | grep -oP '\d+')
    if [ -n "$JOBID" ]; then
        # Check if srun is already running in the session
        if ! tmux list-panes -t salloc_session -F '#{pane_pid}' 2>/dev/null | head -1 | xargs -I{} ps -o ppid= -p {} 2>/dev/null | xargs -I{} pgrep -f "srun.*train_all" 2>/dev/null; then
            echo "[$(date)] Allocation granted (job $JOBID). Starting training via srun..." >> "$LOG"
            tmux send-keys -t salloc_session "srun --job-name=flir_all_trainers $REPO/experiments/01_flir_weather_translation/scripts/train_all_translators.sbatch" Enter
        fi
    fi
else
    # Check squeue for the salloc job
    if echo "$PANE_CONTENT" | grep -q "salloc: Pending"; then
        echo "[$(date)] salloc still pending. Waiting..." >> "$LOG"
    fi
fi

# Also check SLURM squeue for our job
if command -v squeue &> /dev/null; then
    squeue -u "$USER" -h -o "%.18i %.8T" 2>/dev/null | while read jid state; do
        if [ "$state" = "PD" ]; then
            echo "[$(date)] Job $jid still pending in SLURM queue" >> "$LOG"
        fi
    done
fi
