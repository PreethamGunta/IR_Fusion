#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
cd $ROOT

LOGFILE="$CDIR/logs/wait_and_run.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "wait_and_run.sh started (PID $$), checking every 10 min for GPU allocation..."

while true; do
    # Check if pipeline already started
    if [ -f "$REPO/experiments/01_flir_weather_translation/.pipeline_started" ]; then
        log "Pipeline already started. Exiting monitor."
        exit 0
    fi

    # Check for running SLURM jobs owned by preetham_g
    RUNNING=$(squeue -u preetham_g --noheader -o "%.i %.t" 2>/dev/null | grep R | awk '{print $1}')
    
    if [ -n "$RUNNING" ]; then
        ALLOC_JOB=$(echo "$RUNNING" | head -1)
        log "Found running SLURM allocation: job $ALLOC_JOB"
        
        # Mark pipeline as started
        touch "$REPO/experiments/01_flir_weather_translation/.pipeline_started"
        
        # Send pipeline command to srun_training tmux using srun --jobid
        # The srun --jobid targets the existing salloc allocation
        SRUN_CMD="cd $ROOT && source $(conda info --base)/etc/profile.d/conda.sh && conda activate gateir && srun --jobid=$ALLOC_JOB --mem=64G --time=23:00:00 bash experiments/01_flir_weather_translation/run_yolo_full.sh 2>&1 | tee experiments/01_flir_weather_translation/logs/run_yolo_full_\$(date +%s).log"
        
        tmux send-keys -t srun_training "$SRUN_CMD" Enter
        log "Pipeline srun command sent to srun_training tmux (using jobid=$ALLOC_JOB)"
        log "Command: $SRUN_CMD"
        log "Monitor exiting."
        exit 0
    fi
    
    # Still waiting
    PENDING=$(squeue -u preetham_g --noheader -o "%.i %.t" 2>/dev/null | grep PD | wc -l)
    log "Waiting for GPU - $PENDING pending job(s), no running allocations. Will check again in 10 min."
    
    sleep 600
done
