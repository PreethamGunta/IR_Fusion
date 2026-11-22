#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# hourly_health_check.sh — Check CycleDiff training health
# Runs via cron, checks SLURM job, tmux session, and GPU usage

LOG="$ROOT/experiments/01_flir_weather_translation/method4_cyclediff/logs/health_check.log"
REPO="$ROOT"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG"
}

log "=== Health check started ==="

# Dynamically get the latest SLURM job ID for the user
SLURM_JOBID=$(squeue -u preetham_g -O "jobid" --noheader 2>/dev/null | head -1 | tr -d ' ')
if [ -z "$SLURM_JOBID" ]; then
    log "WARNING: No SLURM job found for user preetham_g"
    log "Manual intervention needed: create new salloc in salloc_session tmux"
    SLURM_OK=1
else
    log "Latest SLURM job ID found: $SLURM_JOBID"
    squeue -u preetham_g -O "jobid,state" 2>/dev/null | grep -q "$SLURM_JOBID.*R\|$SLURM_JOBID.*PD\|$SLURM_JOBID.*CG"
    SLURM_OK=$?
    if [ $SLURM_OK -eq 0 ]; then
        log "SLURM job $SLURM_JOBID: RUNNING/PENDING/COMPLETING"
    else
        log "WARNING: SLURM job $SLURM_JOBID not in expected state"
    fi
fi

# Check if srun_training tmux session has an active srun process
SRUN_STATUS=$(tmux capture-pane -t srun_training -p 2>/dev/null | tail -5 | grep -c "Train Step")
if [ $SRUN_STATUS -gt 0 ]; then
    TRAIN_STEP=$(tmux capture-pane -t srun_training -p 2>/dev/null | grep -oP '\[Train Step\] \K[0-9]+' | tail -1)
    log "srun_training: training in progress at step ${TRAIN_STEP:-unknown}"
else
    log "WARNING: No training activity detected in srun_training"
    # Check if we're at a shell prompt (training stopped)
    PROMPT_CHECK=$(tmux capture-pane -t srun_training -p 2>/dev/null | tail -3 | grep -c "preetham_g@\|srun>\|^\$")
    if [ $SLURM_OK -eq 0 ] && [ $SLURM_JOBID -gt 0 ]; then
        log "Training stopped but SLURM job $SLURM_JOBID alive. Training will auto-resume from latest checkpoint."
        log "Manual restart may be needed if auto-restart doesn't trigger."
    fi
fi

# Check GPU usage
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
log "GPU utilization: ${GPU_UTIL:-N/A}%"

# Check for errors in training output
ERRORS=$(tmux capture-pane -t srun_training -p 2>/dev/null | tail -50 | grep -ci "error\|traceback\|exception\|killed\|fail")
if [ $ERRORS -gt 0 ]; then
    log "WARNING: Found $ERRORS error-related messages in recent training output"
    tmux capture-pane -t srun_training -p 2>/dev/null | tail -50 | grep -i "error\|traceback\|exception\|killed\|fail" | tail -5 >> "$LOG"
fi

# Check checkpoint files - LDM clear (should be complete)
LDM_CLEAR_RESULTS="$REPO/experiments/01_flir_weather_translation/method4_cyclediff/results/ddm_const_uncond_unet_ldm_clear"
if [ -d "$LDM_CLEAR_RESULTS" ]; then
    LATEST_CKPT=$(ls -t "$LDM_CLEAR_RESULTS"/model-*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        log "LDM clear latest checkpoint: $(basename $LATEST_CKPT)"
    fi
    if [ -f "$LDM_CLEAR_RESULTS/final_model.pt" ]; then
        log "LDM clear final_model.pt: EXISTS"
    fi
fi

# Check checkpoint files - fog LDM (in progress)
LDM_FOG_RESULTS="$REPO/experiments/01_flir_weather_translation/method4_cyclediff/results/ddm_const_uncond_unet_ldm_fog"
if [ -d "$LDM_FOG_RESULTS" ]; then
    LATEST_CKPT=$(ls -t "$LDM_FOG_RESULTS"/model-*.pt 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        log "LDM fog latest checkpoint: $(basename $LATEST_CKPT)"
    fi
    if [ -f "$LDM_FOG_RESULTS/final_model.pt" ]; then
        log "LDM fog final_model.pt: EXISTS"
    fi
fi

log "=== Health check complete ==="
