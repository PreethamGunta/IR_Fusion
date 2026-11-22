#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# Detector baselines on UNIRI-TID, run strictly sequentially.
#
# The QOS allows one GPU and one running job, so these cannot be parallelised.
# They run inside the already-allocated interactive job via `srun --overlap`.
# If that allocation dies, so does this script.
set -u

PY="python"
CD="$ROOT"
LOG="$CD/bench/logs"
EPOCHS="${EPOCHS:-100}"

# DEBUG=1 exercises the whole sweep -- every model loads, trains, and is scored
# per condition -- on a handful of images in a few minutes. Runs and results are
# suffixed and quarantined so they cannot be confused with the real sweep.
DEBUG="${DEBUG:-}"
case "${DEBUG,,}" in
    1|true|yes|on) SUFFIX="_dbg"; RESDIR="experiments/05_detector_bench/results/debug"; LOG="$LOG/debug"
                   echo "*** DEBUG SWEEP: tiny subset, ${EPOCHS} epochs is ignored ***" ;;
    *)             SUFFIX="";     RESDIR="experiments/05_detector_bench/results" ;;
esac
mkdir -p "$LOG"

cd "$CD" || exit 1

run () {                       # run <name> <model> [extra flags...]
    local name=$1$SUFFIX model=$2; shift 2
    if [ -f "${RESDIR}/${name}.json" ]; then
        echo "[$(date +%T)] SKIP ${name} (result already exists)"
        return
    fi
    # Two attempts. train_detector.py resumes from last.pt, so attempt 2 picks up
    # where attempt 1 died rather than starting the schedule again -- a transient
    # OOM or a dataloader hiccup then costs one epoch, not a whole run.
    local attempt
    for attempt in 1 2; do
        echo "[$(date +%T)] START ${name}  (${model}, ${EPOCHS} epochs, attempt ${attempt})"
        "$PY" bench/train_detector.py --model "$model" --name "$name" \
            --epochs "$EPOCHS" "$@" >> "$LOG/${name}.log" 2>&1
        local rc=$?
        if [ $rc -eq 0 ]; then
            echo "[$(date +%T)] DONE  ${name}"
            return
        fi
        echo "[$(date +%T)] FAIL  ${name} (exit ${rc}, attempt ${attempt}) -- see $LOG/${name}.log"
    done
}

# Ultralytics' nominal batch size is 64 and it gradient-accumulates to reach it,
# so every model below optimises at an effective batch of 64 regardless of what
# fits in memory -- the table compares architectures, not batch schedules.
# Measured peak GPU memory on the 48 GB card: v8s@64 13 GB, p2@64 ~13 GB,
# rtdetr-l@32 ~26 GB. Worst case leaves ~22 GB free.
B_DEFAULT=64
B_P2=64
B_RTDETR=32   # transformer, ~2x the activation memory per image; accumulate=2

# Table 1 row 1: the plain baseline
run v8s          weights/yolov8s.pt      --batch $B_DEFAULT
# "you only compared against an outdated YOLOv8"
run v11s         weights/yolo11s.pt      --batch $B_DEFAULT
run v12s         weights/yolo12s.pt      --batch $B_DEFAULT
# Table 1 row 2: robustness-by-training, the rival to gating+preprocessing
run v8s_weather  weights/yolov8s.pt      --batch $B_DEFAULT --weather-aug
# extra points on the accuracy/latency curve
run v10s         weights/yolov10s.pt     --batch $B_DEFAULT
run v8s_p2       yolov8-p2.yaml          --batch $B_P2
run rtdetrl      weights/rtdetr-l.pt     --batch $B_RTDETR

echo "[$(date +%T)] ALL RUNS COMPLETE"
