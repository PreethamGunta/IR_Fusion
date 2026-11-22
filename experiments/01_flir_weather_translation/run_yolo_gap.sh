#!/bin/bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
set -euo pipefail

# ============================================================
# YOLO Gap Establishment: train on clear+synthetic adverse, evaluate
# Skips clear_baseline. Trains: fog(cyclediff), fog(cyclegan), rain(cyclegan), snow(cyclegan)
# After each model: evaluate on clear test + adverse test.
# ============================================================
REPO="$ROOT"
CDIR="$REPO/experiments/01_flir_weather_translation"
LOGDIR="$CDIR/logs"
mkdir -p "$LOGDIR"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate gateir
cd "$REPO"

EPOCHS=100
BATCH=16
IMGSZ=640
WEIGHTS="yolov8s.pt"

echo "=== YOLO Gap Establishment Started at $(date) ==="

# Helper: run training + eval for a given condition/method
run_gap() {
    local condition=$1
    local method=$2
    local run_name="3a_${method}_${condition}"

    echo ""
    echo "=========================================="
    echo " Training: $run_name"
    echo "=========================================="

    # Train
    python experiments/01_flir_weather_translation/train_detector.py \
        --mode 3a --condition "$condition" --method "$method" \
        --epochs $EPOCHS --batch-size $BATCH --img-size $IMGSZ \
        --weights "$WEIGHTS" \
        2>&1 | tee "$LOGDIR/${run_name}_train.log"
    TRAIN_RC=${PIPESTATUS[0]}
    if [ $TRAIN_RC -ne 0 ]; then
        echo "ERROR: Training $run_name failed (exit $TRAIN_RC)" >&2
        return 1
    fi

    # Evaluate on clear test
    echo "--- Evaluating $run_name on clear test ---"
    python experiments/01_flir_weather_translation/eval_detector.py \
        --weights "experiments/01_flir_weather_translation/runs/${run_name}/weights/best.pt" \
        --data "data/flir_adas_v2/yolo/thermal.yaml" \
        --name "${run_name}_clear" --split test --imgsz $IMGSZ \
        2>&1 | tee "$LOGDIR/${run_name}_eval_clear.log"
    EVAL_CLEAR_RC=${PIPESTATUS[0]}
    if [ $EVAL_CLEAR_RC -ne 0 ]; then
        echo "ERROR: Eval clear for $run_name failed (exit $EVAL_CLEAR_RC)" >&2
        return 1
    fi

    # Evaluate on adverse test
    echo "--- Evaluating $run_name on adverse test ---"
    ADV_DATA="data/flir_adas_v2/synthetic/${method}/${condition}/test/test.yaml"
    if [ -f "$ADV_DATA" ]; then
        python experiments/01_flir_weather_translation/eval_detector.py \
            --weights "experiments/01_flir_weather_translation/runs/${run_name}/weights/best.pt" \
            --data "$ADV_DATA" \
            --name "${run_name}_adv" --split test --imgsz $IMGSZ \
            2>&1 | tee "$LOGDIR/${run_name}_eval_adv.log"
        EVAL_ADV_RC=${PIPESTATUS[0]}
        if [ $EVAL_ADV_RC -ne 0 ]; then
            echo "ERROR: Eval adverse for $run_name failed (exit $EVAL_ADV_RC)" >&2
            return 1
        fi
    else
        echo "WARNING: Adverse test YAML not found at $ADV_DATA, skipping adverse eval"
    fi

    echo "=== $run_name complete at $(date) ==="
}

# Run gap establishment for each condition/method
# (skip clear_baseline per user request)

# 1. Fog (CycleDiff)
run_gap "fog" "cyclediff" || { echo "FAILED: fog cyclediff"; exit 1; }

# 2. Fog (CycleGAN)
run_gap "fog" "cyclegan" || { echo "FAILED: fog cyclegan"; exit 1; }

# 3. Rain (CycleGAN)
run_gap "rain" "cyclegan" || { echo "FAILED: rain cyclegan"; exit 1; }

# 4. Snow (CycleGAN)
run_gap "snow" "cyclegan" || { echo "FAILED: snow cyclegan"; exit 1; }

echo ""
echo "=== All YOLO gap establishment complete at $(date) ==="
