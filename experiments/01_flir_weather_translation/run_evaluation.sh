#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# run_evaluation.sh — Protocol 3A and 3B evaluation for FLIR ADAS v2.
#
# For each generation method (CycleGAN, CycleDiff) and each condition (fog,
# rain, snow), this script:
#   1. Trains the clear-only baseline (3A reference) — reused for all conditions
#   2. Trains the 3A mixed model (clear + synthetic adverse)
#   3. Evaluates 3A: mixed model on adverse test split
#   4. Trains/evaluates 3B: mixed model on translated adverse test
#   5. Computes the gap (clear_test_mAP - adverse_test_mAP)
#
# Idempotent: results JSONs gate re-runs, weights gate re-training.

set -euo pipefail

REPO="$ROOT"
PY="${PY:-python}"

HERE="$REPO/experiments/01_flir_weather_translation"
CGAN="$HERE/method3_cyclegan"
CDATASETS="$CGAN/datasets"
CYCLERES="$HERE/results"
RUNS="$HERE/runs"

mkdir -p "$CYCLERES" "$RUNS"

CONDITIONS=("fog" "rain" "snow")
if [ $# -gt 0 ]; then CONDITIONS=("$@"); fi

EPOCHS="${YOLO_EPOCHS:-100}"
IMGSZ="${YOLO_IMGSZ:-640}"
BATCH="${YOLO_BATCH:-16}"
SEED="${YOLO_SEED:-0}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# --- Helper functions ---

train_if_needed() {
    local mode="$1" cond="$2" method="$3"
    local name="${mode}_${method}_${cond}"
    local weights_dir="$RUNS/${name}"
    local weights="$weights_dir/weights/best.pt"
    local provenance_file="$weights_dir/provenance"

    if [ -f "$weights" ]; then
        echo "  [$name] weights exist. Skipping training."
        return 0
    fi

    echo "  [$name] training..."
    "$PY" "$HERE/train_detector.py" \
        --mode "$mode" \
        --condition "$cond" \
        --method "$method" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --img-size "$IMGSZ" \
        --weights yolov12s.pt
}

eval_model() {
    local name="$1" data_yaml="$2" weights="$3"
    local results_json="$CYCLERES/${name}.json"

    if [ -f "$results_json" ]; then
        echo "  [$name] results JSON exists. Skipping eval."
        return 0
    fi

    echo "  [$name] evaluating on $data_yaml..."
    "$PY" "$HERE/eval_detector.py" \
        --weights "$weights" \
        --data "$data_yaml" \
        --name "$name" \
        --split test \
        --imgsz "$IMGSZ" \
        --out "$results_json"
}

# --- Main ---

say "FLIR ADAS v2 — Protocol 3A/3B Evaluation"
echo " $(date)"
echo " repo:    $REPO"
echo " epochs:  $EPOCHS  imgsz: $IMGSZ  batch: $BATCH  seed: $SEED"
echo " conditions: ${CONDITIONS[*]}"
echo "================================================================"

# Train clear baseline once (reused for all conditions)
if [ ! -f "$RUNS/clear_baseline/weights/best.pt" ]; then
    echo "  [clear_baseline] training clear-only model..."
    "$PY" "$HERE/train_detector.py" \
        --mode clear_baseline \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH" \
        --img-size "$IMGSZ" \
        --weights yolov12s.pt
else
    echo "  [clear_baseline] already trained. Skipping."
fi

CLEAR_W="$RUNS/clear_baseline/weights/best.pt"
if [ ! -f "$CLEAR_W" ]; then
    echo "  ERROR: clear baseline weights missing at $CLEAR_W"
    exit 1
fi

for cond in "${CONDITIONS[@]}"; do
    echo ""
    echo "================================================================"
    echo ">>> Condition: $cond"
    echo "=============================================================="

    # --- CycleGAN ---
    g_ab="$CGAN/runs/cyclegan_${cond}/G_AB.pt"
    if [ -f "$g_ab" ]; then
        echo "  [CycleGAN] $cond: generator found, training 3A..."
        train_if_needed "3a" "$cond" "cyclegan"
        eval_model "3a_cyclegan_${cond}" \
            "$CDATASETS/cyclegan_${cond}_3a.yaml" \
            "$RUNS/3a_cyclegan_${cond}/weights/best.pt"

        echo "  [CycleGAN] $cond: training 3B..."
        train_if_needed "3b" "$cond" "cyclegan"
        eval_model "3b_cyclegan_${cond}_eval" \
            "$CDATASETS/cyclegan_${cond}_3b_adverse_test.yaml" \
            "$RUNS/3b_cyclegan_${cond}/weights/best.pt"
    else
        echo "  [CycleGAN] $cond: no generator at $g_ab. Skipping."
    fi

    # --- CycleDiff ---
    echo "  [CycleDiff] $cond: checking for synthetic images..."
    adv_root="$REPO/data/flir_adas_v2/synthetic/cyclediff/${cond}"
    if [ -d "$adv_root" ]; then
        train_if_needed "3a" "$cond" "cyclediff"
        eval_model "3a_cyclediff_${cond}" \
            "$adv_root/synthetic_${cond}.yaml" \
            "$RUNS/3a_cyclediff_${cond}/weights/best.pt"

        train_if_needed "3b" "$cond" "cyclediff"
        eval_model "3b_cyclediff_${cond}_eval" \
            "$adv_root/test/synthetic_${cond}.yaml" \
            "$RUNS/3b_cyclediff_${cond}/weights/best.pt"
    else
        echo "  [CycleDiff] $cond: no synthetic images at $adv_root. Skipping."
    fi

    echo "  [$cond] done."
done

say "All evaluations complete — $(date)"
