#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# run_all_protocols.sh — Execute full Protocol 3A/3B evaluation.
#
# Protocol 3A: Train detector on clear + synthetic → evaluate on adverse test
# Protocol 3B: Train detector on clear only → evaluate on translated adverse test (gap)
#
# Both protocols need: clear baseline first, then synthetic translations.

set -uo pipefail

REPO="$ROOT"
PY="${PY:-python}"
CONDITIONS=("fog" "rain" "snow")

echo "=============================================================="
echo " Protocol 3A / 3B Evaluation"
echo " started  $(date)"
echo "=============================================================="

# Step 1: Verify clear baseline exists
CLEAR_RUNS="$REPO/experiments/01_flir_weather_translation/runs/clear_baseline"
if [ ! -f "$CLEAR_RUNS/weights/best.pt" ]; then
    echo ">>> Training clear baseline..."
    $PY experiments/01_flir_weather_translation/train_detector.py --mode clear_baseline --epochs 100
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "ERROR: clear baseline training failed ($rc)"
        exit $rc
    fi
fi
echo "  Clear baseline ready at $CLEAR_RUNS/weights/best.pt"

# Step 2: Check translated test sets exist
echo ""
echo ">>> Checking synthetic translations..."
for cond in "${CONDITIONS[@]}"; do
    # CycleGAN
    cg_adv="$REPO/data/flir_adas_v2/synthetic/cyclegan/$cond"
    if [ ! -d "$cg_adv" ]; then
        echo "  CycleGAN $cond: MISSING — run translate.py first"
    else
        echo "  CycleGAN $cond: OK"
    fi
    # CycleDiff
    cd_adv="$REPO/data/flir_adas_v2/synthetic/cyclediff/$cond"
    if [ ! -d "$cd_adv" ]; then
        echo "  CycleDiff $cond: MISSING — run cyclediff sampler first"
    else
        echo "  CycleDiff $cond: OK"
    fi
done

# Step 3: Run Protocol 3A (clear + synthetic training, adverse eval)
echo ""
echo ">>> Protocol 3A: clear + synthetic training"
for cond in "${CONDITIONS[@]}"; do
    for method in cyclegan cyclediff; do
        run_name="3a_${method}_${cond}"
        run_dir="$REPO/experiments/01_flir_weather_translation/runs/$run_name"
        done_marker="$run_dir/DONE"

        if [ -f "$done_marker" ]; then
            echo "  $run_name: DONE (skip)"
            continue
        fi

        adv_root="$REPO/data/flir_adas_v2/synthetic/$method/$cond"
        if [ ! -d "$adv_root" ]; then
            echo "  $run_name: SKIP (no synthetic images at $adv_root)"
            continue
        fi

        echo "  $run_name: training..."
        $PY experiments/01_flir_weather_translation/train_detector.py \
            --mode 3a --condition "$cond" --method "$method" --epochs 100
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "  $run_name: FAILED ($rc)"
            exit $rc
        fi
        touch "$done_marker"
        echo "  $run_name: DONE"
    done
done

# Step 4: Run Protocol 3B (clear baseline → gap on translated test)
echo ""
echo ">>> Protocol 3B: domain gap measurement (clear baseline → adverse)"
for cond in "${CONDITIONS[@]}"; do
    for method in cyclegan cyclediff; do
        run_name="3b_${method}_${cond}"
        run_dir="$REPO/experiments/01_flir_weather_translation/runs/$run_name"
        done_marker="$run_dir/DONE"

        if [ -f "$done_marker" ]; then
            echo "  $run_name: DONE (skip)"
            continue
        fi

        adv_root="$REPO/data/flir_adas_v2/synthetic/$method/$cond/test"
        if [ ! -d "$adv_root" ]; then
            echo "  $run_name: SKIP (no translated test at $adv_root)"
            continue
        fi

        echo "  $run_name: evaluating..."
        $PY experiments/01_flir_weather_translation/train_detector.py \
            --mode 3b --condition "$cond" --method "$method"
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "  $run_name: FAILED ($rc)"
            exit $rc
        fi
        touch "$done_marker"
        echo "  $run_name: DONE"
    done
done

echo ""
echo "=============================================================="
echo " Protocol 3A/3B complete."
echo " finished  $(date)"
echo ""
echo "Results in: $REPO/experiments/01_flir_weather_translation/runs/"
echo "=============================================================="
