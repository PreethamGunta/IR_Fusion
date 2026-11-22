#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# run_cyclegan.sh — Orchestrate all CycleGAN training runs for FLIR ADAS v2.
#
# Runs fog, rain, snow in order (fog first as it is the primary target),
# each resumable via checkpoint or skippable via DONE marker.

set -uo pipefail

REPO="$ROOT"
PY="${PY:-python}"

HERE="$REPO/experiments/01_flir_weather_translation"
CGAN="$HERE/method3_cyclegan"
STATE="$CGAN/logs/state"
mkdir -p "$STATE"

EPOCHS="${CYCLEGAN_EPOCHS:-50}"
CROP="${CYCLEGAN_CROP:-256}"
WORKERS="${CYCLEGAN_WORKERS:-$(python3 -c 'import os;print(max(0,min(int(os.environ.get("SLURM_CPUS_PER_TASK","8"))-2,12)))' 2>/dev/null || echo 6)}"
CALIBRATE_FIRST="${CYCLEGAN_CALIBRATE_FIRST:-1}"

CONDITIONS=("fog" "rain" "snow")
if [ $# -gt 0 ]; then CONDITIONS=("$@"); fi

echo "================================================================"
echo " Method 3 CycleGAN — FLIR ADAS v2"
echo " started $(date)"
echo " python=$PY  epochs=$EPOCHS crop=$CROP workers=$WORKERS"
echo " conditions=${CONDITIONS[*]}"
echo "=============================================================="

for cond in "${CONDITIONS[@]}"; do
    rd="$CGAN/runs/$cond"
    echo ""
    echo "=============================================================="
    echo ">>> $cond  $(date)"
    echo "=============================================================="

    # Skip if already done
    if [ -f "$rd/DONE" ]; then
        echo "  already finished: $(tr -d '\n' < "$rd/DONE" 2>/dev/null || echo "")"
        echo "  skipping."
        continue
    fi

    # Check data availability
    if ! "$PY" "$CGAN/train_cyclegan.py" --config "$cond" --check-data 2>&1 | tee -a "$STATE/${cond}_data_check.log"; then
        echo "  SKIP: target domain unavailable for $cond."
        continue
    fi

    extra=()
    # Calibrate per run on first attempt
    if [ "$CALIBRATE_FIRST" = "1" ] && [ ! -f "$rd/calibration.json" ]; then
        extra+=(--calibrate 30)
    fi

    "$PY" "$CGAN/train_cyclegan.py" \
        --config "$cond" \
        --out "$CGAN/runs" \
        --epochs "$EPOCHS" \
        --crop "$CROP" \
        --batch 8 --workers "$WORKERS" \
        "${extra[@]}"
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "  $cond exited $rc -- stopping."
        exit $rc
    fi
done

echo ""
echo "=============================================================="
echo " finished $(date)"
for cond in "${CONDITIONS[@]}"; do
    rd="$CGAN/runs/$cond"
    if [ -f "$rd/DONE" ]; then
        echo "  [DONE]    $cond  -> $rd/G_AB.pt"
    elif [ -f "$rd/ckpt/latest.pt" ]; then
        echo "  [PARTIAL] $cond  -> $rd/ckpt/latest.pt (resubmit to resume)"
    else
        echo "  [TODO]    $cond"
    fi
done
echo "=============================================================="
