#!/usr/bin/env bash
# Remaining work on allocation 9670: Phase 3 ablations, 100-epoch YOLOv12s main,
# then the CycleDiff-trained U-Net transfer test.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
PY="${PY:-python}"

echo "=== REMAINING START $(date) ==="
( bash scripts/run_phase3_direct.sh ) || echo "PHASE3 FAILED $(date)"
( bash scripts/run_main_100.sh ) || echo "MAIN100 FAILED $(date)"
( bash scripts/run_cyclediff_unet.sh ) || echo "CDUNET FAILED $(date)"
( $PY -m src.make_report ) || true
( $PY -m src.summarize_results ) || true
echo "=== REMAINING DONE $(date) ==="
