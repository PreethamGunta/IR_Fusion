#!/usr/bin/env bash
# Run the proper 100-epoch FLIR comparisons in one allocation:
#   1) CycleGAN: B1 vs G1-U-Net
#   2) CycleDiff-75: B1 vs G1-U-Net (CycleDiff-trained U-Net)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
PY="${PY:-python}"

echo "=== [alloc] job ${SLURM_JOB_ID:-?} start $(date) ==="
( bash scripts/run_main_100.sh ) || echo "CYCLEGAN 100 FAILED $(date)"
( bash scripts/run_main_100_cyclediff.sh ) || echo "CYCLEDIFF 100 FAILED $(date)"
( $PY -m src.make_report ) || true
( $PY -m src.summarize_results ) || true
echo "=== [alloc] done $(date) ==="
