#!/usr/bin/env bash
# Run Phase 1 (CycleGAN main + filter sweep) -> Phase 2 (CycleDiff) -> Phase 3
# (B0/B2/B3 ablations), continuing on stage failures.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
PY="${PY:-python}"

echo "=== ALL REMAINING START $(date) ==="
( STAGE=all bash scripts/run_phase1_cyclegan.sh ) || echo "PHASE1 FAILED $(date)"
( $PY -m src.make_report ) || true
( bash scripts/run_phase2_cyclediff.sh ) || echo "PHASE2 FAILED $(date)"
( $PY -m src.make_report ) || true
( bash scripts/run_phase3_ablations.sh ) || echo "PHASE3 FAILED $(date)"
( $PY -m src.make_report ) || true
echo "=== ALL REMAINING DONE $(date) ==="
