#!/usr/bin/env bash
# Read-only status snapshot for the FLIR + DroneVehicle run.
set -uo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXP" || exit 1
echo "=== $(date) ==="
squeue -u "$USER"
echo
echo "=== orchestrator log ==="
LOG=$(ls -t logs/parallel*.log 2>/dev/null | head -1)
tail -20 "$LOG" 2>/dev/null | tr '\r' '\n' | tail -12
echo
echo "=== run epochs / best mAP50-95 ==="
python3 - <<'PY'
import csv, os, glob
for p in sorted(glob.glob('runs/*/results.csv')):
    n = p.split('/')[1]
    try:
        rows = list(csv.DictReader(open(p)))
        if rows:
            r = rows[-1]
            best = max(float(x['metrics/mAP50-95(B)']) for x in rows)
            print(f"{n:28s} ep={r['epoch']:>3s} last={float(r['metrics/mAP50-95(B)']):.4f} best={best:.4f}")
    except Exception as e:
        print(n, 'ERR', e)
PY
echo
echo "=== DONE markers / results ==="
echo "DONE: $(find runs -maxdepth 2 -name DONE 2>/dev/null | wc -l)   test results: $(ls results/*_test.json 2>/dev/null | wc -l)"
echo
echo "=== live training processes ==="
pgrep -af "graphkd.training.train --name" | grep -v "bin/bash -c" | sed 's/.*--name /--name /' | cut -c1-120 | head -8
echo
echo "=== GPU (our job is the low-index allocated GPU; check memory/util) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
