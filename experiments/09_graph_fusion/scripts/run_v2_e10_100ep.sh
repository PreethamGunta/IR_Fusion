#!/usr/bin/env bash
# E10 flagship: 100 epochs, V2 pipeline, E00's exact recipe.
#
# E10 = concat + 1x1 conv at P3/P4/P5 — the standard *learned mid-level*
# fusion reference (no graph structure, no attention). This gives the
# presentation its "normal fusion network" arm at full budget:
#   E00 (IR-only)  vs  E10 (simple fusion)  vs  E30/E35 (graph fusion)
#
# Comparability matrix (identical to scripts/run_v2_e30_100ep.sh except the
# fuse module):
#   data          full 17,962 paired train
#   labels        thermal (*labelr) supervise training and validation
#   recipe        MuSGD, lr0 0.01, lrf 0.01 (= prefix_lrf(100)), momentum
#                 0.937, warmup_bias_lr 0.1, batch 16, seed 0, workers 8,
#                 patience 30, close_mosaic 10 (matches E00; NOT probe 0)
#   val           1,469-frame training validator, per epoch
#   variable      the fuse module only (E10 vs E30: simple mixing vs graph)
# Note: E10 has no `--d` (no node-projection width); the module is a plain
# conv, matching the ladder spec and tonight's e10_v2_10ep screen arm.
# Leanest fused arm: expect ~3-4.5 min/ep => ~5-7.5 h for 100 ep. Solo.
#
# Usage: bash scripts/run_v2_e10_100ep.sh [JOBID]
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python
JOBID="${1:-10230}"
MAX_USED_MIB="${MAX_USED_MIB:-2500}"
DATA="configs/dronevehicle_v2_paired.yaml"
RGB="configs/dronevehicle_v2_rgb.yaml"
NAME="e10_v2_100ep"

for f in "$EXP/$DATA" "$EXP/$RGB" "$PY"; do
  [[ -e "$f" ]] || { echo "[e10flag] missing required path: $f"; exit 1; }
done

if [[ -f "$EXP/runs/${NAME}/DONE" ]]; then
  echo "[e10flag] SKIP: runs/${NAME}/DONE exists"
  exit 0
fi

used="$(srun --jobid="$JOBID" bash -c 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' | head -1 | tr -d ' ')"
if (( used > MAX_USED_MIB )); then
  echo "[e10flag] REFUSING to start: GPU already holds ${used} MiB (> ${MAX_USED_MIB})"
  exit 1
fi

echo "[e10flag] $(date -u) START ${NAME} (100 ep, close_mosaic 10, E10 concat+1x1)"
rc=0
srun --jobid="$JOBID" bash -c \
  "cd '$EXP' && PYTHONPATH=src '$PY' scripts/train_unified_v2.py \
   --name '$NAME' --mode pair --fuse E10 \
   --data '$DATA' --rgb-yaml '$RGB' \
   --epochs 100 --close-mosaic 10 --workers 8 \
   2>&1 | tee '$EXP/logs/v2_${NAME}.log'" || rc=$?
if (( rc != 0 )); then
  echo "[e10flag] $(date -u) FAILED ${NAME} (rc=${rc})"
  exit 1
fi
echo "[e10flag] $(date -u) DONE ${NAME}"
