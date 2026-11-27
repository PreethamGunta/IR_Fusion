#!/usr/bin/env bash
# E30 flagship: 100 epochs, V2 pipeline, E00's exact recipe.
#
# The one pre-registered question: does the window-graph GAT fusion module,
# trained at E00's full budget, beat the IR-only reference E00
# (training-val 0.6306 / locked test 0.57828)?
#
# Comparability matrix (every row matches E00 except the intended variable):
#   data          full 17,962 paired train (E00: same images, IR only)
#   labels        thermal (*labelr) supervise training and validation
#   recipe        MuSGD, lr0 0.01, lrf 0.01 (= prefix_lrf(100)), momentum
#                 0.937, warmup_bias_lr 0.1, batch 16, seed 0, workers 8,
#                 patience 30
#   close_mosaic  10 — matches E00 (mosaic closes at ep90). NOT the probe
#                 default 0; passing 0 would silently change the schedule.
#   val           1,469-frame training validator, per epoch
#   variable      the fused architecture (dual stream + E30 fuse, d=128)
# Controls: E00 itself (same budget/recipe/seed), plus the 10-epoch
# inert-fuse control e00_fusedctl_v2_10ep (0.53625) for the early trajectory.
# Interpretable either way: a loss refutes mid-level graph fusion at full
# budget on this benchmark; a win goes to locked-test eval + bootstrap before
# any test claim (E22 capacity control before any "cross-modal information"
# claim).
#
# Usage: bash scripts/run_v2_e30_100ep.sh [JOBID]
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python
JOBID="${1:-10230}"
MAX_USED_MIB="${MAX_USED_MIB:-2500}"
DATA="configs/dronevehicle_v2_paired.yaml"
RGB="configs/dronevehicle_v2_rgb.yaml"
NAME="e30_v2_100ep"

# Fail before burning a GPU step on a typo (lesson from batch attempt 1).
for f in "$EXP/$DATA" "$EXP/$RGB" "$PY"; do
  [[ -e "$f" ]] || { echo "[e30flag] missing required path: $f"; exit 1; }
done

if [[ -f "$EXP/runs/${NAME}/DONE" ]]; then
  echo "[e30flag] SKIP: runs/${NAME}/DONE exists"
  exit 0
fi

# Solo rule: refuse to start while the GPU holds memory (postmortem B8 — a
# concurrent arm silently halves the batch and breaks the recipe).
used="$(srun --jobid="$JOBID" bash -c 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' | head -1 | tr -d ' ')"
if (( used > MAX_USED_MIB )); then
  echo "[e30flag] REFUSING to start: GPU already holds ${used} MiB (> ${MAX_USED_MIB})"
  exit 1
fi

echo "[e30flag] $(date -u) START ${NAME} (100 ep, close_mosaic 10, E30 d=128)"
rc=0
srun --jobid="$JOBID" bash -c \
  "cd '$EXP' && PYTHONPATH=src '$PY' scripts/train_unified_v2.py \
   --name '$NAME' --mode pair --fuse E30 --d 128 \
   --data '$DATA' --rgb-yaml '$RGB' \
   --epochs 100 --close-mosaic 10 --workers 8 \
   2>&1 | tee '$EXP/logs/v2_${NAME}.log'" || rc=$?
if (( rc != 0 )); then
  echo "[e30flag] $(date -u) FAILED ${NAME} (rc=${rc})"
  exit 1
fi
echo "[e30flag] $(date -u) DONE ${NAME}"
