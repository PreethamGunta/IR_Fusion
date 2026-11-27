#!/usr/bin/env bash
# Generic single-arm driver for follow-up experiments (V2 pipeline).
#
# Usage: bash scripts/run_v2_arm.sh JOBID NAME FUSE EPOCHS [extra launcher args...]
# Example (warm-start fine-tune):
#   bash scripts/run_v2_arm.sh 10312 e30ws_v2_50ep E30 50 \
#     --d 128 --lr0 0.002 --lrf 0.1 --patience 20 --close-mosaic 5 \
#     --fuse-warmup 2 --fuse-ramp 2 \
#     --init-ir runs/e00_ir/weights/best.pt \
#     --init-rgb runs/e01_rgb/weights/best.pt
#
# Guards: DONE-marker skip, GPU-free check (solo rule), loud failure.
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python
JOBID="${1:?usage: run_v2_arm.sh JOBID NAME FUSE EPOCHS [args...]}"
NAME="${2:?name}"
FUSE="${3:?fuse}"
EPOCHS="${4:?epochs}"
shift 4
EXTRA="$*"

DATA="configs/dronevehicle_v2_paired.yaml"
RGB="configs/dronevehicle_v2_rgb.yaml"
PROJECT="${PROJECT:-runs_g2}"
MAX_USED_MIB="${MAX_USED_MIB:-2500}"
CPUS="${CPUS:-8}"

for f in "$EXP/$DATA" "$EXP/$RGB" "$PY"; do
  [[ -e "$f" ]] || { echo "[arm] missing required path: $f"; exit 1; }
done

if [[ -f "$EXP/$PROJECT/${NAME}/DONE" ]]; then
  echo "[arm] SKIP: $PROJECT/${NAME}/DONE exists"
  exit 0
fi

# Busy check: only *our* user's GPU memory counts (other users' jobs on other
# GPUs must not block us). Explicit --ntasks=1 and --cpus-per-task so this
# step does not inherit the allocation's task count (a 32-CPU allocation
# otherwise turns one command into four parallel tasks) and does not
# deadlock behind other running steps.
used="$(srun --jobid="$JOBID" --ntasks=1 --cpus-per-task=1 bash -c '
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null |
  while IFS=, read -r pid mem; do
    pid=${pid// /}; u=$(ps -o user= -p "$pid" 2>/dev/null | tr -d " ")
    [ "$u" = "$USER" ] && echo "${mem// /}"
  done' 2>/dev/null | awk '{s+=$1} END {print s+0}')"
if (( used > MAX_USED_MIB )); then
  echo "[arm] REFUSING to start: our user already holds ${used} MiB on a GPU (> ${MAX_USED_MIB})"
  exit 1
fi

echo "[arm] $(date -u) START ${NAME} fuse=${FUSE} epochs=${EPOCHS} project=${PROJECT} extras: ${EXTRA}"
rc=0
srun --jobid="$JOBID" --ntasks=1 --overlap --cpus-per-task="$CPUS" bash -c \
  "cd '$EXP' && PYTHONPATH=src:testbed_gfyolo '$PY' scripts/train_unified_v2.py \
   --name '$NAME' --mode pair --fuse '$FUSE' --project '$PROJECT' \
   --data '$DATA' --rgb-yaml '$RGB' \
   --epochs $EPOCHS --workers 8 $EXTRA \
   2>&1 | tee '$EXP/logs/v2_${NAME}.log'" || rc=$?
if (( rc != 0 )); then
  echo "[arm] $(date -u) FAILED ${NAME} (rc=${rc})"
  exit 1
fi
echo "[arm] $(date -u) DONE ${NAME}"
