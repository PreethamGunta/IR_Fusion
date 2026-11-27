#!/usr/bin/env bash
# V2 fusion screen — 10 epochs per arm, sequential, solo on the GPU.
#
# Every arm runs as its own `srun` step inside an existing allocation, so no
# training ever touches the login node and each arm is separately visible in
# `squeue -s`.
#
# Why sequential-solo: two concurrent graph/attention arms previously forced an
# Ultralytics OOM fallback that silently halved the batch (16 -> 8) and broke
# the recipe (postmortem B8). `guard_gpu` refuses to start an arm while the GPU
# already holds memory, which turns that silent failure into a loud one.
#
# Why the control first: the fused arms are a *different architecture* (dual
# backbone, 6-channel input, adapted IR stem), so comparing them to the IR-only
# run conflates "fusion" with "the 6-channel dual-stream init". The control runs
# the same architecture with the fuse held at alpha=0 for every epoch
# (`--fuse-warmup 10`), which is provably inert. Fusion is then measured against
# the architecture, not against E00.
#
# Order = priority for the research question:
#   1. control   architecture-matched IR-only reference
#   2. E14       IR-anchored gated residual (starts as IR-only; most likely to
#                preserve IR performance, cheapest to defend)
#   3. E30       window-graph GAT (the rung the experiment is named for)
#   4. E10       concat + 1x1 conv (simple learned-fusion baseline)
#   5. E21       bidirectional cross-attention (attention representative)
#   6. E35       semantic-kNN graph (core novelty) — only if the window allows
#
# These are a PLUMBING + GROSS-FAILURE screen, not a ranking: at 10 epochs no
# rung has converged, and the observed single-seed spread is 1.4-3.7 pp. Do not
# quote a rank from this batch; a ranking needs >=40 epochs.
#
# Usage: bash scripts/run_v2_fusion_10ep.sh [JOBID]
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python
JOBID="${1:-10187}"
EPOCHS="${EPOCHS:-10}"
MAX_USED_MIB="${MAX_USED_MIB:-2500}"

DATA="configs/dronevehicle_v2_paired.yaml"
RGB="configs/dronevehicle_v2_rgb.yaml"

# Fail before burning a GPU step on a typo (this exact class of error already
# cost one full batch: the driver was missing --data and every arm died in <1 s).
for f in "$EXP/$DATA" "$EXP/$RGB" "$PY"; do
  [[ -e "$f" ]] || { echo "[v2fusion] missing required path: $f"; exit 1; }
done

guard_gpu() {
  local used
  used="$(srun --jobid="$JOBID" bash -c 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' \
          | head -1 | tr -d ' ')"
  if (( used > MAX_USED_MIB )); then
    echo "[v2fusion] REFUSING to start $1: GPU already holds ${used} MiB" \
         "(> ${MAX_USED_MIB}); a concurrent arm silently changes the recipe"
    exit 1
  fi
}

# Wait (up to 90 min) for any in-flight run to release the GPU before the batch
# starts. Script-side wait, not agent-side polling.
wait_gpu_free() {
  local used i
  for i in $(seq 1 90); do
    used="$(srun --jobid="$JOBID" bash -c 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' \
            | head -1 | tr -d ' ')"
    if (( used <= MAX_USED_MIB )); then
      echo "[v2fusion] $(date -u +%H:%M:%S) GPU free (${used} MiB) — starting batch"
      return 0
    fi
    echo "[v2fusion] $(date -u +%H:%M:%S) waiting for GPU (${used} MiB in use)"
    sleep 60
  done
  echo "[v2fusion] GPU still busy after 90 min — aborting without launching"
  exit 1
}

run() {
  local name="$1"; local eps="$2"; shift 2
  if [[ -f "$EXP/runs/${name}/DONE" ]]; then
    echo "[v2fusion] SKIP ${name} (DONE exists)"
    return 0
  fi
  guard_gpu "$name"
  echo "[v2fusion] $(date -u +%H:%M:%S) START ${name} (${eps} ep) $*"
  local rc=0
  srun --jobid="$JOBID" bash -c \
    "cd '$EXP' && PYTHONPATH=src '$PY' scripts/train_unified_v2.py \
     --name '$name' --mode pair --data '$DATA' --rgb-yaml '$RGB' \
     --epochs '$eps' --workers 8 $* \
     2>&1 | tee '$EXP/logs/v2_${name}.log'" || rc=$?
  if (( rc != 0 )); then
    echo "[v2fusion] $(date -u +%H:%M:%S) FAILED ${name} (rc=${rc}) — stopping batch"
    exit 1
  fi
  echo "[v2fusion] $(date -u +%H:%M:%S) DONE ${name}"
}

wait_gpu_free

# Smoke: the pair path (V2GFTrainer) has never trained end-to-end. One epoch of
# the cheapest rung proves dataset pairing, the 6-channel forward, the paired
# validator and the fused loss all work before committing ~3 h.
run smoke_pair_v2_1ep 1 --fuse E10

run e00_fusedctl_v2_10ep "$EPOCHS" --fuse E30 --fuse-warmup 10
run e14_v2_10ep          "$EPOCHS" --fuse E14
run e30_v2_10ep          "$EPOCHS" --fuse E30 --d 128
run e10_v2_10ep          "$EPOCHS" --fuse E10
run e21_v2_10ep          "$EPOCHS" --fuse E21 --d 128
run e35_v2_10ep          "$EPOCHS" --fuse E35 --d 128

echo "[v2fusion] $(date -u +%H:%M:%S) BATCH COMPLETE"
