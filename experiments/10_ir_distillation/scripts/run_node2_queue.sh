#!/usr/bin/env bash
# Priority queue N0..N6 for experiment 10 (IR-only distillation).
#
#   bash scripts/run_node2_queue.sh                 # direct python (interactive GPU)
#   bash scripts/run_node2_queue.sh --srun-jobid 12345
#   bash scripts/run_node2_queue.sh --stages N0,N1 --dry-run
#
# The default is *direct* execution: the target node already has an interactive
# srun shell with a GPU (tmux). Do not submit a new allocation from here.
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
E09="$REPO/experiments/09_graph_fusion"
# Interpreter: pass PY=/path/to/env/bin/python on the target node. The dev
# node's path is only a fallback when it exists.
PY="${PY:-}"
if [[ -z "$PY" ]]; then
  if [[ -x /home2/home/preetham_g/.conda/envs/gateir/bin/python ]]; then
    PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python
  else
    PY=python
  fi
fi
STUDENTS="$REPO/models/pretrained"
TEACHER="$E09/runs/e30_v2_100ep/weights/best.pt"
GT="$E09/results/gt/dv_v2_ir_test.json"
DV_ROOT="${DV_ROOT:-$E09/data/dronevehicle_v2/yolo}"
RAW_DV="${RAW_DV:-$REPO/data/dronevehicle/raw}"

STAGES="N0,N1,N2,N3,N4,N5,N6"
JOBID=""
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages) STAGES="$2"; shift 2 ;;
    --srun-jobid) JOBID="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --python) PY="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

cd "$EXP"
QD="$EXP/logs/queue"
mkdir -p "$QD" "$EXP/results/preds" "$EXP/results"
LOG="$QD/queue_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[n2 $(date -u +%H:%M:%S)] $*"; }
done_p() { [[ -f "$QD/$1.done" ]]; }
mark() { (( DRY )) && return 0; date -u > "$QD/$1.done"; }

run() {
  if (( DRY )); then
    if [[ -n "$JOBID" ]]; then
      echo "+ srun --jobid=$JOBID --ntasks=1 --overlap --cpus-per-task=8 $*"
    else
      echo "+ $*"
    fi
    return 0
  fi
  if [[ -n "$JOBID" ]]; then
    srun --jobid="$JOBID" --ntasks=1 --overlap --cpus-per-task=8 "$@"
  else
    "$@"
  fi
}

preflight() {
  say "python: $PY"
  [[ -x "$PY" ]] || { say "ERROR: interpreter not executable: $PY"; exit 1; }
  for f in "$TEACHER" "$GT" "$STUDENTS/yolo12s.pt" "$STUDENTS/yolo12n.pt"; do
    [[ -f "$f" ]] || { say "ERROR: missing $f"; exit 1; }
  done
  if [[ ! -d "$DV_ROOT/ir/images/train" ]]; then
    say "dataset missing at $DV_ROOT; building it from $RAW_DV (one-time)"
    run "$PY" "$EXP/scripts/build_dv_from_raw.py" --raw "$RAW_DV" \
      --out "$(dirname "$DV_ROOT")"
  fi
  (( DRY )) || [[ -d "$DV_ROOT/ir/images/train" ]] || {
    say "ERROR: DroneVehicle build failed or DV_ROOT is wrong: $DV_ROOT"; exit 1; }
  run "$PY" scripts/make_data_configs.py --root "$DV_ROOT"
  if (( ! DRY )); then
    "$PY" - <<'EOF'
import torch, sys
print("[n2] torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "devices", torch.cuda.device_count())
if not torch.cuda.is_available():
    sys.exit("ERROR: no CUDA device visible")
EOF
  fi
}

train_student() {  # name student_file kd seed extra...
  local name=$1 student=$2 kd=$3 seed=$4; shift 4
  if done_p "train_$name"; then say "skip train $name (marker)"; return 0; fi
  say "TRAIN $name (student=$student kd=$kd seed=$seed)"
  run "$PY" scripts/train_student_kd.py \
    --name "$name" --student-cfg "$STUDENTS/$student" \
    --teacher-ckpt "$TEACHER" --kd-mode "$kd" --seed "$seed" \
    --workers 8 "$@"
  mark "train_$name"
}

export_student() {  # name split
  local name=$1 split=${2:-test}
  local out="$EXP/results/preds/${name}_${split}.json"
  if [[ -f "$out" ]]; then say "skip export $name/$split"; return 0; fi
  run "$PY" "$EXP/scripts/export_ir.py" \
    --weights "$EXP/runs/$name/weights/best.pt" \
    --dataset-root "$DV_ROOT" --split "$split" --out "$out"
}

eval_student() {  # name
  local name=$1
  if [[ ! -f "$EXP/runs/$name/weights/best.pt" ]]; then
    say "SKIP eval $name (no best.pt)"; return 0
  fi
  export_student "$name" test
  local best
  best="$("$PY" "$EXP/scripts/best_val.py" "$EXP/runs/$name/results.csv")"
  if [[ "$best" == "0" ]]; then say "SKIP eval $name (no val metric)"; return 0; fi
  if [[ ! -f "$EXP/results/LOCK_${name}.json" ]]; then
    run "$PY" "$E09/scripts/make_lock.py" --exp "$name" \
      --config "$EXP/runs/$name/ref_meta.json" \
      --checkpoint "$EXP/runs/$name/weights/best.pt" \
      --val-metric "$best" --results-dir "$EXP/results"
  fi
  if [[ ! -f "$EXP/results/${name}_test_metrics.json" ]]; then
    run "$PY" "$E09/scripts/final_eval.py" --exp "$name" --gt "$GT" \
      --preds "$EXP/results/preds/${name}_test.json" \
      --out "$EXP/results/${name}_test_metrics.json" \
      --results-dir "$EXP/results"
  fi
  for ref in e00 e30; do
    local out="$EXP/results/bootstrap_${name}_vs_${ref}.json"
    [[ -f "$out" ]] && continue
    if [[ "$ref" == "e00" ]]; then
      refpred="$E09/results/preds/e00_ir_test.json"; refname=e00
    else
      refpred="$E09/results/preds/e30_v2_100ep_test.json"; refname=e30
    fi
    run "$PY" "$E09/scripts/paired_bootstrap.py" --gt "$GT" --ref "$refname" \
      --model "$refname=$refpred" \
      --model "$name=$EXP/results/preds/${name}_test.json" \
      --n 500 --out "$out"
  done
  say "evaluated $name (val $best)"
}

stage_N0() {
  say "N0: unit tests + 2-epoch smoke"
  run env "PYTHONPATH=$EXP/src:$E09/src:$E09" "$PY" -m pytest "$EXP/tests/test_irkd.py" -q
  train_student smoke_12s_decomp yolo12s.pt decomposed 0 \
    --epochs 2 --close-mosaic 0 --patience 2 --fresh
  export_student smoke_12s_decomp val
  if (( ! DRY )); then
    "$PY" - "$EXP/results/preds/smoke_12s_decomp_val.json" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
print(f"[n2] smoke val predictions: {len(p)}")
if len(p) == 0:
    raise SystemExit("ERROR: smoke export produced zero predictions")
EOF
  fi
  mark N0
  say "N0 done"
}

stage_N1() { say "N1: 12s decomposed";  train_student student_12s_decomp yolo12s.pt decomposed 0 --epochs 100 --close-mosaic 10; mark N1; }
stage_N2() { say "N2: 12n no-KD";      train_student student_12n_gt yolo12n.pt none 0 --epochs 100 --close-mosaic 10; mark N2; }
stage_N3() { say "N3: 12n decomposed"; train_student student_12n_decomp yolo12n.pt decomposed 0 --epochs 100 --close-mosaic 10; mark N3; }
stage_N4() { say "N4: 12s direct";     train_student student_12s_direct yolo12s.pt direct 0 --epochs 100 --close-mosaic 10; mark N4; }

stage_N5() {
  say "N5: evaluation + profiling"
  for n in student_12s_decomp student_12n_gt student_12n_decomp student_12s_direct; do
    eval_student "$n"
  done
  run "$PY" scripts/profile_students.py --out "$EXP/results/profiling_students.json"
  run "$PY" paper/make_figures.py
  mark N5
  say "N5 done"
}

stage_N6() { say "N6: 12s decomposed seed 1"; train_student student_12s_decomp_seed1 yolo12s.pt decomposed 1 --epochs 100 --close-mosaic 10; mark N6; }

preflight
say "queue start: stages=$STAGES jobid=${JOBID:-none} dry_run=$DRY"
IFS=',' read -r -a WANT <<< "$STAGES"
for s in "${WANT[@]}"; do
  case "$s" in
    N0) stage_N0 ;;
    N1) stage_N1 ;;
    N2) stage_N2 ;;
    N3) stage_N3 ;;
    N4) stage_N4 ;;
    N5) stage_N5 ;;
    N6) stage_N6 ;;
    *) echo "unknown stage $s"; exit 2 ;;
  esac
done
say "queue finished"
