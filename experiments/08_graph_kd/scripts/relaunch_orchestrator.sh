#!/usr/bin/env bash
# Relaunch the resume-safe orchestrator in a fresh allocation.
# Usage: relaunch_orchestrator.sh <jobid> [logname]
set -uo pipefail
JOB="${1:?job id required}"
LOG="${2:-logs/parallel_next.log}"
EXP="/home2/home/preetham_g/gate_ir_release/experiments/08_graph_kd"
PY="/home2/home/preetham_g/.conda/envs/gateir/bin/python"
cd "$EXP" || exit 1
tmux send-keys -t srun_s "cd $EXP && srun --jobid=$JOB --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=32 bash -lc \"$PY scripts/parallel_orchestrator.py\" > $LOG 2>&1" Enter
echo "launched orchestrator on job $JOB -> $LOG"
