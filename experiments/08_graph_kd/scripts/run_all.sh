#!/usr/bin/env bash
# Full experiment chain.  Run inside a live salloc allocation via srun.
set -uo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXP" || exit 1
bash scripts/make_configs.sh
bash scripts/train_teachers.sh
bash scripts/cache_teachers.sh
bash scripts/train_students.sh
bash scripts/evaluate_all.sh
echo "[$(date)] all phases finished"
