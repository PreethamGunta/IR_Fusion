#!/usr/bin/env bash
# Copy the non-git assets experiment 10 needs from this node to the target node.
#
#   bash scripts/sync_assets.sh --check
#   bash scripts/sync_assets.sh --to yash_gupta@node:/abs/path/to/IR_Fusion
#
# Why this exists: AGENTS.md forbids committing *.pt / results/ to git, so the
# teacher, pretrained students, locked-test GT and reference predictions travel
# out-of-band. `--to` uses `rsync -R`, preserving the repo-relative paths.
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$EXP/../.." && pwd)"
E09="$REPO/experiments/09_graph_fusion"

FILES=(
  "models/pretrained/yolo12s.pt"
  "models/pretrained/yolo12n.pt"
  "experiments/09_graph_fusion/runs/e30_v2_100ep/weights/best.pt"
  "experiments/09_graph_fusion/runs/e00_ir/weights/best.pt"
  "experiments/09_graph_fusion/results/gt/dv_v2_ir_test.json"
  "experiments/09_graph_fusion/results/preds/e00_ir_test.json"
  "experiments/09_graph_fusion/results/preds/e30_v2_100ep_test.json"
  "experiments/09_graph_fusion/results/e00_ir_test_metrics.json"
  "experiments/09_graph_fusion/results/e30_v2_100ep_test_metrics.json"
  "experiments/09_graph_fusion/results/bootstrap_flagships_vs_e00.json"
  "experiments/09_graph_fusion/runs/e00_ir/results.csv"
  "experiments/09_graph_fusion/runs/e30_v2_100ep/results.csv"
)

check() {
  local rc=0
  for rel in "${FILES[@]}"; do
    local f="$REPO/$rel"
    if [[ -f "$f" ]]; then
      printf "OK   %8s  %s\n" "$(du -h "$f" | cut -f1)" "$rel"
    else
      printf "MISS %8s  %s\n" "-" "$rel"; rc=1
    fi
  done
  return $rc
}

to=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) check; exit $? ;;
    --to) to="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

[[ -n "$to" ]] || { echo "usage: $0 --to user@host:/abs/path/to/IR_Fusion | --check"; exit 2; }
check || { echo "ERROR: missing assets above; copy them first"; exit 1; }

cmd=(rsync -avR --progress)
for rel in "${FILES[@]}"; do cmd+=("$rel"); done
cmd+=("$to/")
echo "+ ${cmd[*]}"
cd "$REPO"
"${cmd[@]}"
echo "[sync] done. On the target node run: bash experiments/10_ir_distillation/scripts/run_node2_queue.sh"
