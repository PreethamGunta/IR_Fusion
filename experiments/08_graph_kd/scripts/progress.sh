#!/usr/bin/env bash
# One-screen progress report for experiment 08.
set -uo pipefail
EXP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$EXP" || exit 1
printf "%-22s %6s %8s %8s %s\n" run epoch mAP50-95 status
for n in ir_teacher rgb_teacher fused_teacher a0_gt a2_ir_control a1_pred_fix a2_node \
         a3_node_edge a5_fgw a7_reliability b_consensus a4_sinkhorn d_gt_anchor \
         e_condense f_decomposed c_three_teacher a6_partial; do
  f="runs/$n/results.csv"
  if [ -f "$f" ]; then
    read -r ep map <<<"$(python3 - "$f" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
if rows:
    r=rows[-1]; print(r['epoch'], r['metrics/mAP50-95(B)'])
PY
)"
  else
    ep="-"; map="-"
  fi
  st="-"
  [ -f "runs/$n/DONE" ] && st="DONE"
  [ -f "results/${n}_test.json" ] && st="EVAL"
  printf "%-22s %6s %8s %8s %s\n" "$n" "$ep" "$map" "" "$st"
done
echo
echo "teacher caches:"
for t in ir rgb fused; do
  if [ -f "data/cache/$t/meta.json" ]; then echo "  $t ok"; else echo "  $t missing"; fi
done
echo "results: $(ls results/*_test.json 2>/dev/null | wc -l)"
