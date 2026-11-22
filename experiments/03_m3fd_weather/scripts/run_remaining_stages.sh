#!/usr/bin/env bash
# ============================================================================
# run_remaining_stages.sh -- every stage EXCEPT stage 5, with per-stage
# failure isolation.
#
# WHY THIS EXISTS
# ---------------
# run_all_methods.sh runs under `set -euo pipefail`, so the first failing command
# aborts the whole job. That has now cost two runs:
#
#   job 8735  died in stage 5  (3B training). Nothing after it ran.
#   job 8752  died in stage 6  (CycleGAN 3B training, refused by the provenance
#              guard). Stages 6 and 7 both still had work to do, and stage 7 --
#              the gap table, the only thing anyone reads -- never ran.
#
# Stages 3, 4, 6 and 7 have no ordering dependency on each other: 3 and 4 only
# need the split, 6 needs the clear weights from 3 plus the CycleGAN generators,
# 7 needs nothing and degrades gracefully on missing results. So a failure in one
# should not cost the others. This script runs each requested stage in isolation,
# records the outcome, and continues.
#
# Stage 5 is deliberately NOT here -- it has its own script,
# run_stage5_method2.sh, with fine-grained sub-step skipping so a rerun does not
# repeat half an hour of depth estimation.
#
# Usage:
#   bash experiments/03_m3fd_weather/scripts/run_remaining_stages.sh          # 3 4 6 7
#   bash experiments/03_m3fd_weather/scripts/run_remaining_stages.sh 6 7      # just these
#   STAGES="4 6" bash experiments/03_m3fd_weather/scripts/run_remaining_stages.sh
#
# Exit status: 0 only if every requested stage succeeded.
# ============================================================================

# Deliberately NOT `set -e`. A stage failing is data, not an abort: this script
# exists to keep going. `set -u` and `pipefail` stay on.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
EXP="$REPO/experiments/03_m3fd_weather"
CGAN="$EXP/method3_cyclegan"
STATE="$EXP/logs/state"
CLEAR_YAML="$EXP/data/split/m3fd_clear.yaml"
CLEAR_WEIGHTS="$EXP/runs/v12s_m3fd_clear/weights/best.pt"

# Must match run_all_methods.sh, or the two scripts disagree about which stages
# are complete and each invalidates the other's markers.
PIPELINE_GENERATION=5

# Positional args select stages; STAGES=... is still honoured as a default.
# These used to be ignored entirely, so `run_remaining_stages.sh 6 7` silently
# ran the full "3 4 6 7" list and spent its time re-running the clear baseline.
STAGES="${*:-${STAGES:-3 4 6 7}}"

mkdir -p "$STATE"
cd "$REPO"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '  WARN: %s\n' "$*"; }
die()  { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

if [ ! -f "$EXP/data/split/split_manifest.json" ]; then
    die "split_manifest.json missing -- run stage 1 (run_all_methods.sh) first"
fi
SPLIT_HASH="$("$PY" - "$EXP/data/split/split_manifest.json" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1]))["split_hash"])
EOF
)"
[ -n "$SPLIT_HASH" ] || die "could not read split_hash from the manifest"

say "M3FD remaining stages $(date)"
echo " repo:      $REPO"
echo " python:    $PY"
echo " split:     $SPLIT_HASH"
echo " stages:    $STAGES"
echo " job:       ${SLURM_JOB_ID:-none}"
echo "========================================================================"

FAILED=()
SKIPPED=()

# --- stage isolation ---------------------------------------------------------
# mark_stage writes the marker only on success, and deletes any stale marker
# first. That ordering matters: if a stage is killed mid-flight the marker must
# be absent, or the next run would skip work that never finished.
mark_stage() {
    printf 'stage=%s\npipeline=%s\nsplit_hash=%s\nfinished=%s\n' \
        "$1" "$PIPELINE_GENERATION" "$SPLIT_HASH" "$(date -Is)" \
        > "$STATE/$1.done"
    rm -f "$STATE/$1.FAILED"
}

run_stage() {
    local name="$1" fn="$2" rc
    rm -f "$STATE/$name.done"
    say "Stage $name"
    if "$fn"; then
        mark_stage "$name"
        printf '  OK    %s\n' "$name"
    else
        rc=$?
        printf 'stage=%s\npipeline=%s\nsplit_hash=%s\nfailed=%s\nexit=%s\n' \
            "$name" "$PIPELINE_GENERATION" "$SPLIT_HASH" "$(date -Is)" "$rc" \
            > "$STATE/$name.FAILED"
        printf '  FAIL  %s (exit %s) -- continuing to the next stage\n' "$name" "$rc"
        FAILED+=("$name")
    fi
}

# --- small shared helpers ----------------------------------------------------
assert_labels_found() {
    local name="$1" log
    log="$(find "$EXP/runs/$name" \( -name '*.log' -o -name 'log.txt' \) 2>/dev/null | head -1)"
    if [ -z "$log" ]; then
        warn "no log under runs/$name -- cannot verify labels were read"
        return 0
    fi
    if grep -qiE 'backgrounds' "$log"; then
        echo "  label check ($log):"
        grep -iE 'backgrounds' "$log" | tail -2 | sed 's/^/    /'
    else
        warn "'backgrounds' never appeared in $log -- inspect manually before trusting any mAP"
    fi
}

count_split() {   # count_split <yaml> <train|val|test> -> integer on stdout
    "$PY" -c '
import sys, pathlib
sys.path.insert(0, sys.argv[1])
import provenance
print(provenance.count_images(pathlib.Path(sys.argv[2]), sys.argv[3]))
' "$EXP" "$1" "$2" 2>/dev/null || echo 0
}

json_field() {     # json_field <file> <key> -> value, or empty if absent/unreadable
    "$PY" -c '
import json, sys
try:
    print(json.load(open(sys.argv[1])).get(sys.argv[2]) or "")
except Exception:
    print("")
' "$1" "$2"
}

# --- Stage 3: clear baseline -------------------------------------------------
# The single reference model. Every 3A number in the project is this weight file
# evaluated on a degraded test set, so nothing else can be interpreted without it.
stage3() {
    "$PY" "$EXP/train_m3fd.py" \
        --model weights/yolo12s.pt \
        --name v12s_m3fd_clear \
        --data "$CLEAR_YAML" \
        --epochs 100 --batch 64 --cache disk \
        --no-resume || return 1
    assert_labels_found v12s_m3fd_clear
    [ -f "$CLEAR_WEIGHTS" ] || { echo "  FATAL: $CLEAR_WEIGHTS missing"; return 1; }
    # train_m3fd.py writes this unless it was killed; the pipeline needs it as the
    # 3A reference, so a missing file is a stage failure, not a warning.
    [ -f "$EXP/results/v12s_m3fd_clear.json" ] \
        || { echo "  FATAL: results/v12s_m3fd_clear.json missing -- no 3A reference"; return 1; }
}

# --- Stage 4: Method 1, Protocol 3A -----------------------------------------
# No training. Six evals of the clear baseline on degraded val/test sets.
stage4() {
    rm -rf "$EXP/data/synthetic"
    "$PY" "$EXP/method1_classical/generate_weather_splits.py" || return 1
    for cond in fog rain; do
        for sev in light medium heavy; do
            local yaml="$EXP/data/synthetic/${cond}_${sev}/m3fd_${cond}_${sev}.yaml"
            [ -f "$yaml" ] || { echo "  FATAL: $yaml not generated"; return 1; }
            echo "  evaluating ${cond}_${sev} ..."
            "$PY" "$EXP/eval_m3fd.py" \
                --weights "$CLEAR_WEIGHTS" \
                --data "$yaml" \
                --name "${cond}_${sev}" \
                --out "$EXP/method1_classical/results_${cond}_${sev}.json" || return 1
        done
    done
}

# --- Stage 6: Method 3, Protocols 3A + 3B ------------------------------------
# run_eval_3ab.sh translates both generators, runs the two 3A evals, then trains
# two 3B models. The dataset YAMLs it builds are now stamped with the split
# (make_cyclegan_dataset.write_yaml), which is what job 8752 tripped over.
stage6() {
    [ -f "$CLEAR_WEIGHTS" ] || { echo "  FATAL: stage 3 weights missing ($CLEAR_WEIGHTS)"; return 1; }
    for r in thermal_fog thermal_rain; do
        [ -f "$CGAN/runs/$r/G_AB.pt" ] || { echo "  FATAL: $r/G_AB.pt missing -- run stage 2"; return 1; }
        [ -f "$CGAN/runs/$r/DONE" ]   || { echo "  FATAL: $r never finished training"; return 1; }
        [ -f "$CGAN/runs/$r/SOURCE.json" ] \
            || { echo "  FATAL: $r/SOURCE.json missing -- cannot prove the generator saw this split"; return 1; }
    done
    bash "$CGAN/scripts/run_eval_3ab.sh" || return 1
    assert_labels_found v12s_m3fd_3b_cgan_fog
}

# --- Stage 7: unified gap table ----------------------------------------------
# The summary block is extracted from run_all_methods.sh rather than copied, so
# there is exactly one definition of the table. If the extraction finds nothing
# the stage fails loudly instead of printing an empty table.
stage7() {
    local py_src
    # Collect only the lines between the heredoc opener and its `PYEOF`
    # terminator. Matching the terminator literally is safe: every other heredoc
    # in run_all_methods.sh closes on `EOF`, not `PYEOF`.
    py_src="$(awk '/^"\$PY" - <<.PYEOF.$/{f=1;next} f&&/^PYEOF$/{exit} f' \
                  "$EXP/scripts/run_all_methods.sh")"
    [ -n "$py_src" ] || { echo "  FATAL: could not extract the summary block from run_all_methods.sh"; return 1; }
    printf '%s\n' "$py_src" | "$PY" - || return 1
}

# --- dispatch ----------------------------------------------------------------
for s in $STAGES; do
    case "$s" in
        3) run_stage clear_baseline stage3 ;;
        4) run_stage method1       stage4 ;;
        6) run_stage method3       stage6 ;;
        7) run_stage summary       stage7 ;;
        5) SKIPPED+=("5 (use run_stage5_method2.sh)") ;;
        *) echo "unknown stage '$s' (want 3, 4, 6 or 7)" >&2 ;;
    esac
done

say "done $(date)"
for s in "${SKIPPED[@]:-}"; do [ -n "$s" ] && echo "  delegated: $s"; done
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "  all requested stages OK"
    exit 0
fi
echo "  FAILED: ${FAILED[*]}"
echo "  fix and rerun this script; completed stages are marker-skipped."
exit 1