#!/usr/bin/env bash
# ============================================================================
# run_stage5_method2.sh -- Stage 5 (Method 2, depth-aware) alone, with
# per-sub-step skipping driven by artefact inspection rather than a blanket wipe.
#
# WHY THIS IS SEPARATE
# --------------------
# run_all_methods.sh opens stage 5 with `rm -rf data/synthetic_depth
# data/synthetic_depth_3b data/3b_fog_heavy`. That is correct for a cold start
# and wasteful for a warm one: every rerun redoes ~30 minutes of Depth-Anything
# inference over 1680 frames, plus a second full pass over train+val+test, even
# when the only thing missing is one evaluation. Jobs 8735 and 8752 both lost
# their remaining work here.
#
# So each of the eight sub-steps below asks "is this specific artefact complete
# and traceable to the current split?" and skips if so. A sub-step is only
# redone when its own output is absent, short, or carries a different split hash.
# Nothing here deletes a directory unless that directory's own sub-step is being
# redone, and then only that one.
#
# SUB-STEPS
#   gen3a_<cond>_<sev>   apply_weather_depth.py  val+test -> data/synthetic_depth
#   eval3a_<cond>_<sev>  eval_m3fd.py with the clear baseline
#   gen3b                apply_weather_depth.py  train+val+test -> data/synthetic_depth_3b
#   yaml3b               make_3b_dataset.py -> data/3b_fog_heavy/m3fd_3b.yaml
#   train3b              train_m3fd.py -> runs/v12s_m3fd_3b_fog_heavy
#   eval3b               eval_m3fd.py on the fog_heavy test set
#
# Usage:
#   bash experiments/03_m3fd_weather/scripts/run_stage5_method2.sh            # skip what is done
#   bash experiments/03_m3fd_weather/scripts/run_stage5_method2.sh --force    # redo everything
#   FORCE=1 bash experiments/03_m3fd_weather/scripts/run_stage5_method2.sh
#   ONLY=gen3b,train3b,eval3b bash experiments/03_m3fd_weather/scripts/run_stage5_method2.sh
# ============================================================================

# No `set -e`: a failed sub-step should be reported and stop the stage, but the
# summary of what did and did not run still has to print.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
EXP="$REPO/experiments/03_m3fd_weather"
STATE="$EXP/logs/state"
CLEAR_YAML="$EXP/data/split/m3fd_clear.yaml"
CLEAR_WEIGHTS="$EXP/runs/v12s_m3fd_clear/weights/best.pt"

DEPTH_3A="$EXP/data/synthetic_depth"          # 3A evaluation sets (val + test)
DEPTH_3B="$EXP/data/synthetic_depth_3b"       # 3B training sets (train + val + test)
YAML_3B="$EXP/data/3b_fog_heavy/m3fd_3b.yaml"
RUN_3B="$EXP/runs/v12s_m3fd_3b_fog_heavy"
RES_3B_TRAIN="$EXP/results/v12s_m3fd_3b_fog_heavy.json"
RES_3B_EVAL="$EXP/method2_depth/results_3b_fog_heavy.json"

SEV_3A="medium heavy"       # 3A sweeps only medium + heavy (matches the pipeline)
# Read the inherited values before defaulting. `FORCE=0` on its own line wiped
# any exported FORCE/PLAN first, so the FORCE=1 form documented in the header
# below was silently ignored and every step ran under FORCE=0.
FORCE="${FORCE:-0}"
PLAN="${PLAN:-0}"
case "${1:-}" in
    --force) FORCE=1 ;;
    --plan)  PLAN=1 ;;
esac
ONLY="${ONLY:-}"

mkdir -p "$STATE"
cd "$REPO"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
die()  { printf 'FATAL: %s\n' "$*" >&2; exit 1; }

[ -f "$EXP/data/split/split_manifest.json" ] \
    || die "no split -- run stage 1 first"
SPLIT_HASH="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["split_hash"])' \
             "$EXP/data/split/split_manifest.json")"
[ -n "$SPLIT_HASH" ] || die "could not read split_hash"
[ -f "$CLEAR_WEIGHTS" ] || die "stage 3 weights missing ($CLEAR_WEIGHTS) -- run stage 3 first"

want() {   # want <substep-name>
    # ONLY, when given, is authoritative. It used to be checked only when
    # FORCE=0, so `FORCE=1 ONLY=eval3a_fog_medium` selected *every* sub-step and
    # only the completeness gates happened to hold the expensive ones back.
    if [ -n "$ONLY" ]; then
        case ",$ONLY," in *",$1,"*) return 0 ;; esac
        return 1
    fi
    return 0
}

# --- inspection helpers ------------------------------------------------------
count_split() {   # count_split <yaml> <train|val|test>
    "$PY" -c '
import sys, pathlib
sys.path.insert(0, sys.argv[1])
import provenance
print(provenance.count_images(pathlib.Path(sys.argv[2]), sys.argv[3]))
' "$EXP" "$1" "$2" 2>/dev/null || echo 0
}

json_field() {
    "$PY" -c '
import json, sys
try:
    print(json.load(open(sys.argv[1])).get(sys.argv[2]) or "")
except Exception:
    print("")
' "$1" "$2"
}

hash_matches() {  # hash_matches <file.json>
    [ "$FORCE" = "0" ] || return 1
    [ "$(json_field "$1" split_hash)" = "$SPLIT_HASH" ]
}

weights_match() {   # weights_match <result.json> <weights.pt>
    # A result is only meaningful for the weights it was actually computed from.
    # eval_m3fd.py records weights_sha256; comparing it against the checkpoint on
    # disk catches the case where a result survives a retrain of its own model --
    # which is exactly how a 4-epoch diagnostic checkpoint was about to be
    # reported as a trained 3B model.
    [ "$FORCE" = "0" ] || return 1
    local recorded actual
    recorded="$(json_field "$1" weights_sha256)"
    [ -n "$recorded" ] || return 1
    actual="$("$PY" -c '
import sys, pathlib
sys.path.insert(0, sys.argv[1])
import provenance
print(provenance.sha256_file(pathlib.Path(sys.argv[2])) or "")
' "$EXP" "$2" 2>/dev/null)"
    [ -n "$actual" ] && [ "$recorded" = "$actual" ]
}

skipping() { printf '  skip  %-22s %s\n' "$1" "$2"; }
doing()    { printf '  RUN   %-22s %s\n' "$1" "$2"; }

say "Stage 5 — Method 2 (depth-aware) $(date)"
echo " repo:    $REPO"
echo " split:   $SPLIT_HASH"
echo " force:   $FORCE"
echo " only:    ${ONLY:-<all>}"
echo "========================================================================"

FAILED=()
DONE_STEPS=()
SKIP_STEPS=()
PLAN_RUN=()

step() {   # step <name> <description> [command ...]
    # The 3A sub-steps are named gen3a_<cond>_<sev> / eval3a_<cond>_<sev> for
    # logging, but their workers are make_gen3a / run_eval3a, which take the
    # condition and severity as arguments. Calling "$name" therefore failed with
    # 127 for every 3A sub-step and only the unparameterised ones (gen3b,
    # yaml3b, train3b, eval3b) could ever run. Pass the worker explicitly.
    local name="$1" desc="$2"
    shift 2
    doing "$name" "$desc"
    if [ "$PLAN" = "1" ]; then
        PLAN_RUN+=("$name")
        return 0
    fi
    if [ "$#" -gt 0 ]; then "$@"; else "$name"; fi && { DONE_STEPS+=("$name"); return 0; }
    local rc=$?
    FAILED+=("$name($rc)")
    printf '  FAIL  %-22s exit=%s\n' "$name" "$rc"
    return 1
}

# ── 3A generation ────────────────────────────────────────────────────────────
# Complete = the YAML exists, val AND test both have images, and generation.json
# is stamped with the split that is live now. A dataset left over from a previous
# split fails the hash test and is regenerated.
gen_is_complete() {   # gen_is_complete <dest_root> <cond> <sev> <splits...>
    local root="$1" cond="$2" sev="$3"; shift 3
    local dir="$root/${cond}_${sev}"
    local yaml="$dir/m3fd_${cond}_${sev}.yaml"
    [ -f "$yaml" ] || return 1
    hash_matches "$dir/generation.json" || return 1
    local sp
    for sp in "$@"; do
        [ "$(count_split "$yaml" "$sp")" -gt 0 ] || return 1
    done
    return 0
}

make_gen3a() {   # make_gen3a <cond> <sev>
    local cond="$1" sev="$2"
    local dir="$DEPTH_3A/${cond}_${sev}"
    # Targeted wipe: only this condition, and only because we decided to redo it.
    rm -rf "$dir"
    "$PY" "$EXP/method2_depth/apply_weather_depth.py" \
        --src "$EXP/data/split/IR" \
        --dest "$DEPTH_3A" \
        --splits val test --condition "$cond" --severity "$sev" || return 1
    gen_is_complete "$DEPTH_3A" "$cond" "$sev" val test \
        || { echo "  generation finished but the dataset is still incomplete"; return 1; }
}

eval3a_is_complete() {   # eval3a_is_complete <cond> <sev>
    local res="$EXP/method2_depth/results_${1}_${2}_depth.json"
    [ -f "$res" ] || return 1
    hash_matches "$res" || return 1
    # Also tied to the specific baseline checkpoint: retraining stage 3 must
    # invalidate every 3A number computed from the old weights.
    weights_match "$res" "$CLEAR_WEIGHTS" || return 1
    [ "$(json_field "$res" n_images)" -gt 0 ] 2>/dev/null || return 1
    return 0
}

run_eval3a() {   # run_eval3a <cond> <sev>
    local cond="$1" sev="$2"
    "$PY" "$EXP/eval_m3fd.py" \
        --weights "$CLEAR_WEIGHTS" \
        --data "$DEPTH_3A/${cond}_${sev}/m3fd_${cond}_${sev}.yaml" \
        --name "${cond}_${sev}_depth" \
        --out "$EXP/method2_depth/results_${cond}_${sev}_depth.json" || return 1
    eval3a_is_complete "$cond" "$sev"
}

# ── 3B ───────────────────────────────────────────────────────────────────────
# The 3B dataset gets its own destination (data/synthetic_depth_3b) so it cannot
# overwrite the 3A YAML the way it used to, which silently repointed `test:` from
# the 600-frame test set to the 240-frame val set.
gen3b_is_complete() {
    local yaml="$DEPTH_3B/fog_heavy/m3fd_fog_heavy.yaml"
    [ -f "$yaml" ] || return 1
    hash_matches "$DEPTH_3B/fog_heavy/generation.json" || return 1
    local sp
    for sp in train val test; do
        [ "$(count_split "$yaml" "$sp")" -gt 0 ] || return 1
    done
    return 0
}

gen3b() {
    rm -rf "$DEPTH_3B/fog_heavy"
    "$PY" "$EXP/method2_depth/apply_weather_depth.py" \
        --src "$EXP/data/split/IR" \
        --dest "$DEPTH_3B" \
        --splits train val test --condition fog --severity heavy || return 1
    gen3b_is_complete || { echo "  3B generation finished but the dataset is incomplete"; return 1; }
}

yaml3b_is_complete() {
    [ -f "$YAML_3B" ] || return 1
    hash_matches "$EXP/data/3b_fog_heavy/split_manifest.json" || return 1
    # train must be the clear set PLUS the foggy copy: 2520 + 2520. Anything at
    # or below the clear-only count means the second directory is missing and the
    # model would silently train on clear data only.
    local ntr nva nte
    ntr="$(count_split "$YAML_3B" train)"
    nva="$(count_split "$YAML_3B" val)"
    nte="$(count_split "$YAML_3B" test)"
    [ "$ntr" -gt 4000 ] || { echo "    3B YAML train=$ntr, expected ~5040 (clear + foggy)"; return 1; }
    [ "$nva" -gt 0 ] && [ "$nte" -gt 0 ] || return 1
    return 0
}

yaml3b() {
    M3B_DEPTH_ROOT="$DEPTH_3B" "$PY" "$EXP/make_3b_dataset.py" || return 1
    yaml3b_is_complete || { echo "  3B YAML written but still incomplete"; return 1; }
}

train3b_is_complete() {
    [ -f "$RUN_3B/weights/best.pt" ] || return 1
    [ -f "$RES_3B_TRAIN" ] || return 1
    hash_matches "$RES_3B_TRAIN" || return 1
    [ "$(json_field "$RES_3B_TRAIN" epochs_trained)" -gt 0 ] 2>/dev/null || return 1
    return 0
}

train3b() {
    "$PY" "$EXP/train_m3fd.py" \
        --model weights/yolo12s.pt \
        --name v12s_m3fd_3b_fog_heavy \
        --data "$YAML_3B" \
        --epochs 100 --batch 64 --cache disk \
        --no-resume || return 1
    [ -f "$RUN_3B/weights/best.pt" ] || { echo "  training left no best.pt"; return 1; }
    train3b_is_complete
}

eval3b_is_complete() {
    [ -f "$RES_3B_EVAL" ] || return 1
    hash_matches "$RES_3B_EVAL" || return 1
    # Must also be the eval OF the trained 3B weights currently on disk.
    weights_match "$RES_3B_EVAL" "$RUN_3B/weights/best.pt" || return 1
    [ "$(json_field "$RES_3B_EVAL" n_images)" -gt 0 ] 2>/dev/null || return 1
    return 0
}

eval3b() {
    "$PY" "$EXP/eval_m3fd.py" \
        --weights "$RUN_3B/weights/best.pt" \
        --data "$DEPTH_3B/fog_heavy/m3fd_fog_heavy.yaml" \
        --name 3b_fog_heavy \
        --out "$RES_3B_EVAL" || return 1
    eval3b_is_complete
}

# ── run the sub-steps ────────────────────────────────────────────────────────
FAILED_HARD=0
for cond in fog rain; do
    for sev in $SEV_3A; do
        sname="gen3a_${cond}_${sev}"
        if want "$sname"; then
            if gen_is_complete "$DEPTH_3A" "$cond" "$sev" val test; then
                skipping "$sname" "dataset complete for this split"
                SKIP_STEPS+=("$sname")
            else
                step "$sname" "depth fog/$sev -> val+test" make_gen3a "$cond" "$sev" && :
            fi
        fi
        ename="eval3a_${cond}_${sev}"
        if want "$ename"; then
            if eval3a_is_complete "$cond" "$sev"; then
                skipping "$ename" "result present for this split"
                SKIP_STEPS+=("$ename")
            else
                step "$ename" "clear baseline on depth ${cond}_${sev}" run_eval3a "$cond" "$sev" && :
            fi
        fi
    done
done

if want gen3b; then
    if gen3b_is_complete; then
        skipping gen3b "train+val+test depth fog_heavy complete"
        SKIP_STEPS+=("gen3b")
    else
        step gen3b "depth fog_heavy -> train+val+test" || FAILED_HARD=1
    fi
fi

if want yaml3b; then
    if yaml3b_is_complete; then
        skipping yaml3b "3B YAML complete (train >4000, all splits present)"
        SKIP_STEPS+=("yaml3b")
    else
        step yaml3b "build 3B combined YAML" || FAILED_HARD=1
    fi
fi

if want train3b; then
    if train3b_is_complete; then
        skipping train3b "weights + result JSON present"
        SKIP_STEPS+=("train3b")
    else
        step train3b "train YOLOv12s on clear + depth fog_heavy" || FAILED_HARD=1
    fi
fi

if want eval3b; then
    if eval3b_is_complete; then
        skipping eval3b "3B gap result present"
        SKIP_STEPS+=("eval3b")
    else
        step eval3b "evaluate 3B model on depth fog_heavy test" || FAILED_HARD=1
    fi
fi

# Only claim the stage when every sub-step is complete, so run_all_methods.sh
# will not skip a partially-run stage.
say "Stage 5 summary $(date)"
if [ "$PLAN" = "1" ]; then
    echo "  PLAN ONLY -- nothing was executed."
    [ "${#SKIP_STEPS[@]}" -gt 0 ] && echo "  would skip: ${SKIP_STEPS[*]}"
    [ "${#PLAN_RUN[@]}" -gt 0 ] && echo "  would run:  ${PLAN_RUN[*]}"
    exit 0
fi
if [ "${#SKIP_STEPS[@]}" -gt 0 ]; then
    echo "  skipped (already done): ${SKIP_STEPS[*]}"
fi
if [ "${#DONE_STEPS[@]}" -gt 0 ]; then
    echo "  ran this invocation:    ${DONE_STEPS[*]}"
fi

ALL_OK=1
for cond in fog rain; do
    for sev in $SEV_3A; do
        gen_is_complete "$DEPTH_3A" "$cond" "$sev" val test || { echo "  INCOMPLETE: gen3a_${cond}_${sev}"; ALL_OK=0; }
        eval3a_is_complete "$cond" "$sev" || { echo "  INCOMPLETE: eval3a_${cond}_${sev}"; ALL_OK=0; }
    done
done
gen3b_is_complete   || { echo "  INCOMPLETE: gen3b";   ALL_OK=0; }
yaml3b_is_complete  || { echo "  INCOMPLETE: yaml3b";  ALL_OK=0; }
train3b_is_complete || { echo "  INCOMPLETE: train3b"; ALL_OK=0; }
eval3b_is_complete  || { echo "  INCOMPLETE: eval3b";  ALL_OK=0; }

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "  FAILED sub-steps: ${FAILED[*]}"
    ALL_OK=0
fi

if [ "$ALL_OK" = "1" ] && [ "$FAILED_HARD" = "0" ]; then
    printf 'stage=method2\npipeline=5\nsplit_hash=%s\nfinished=%s\n' \
        "$SPLIT_HASH" "$(date -Is)" > "$STATE/method2.done"
    rm -f "$STATE/method2.FAILED"
    echo "  Stage 5 COMPLETE -- marker method2.done written"
    exit 0
fi

echo "  Stage 5 INCOMPLETE -- no marker written; rerun to continue from here."
exit 1