#!/usr/bin/env bash
# ============================================================================
# Unified M3FD Pipeline — all three synthetic-weather methods on one clean split
# ============================================================================
#
# Stages, in order:
#   1  split           fresh chunk-disjoint temporal split  (resplit_m3fd.py)
#   2  cyclegan        retrain thermal_fog + thermal_rain against this split
#   3  clear_baseline  YOLOv12s on clear IR
#   4  method1         classical synthetic weather — Protocol 3A
#   5  method2         depth-aware synthetic weather — Protocols 3A + 3B
#   6  method3         CycleGAN — Protocols 3A + 3B
#   7  summary         unified gap table
#
# WHY STAGES, WHY MARKERS
# -----------------------
# This script runs under `--requeue`, because nodes fail and a multi-hour job
# cannot survive that otherwise.  The previous version re-ran from step 1 on
# every requeue, which re-did the resplit and then discarded *all* training
# (`--no-resume`), wiped Method 3's translations, and threw away a full day.
# So each stage drops a marker in logs/state/ recording the split hash and the
# pipeline generation; a stage is skipped only when both match.  Change the
# split and every marker invalidates; requeue with the same split and finished
# stages are skipped.  Bump PIPELINE_GENERATION when a stage's code or
# definition changes, which invalidates everything by design.
#
# This is deliberately *not* the "if the result file exists, skip" guard the
# previous audit removed: that one keyed on mere existence, so a stale JSON
# from a previous split was silently accepted. This one keys on the split hash.
#
# Submit via:   sbatch experiments/03_m3fd_weather/scripts/submit_all_methods.sbatch
# Or inside a held allocation:
#                srun --jobid=<JOBID> bash experiments/03_m3fd_weather/scripts/run_all_methods.sh
#
# Runtime on one GPU, measured: ~9h CycleGAN (2 x ~4.2h) + ~0.5h clear baseline
# + ~0.5h Method 1 + ~2h Method 2 (depth inference dominates) + ~3h Method 3.
# Budget 13-16h, so a single 24h slot fits with room for one requeue.

set -euo pipefail

# Resolve the repo from this script's own location rather than hardcoding it: a
# hardcoded path means the copy of the code you just edited is not the copy that
# runs. Relevant because worktrees and the primary checkout both exist here.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PY="${PY:-python}"
EXP="$REPO/experiments/03_m3fd_weather"
CGAN="$EXP/method3_cyclegan"
STATE="$EXP/logs/state"
CLEAR_YAML="$EXP/data/split/m3fd_clear.yaml"
CLEAR_WEIGHTS="$EXP/runs/v12s_m3fd_clear/weights/best.pt"

# Bump when a stage's code or definition changes: every marker is invalidated.
# 5: the thermal rain degradation was replaced. Both methods modelled rain as a
#    Gaussian blur, which on M3FD thermal changed the mean pixel by 0.6-0.8 of
#    255 and the contrast by 0.1 grey levels at every severity, and Method 2's
#    blur took no depth term, so its rain IR was byte-identical to Method 1's.
#    Rain is now droplet streaks plus a contrast loss, depth-modulated in Method 2.
PIPELINE_GENERATION=5

mkdir -p "$STATE"
cd "$REPO"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ── stage markers ─────────────────────────────────────────────────────────────
# SPLIT_HASH is not known until stage 1 has run, but stage_done is called on
# stage 1's own marker before that. Under `set -u` a bare $SPLIT_HASH there is a
# fatal unbound-variable error, reached only when the marker already exists --
# which is to say on every requeue, the one case the markers exist to handle.
# Hence ${SPLIT_HASH:-} throughout, and an empty hash matches only a marker
# written with an empty hash.
mark_stage() {
    printf 'stage=%s\npipeline=%s\nsplit_hash=%s\nfinished=%s\n' \
        "$1" "$PIPELINE_GENERATION" "${SPLIT_HASH:-}" "$(date -Is)" \
        > "$STATE/$1.done"
}
stage_done() {
    local m="$STATE/$1.done" want="${SPLIT_HASH:-}"
    [ -f "$m" ] || return 1
    grep -qx "pipeline=$PIPELINE_GENERATION" "$m" || return 1
    grep -qx "split_hash=$want" "$m"
}

read_split_hash() {
    "$PY" - "$1" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    print(json.loads((p.parent / "split_manifest.json").read_text())["split_hash"])
except Exception:
    print("")
EOF
}

# ── the silent-label check ────────────────────────────────────────────────────
# YOLOv12 does not crash when labels are in the wrong place. It logs
# "0 images, N backgrounds", trains on nothing and reports 0.000 mAP. Every
# training run below is followed by this grep.
assert_labels_found() {
    local name="$1"
    local log
    log="$(find "$EXP/runs/$name" -name '*.log' -o -name 'log.txt' 2>/dev/null | head -1)"
    if [ -z "$log" ]; then
        echo "  WARN: no log found under runs/$name -- cannot verify labels were read"
        return 0
    fi
    if grep -qiE 'backgrounds' "$log"; then
        echo "  label check ($log):"
        grep -iE 'backgrounds' "$log" | tail -2 | sed 's/^/    /'
    else
        echo "  WARN: 'backgrounds' never appeared in $log -- cannot confirm the"
        echo "       labels were found. Inspect manually before trusting any mAP."
    fi
}

# ============================================================================
say "M3FD Unified Pipeline — all methods"
echo " $(date)"
echo " repo:     $REPO"
echo " python:   $PY"
echo " generation: $PIPELINE_GENERATION"
echo " job:      ${SLURM_JOB_ID:-interactive}"
echo "========================================================================"

# ── Stage 1: split ───────────────────────────────────────────────────────────
if stage_done split; then
    say "Stage 1/7 split — already done for split $(cat "$STATE/split.done" | grep split_hash= | cut -d= -f2)"
else
    say "Stage 1/7 split"
    "$PY" "$EXP/resplit_m3fd.py" --dry-run
    "$PY" "$EXP/resplit_m3fd.py"
fi

if [ ! -f "$EXP/data/split/split_manifest.json" ]; then
    echo "FATAL: split_manifest.json missing after the split stage"
    exit 1
fi
SPLIT_HASH="$(read_split_hash "$CLEAR_YAML")"
if [ -z "$SPLIT_HASH" ]; then
    echo "FATAL: could not read split_hash from the manifest"
    exit 1
fi
echo ""
echo "split hash: $SPLIT_HASH"

# Image/label parity. A count mismatch here is the YOLO silent-label bug in its
# earliest form, so it is fatal rather than a warning.
for split in train val test; do
    n_imgs=$(ls "$EXP/data/split/IR/images/$split/"*.png 2>/dev/null | wc -l)
    n_lbls=$(ls "$EXP/data/split/IR/labels/$split/"*.txt 2>/dev/null | wc -l)
    printf '  %-5s %5d images %5d labels\n' "$split" "$n_imgs" "$n_lbls"
    [ "$n_imgs" -eq "$n_lbls" ] || { echo "FATAL: image/label mismatch in $split"; exit 1; }
done
"$PY" - "$EXP/data/split/split_manifest.json" <<'EOF'
import json, sys
m = json.loads(open(sys.argv[1]).read())
bl = m["boundary_leakage"]["contamination"]
print(f"  leakage: {bl['contaminated_frames']}/{bl['eval_frames']} eval frames "
      f"({bl['fraction'] * 100:.2f}%) carry a train box at IoU > {bl['iou_threshold']} "
      f"within +-{bl['window_frames']} frames")
print(f"  min cross-split temporal gap: {m['min_cross_split_gap_frames']} frames")
EOF
mark_stage split

# ── Stage 2: CycleGAN generators ─────────────────────────────────────────────
# Retraining is not optional.  The generators on disk were trained when
# data/split/IR/images/train held the OLD random split, and 432 of the 616 new
# test frames were in that pool (70.1%).  Translating them through a generator
# that already saw them would measure its memory of its own training data.
if stage_done cyclegan; then
    say "Stage 2/7 cyclegan — already trained for this split"
else
    say "Stage 2/7 cyclegan — retrain both generators against $SPLIT_HASH"
    for run in thermal_fog thermal_rain; do
        echo ""
        echo "--- $run ---"
        fp="$("$PY" "$CGAN/train_cyclegan.py" --run "$run" --print-fingerprint)"
        cur="$(echo "$fp" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["source_hash"])')"
        src_json="$CGAN/runs/$run/SOURCE.json"
        if [ -f "$src_json" ]; then
            old="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("source_hash"))' "$src_json")"
        else
            # No provenance at all is *not* a pass. train_cyclegan.py exits 1 on a
            # missing SOURCE.json precisely because it cannot prove the generator
            # saw the current frame pool, and under `set -e` that would abort the
            # whole pipeline here. Archive instead, so the retrain actually
            # happens. The pre-audit runs on disk are exactly this case: a DONE
            # marker with no SOURCE.json.
            old="<no SOURCE.json>"
        fi
        if [ "$old" != "$cur" ] && [ -d "$CGAN/runs/$run" ]; then
            stamp="$(date +%s)"
            echo "  source pool unverifiable or changed ($old -> $cur): archiving the old run"
            mv "$CGAN/runs/$run" "$CGAN/runs/${run}.stale-${stamp}"
        fi
        # No --no-resume: a requeue mid-training must continue, not restart. The
        # stale case above is handled by moving the directory instead.
        "$PY" "$CGAN/train_cyclegan.py" --run "$run" --epochs 50
        if [ ! -f "$CGAN/runs/$run/DONE" ]; then
            echo "FATAL: $run did not reach its epoch budget (no DONE marker)"
            exit 1
        fi
    done
fi
for run in thermal_fog thermal_rain; do
    [ -f "$CGAN/runs/$run/G_AB.pt" ] || { echo "FATAL: $run/G_AB.pt missing"; exit 1; }
done
mark_stage cyclegan

# ── Stage 3: clear baseline ──────────────────────────────────────────────────
if stage_done clear_baseline && [ -f "$CLEAR_WEIGHTS" ]; then
    say "Stage 3/7 clear_baseline — weights present, skipping"
else
    say "Stage 3/7 clear_baseline"
    "$PY" "$EXP/train_m3fd.py" \
        --model weights/yolo12s.pt \
        --name v12s_m3fd_clear \
        --data "$CLEAR_YAML" \
        --epochs 100 --batch 64 --cache disk \
        --no-resume
    assert_labels_found v12s_m3fd_clear
fi
[ -f "$CLEAR_WEIGHTS" ] || { echo "FATAL: $CLEAR_WEIGHTS missing"; exit 1; }
# The result JSON is the reference every 3A gap is measured against. Its absence
# used to be invisible: the pipeline only checked for weights, so the summary
# printed "Clear baseline: pending" and carried on.
[ -f "$EXP/results/v12s_m3fd_clear.json" ] \
    || { echo "FATAL: results/v12s_m3fd_clear.json missing — no 3A reference"; exit 1; }
mark_stage clear_baseline

# ── Stage 4: Method 1, Protocol 3A ───────────────────────────────────────────
if stage_done method1; then
    say "Stage 4/7 method1 — already generated and evaluated for this split"
else
    say "Stage 4/7 method1 (classical, Protocol 3A)"
    rm -rf "$EXP/data/synthetic"
    "$PY" "$EXP/method1_classical/generate_weather_splits.py"
    for cond in fog rain; do
        for sev in light medium heavy; do
            yaml="$EXP/data/synthetic/${cond}_${sev}/m3fd_${cond}_${sev}.yaml"
            [ -f "$yaml" ] || { echo "FATAL: $yaml not generated"; exit 1; }
            echo "  evaluating ${cond}_${sev} ..."
            "$PY" "$EXP/eval_m3fd.py" \
                --weights "$CLEAR_WEIGHTS" \
                --data "$yaml" \
                --name "${cond}_${sev}" \
                --out "$EXP/method1_classical/results_${cond}_${sev}.json"
        done
    done
fi
mark_stage method1

# ── Stage 5: Method 2, Protocols 3A + 3B ─────────────────────────────────────
if stage_done method2; then
    say "Stage 5/7 method2 — already generated and evaluated for this split"
else
    say "Stage 5/7 method2 (depth-aware, Protocols 3A + 3B)"
    rm -rf "$EXP/data/synthetic_depth" "$EXP/data/synthetic_depth_3b" \
           "$EXP/data/3b_fog_heavy"
    # val+test for the 3A screening sweep, then train+val for the 3B dataset.
    # No --drop-obscured: filtering the labels of an evaluation set would make
    # the measured gap partly a change of denominator.
    for cond in fog rain; do
        for sev in medium heavy; do
            echo "  generating depth ${cond}_${sev} (val test) ..."
            "$PY" "$EXP/method2_depth/apply_weather_depth.py" \
                --src "$EXP/data/split/IR" \
                --dest "$EXP/data/synthetic_depth" \
                --splits val test --condition "$cond" --severity "$sev"
        done
    done
    for cond in fog rain; do
        for sev in medium heavy; do
            yaml="$EXP/data/synthetic_depth/${cond}_${sev}/m3fd_${cond}_${sev}.yaml"
            [ -f "$yaml" ] || { echo "FATAL: $yaml not generated"; exit 1; }
            echo "  evaluating depth ${cond}_${sev} (3A) ..."
            "$PY" "$EXP/eval_m3fd.py" \
                --weights "$CLEAR_WEIGHTS" \
                --data "$yaml" --name "${cond}_${sev}_depth" \
                --out "$EXP/method2_depth/results_${cond}_${sev}_depth.json"
        done
    done

    echo ""
    echo "  --- Method 2 Protocol 3B: fog_heavy (train+val mixed, test for the gap) ---"
    # Its own destination, and it includes test. It used to regenerate
    # synthetic_depth/fog_heavy in place with `--splits train val`, which rewrote
    # that YAML's `test:` from images/test to images/val. The 3A result above was
    # computed against the 3A YAML, so it then stopped being reproducible from
    # anything on disk, and the 3B gap below was scored on 240 val frames while its
    # reference clear-test mAP came from 600 test frames.
    "$PY" "$EXP/method2_depth/apply_weather_depth.py" \
        --src "$EXP/data/split/IR" \
        --dest "$EXP/data/synthetic_depth_3b" \
        --splits train val test --condition fog --severity heavy
    M3B_DEPTH_ROOT="$EXP/data/synthetic_depth_3b" \
    "$PY" "$EXP/make_3b_dataset.py" \
    "$PY" "$EXP/train_m3fd.py" \
        --model weights/yolo12s.pt \
        --name v12s_m3fd_3b_fog_heavy \
        --data "$EXP/data/3b_fog_heavy/m3fd_3b.yaml" \
        --epochs 100 --batch 64 --cache disk \
        --no-resume
    assert_labels_found v12s_m3fd_3b_fog_heavy
    "$PY" "$EXP/eval_m3fd.py" \
        --weights "$EXP/runs/v12s_m3fd_3b_fog_heavy/weights/best.pt" \
        --data "$EXP/data/synthetic_depth_3b/fog_heavy/m3fd_fog_heavy.yaml" \
        --name 3b_fog_heavy \
        --out "$EXP/method2_depth/results_3b_fog_heavy.json"
fi
mark_stage method2

# ── Stage 6: Method 3, Protocols 3A + 3B ─────────────────────────────────────
# No `rm -rf translated/` here: make_cyclegan_dataset.py stamps each translation
# with the split hash and re-translates on mismatch, so wiping it would throw
# away ~40 min of generator work on every requeue for no benefit.
if stage_done method3; then
    say "Stage 6/7 method3 — already evaluated for this split"
else
    say "Stage 6/7 method3 (CycleGAN, Protocols 3A + 3B)"
    bash "$CGAN/scripts/run_eval_3ab.sh"
    assert_labels_found v12s_m3fd_3b_cgan_fog
fi
mark_stage method3

# ── Stage 7: summary ──────────────────────────────────────────────────────────
say "Stage 7/7 summary"
"$PY" - <<'PYEOF'
import json
from pathlib import Path

EXP = Path("experiments/03_m3fd_weather")
CGAN = EXP / "method3_cyclegan"


def load(path):
    p = Path(path)
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    if "mAP50_95" in d:
        return d["mAP50_95"], d.get("split_hash")
    try:
        return d["metrics"]["test"]["mAP50_95"], d.get("split_hash")
    except (KeyError, TypeError):
        return None, None


clear, clear_hash = load(EXP / "results" / "v12s_m3fd_clear.json")
print(f"\nClear baseline mAP@50-95: {clear:.4f}  (split {clear_hash})" if clear
      else "\nClear baseline: MISSING")
print()
print("Thresholds: 3A gap >= 0.15 PASS, >= 0.05 WEAK; 3B gap >= 0.10 PASS, >= 0.05 WEAK")
print()

rows = [("Method", "Condition", "Prot", "mAP@50-95", "Gap", "Verdict", "split")]


def add(method, cond, prot, path, reference, pass_at):
    m, sh = load(path)
    if m is None:
        rows.append((method, cond, prot, "missing", "-", "-", sh or "-"))
        return
    if reference is None:
        rows.append((method, cond, prot, f"{m:.4f}", "-", "no ref", sh or "-"))
        return
    gap = reference - m
    verdict = "PASS" if gap >= pass_at else "WEAK" if gap >= pass_at / 3 else "FAIL"
    rows.append((method, cond, prot, f"{m:.4f}", f"{gap:+.4f}", verdict, sh or "-"))


for cond in ("fog", "rain"):
    for sev in ("light", "medium", "heavy"):
        add("M1 classical", f"{cond}_{sev}", "3A",
            EXP / "method1_classical" / f"results_{cond}_{sev}.json", clear, 0.15)
for cond in ("fog", "rain"):
    for sev in ("medium", "heavy"):
        add("M2 depth", f"{cond}_{sev}", "3A",
            EXP / "method2_depth" / f"results_{cond}_{sev}_depth.json", clear, 0.15)
m2_train, _ = load(EXP / "results" / "v12s_m3fd_3b_fog_heavy.json")
add("M2 depth", "fog_heavy", "3B",
    EXP / "method2_depth" / "results_3b_fog_heavy.json", m2_train, 0.10)
for name in ("fog", "rain"):
    add("M3 cyclegan", name, "3A", CGAN / "results" / f"3a_{name}_eval.json", clear, 0.15)
    m3_train, _ = load(EXP / "results" / f"v12s_m3fd_3b_cgan_{name}.json")
    add("M3 cyclegan", name, "3B", CGAN / "results" / f"3b_{name}_eval.json", m3_train, 0.10)

w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
fmt = "  ".join(f"{{:<{x}}}" for x in w)
print(fmt.format(*rows[0]))
print(fmt.format(*["-" * x for x in w]))
for r in rows[1:]:
    print(fmt.format(*r))

mixed = {r[-1] for r in rows[1:] if r[-1] != "-"} - {clear_hash}
if mixed:
    print(f"\nWARNING: results carry more than one split_hash: {sorted(mixed | ({clear_hash} if clear_hash else set()))}")
else:
    print(f"\nAll results carry split_hash {clear_hash}.")
PYEOF

say "Pipeline complete — $(date)"
echo "Results are under $EXP/results, $EXP/method{1_classical,2_depth,3_cyclegan}/results"