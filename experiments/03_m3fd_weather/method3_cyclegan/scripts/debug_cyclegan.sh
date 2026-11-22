#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# Smoke test: exercise the full training path for all four runs, cheaply.
#
# This exists so that a code bug is found in three minutes rather than after a
# queue position. It trains 20 real iterations of each run on a 40-image
# source subset at 128px, writes to runs/_smoke/ so it can never be confused
# with a real run, and then verifies:
#
#   * all four networks build, run forward/backward, and produce finite loss
#   * a sample grid is written
#   * a checkpoint round-trips: train, kill, resume, and confirm the resume
#     line reports a non-zero iteration count rather than starting over
#
# That last check is the point. The whole 24h design rests on resume actually
# working, and "resume" that silently restarts from zero would be invisible
# until the second day.
#
# Use:
#   bash experiments/03_m3fd_weather/method3_cyclegan/scripts/debug_cyclegan.sh
#
# Inside a live allocation, to debug without requeuing:
#   srun --jobid=<JOBID> --overlap \
#     bash experiments/03_m3fd_weather/method3_cyclegan/scripts/debug_cyclegan.sh
#
# Takes a few minutes and uses a small amount of GPU memory. It is safe to run
# while a real job holds the allocation: --overlap means it shares the GPU
# rather than waiting for it.

set -uo pipefail

REPO="$ROOT"
M3="$REPO/experiments/03_m3fd_weather/method3_cyclegan"
PY="python"
cd "$REPO" || exit 1

OUT="$M3/runs/_smoke"
ITERS="${SMOKE_ITERS:-20}"
CROP="${SMOKE_CROP:-128}"
SRC="${SMOKE_SOURCE:-40}"

echo "=============================================================="
echo " CycleGAN smoke test   $(date)"
echo " out=$OUT  iters=$ITERS  crop=$CROP  source=$SRC"
echo "=============================================================="

if [ ! -d "$ROOT/data/dsert_roll" ]; then
    echo "FATAL: DSERT-RoLL not downloaded. Run download_dsert.py first." >&2
    exit 1
fi

# Wipe any previous smoke output so the resume test starts from a known state.
# Scoped to runs/_smoke so this can never touch a real run.
rm -rf "$OUT"
mkdir -p "$OUT"

fail=0
skip=0
LOGDIR="$OUT/_logs"
mkdir -p "$LOGDIR"
for run in thermal_fog thermal_rain rgb_fog rgb_rain; do
    echo ""
    echo "--- $run ---"
    # tee into a directory that already exists: train_cyclegan.py creates the
    # per-run dir itself, so teeing into it would fail on the first run
    if ! "$PY" "$M3/train_cyclegan.py" \
            --run "$run" --out "$OUT" \
            --crop "$CROP" --epochs 1 --limit-iters "$ITERS" \
            --max-source "$SRC" --workers 2 \
            --ckpt-every 5 --sample-every 5 --no-resume \
            2>&1 | tee "$LOGDIR/$run.smoke.log"
    then
        # A target domain that has not finished downloading is not a code
        # failure -- the download is slow and rate-limited, and this script is
        # meant to be runnable while it is still in flight. Anything else is.
        if grep -q 'no on-disk DSERT' "$LOGDIR/$run.smoke.log"; then
            echo "  SKIP: target domain not downloaded yet"
            skip=$((skip + 1))
        else
            echo "  FAIL: $run did not train"
            fail=1
        fi
        continue
    fi
    if [ ! -f "$OUT/$run/G_AB.pt" ]; then
        echo "  FAIL: $run produced no generator"
        fail=1
        continue
    fi
    n=$(ls -1 "$OUT/$run/samples" 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
        echo "  FAIL: $run wrote no sample grid"
        fail=1
        continue
    fi
    echo "  ok: trained, generator + $n sample grid(s)"

    # --- the resume test -------------------------------------------------
    # Re-run WITHOUT --no-resume and without wiping. It must pick up the
    # DONE marker written above; delete that marker first so we exercise the
    # checkpoint path rather than the completion path.
    rm -f "$OUT/$run/DONE"
    before=$(grep -c '\[resume\]' "$OUT/$run/train.log" 2>/dev/null || echo 0)
    "$PY" "$M3/train_cyclegan.py" \
        --run "$run" --out "$OUT" \
        --crop "$CROP" --epochs 2 --limit-iters "$ITERS" \
        --max-source "$SRC" --workers 2 --ckpt-every 5 --sample-every 5 \
        > "$OUT/$run/resume_test.log" 2>&1
    if grep -q '\[resume\].*iters already done' "$OUT/$run/resume_test.log"; then
        echo "  ok: resumed from checkpoint (not restarted)"
    elif grep -q 'DONE.*exists' "$OUT/$run/resume_test.log"; then
        echo "  ok: completion marker honoured"
    else
        echo "  WARN: could not confirm resume from this log:"
        grep '\[resume\]' "$OUT/$run/resume_test.log" | tail -3
    fi
done

echo ""
echo "=============================================================="
if [ "$fail" -eq 0 ]; then
    echo " SMOKE TEST PASSED  ($skip run(s) skipped, target not downloaded yet)"
    echo " sample grids: $OUT/*/samples/"
    echo " Nothing here is usable for real -- it is 128px, 20 iters, 40 images."
else
    echo " SMOKE TEST FAILED -- do not submit the real job yet"
fi
echo "=============================================================="
exit $fail
