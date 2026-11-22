#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -d "$ROOT/data/flir_adas_v2" ]; do ROOT="$(dirname "$ROOT")"; done
# Method 3: train the four CycleGANs, in priority order, resumably.
#
# Order is not arbitrary. thermal_fog is first because thermal fog is exactly
# what Method 2 failed on (Protocol 3B gap -0.002, method2_depth/report.md:106)
# and because every M3FD metric is IR-only -- the RGB generators cannot move
# any number currently on the books. If this job is cut off, the one that got
# furthest is the one that mattered.
#
# Idempotent at every level:
#   * a run with a DONE marker is skipped
#   * a run with a checkpoint resumes from it
#   * so a --requeue restart continues rather than repeating
#
# Run via submit_cyclegan.sbatch, or directly:
#   bash experiments/03_m3fd_weather/method3_cyclegan/scripts/run_cyclegan.sh
#   bash experiments/03_m3fd_weather/method3_cyclegan/scripts/run_cyclegan.sh thermal_fog

set -uo pipefail

REPO="$ROOT"
M3="$REPO/experiments/03_m3fd_weather/method3_cyclegan"
PY="python"
cd "$REPO" || exit 1

EPOCHS="${CYCLEGAN_EPOCHS:-50}"
CROP="${CYCLEGAN_CROP:-512}"
WORKERS="${CYCLEGAN_WORKERS:-$(python3 -c 'import os;print(max(0,min(int(os.environ.get("SLURM_CPUS_PER_TASK","8"))-2,12)))' 2>/dev/null || echo 6)}"
CALIBRATE_FIRST="${CYCLEGAN_CALIBRATE_FIRST:-1}"

# Escape hatches, all empty in the real job. They exist so the whole
# orchestrator can be rehearsed against a throwaway run directory without
# touching a real one -- which is the only way to be confident the 24h job
# will not die on a shell-level bug after burning a queue position.
OUT="${CYCLEGAN_OUT:-$M3/runs}"
MAX_SOURCE="${CYCLEGAN_MAX_SOURCE:-}"
LIMIT_ITERS="${CYCLEGAN_LIMIT_ITERS:-}"

# priority order; a single run may be passed as $1
ALL_RUNS=(thermal_fog thermal_rain rgb_fog rgb_rain)
if [ $# -gt 0 ]; then RUNS=("$@"); else RUNS=("${ALL_RUNS[@]}"); fi

echo "=============================================================="
echo " Method 3 CycleGANs"
echo " started $(date)"
echo " host=$(hostname)  job=${SLURM_JOB_ID:-none}  attempt=${SLURM_RESTART_COUNT:-0}"
echo " python=$PY"
echo " epochs=$EPOCHS crop=$CROP workers=$WORKERS"
echo " out=$OUT"
echo " runs=${RUNS[*]}"
echo "=============================================================="

# ---------------------------------------------------------------------------
# Preconditions. Fail fast and loudly rather than hang.
# ---------------------------------------------------------------------------
DSERT="$ROOT/data/dsert_roll"
if [ ! -d "$DSERT" ]; then
    cat >&2 <<EOF

FATAL: DSERT-RoLL target domain is missing at $DSERT

Download it FIRST, from the login node:

    $PY $M3/download_dsert.py

This job deliberately does not download. Compute nodes on this cluster may
have no outbound route to huggingface.co, in which case a job that tries to
download hangs until the allocation expires and wastes the queue position.

EOF
    exit 1
fi

MANIFEST="$M3/dsert_files.json"
if [ ! -f "$MANIFEST" ]; then
    echo "FATAL: $MANIFEST missing -- run download_dsert.py first." >&2
    exit 1
fi

n_on_disk=$(find "$DSERT" -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l)
echo "DSERT files on disk: $n_on_disk"
if [ "$n_on_disk" -lt 100 ]; then
    echo "FATAL: only $n_on_disk DSERT files on disk; the download did not complete." >&2
    exit 1
fi

# Report the resolved image format once, at the top, so the numbers are in the
# job log next to everything else. Advisory -- it never stops a run.
echo ""
echo "--- DSERT format check (advisory) ---"
"$PY" "$M3/inspect_dsert.py" 2>&1 | tail -25 || true
echo "-----------------------------------"
echo ""

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
skipped_runs=()

for run in "${RUNS[@]}"; do
    rd="$OUT/$run"
    echo ""
    echo "=============================================================="
    echo ">>> $run  $(date)"
    echo "=============================================================="

    if [ -f "$rd/DONE" ]; then
        echo "  already finished: $(tr -d '\n' < "$rd/DONE")"
        echo "  skipping."
        continue
    fi

    # A target domain that has not finished downloading must not abort the
    # queue. The run order is thermal_fog, thermal_rain, rgb_fog, rgb_rain and
    # the rain domain arrives last, so failing hard on it would stop rgb_fog
    # from ever being attempted even though its data is ready. Skip loudly and
    # carry on; the final summary lists what is still outstanding.
    if ! "$PY" "$M3/train_cyclegan.py" --run "$run" --check-data; then
        echo "  SKIP: target domain unavailable for $run."
        echo "        Finish download_dsert.py and resubmit; this run will be"
        echo "        picked up then."
        skipped_runs+=("$run")
        continue
    fi

    extra=()
    # Calibrate per run, writing runs/<run>/calibration.json. The cost is ~30
    # iterations each, which is negligible next to 147,000, and it means each
    # run's own dir records the rate actually measured on it.
    if [ "$CALIBRATE_FIRST" = "1" ] && [ ! -f "$rd/calibration.json" ]; then
        extra+=(--calibrate 30)
    fi
    [ -n "$MAX_SOURCE" ] && extra+=(--max-source "$MAX_SOURCE")
    [ -n "$LIMIT_ITERS" ] && extra+=(--limit-iters "$LIMIT_ITERS")

    "$PY" "$M3/train_cyclegan.py" \
        --run "$run" \
        --out "$OUT" \
        --epochs "$EPOCHS" \
        --crop "$CROP" \
        --workers "$WORKERS" \
        "${extra[@]}"
    rc=$?

    if [ $rc -ne 0 ]; then
        echo "  $run exited $rc -- stopping the queue of runs."
        echo "  Fix and resubmit; finished runs keep their DONE marker and the"
        echo "  partial one resumes from its last checkpoint."
        exit $rc
    fi
done

echo ""
echo "=============================================================="
echo " finished $(date)"
for run in "${RUNS[@]}"; do
    rd="$OUT/$run"
    skip_note=""
    for s in "${skipped_runs[@]}"; do
        [ "$s" = "$run" ] && skip_note=" (target data not downloaded yet)"
    done
    if [ -f "$rd/DONE" ]; then
        echo "  [DONE]    $run  -> $rd/G_AB.pt"
    elif [ -n "$skip_note" ]; then
        echo "  [SKIP]    $run$skip_note"
    elif [ -f "$rd/ckpt/latest.pt" ]; then
        echo "  [PARTIAL] $run  -> $rd/ckpt/latest.pt (resubmit to resume)"
    else
        echo "  [TODO]    $run  (not started)"
    fi
done
echo "=============================================================="

if [ ${#skipped_runs[@]} -gt 0 ]; then
    echo ""
    echo "skipped for missing target-domain data: ${skipped_runs[*]}"
    echo "run download_dsert.py to completion, then resubmit this script."
fi
exit 0
