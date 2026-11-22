#!/usr/bin/env bash
# Protocol 3A and 3B evaluation for Method 3 (CycleGAN).
#
# Steps:
#   1. Translate thermal_fog / thermal_rain (train/val/test) and write YAMLs.
#      Translations are stamped with the split hash and re-made on mismatch, so
#      this is safe to re-run after a resplit.
#   2. Protocol 3A: evaluate the clear-only baseline on translated test.
#   3. Protocol 3B (fog): train YOLOv12s on clear + CycleGAN fog, eval on fog test.
#   4. Protocol 3B (rain): same for rain.
#   5. Gap summary.
#
# No idempotency guards keyed on result-file existence: an earlier version had
# them, and they meant a stale JSON from a contaminated split suppressed a fresh
# evaluation. Regeneration is decided by the pipeline's split-hash stage markers
# (scripts/run_all_methods.sh), which is a different and sounder mechanism --
# these scripts always regenerate whatever they are asked to regenerate.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CGAN="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$CGAN/../.." && pwd)"
PY="${PY:-python}"
EXP="$REPO/experiments/03_m3fd_weather"

cd "$REPO"

echo "==========================================================="
echo " Method 3 CycleGAN — Protocol 3A / 3B Evaluation"
echo " $(date)"
echo " repo: $REPO"
echo "==========================================================="
echo ""

# ---------------------------------------------------------------------------
# Step 1 — Translate and write dataset YAMLs
# ---------------------------------------------------------------------------
echo "--- Step 1: translate CycleGAN outputs and write dataset YAMLs ---"
"$PY" "$CGAN/make_cyclegan_dataset.py" \
    --runs thermal_fog thermal_rain \
    --splits train val test

echo ""
echo "  Dataset YAMLs:"
ls "$CGAN/datasets/"*.yaml 2>/dev/null || { echo "  (none written -- see errors above)"; exit 1; }
echo ""

CLEAR_WEIGHTS="$EXP/runs/v12s_m3fd_clear/weights/best.pt"
if [ ! -f "$CLEAR_WEIGHTS" ]; then
    echo "ERROR: clear baseline weights not found: $CLEAR_WEIGHTS"
    echo "Train a clear-only model first (train_m3fd.py --name v12s_m3fd_clear)"
    exit 1
fi

mkdir -p "$CGAN/results"

# ---------------------------------------------------------------------------
# Step 2 — Protocol 3A
# ---------------------------------------------------------------------------
echo "--- Step 2a: Protocol 3A — clear baseline on CycleGAN fog test ---"
"$PY" "$EXP/eval_m3fd.py" \
    --weights "$CLEAR_WEIGHTS" \
    --data "$CGAN/datasets/cyclegan_fog_3a.yaml" \
    --name cyclegan_3a_fog \
    --out "$CGAN/results/3a_fog_eval.json"

if [ -f "$CGAN/datasets/cyclegan_rain_3a.yaml" ]; then
    echo "--- Step 2b: Protocol 3A — clear baseline on CycleGAN rain test ---"
    "$PY" "$EXP/eval_m3fd.py" \
        --weights "$CLEAR_WEIGHTS" \
        --data "$CGAN/datasets/cyclegan_rain_3a.yaml" \
        --name cyclegan_3a_rain \
        --out "$CGAN/results/3a_rain_eval.json"
fi

# ---------------------------------------------------------------------------
# Steps 3 & 4 — Protocol 3B, one condition at a time
#
# fog and rain are independent: neither reads the other's weights, YAML, or
# result. Under `set -e` a failed fog training used to abort the whole script
# before rain ever started, which is how job 8752 lost both. Each condition is
# therefore run in a function whose failure is recorded and stepped over.
# ---------------------------------------------------------------------------
CGAN_3B_FAILED=()

run_3b () {
    local short="$1"          # fog | rain
    local yaml="$CGAN/datasets/cyclegan_${short}_3b.yaml"
    local test_yaml="$CGAN/datasets/cyclegan_${short}_3b_fogtest.yaml"
    local run="v12s_m3fd_3b_cgan_${short}"

    if [ ! -f "$yaml" ]; then
        echo "--- ${short} 3B YAML missing, skipping ${short} 3B ---"
        return 0
    fi

    echo "--- Protocol 3B ${short}: train on clear + CycleGAN ${short} ---"
    echo "    val = mixed (clear + CycleGAN ${short}) so early stopping does not pick a"
    echo "          checkpoint that is good at one domain and useless at the other"
    if ! "$PY" "$EXP/train_m3fd.py" \
        --model weights/yolo12s.pt \
        --name "$run" \
        --data "$yaml" \
        --epochs 100 --batch 64 --cache disk \
        --no-resume; then
        echo "!!! 3B ${short} TRAINING FAILED -- carrying on to the next condition"
        return 1
    fi

    # The rain dataset file is named _fogtest for historical reasons.
    echo "--- Protocol 3B ${short}: evaluate on CycleGAN ${short} test ---"
    if ! "$PY" "$EXP/eval_m3fd.py" \
        --weights "$EXP/runs/$run/weights/best.pt" \
        --data "$test_yaml" \
        --name "cyclegan_3b_${short}_eval" \
        --out "$CGAN/results/3b_${short}_eval.json"; then
        echo "!!! 3B ${short} EVAL FAILED -- carrying on to the next condition"
        return 1
    fi
    return 0
}

run_3b fog  || CGAN_3B_FAILED+=("fog")
run_3b rain || CGAN_3B_FAILED+=("rain")

# ---------------------------------------------------------------------------
# Step 5 — Gap summary
# ---------------------------------------------------------------------------
echo ""
echo "==========================================================="
echo " Method 3 Protocol 3A / 3B — Gap Summary"
echo "==========================================================="

"$PY" - <<'PYEOF'
import json
from pathlib import Path

EXP = Path("experiments/03_m3fd_weather")
CGAN = EXP / "method3_cyclegan"
RESULTS = CGAN / "results"
M3FD_RES = EXP / "results"


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if "mAP50_95" in d:
        return d["mAP50_95"]
    try:
        return d["metrics"]["test"]["mAP50_95"]
    except (KeyError, TypeError):
        return None


clear = load(M3FD_RES / "v12s_m3fd_clear.json")

rows = [("Protocol", "Condition", "mAP@50-95", "Gap", "Verdict", "split")]


def split_of(path):
    p = Path(path)
    if not p.exists():
        return "-"
    return json.loads(p.read_text()).get("split_hash") or "-"


for name, label in (("fog", "Fog"), ("rain", "Rain")):
    f3a = RESULTS / f"3a_{name}_eval.json"
    f3b = RESULTS / f"3b_{name}_eval.json"
    f3bt = M3FD_RES / f"v12s_m3fd_3b_cgan_{name}.json"

    m3a, m3b, m3bt = load(f3a), load(f3b), load(f3bt)

    if m3a is None:
        rows.append(("3A (clear-only model)", label, "missing", "-", "-", split_of(f3a)))
    elif clear is None:
        rows.append(("3A (clear-only model)", label, f"{m3a:.4f}", "-", "no baseline", split_of(f3a)))
    else:
        gap = clear - m3a
        v = "PASS" if gap >= 0.15 else "WEAK" if gap >= 0.05 else "FAIL"
        rows.append(("3A (clear-only model)", label, f"{m3a:.4f}", f"{gap:+.4f}", v, split_of(f3a)))

    if m3bt is None:
        rows.append(("3B clear test (ref)", label, "missing", "-", "-", split_of(f3bt)))
    else:
        rows.append(("3B clear test (ref)", label, f"{m3bt:.4f}", "ref", "-", split_of(f3bt)))

    if m3b is None or m3bt is None:
        rows.append(("3B fog/rain test", label, "missing", "-", "-", split_of(f3b)))
    else:
        gap = m3bt - m3b
        v = "PASS" if gap >= 0.10 else "WEAK" if gap >= 0.05 else "FAIL"
        rows.append(("3B fog/rain test", label, f"{m3b:.4f}", f"{gap:+.4f}", v, split_of(f3b)))

w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
fmt = "  ".join(f"{{:<{x}}}" for x in w)
print(fmt.format(*rows[0]))
print(fmt.format(*["-" * x for x in w]))
for r in rows[1:]:
    print(fmt.format(*r))

print()
print("Bar: 3B gap >= 0.10 declares Method 3 successful.")
print("Cross-method comparison lives in scripts/run_all_methods.sh; nothing is")
print("printed here from an earlier run, so no archived number can appear")
print("beside a freshly computed one.")
PYEOF

echo ""
echo "==========================================================="
if [ "${#CGAN_3B_FAILED[@]}" -gt 0 ]; then
    echo " Finished with failures — $(date)"
    echo "   3B conditions that failed: ${CGAN_3B_FAILED[*]}"
    echo "   Conditions that succeeded are intact; rerun to retry only the missing."
else
    echo " Done — $(date)"
fi
echo "==========================================================="

[ "${#CGAN_3B_FAILED[@]}" -eq 0 ]