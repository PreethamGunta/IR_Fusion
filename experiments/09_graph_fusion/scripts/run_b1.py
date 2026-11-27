#!/usr/bin/env python3
"""B1 — fuse-free warm-up + ramped fuse engagement (IDEAS_BACKLOG.md §3).

Pre-flight (rules.md PRE-FLIGHT REVIEW RULE)
- **Goal.** Does the fused model lose the pretrained IR capability because the
  fuse sits in the gradient path from step 0? B1 holds the fuse contribution at
  exactly zero for 5 epochs, then ramps it in. Decision it drives: whether the
  fusion line continues (promote to a longer budget) or is dropped in favour of
  the deferred variants (B3/B5/B9).
- **Duplicate check.** No existing run has a fuse schedule: every `runs/e*`
  applies the fuse from step 0 (verified in `args.yaml` / `code_state.json`).
  This is not a repeat of `e30_unified`; the fuse schedule is the only variable.
- **Comparability.** Same data (`configs/dronevehicle_v2_paired.yaml`), same
  pinned optimizer (MuSGD lr0 1e-2, momentum 0.9, warmup_bias_lr 0), same
  batch 16, same seed 0, same workers as the control's step. Reference is
  `e00_ir_unified` (same pipeline, 80-epoch schedule) read at the *same
  epochs*; `close_mosaic=0` keeps mosaic on for the whole probe so its early
  phase matches epochs 0-20 of the 80-epoch schedule (a 20-epoch run with the
  default close_mosaic would drop mosaic at ep10 and flatter the probe).
  Documented residual mismatch: the 20-epoch cosine anneals slightly faster
  (~15% lower LR at ep5), which biases the probe *upward*, so a failure is
  decisive and a pass is only suggestive.
- **Symmetry.** Unchanged from the unified pipeline: the pair is stacked before
  augmentation, so both modalities get the identical geometric transform.
- **Resource profile.** 2-epoch smoke (~0.2 h) + 20 epochs at ~5 min/epoch
  (~1.8 h) inside allocation 10187 (ends 08:14 UTC).
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
PY = sys.executable
RUNS = EXP / "runs"
LOGS = EXP / "logs"

EPOCHS = 20
WARMUP = 5
RAMP = 3          # alpha: 0 for ep0-4, 1/3 at ep5, 2/3 at ep6, 1 from ep7


def launch(name: str, epochs: int, warmup: int, ramp: int,
           close_mosaic: int, workers: int) -> int:
    argv = [PY, str(EXP / "scripts" / "train_unified.py"),
            "--name", name, "--mode", "pair", "--fuse", "E30",
            "--data", "configs/dronevehicle_v2_paired.yaml",
            "--rgb-yaml", "configs/dronevehicle_v2_rgb.yaml",
            "--epochs", str(epochs), "--workers", str(workers),
            "--fuse-warmup", str(warmup), "--fuse-ramp", str(ramp),
            "--close-mosaic", str(close_mosaic)]
    log = open(LOGS / f"clean_{name}.log", "a")
    log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {name} "
              f"(epochs={epochs} warmup={warmup} ramp={ramp} "
              f"close_mosaic={close_mosaic}) ===\n")
    log.flush()
    rc = subprocess.call(argv, cwd=str(EXP), stdout=log, stderr=subprocess.STDOUT)
    log.close()
    return rc


def smoke_ok(name: str) -> tuple[bool, str]:
    """A 2-epoch run must exercise alpha=0 (ep0) and alpha=1 (ep1) and validate."""
    f = RUNS / name / "results.csv"
    if not f.exists():
        return False, "no results.csv"
    rows = list(csv.DictReader(f.open()))
    if len(rows) < 2:
        return False, f"only {len(rows)} epoch(s) recorded"
    aps = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    if any(a != a for a in aps):                      # NaN check
        return False, f"NaN in val AP: {aps}"
    if max(aps) <= 0:
        return False, f"val AP never left zero: {aps}"
    return True, f"{len(rows)} epochs, AP50-95={[round(a, 4) for a in aps]}"


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)

    # 1) smoke: exercises alpha=0 then alpha=1 and a validation pass
    smoke = "b1_smoke"
    if not (RUNS / smoke / "DONE").exists():
        print(f"[{time.strftime('%H:%M:%S')}] smoke: {smoke}", flush=True)
        rc = launch(smoke, epochs=2, warmup=1, ramp=1,
                    close_mosaic=1, workers=16)
        ok, why = smoke_ok(smoke)
        print(f"[{time.strftime('%H:%M:%S')}] smoke exit code {rc}: {why}",
              flush=True)
        if rc != 0 or not ok:
            print(f"B1 RUN COMPLETE failed={[smoke]}", flush=True)
            return

    # 2) the probe
    name = "b1_warmup_probe"
    if (RUNS / name / "DONE").exists():
        print(f"[{time.strftime('%H:%M:%S')}] {name} already DONE", flush=True)
    else:
        print(f"[{time.strftime('%H:%M:%S')}] probe: {name}", flush=True)
        rc = launch(name, epochs=EPOCHS, warmup=WARMUP, ramp=RAMP,
                    close_mosaic=0, workers=24)
        print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc}", flush=True)
        if rc != 0:
            print(f"B1 RUN COMPLETE failed={[name]}", flush=True)
            return

    print("B1 RUN COMPLETE failed=[]", flush=True)


if __name__ == "__main__":
    main()
