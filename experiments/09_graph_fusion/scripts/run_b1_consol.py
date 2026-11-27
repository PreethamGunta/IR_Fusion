#!/usr/bin/env python3
"""B1 consolidation — 40 epochs of the warm-up design, sized to the window.

Run only after `scripts/gate_b1.py` reports CONFIRMED or PARTIAL.

Design identical to `b1_warmup_probe` except the budget: up to 40 epochs with
`close_mosaic = 12.5 % of the schedule` (5 epochs), matching the fraction the
80-epoch control uses with its default 10.

**Dynamic sizing.** The budget is capped by the actual remaining allocation
time and the *observed* pace of the probe, not by a planning assumption: a run
that cannot finish leaves a resumable checkpoint, but a 40-epoch run launched
at 04:00 would only reach ~33 and waste the tail. If fewer than 10 epochs fit,
the script refuses to launch rather than starting a run that cannot say
anything.

**Comparison caveat.** This run's cosine anneals fully by its last epoch; the
control at the same *epoch number* has not annealed. A win is therefore
inconclusive (the anneal could explain it) while a loss is decisive. Recorded
in `PREFLIGHT_B1.md` §5.
"""
from __future__ import annotations

import csv
import datetime as dt
import math
import os
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
PY = sys.executable
RUNS = EXP / "runs"
LOGS = EXP / "logs"

NAME = "b1_warmup_40ep"
MAX_EPOCHS = 40
DEADLINE_HHMM = os.environ.get("B1_DEADLINE", "07:20")   # UTC, ~54 min margin
FALLBACK_PACE = 6.0                                      # min/epoch, conservative


def observed_pace(name: str = "b1_warmup_probe") -> tuple[float, str]:
    """min/epoch from the run's own clock; falls back conservatively."""
    f = RUNS / name / "results.csv"
    if not f.exists():
        return FALLBACK_PACE, "fallback (no probe results)"
    ts = [float(r["time"]) for r in csv.DictReader(f.open())][-6:]
    if len(ts) < 3:
        return FALLBACK_PACE, "fallback (too few epochs)"
    return (ts[-1] - ts[0]) / (len(ts) - 1) / 60.0, f"measured over {len(ts)-1} epochs"


def epochs_that_fit(pace: float) -> int:
    now = dt.datetime.now(dt.timezone.utc)
    hh, mm = (int(x) for x in DEADLINE_HHMM.split(":"))
    deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if deadline <= now:                       # the window ends the next morning
        deadline += dt.timedelta(days=1)
    avail = (deadline - now).total_seconds() / 60.0
    return max(0, math.floor(avail / pace) - 1)   # 1-epoch buffer


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    if (RUNS / NAME / "DONE").exists():
        print(f"[{time.strftime('%H:%M:%S')}] {NAME} already DONE", flush=True)
        print("B1C RUN COMPLETE failed=[]", flush=True)
        return

    pace, how = observed_pace()
    fit = epochs_that_fit(pace)
    epochs = min(MAX_EPOCHS, fit)
    print(f"[{time.strftime('%H:%M:%S')}] pace {pace:.2f} min/ep ({how}); "
          f"{fit} epochs fit before {DEADLINE_HHMM} UTC -> budget {epochs}",
          flush=True)
    if epochs < 10:
        print(f"[{time.strftime('%H:%M:%S')}] refusing to launch: only "
              f"{epochs} epoch(s) fit before the deadline", flush=True)
        print("B1C RUN COMPLETE skipped=[insufficient_window]", flush=True)
        return

    argv = [PY, str(EXP / "scripts" / "train_unified.py"),
            "--name", NAME, "--mode", "pair", "--fuse", "E30",
            "--data", "configs/dronevehicle_v2_paired.yaml",
            "--rgb-yaml", "configs/dronevehicle_v2_rgb.yaml",
            "--epochs", str(epochs), "--workers", "24",
            "--fuse-warmup", "5", "--fuse-ramp", "3",
            "--close-mosaic", str(max(1, round(0.125 * epochs)))]
    log = open(LOGS / f"clean_{NAME}.log", "a")
    log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {NAME} "
              f"epochs={epochs} pace={pace:.2f} ===\n")
    log.flush()
    print(f"[{time.strftime('%H:%M:%S')}] launching {NAME} ({epochs} ep)",
          flush=True)
    rc = subprocess.call(argv, cwd=str(EXP), stdout=log, stderr=subprocess.STDOUT)
    log.close()
    print(f"[{time.strftime('%H:%M:%S')}] {NAME} rc={rc}", flush=True)
    print(f"B1C RUN COMPLETE failed={[] if rc == 0 else [NAME]}", flush=True)


if __name__ == "__main__":
    main()
