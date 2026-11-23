#!/usr/bin/env python3
"""Sequential screening ladder driver (resume-safe, continue-on-failure).

Runs the rungs in order, skipping any with a DONE marker, and continues to
the next rung if one fails (each rung is independent). Per-rung logs:
logs/ladder_<name>.log. A final summary line is printed:
"LADDER BATCH COMPLETE failed=[...]".

E22 is deliberately NOT in the list: it is conditional on E21 > E14 and is
decided after these results are in.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
PY = sys.executable
RUNS = EXP / "runs"
LOGS = EXP / "logs"

RUNGS = ["E10", "E14", "E21", "E30", "E35"]
DATA = "configs/dronevehicle_v2_paired_screening.yaml"
RGB = "configs/dronevehicle_v2_rgb_screening.yaml"


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    failed, ran = [], []
    for rung in RUNGS:
        name = f"{rung.lower()}_screening"
        if (RUNS / name / "DONE").exists():
            print(f"[{time.strftime('%H:%M:%S')}] {name} already DONE, skipping", flush=True)
            continue
        argv = [PY, str(EXP / "scripts" / "train_gfyolo.py"),
                "--name", name, "--fuse", rung,
                "--data", DATA, "--rgb-yaml", RGB,
                "--paired-rot90", "--workers", "12"]
        log = open(LOGS / f"ladder_{name}.log", "a")
        log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {name} ===\n")
        log.flush()
        print(f"[{time.strftime('%H:%M:%S')}] starting {name}", flush=True)
        t0 = time.time()
        rc = subprocess.call(argv, cwd=str(EXP), stdout=log, stderr=subprocess.STDOUT)
        log.close()
        dt = (time.time() - t0) / 3600
        print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc} wall={dt:.2f}h", flush=True)
        ran.append((name, rc, round(dt, 2)))
        if rc != 0 and not (RUNS / name / "DONE").exists():
            failed.append(name)
    print(f"LADDER BATCH COMPLETE failed={failed} ran={ran}", flush=True)


if __name__ == "__main__":
    main()
