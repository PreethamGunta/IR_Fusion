#!/usr/bin/env python3
"""Run the E00 + E01 reference pair concurrently inside one srun step.

Resume-safe: DONE runs are skipped, partial runs resume from last.pt.
Phase 0.6 timing is read from the produced results.csv files.
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

JOBS = {
    "e00_ir": [PY, str(EXP / "scripts" / "train_ref.py"),
               "--name", "e00_ir", "--data", "configs/dronevehicle_v2_ir.yaml"],
    "e01_rgb": [PY, str(EXP / "scripts" / "train_ref.py"),
                "--name", "e01_rgb", "--data", "configs/dronevehicle_v2_rgb.yaml"],
}


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    procs = {}
    for name, argv in JOBS.items():
        if (RUNS / name / "DONE").exists():
            print(f"[{time.strftime('%H:%M:%S')}] {name} already DONE, skipping", flush=True)
            continue
        log = open(LOGS / f"ref_{name}.log", "a")
        log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {name} ===\n")
        log.flush()
        procs[name] = subprocess.Popen(argv, cwd=str(EXP), stdout=log, stderr=subprocess.STDOUT)
        print(f"[{time.strftime('%H:%M:%S')}] started {name}", flush=True)
    failed = []
    for name, p in procs.items():
        rc = p.wait()
        print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc}", flush=True)
        if rc != 0 and not (RUNS / name / "DONE").exists():
            failed.append(name)
    if failed:
        print(f"FAILED: {failed}", flush=True)
        sys.exit(1)
    print("all reference runs complete", flush=True)


if __name__ == "__main__":
    main()
