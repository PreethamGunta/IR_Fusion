#!/usr/bin/env python3
"""Final screening phase on allocation 10187 (32 CPUs, one GPU).

Stage 1: E35 resume (ep95 -> 100; full-state checkpoint from 10157)
Stage 2: E10 (resume ep17) + E14 (fresh) + IR-only screening control,
         three lean runs concurrently (10 workers each; ~34 GB total)

The IR-only control (train_ref.py, configs/dronevehicle_v2_ir_screening.yaml)
is the missing same-budget baseline: without it the ladder ranks fusion
designs but cannot claim fusion beats IR-only on the 1/3-train budget.

Skip-if-DONE, continue-on-failure. Logs: logs/final_<name>.log.
Final line: "SCREENING FINAL COMPLETE failed=[...]".
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
DATA = "configs/dronevehicle_v2_paired_screening.yaml"
RGB = "configs/dronevehicle_v2_rgb_screening.yaml"


def gfyolo_argv(name: str, fuse: str, workers: int) -> list:
    return [PY, str(EXP / "scripts" / "train_gfyolo.py"),
            "--name", name, "--fuse", fuse, "--data", DATA,
            "--rgb-yaml", RGB, "--paired-rot90", "--workers", str(workers)]


def ref_argv(name: str, workers: int) -> list:
    return [PY, str(EXP / "scripts" / "train_ref.py"),
            "--name", name, "--data", "configs/dronevehicle_v2_ir_screening.yaml",
            "--epochs", "100", "--workers", str(workers)]


def done(name: str) -> bool:
    return (RUNS / name / "DONE").exists()


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    failed = []

    def run_stage(runs):
        """runs: list of (run_name, argv). Waits for all, records failures."""
        procs = []
        for name, argv in runs:
            if done(name):
                print(f"[{time.strftime('%H:%M:%S')}] {name} already DONE, skipping", flush=True)
                continue
            log = open(LOGS / f"final_{name}.log", "a")
            log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {name} ===\n")
            log.flush()
            print(f"[{time.strftime('%H:%M:%S')}] launching {name}", flush=True)
            procs.append((name, subprocess.Popen(argv, cwd=str(EXP),
                                                 stdout=log, stderr=subprocess.STDOUT)))
        for name, p in procs:
            rc = p.wait()
            print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc}", flush=True)
            if rc != 0 and not done(name):
                failed.append(name)

    print(f"[{time.strftime('%H:%M:%S')}] stage 1: E35 resume ep95->100 (solo)", flush=True)
    run_stage([("e35_screening", gfyolo_argv("e35_screening", "E35", 16))])

    print(f"[{time.strftime('%H:%M:%S')}] stage 2: E10 + E14 + IR-only control (3-way)", flush=True)
    run_stage([
        ("e10_screening", gfyolo_argv("e10_screening", "E10", 10)),
        ("e14_screening", gfyolo_argv("e14_screening", "E14", 10)),
        ("e00_ir_screening", ref_argv("e00_ir_screening", 10)),
    ])
    print(f"SCREENING FINAL COMPLETE failed={failed}", flush=True)


if __name__ == "__main__":
    main()
