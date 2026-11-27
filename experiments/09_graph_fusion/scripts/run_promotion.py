#!/usr/bin/env python3
"""Promotion phase on allocation 10187 (32 CPUs, one GPU).

Stage 1 (concurrent):
  - e00_ir_noaug_screening: IR-only on the 1/3 screening subset with ALL
    augmentation disabled. This is the matched-recipe control for the
    paired ladder: the earlier control (e00_ir_screening) used the full
    default recipe (mosaic, flips, HSV) while the paired pipeline trains
    with no augmentation, confounding the fusion-vs-IR comparison.
  - e30_full: the ladder leader (window-graph GAT fuse) on the FULL
    training split -- the promotion run for the locked-test claim.

E21 full-train does not fit this allocation (attention ~7 min/epoch at
full data); it is queued for the next window.

Skip-if-DONE, continue-on-failure. Logs: logs/promo_<name>.log.
Final line: "PROMOTION BATCH COMPLETE failed=[...]".
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


def gfyolo_full(name: str, fuse: str, workers: int) -> list:
    return [PY, str(EXP / "scripts" / "train_gfyolo.py"),
            "--name", name, "--fuse", fuse,
            "--data", "configs/dronevehicle_v2_paired.yaml",
            "--rgb-yaml", "configs/dronevehicle_v2_rgb.yaml",
            "--paired-rot90", "--workers", str(workers)]


def ref_noaug(name: str, workers: int) -> list:
    return [PY, str(EXP / "scripts" / "train_ref.py"),
            "--name", name,
            "--data", "configs/dronevehicle_v2_ir_screening.yaml",
            "--epochs", "100", "--no-aug", "--workers", str(workers)]


def done(name: str) -> bool:
    return (RUNS / name / "DONE").exists()


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    failed = []

    def run_stage(runs):
        procs = []
        for name, argv in runs:
            if done(name):
                print(f"[{time.strftime('%H:%M:%S')}] {name} already DONE, skipping", flush=True)
                continue
            log = open(LOGS / f"promo_{name}.log", "a")
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

    print(f"[{time.strftime('%H:%M:%S')}] stage 1: e30_full + IR no-aug control", flush=True)
    run_stage([
        ("e30_full", gfyolo_full("e30_full", "E30", 18)),
        ("e00_ir_noaug_screening", ref_noaug("e00_ir_noaug_screening", 12)),
    ])
    print(f"PROMOTION BATCH COMPLETE failed={failed}", flush=True)


if __name__ == "__main__":
    main()
