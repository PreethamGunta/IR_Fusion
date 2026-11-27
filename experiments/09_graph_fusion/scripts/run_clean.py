#!/usr/bin/env python3
"""Clean full-data GAT run through the unified pipeline.

IR-only on full data already exists: E00 (runs/e00_ir, training-val 0.6306,
locked test 0.5783). The unified-pipeline IR arm is therefore NOT rerun here;
E00 is the reference. The only missing number is the fused arm under the
unified loader:

  e30_unified  -- GF-YOLO GAT (E30) on the full paired split, full
                  augmentation (mosaic + close_mosaic, flips, affine,
                  HSV/value jitter, erasing), same pretrained init.

Comparability note: E00 used Ultralytics' native augmentation; the GAT uses
the unified implementation with the same hyperparameters. If time remains, an
IR-only run through the unified loader quantifies that residual difference.

Skip-if-DONE, continue-on-failure. Logs: logs/clean_<name>.log.
Final line: "CLEAN RUN COMPLETE failed=[...]".
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
EPOCHS = "80"
IR_DATA = "configs/dronevehicle_v2_ir.yaml"
PAIR_DATA = "configs/dronevehicle_v2_paired.yaml"
RGB_DATA = "configs/dronevehicle_v2_rgb.yaml"


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    failed = []

    def run(name: str, argv: list, workers_note: str) -> None:
        if (RUNS / name / "DONE").exists():
            print(f"[{time.strftime('%H:%M:%S')}] {name} already DONE, skipping", flush=True)
            return
        log = open(LOGS / f"clean_{name}.log", "a")
        log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {name} ===\n")
        log.flush()
        print(f"[{time.strftime('%H:%M:%S')}] launching {name} ({workers_note})", flush=True)
        rc = subprocess.call([PY, str(EXP / "scripts" / "train_unified.py")] + argv,
                             cwd=str(EXP), stdout=log, stderr=subprocess.STDOUT)
        log.close()
        print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc}", flush=True)
        if rc != 0 and not (RUNS / name / "DONE").exists():
            failed.append(name)

    run("e30_unified", [
        "--name", "e30_unified", "--mode", "pair", "--fuse", "E30",
        "--data", PAIR_DATA, "--rgb-yaml", RGB_DATA,
        "--epochs", EPOCHS, "--workers", "20",
    ], "GAT E30, 20 workers")

    print(f"CLEAN RUN COMPLETE failed={failed}", flush=True)


if __name__ == "__main__":
    main()
