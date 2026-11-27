#!/usr/bin/env python3
"""Matched IR-only control for the unified full-data comparison.

Pre-flight (rules.md PRE-FLIGHT REVIEW RULE):
- **Goal**: IR-only YOLO12s on the FULL IR split through the *same* unified
  pipeline as `e30_unified` — same augmentation implementation, same label
  path, same 80 epochs, same pinned optimizer (MuSGD lr 1e-2, momentum 0.9,
  warmup_bias_lr 0), same seed — so the only intended difference between the
  arms is fusion.
- **Duplicate check**: E00 exists, but it trained with the *native*
  Ultralytics augmentation and an auto-resolved optimizer; this control
  isolates the unified-pipeline delta (rules.md A1–A4 in the post-mortem).
- **Resource profile**: 80 epochs at ~2.5–3 min/epoch ≈ 3.5–4 h; the window
  ends 08:14 UTC with the GAT finishing ~21:00, leaving ≥ 6 h of margin.
- **Comparability**: e00_ir_unified (control) vs e30_unified (treatment);
  data volume, frame set, schedule, seed and optimizer identical.
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


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    name = "e00_ir_unified"
    if (RUNS / name / "DONE").exists():
        print(f"[{time.strftime('%H:%M:%S')}] {name} already DONE, skipping", flush=True)
        return
    log = open(LOGS / f"control_{name}.log", "a")
    log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {name} ===\n")
    log.flush()
    argv = [PY, str(EXP / "scripts" / "train_unified.py"),
            "--name", name, "--mode", "ir",
            "--data", "configs/dronevehicle_v2_ir.yaml",
            "--epochs", "80", "--workers", "24"]
    print(f"[{time.strftime('%H:%M:%S')}] launching {name}", flush=True)
    rc = subprocess.call(argv, cwd=str(EXP), stdout=log, stderr=subprocess.STDOUT)
    log.close()
    print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc}", flush=True)
    print(f"CONTROL RUN COMPLETE failed={[] if rc == 0 else [name]}", flush=True)


if __name__ == "__main__":
    main()
