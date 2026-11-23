#!/usr/bin/env python3
"""Priority-ordered screening ladder (one training per GPU; lean rungs pair).

GPU-memory-aware ordering — the 2026-11-22 OOM lesson: two graph rungs
sharing 49 GB is not safe; E35 auto-reduced itself to batch=8 (recipe and
pace broken: ~10 min/epoch vs ~3.5 for E30 at batch=16). Graph/attention
rungs therefore run SOLO with the full GPU; only the lean baselines pair.

  stage 1: E35 solo (12 workers, fresh)         - core novelty, with the
           BN-stat watchdog + fuse input clamping + fp32-always fuse
  stage 2: E10 + E14 concurrent (8 workers each; E10 auto-resumes ep17,
           E14 fresh)

If the allocation ends during stage 4 both auto-resume next allocation; the
salvaged ep80 baselines (E10 0.2258 / E14 0.2068, trained pre-collapse under
identical transforms) cover the comparison meanwhile.

Each run is launched via scripts/train_gfyolo.py (auto-resume from last.pt +
DONE marker). Failures don't block later stages. Per-run logs:
logs/pair_<name>.log. Final line: "PAIR BATCH COMPLETE failed=[...]".
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


def argv_for(name: str, fuse: str, workers: int) -> list:
    return [PY, str(EXP / "scripts" / "train_gfyolo.py"),
            "--name", name, "--fuse", fuse, "--data", DATA,
            "--rgb-yaml", RGB, "--paired-rot90", "--workers", str(workers)]


def done(name: str) -> bool:
    return (RUNS / name / "DONE").exists()


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    failed = []

    def run_stage(rungs, workers):
        """rungs: list of (run_name, fuse_id). Returns when all exit."""
        procs = []
        for name, fuse in rungs:
            if done(name):
                print(f"[{time.strftime('%H:%M:%S')}] {name} already DONE, skipping", flush=True)
                continue
            log = open(LOGS / f"pair_{name}.log", "a")
            log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%S')} {name} ===\n")
            log.flush()
            print(f"[{time.strftime('%H:%M:%S')}] launching {name} (workers={workers})", flush=True)
            procs.append((name, subprocess.Popen(argv_for(name, fuse, workers), cwd=str(EXP),
                                                 stdout=log, stderr=subprocess.STDOUT)))
        for name, p in procs:
            rc = p.wait()
            print(f"[{time.strftime('%H:%M:%S')}] {name} rc={rc}", flush=True)
            if rc != 0 and not done(name):
                failed.append(name)

    print(f"[{time.strftime('%H:%M:%S')}] stage 1: E35 solo (core novelty, fresh, hardened)", flush=True)
    run_stage([("e35_screening", "E35")], 12)
    print(f"[{time.strftime('%H:%M:%S')}] stage 2: E10 + E14 concurrent (lean baselines)", flush=True)
    run_stage([("e10_screening", "E10"), ("e14_screening", "E14")], 8)
    print(f"PAIR BATCH COMPLETE failed={failed}", flush=True)


if __name__ == "__main__":
    main()
