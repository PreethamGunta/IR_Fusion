#!/usr/bin/env python3
"""Wake-on-critical monitor for the pair ladder batch (step 10157.3).

Ticks every 10 min. INFORMATIONAL events keep looping; CRITICAL events
print details and exit(1) so the harness wakes the agent immediately:
  - a rung exits with rc != 0 and no DONE marker
  - CUDA OOM / FloatingPointError / Traceback in any rung log
  - collapse recurrence (>=91 epochs, last AP < 0.05, best > 0.15)
  - no epoch progress for ~30 min while the batch is incomplete
    (dead driver / dataloader stall)
Batch completion and allocation end exit(0) with a full report.
"""
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

EXP = Path("/home2/home/preetham_g/gate_ir_release/experiments/09_graph_fusion")
RUNS = EXP / "runs"
# Supervised-run binding: every instance declares the batch log it watches, the
# rungs it expects, and the completion marker; stages hand off by starting a
# new instance with new env values (allocation 10187).
ALLOC = os.environ.get("MON_ALLOC", "10187")
LOG = Path(os.environ.get("MON_LOG", str(EXP / "logs" / "clean_batch.log")))
RUNGS = os.environ.get("MON_RUNGS", "e30_unified").split(",")
COMPLETE_MARKER = os.environ.get("MON_COMPLETE", "CLEAN RUN COMPLETE")
ERR_PREFIX = os.environ.get("MON_ERR_PREFIX", "clean_")
FLAG = EXP / "logs" / ".mon_flags"
FLAG.mkdir(exist_ok=True)


def read_results(name):
    f = RUNS / name / "results.csv"
    if not f.exists():
        return []
    with open(f) as fh:
        return [(int(float(r["epoch"])), float(r["metrics/mAP50-95(B)"]))
                for r in csv.DictReader(fh)]


def tail(path, n=15):
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except Exception:
        return "(unreadable)"


def batch_text():
    try:
        return LOG.read_text(errors="replace")
    except Exception:
        return ""


def alloc_alive():
    out = subprocess.run(["squeue", "-h", "-j", ALLOC],
                        capture_output=True, text=True).stdout
    return out.strip() != ""


prev_sum, stall, prev_rcs = -1, 0, 0
prev_rcs = batch_text().count(" rc=")  # don't re-alert on historical exits
prev_ep = {}
pace_bad = {}

while True:
    time.sleep(600)
    t = time.strftime("%H:%M", time.gmtime())
    bl = batch_text()
    blines = bl.splitlines()
    eps = {n: read_results(n) for n in RUNGS}

    # informational: collapse fix proven in training
    for n, aps in eps.items():
        if aps and aps[-1][0] >= 91 and aps[-1][1] > 0.05 and not (FLAG / n).exists():
            (FLAG / n).write_text("ok")
            print(f"FIX PROVEN: {n} passed ep91 healthy "
                  f"(ep{aps[-1][0]} AP={aps[-1][1]:.4f})", flush=True)

    # critical: collapse recurrence
    for n, aps in eps.items():
        if len(aps) >= 91:
            best = max(a for _, a in aps)
            if aps[-1][1] < 0.05 and best > 0.15:
                print(f"CRITICAL: COLLAPSE RECURRED in {n} "
                      f"(best={best:.4f}, ep{aps[-1][0]}={aps[-1][1]:.4f})", flush=True)
                sys.exit(1)

    # batch complete -> full report
    if COMPLETE_MARKER in bl:
        print("ALERT: clean batch COMPLETE", flush=True)
        print("\n".join(l for l in blines
                        if "rc=" in l or "stage" in l or "launching" in l), flush=True)
        for n, aps in eps.items():
            if aps:
                b = max(aps, key=lambda p: p[1])
                print(f"{n}: {len(aps)} epochs, best ep{b[0]} AP={b[1]:.4f}, "
                      f"last {aps[-1][1]:.4f}", flush=True)
        sys.exit(0)

    # rung exit lines: wake the agent at every transition (rc 0 or not)
    rcs = [l for l in blines if " rc=" in l]
    if len(rcs) > prev_rcs:
        prev_rcs = len(rcs)
        m = re.search(r"\] (\S+) rc=(\d+)", rcs[-1])
        name, rc = (m.group(1), int(m.group(2))) if m else ("?", -1)
        if rc == 0 or (RUNS / name / "DONE").exists():
            print(f"NOTIFY: {name} finished rc={rc} (driver continues if more stages)", flush=True)
        else:
            print(f"CRITICAL: {name} FAILED rc={rc}. Log tail:\n"
                  f"{tail(EXP / 'logs' / f'promo_{name}.log')}", flush=True)
        sys.exit(0)

    # critical: hard errors in any rung log
    for n in RUNGS:
        p = EXP / "logs" / f"{ERR_PREFIX}{n}.log"
        if p.exists():
            # scan only recent output: historical (already-triaged) tracebacks
            # must not re-alert every tick
            txt = p.read_text(errors="replace")[-4000:]
            if ("CUDA out of memory" in txt or "FloatingPointError" in txt
                    or "\nTraceback" in txt):
                print(f"CRITICAL: error in {n}. Log tail:\n{tail(p, 20)}", flush=True)
                sys.exit(1)

    # critical: stall / dead driver (no new epochs for ~30 min)
    tot = sum(len(a) for a in eps.values())
    done_all = all((RUNS / n / "DONE").exists() for n in RUNGS)
    stall = stall + 1 if (tot == prev_sum and not done_all) else 0
    prev_sum = tot
    if stall >= 3:
        print(f"CRITICAL: no epoch progress for ~30 min "
              f"(driver dead or dataloader stall). Batch log tail:\n{tail(LOG, 8)}",
              flush=True)
        sys.exit(1)

    # pace: sustained slowdown only. Tick-count comparisons alias badly at
    # ~5 min/epoch: a 10-min tick legitimately contains 1 or 3 epochs, and a
    # single 1-epoch tick previously killed the supervisor on a healthy run
    # (2026-11-23, e30_unified ep34, actual pace 4.95 min/ep). Use the run's
    # own clock (results.csv `time`) and require two consecutive bad ticks.
    for n in RUNGS[:2]:
        cur = len(eps[n])
        if not (3 <= cur < 50) or (RUNS / n / "DONE").exists():
            pace_bad.pop(n, None)
            continue
        try:
            rows = list(csv.DictReader((RUNS / n / "results.csv").open()))
            ts = [float(r["time"]) for r in rows][-6:]
        except Exception:
            ts = []
        if len(ts) >= 6:
            per_ep = (ts[-1] - ts[0]) / 5 / 60.0
            pace_bad[n] = pace_bad.get(n, 0) + 1 if per_ep > 8.0 else 0
            if pace_bad[n] >= 2:
                print(f"CRITICAL: {n} sustained slowdown ({per_ep:.1f} min/ep "
                      f"over the last 5 epochs, threshold 8.0)", flush=True)
                sys.exit(1)

    prog = " ".join(f"{n.split('_')[0][1:]}:{len(a)}ep" for n, a in eps.items() if a)
    print(f"[{t}] {prog} | {blines[-1][:60] if blines else ''}", flush=True)

    if not alloc_alive():
        print(f"ALERT: allocation {ALLOC} ended", flush=True)
        sys.exit(0)
