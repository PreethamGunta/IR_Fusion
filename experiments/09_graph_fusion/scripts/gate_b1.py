#!/usr/bin/env python3
"""Print the B1 gate comparison: probe vs the matched IR-only control.

Read-only. Encodes the pre-registered thresholds from `PREFLIGHT_B1.md` §6 so
the 03:00 decision is not made from memory or from a moving bar.

    PYTHONPATH=src:. python scripts/gate_b1.py
"""
from __future__ import annotations

import csv
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
RUNS = EXP / "runs"

BARS = {5: 0.25, 10: 0.33}      # pre-registered pass bar (probe)
COLS = [5, 10, 15, 20]


def curve(name: str) -> dict[int, float]:
    f = RUNS / name / "results.csv"
    if not f.exists():
        return {}
    return {int(float(r["epoch"])): float(r["metrics/mAP50-95(B)"])
            for r in csv.DictReader(f.open())}


def row(name: str, c: dict[int, float]) -> str:
    cells = "".join(f"{c[k]:9.4f}" if k in c else f"{'-':>9s}" for k in COLS)
    return f"{name:22s}{cells}"


def main() -> None:
    ctrl = curve("e00_ir_unified")
    probe = curve("b1_warmup_probe")
    anc = curve("e30_unified")
    scr = curve("e30_screening")

    print(f"{'run':22s}" + "".join(f"{'ep' + str(k):>9s}" for k in COLS))
    for n, c in (("e00_ir_unified (ref)", ctrl), ("b1_warmup_probe", probe),
                 ("e30_unified (no warmup)", anc), ("e30_screening (1/3)", scr)):
        print(row(n, c))

    print()
    missing = [k for k in BARS if k not in probe]
    if missing:
        print(f"probe incomplete: no ep{missing} yet — decision not available")
        return

    ok = True
    for k, bar in BARS.items():
        p, c = probe.get(k), ctrl.get(k)
        frac = (p / c) if (p is not None and c) else float("nan")
        passed = p is not None and p >= bar
        ok &= passed
        print(f"ep{k:<3d} probe={p:.4f} bar={bar:.2f} "
              f"control={c:.4f} ratio={frac:.2f}x -> "
              f"{'PASS' if passed else 'FAIL'}")

    rising = (20 in probe and 15 in probe and probe[20] >= probe[15])
    print(f"\nstill rising ep15->ep20: {'yes' if rising else 'no'}")

    if ok and rising:
        verdict = ("MECHANISM CONFIRMED -> launch the 40-epoch consolidation "
                   "(scripts/run_b1_consol.py)")
    elif ok or (probe.get(5, 0) >= 0.12):
        verdict = ("PARTIAL -> consolidate the same design "
                   "(scripts/run_b1_consol.py)")
    else:
        verdict = ("MECHANISM REFUTED -> no consolidation; run the A3 ceiling "
                   "probe and record the refutation")
    print(verdict)


if __name__ == "__main__":
    main()
