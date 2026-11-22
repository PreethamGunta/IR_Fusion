"""Measure how long the real sweep will actually take, before committing to it.

Runs a few real epochs of each model on the real split and reads the *marginal*
epoch time out of Ultralytics' own results.csv, so one-off startup (weight
download, dataset scan, AMP check, cache build) is excluded from the per-epoch
figure and charged once instead. Each model runs in its own subprocess so GPU
memory from one does not distort the next.

    python bench/scripts/estimate_runtime.py --epochs 3 --target 100
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
ROOT = BENCH.parent
DATA = ROOT / "data" / "UNIRI-TID" / "yolo" / "uniri.yaml"
PROBE = BENCH / "runs" / "_probe"

MODELS = [
    ("v8s",         "weights/yolov8s.pt",  []),
    ("v11s",        "weights/yolo11s.pt",  []),
    ("v12s",        "weights/yolo12s.pt",  []),
    ("v8s_weather", "weights/yolov8s.pt",  ["--weather-aug"]),
    ("v10s",        "weights/yolov10s.pt", []),
    ("v8s_p2",      "yolov8-p2.yaml",      []),
    ("rtdetrl",     "weights/rtdetr-l.pt", []),
]


def probe_one(model, name, epochs, imgsz, batch, weather):
    import os
    os.environ.setdefault("YOLO_CONFIG_DIR", str(BENCH / ".ultralytics"))
    from ultralytics import YOLO

    kw = dict(data=str(DATA), epochs=epochs, imgsz=imgsz, batch=batch, seed=0,
              project=str(PROBE), name=name, exist_ok=True,
              pretrained=model.endswith(".pt"), val=True, plots=False, verbose=False,
              hsv_h=0.0, hsv_s=0.0, hsv_v=0.4, degrees=0.0, fliplr=0.5, flipud=0.0)
    if weather:
        kw.update(hsv_v=0.8, scale=0.7, translate=0.2, erasing=0.4, mosaic=1.0)
    t0 = time.time()
    YOLO(model).train(**kw)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3, help="probe epochs per model")
    ap.add_argument("--target", type=int, default=100, help="epochs the real run will use")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--one", help="internal: probe a single model in this process")
    ap.add_argument("--model")
    ap.add_argument("--weather", action="store_true")
    args = ap.parse_args()

    if args.one:
        wall = probe_one(args.model, args.one, args.epochs, args.imgsz, args.batch, args.weather)
        print(f"__WALL__ {wall:.1f}")
        return

    rows = []
    for name, model, extra in MODELS:
        cmd = [sys.executable, __file__, "--one", name, "--model", model,
               "--epochs", str(args.epochs), "--imgsz", str(args.imgsz),
               "--batch", str(args.batch)]
        if "--weather-aug" in extra:
            cmd.append("--weather")
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        wall = time.time() - t0
        if p.returncode != 0:
            print(f"{name:>12}  FAILED\n{p.stdout[-800:]}\n{p.stderr[-800:]}")
            rows.append(dict(name=name, model=model, failed=True))
            continue

        csv_path = PROBE / name / "results.csv"
        times = [float(r["time"]) for r in csv.DictReader(open(csv_path))]
        # cumulative seconds -> marginal cost of one steady-state epoch
        per_epoch = (times[-1] - times[0]) / max(len(times) - 1, 1) if len(times) > 1 else times[0]
        startup = wall - times[-1]
        est = startup + per_epoch * args.target
        rows.append(dict(name=name, model=model, per_epoch_s=round(per_epoch, 1),
                         startup_s=round(startup, 1), probe_wall_s=round(wall, 1),
                         est_target_s=round(est, 1)))
        print(f"{name:>12}  {per_epoch:6.1f} s/epoch  +{startup:5.1f} s startup  "
              f"-> {est / 3600:5.2f} h for {args.target} epochs", flush=True)

    ok = [r for r in rows if not r.get("failed")]
    total = sum(r["est_target_s"] for r in ok)
    print(f"\n{'model':>12} {'s/epoch':>9} {'startup':>9} {f'{args.target}ep (h)':>12}")
    print("-" * 45)
    for r in ok:
        print(f"{r['name']:>12} {r['per_epoch_s']:9.1f} {r['startup_s']:9.1f} {r['est_target_s'] / 3600:12.2f}")
    print("-" * 45)
    print(f"{'TOTAL':>12} {'':>9} {'':>9} {total / 3600:12.2f}")
    print("\n(excludes the per-condition eval pass at the end of each run)")

    out = BENCH / "results" / "runtime_estimate.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(probe_epochs=args.epochs, target_epochs=args.target,
                                   imgsz=args.imgsz, batch=args.batch,
                                   total_est_hours=round(total / 3600, 2), models=rows), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
