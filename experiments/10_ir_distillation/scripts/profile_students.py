#!/usr/bin/env python3
"""Profile student checkpoints (params, GFLOPs, A100 latency/FPS).

Forward only (``_predict_once``), batch 1, native 512x640 IR input. Includes the
E00 reference so the latency claim is against the exact baseline.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

EXP10 = Path(__file__).resolve().parents[1]

DEFAULTS = [
    ("e00_ir", "../09_graph_fusion/runs/e00_ir/weights/best.pt", 3),
    ("student_12s_decomp", "runs/student_12s_decomp/weights/best.pt", 3),
    ("student_12s_direct", "runs/student_12s_direct/weights/best.pt", 3),
    ("student_12n_gt", "runs/student_12n_gt/weights/best.pt", 3),
    ("student_12n_decomp", "runs/student_12n_decomp/weights/best.pt", 3),
]


def bench(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    if not str(device).startswith("cuda"):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) * 1000.0 / iters
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def profile(label, path, ch, device, warmup, iters, h, w):
    from ultralytics import YOLO
    from torch.utils.flop_counter import FlopCounterMode

    p = Path(path)
    if not p.is_absolute():
        p = (EXP10 / p).resolve()
    if not p.exists():
        return None
    yolo = YOLO(str(p))
    model = yolo.model.to(device).float().eval()
    model.fp16 = False
    params = sum(x.numel() for x in model.parameters())
    x32 = torch.zeros(1, ch, h, w, device=device)

    def fwd(t):
        with torch.inference_mode():
            return model._predict_once(t)

    with FlopCounterMode(display=False) as fc:
        fwd(x32)
    gflops = fc.get_total_flops() / 1e9
    lat32 = bench(lambda: fwd(x32), warmup, iters, device)

    model.half()
    x16 = x32.half()
    lat16 = bench(lambda: fwd(x16), warmup, iters, device)
    model.float()

    rec = {
        "model": label, "weights": str(p),
        "input": [1, ch, h, w],
        "params_total_m": round(params / 1e6, 3),
        "gflops_fp32": round(gflops, 2),
        "latency_fp32_ms": round(lat32, 3),
        "latency_fp16_ms": round(lat16, 3),
        "fps_fp32": round(1000.0 / lat32, 1),
        "fps_fp16": round(1000.0 / lat16, 1),
    }
    print(f"[profile] {label:22s} {rec['params_total_m']:6.2f} M "
          f"{gflops:7.2f} GFLOPs  fp32 {lat32:6.2f} ms ({rec['fps_fp32']:6.1f} FPS)  "
          f"fp16 {lat16:6.2f} ms", flush=True)
    del model, yolo, x32, x16
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/profiling_students.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    rows = [r for r in (
        profile(label, path, ch, args.device, args.warmup, args.iters,
                args.height, args.width)
        for label, path, ch in DEFAULTS
    ) if r]
    out = Path(args.out)
    if not out.is_absolute():
        out = EXP10 / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "device": args.device, "input_hw": [args.height, args.width],
        "warmup": args.warmup, "iters": args.iters,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": rows,
    }, indent=1))
    print(f"[profile] wrote {out}")


if __name__ == "__main__":
    main()
