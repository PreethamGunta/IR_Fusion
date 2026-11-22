"""Cache teacher prediction graphs for the training split.

Usage (inside the salloc allocation):
  srun --jobid="$SLURM_JOB_ID" --overlap --gres=gpu:1 \
    python -m graphkd.graph.cache_teacher --weights runs/ir_teacher/weights/best.pt \
    --data configs/ir.yaml --split train --out data/cache/ir --name ir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from graphkd import paths
from graphkd.data.dataset import make_dataset, make_paired_dataset
from graphkd.graph.cache import CacheWriter
from graphkd.graph.candidates import decode_predictions


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(paths.CONFIGS / "ir.yaml"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="teacher")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--paired-yaml", default=None)
    ap.add_argument("--rgb-yaml", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(args.weights)
    model = yolo.model.to(device).eval()
    head = model.model[-1]
    head.training = True  # return per-scale raw predictions, not NMS output
    nc = int(model.nc)
    strides = [int(s) for s in model.stride.tolist()]

    if args.paired:
        ds = make_paired_dataset(args.paired_yaml or str(paths.CONFIGS / "paired.yaml"),
                                 args.rgb_yaml or str(paths.CONFIGS / "rgb.yaml"),
                                 args.split, mode="train", imgsz=args.imgsz, batch=16)
    else:
        ds = make_dataset(args.data, args.split, mode="train", imgsz=args.imgsz, batch=16)
    n = len(ds) if not args.limit else min(args.limit, len(ds))

    feat_dim = None
    writer = CacheWriter(Path(args.out), {
        "name": args.name, "weights": str(Path(args.weights).resolve()),
        "weights_sha256": sha256(Path(args.weights)), "nc": nc, "k": args.k,
        "imgsz": args.imgsz, "split": args.split, "paired": args.paired,
        "n": n, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "strides": strides,
    })
    t0 = time.time()
    with torch.no_grad():
        for i in range(n):
            batch = ds[i]
            img = batch["img"].float().unsqueeze(0).to(device) / 255.0
            feats = []
            hook = head.register_forward_hook(lambda m, inp, out: feats.append(inp[0]))
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model(img)
            finally:
                hook.remove()
            raw = out[1] if isinstance(out, tuple) else out
            if isinstance(raw, dict) and "boxes" not in raw:
                raw = raw.get("one2many", raw)
            nodes, dense = decode_predictions(raw, feats[0], torch.tensor(strides),
                                              nc, k=args.k, conf=args.conf)
            nd = nodes[0]
            rec = {
                "boxes": nd["boxes"].cpu(), "cls": nd["cls"].cpu(),
                "obj": nd["obj"].cpu(), "feat": nd["feat"].cpu(),
                "anchor": nd["anchor"].cpu(),
                "dense_obj": dense["obj"][0].cpu(), "dense_cls": dense["cls"][0].cpu(),
                "dense_box": dense["box"][0].cpu(),
            }
            feat_dim = int(rec["feat"].shape[1])
            writer.add(Path(batch["im_file"]).stem.replace("_PreviewData", "").replace("_RGB", ""), rec)
            if (i + 1) % 100 == 0:
                dt = time.time() - t0
                print(f"[{args.name}] {i+1}/{n} {dt:.1f}s ({dt/(i+1):.3f}s/img)", flush=True)
    writer.meta["feat_dim"] = feat_dim
    writer.close()
    print(f"[{args.name}] wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
