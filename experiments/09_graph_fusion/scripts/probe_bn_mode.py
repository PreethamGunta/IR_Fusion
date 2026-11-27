#!/usr/bin/env python3
"""A2 probe — is the near-zero validation a BN running-statistics artefact?

Train mode normalises with **batch** statistics; validation with the stored
**running** statistics. If the fuse path shifts the feature distribution, the
running statistics can be badly stale while training loss looks healthy — which
is exactly the pattern the fused runs show (normal train loss, val AP near
zero, slow recovery).

The probe computes the same detection loss twice on identical real batches:

  * `running` — model in train mode (so the loss head receives raw features)
    but every BatchNorm forced to eval(), i.e. using the stored statistics;
  * `batch`   — everything in train mode, i.e. using per-batch statistics.

A large `running / batch` ratio means the stored statistics do not describe the
features the model is actually producing: a train/eval mismatch, not a feature
quality problem. Running statistics are snapshotted and restored, so a
checkpoint is never modified.

Usage (CPU is enough and avoids contending with a live training job):
    PYTHONPATH=src:. python scripts/probe_bn_mode.py \
        --weights runs/e30_screening/weights/best.pt --batches 6
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from ultralytics.data.dataset import YOLODataset

EXP = Path(__file__).resolve().parents[1]


def load_model(weights: Path):
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    # in-progress checkpoints store the EMA weights and leave `model` as None
    model = ckpt.get("model") or ckpt.get("ema") or ckpt
    model = model.float().cpu()
    model.args = type("A", (), {
        "box": 7.5, "cls": 0.5, "dfl": 1.5, "overlap_mask": True,
        "mask_ratio": 4, "nc": 5, "single_cls": False, "data": None})()
    model.eval()
    return model


def make_batches(n_batches: int, batch: int, split: str = "val"):
    from gfdata.unified import make_dataset
    ds = make_dataset(str(EXP / "configs/dronevehicle_v2_paired.yaml"), split,
                      modality="pair", mode=split, imgsz=640, batch=batch,
                      rgb_yaml=str(EXP / "configs/dronevehicle_v2_rgb.yaml"))
    ds.mosaic = False
    out = []
    for b in range(n_batches):
        samples = [ds[i] for i in range(b * batch, (b + 1) * batch)]
        out.append(YOLODataset.collate_fn(samples))
    for b in out:                      # the trainer normalises; do it here
        b["img"] = b["img"].float() / 255.0
    return out


def bn_layers(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]


def set_bn_training(model, training: bool):
    for m in bn_layers(model):
        m.training = training


def loss_of(model, batch) -> dict:
    with torch.no_grad():
        total, items = model.loss(batch)
    d = {"total": float(total.detach().sum())}
    if isinstance(items, dict):
        for k, v in items.items():
            d[k] = float(torch.as_tensor(v).detach().sum())
    else:
        for i, v in enumerate(items):
            d[f"item{i}"] = float(torch.as_tensor(v).detach().sum())
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", default="train", choices=["train", "val"],
                    help="train is the meaningful split: running statistics are "
                         "estimated there, so a stale-stats artefact shows up as "
                         "running-loss >> batch-loss on training batches")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    model = load_model(Path(args.weights))
    batches = make_batches(args.batches, args.batch, args.split)
    print(f"loaded {args.weights}; {len(batches)} batches of {args.batch}",
          flush=True)

    # snapshot BN buffers so the checkpoint's state is never mutated
    snap = {n: (b.running_mean.clone(), b.running_var.clone(),
                b.num_batches_tracked.clone())
            for n, b in model.named_modules()
            if isinstance(b, torch.nn.BatchNorm2d)}

    model.train()                 # loss head needs raw features
    set_bn_training(model, False)  # ... but normalise with running stats
    run = [loss_of(model, b) for b in batches]

    # restore stats that the train-mode pass above may have touched
    for n, b in model.named_modules():
        if isinstance(b, torch.nn.BatchNorm2d):
            rm, rv, nb = snap[n]
            b.running_mean.copy_(rm); b.running_var.copy_(rv)
            b.num_batches_tracked.copy_(nb)

    set_bn_training(model, True)   # per-batch statistics
    bat = [loss_of(model, b) for b in batches]

    def mean(xs, k):
        return sum(x[k] for x in xs) / len(xs)

    r, b_ = mean(run, "total"), mean(bat, "total")
    print(f"\nrunning-stat loss: {r:.3f}   batch-stat loss: {b_:.3f}   "
          f"ratio: {r / b_ if b_ else float('inf'):.2f}x")
    for k in run[0]:
        if k != "total" and k in bat[0]:
            print(f"  {k:12s} running={mean(run, k):7.3f}  "
                  f"batch={mean(bat, k):7.3f}")
    verdict = ("MISMATCH: stored statistics do not describe the features"
               if b_ and r / b_ > 1.5 else
               "no strong mismatch on this checkpoint")
    print(f"\nverdict: {verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "weights": str(args.weights),
            "running": {k: mean(run, k) for k in run[0]},
            "batch": {k: mean(bat, k) for k in bat[0]},
            "ratio": (r / b_) if b_ else None,
            "verdict": verdict,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
