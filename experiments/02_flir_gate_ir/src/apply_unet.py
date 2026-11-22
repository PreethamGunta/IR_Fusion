"""Apply the trained U-Net dehazer to a directory of images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .paths import RESULTS
from .unet_dehazer import UNetDehazer

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _pad_to_multiple(x: torch.Tensor, m: int = 16) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, h, w = x.shape
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply the U-Net dehazer")
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--weights", type=Path, default=RESULTS / "unet_dehazer.pt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model = UNetDehazer(base=32).to(device).eval()
    model.load_state_dict(ckpt["model"])

    srcs = sorted(p for p in args.src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        srcs = srcs[: args.limit]
    args.dst.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(srcs), args.batch):
        chunk = srcs[start:start + args.batch]
        imgs, shapes = [], []
        for p in chunk:
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                imgs.append(None)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb).permute(2, 0, 1)
            imgs.append(t)
            shapes.append(t.shape[1:])
        if not imgs:
            continue
        valid = [i for i, t in enumerate(imgs) if t is not None]
        if not valid:
            continue
        batch = torch.stack([imgs[i] for i in valid]).to(device)
        batch, (h, w) = _pad_to_multiple(batch)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(batch)
        out = out[:, :, :h, :w].float().clamp(0, 1).cpu()
        for j, i in enumerate(valid):
            rgb = (out[j].permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(args.dst / f"{chunk[i].stem}.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        if start % (args.batch * 50) == 0:
            print(f"[unet-apply] {min(start + args.batch, len(srcs))}/{len(srcs)}")

    print(f"[unet-apply] {args.src} -> {args.dst} ({len(srcs)} images)")


if __name__ == "__main__":
    main()
