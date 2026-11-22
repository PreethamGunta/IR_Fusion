"""Train the paired U-Net dehazer on (CycleGAN fog, clear) pairs."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .paths import CLEAR_TRAIN_IMG, FOG, RESULTS, ensure_dirs
from .unet_dehazer import UNetDehazer, ssim_loss

VIDEO_RE = re.compile(r"video-([^-]+)-frame-")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def video_id(path: Path) -> str:
    m = VIDEO_RE.search(path.name)
    return m.group(1) if m else path.name


def build_pairs(condition: str) -> list[tuple[Path, Path]]:
    clear = {p.stem: p for p in CLEAR_TRAIN_IMG.iterdir() if p.suffix.lower() in IMG_EXTS}
    fog_dir = FOG[condition]["train_img"]
    pairs = []
    for f in sorted(fog_dir.iterdir()):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        c = clear.get(f.stem)
        if c is not None:
            pairs.append((c, f))
    return pairs


def group_split(pairs, val_frac=0.1, seed=42):
    groups = sorted({video_id(p[1]) for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_frac)))
    val_groups = set(groups[:n_val])
    train = [p for p in pairs if video_id(p[1]) not in val_groups]
    val = [p for p in pairs if video_id(p[1]) in val_groups]
    return train, val


class PairedDataset(Dataset):
    def __init__(self, pairs, crop=256, train=True):
        self.pairs = pairs
        self.crop = crop
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def _read(self, p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        clear_p, fog_p = self.pairs[idx]
        clear = self._read(clear_p)
        fog = self._read(fog_p)
        h, w = fog.shape[:2]

        if self.train:
            ch, cw = self.crop, self.crop
            y = random.randint(0, max(0, h - ch))
            x = random.randint(0, max(0, w - cw))
            clear = clear[y:y + ch, x:x + cw]
            fog = fog[y:y + ch, x:x + cw]
            if random.random() < 0.5:
                clear = clear[:, ::-1]
                fog = fog[:, ::-1]
            if random.random() < 0.5:
                clear = clear[::-1]
                fog = fog[::-1]
        else:
            ch, cw = min(self.crop, h), min(self.crop, w)
            y = (h - ch) // 2
            x = (w - cw) // 2
            clear = clear[y:y + ch, x:x + cw]
            fog = fog[y:y + ch, x:x + cw]

        clear = np.ascontiguousarray(clear).astype(np.float32) / 255.0
        fog = np.ascontiguousarray(fog).astype(np.float32) / 255.0
        clear = torch.from_numpy(clear).permute(2, 0, 1)
        fog = torch.from_numpy(fog).permute(2, 0, 1)
        return fog, clear


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a paired U-Net dehazer")
    ap.add_argument("--condition", default="cyclegan", choices=list(FOG))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=RESULTS / "unet_dehazer.pt",
                    help="checkpoint path (use a distinct name for a new generator)")
    args = ap.parse_args()

    ensure_dirs()
    pairs = build_pairs(args.condition)
    if not pairs:
        raise RuntimeError(f"no pairs found for condition {args.condition}")
    train_pairs, val_pairs = group_split(pairs)
    print(f"[unet] pairs: train={len(train_pairs)} val={len(val_pairs)}")

    train_ds = PairedDataset(train_pairs, args.crop, train=True)
    val_ds = PairedDataset(val_pairs, args.crop, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = UNetDehazer(base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for fog, clear in train_dl:
            fog, clear = fog.to(device), clear.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(fog)
                loss = torch.nn.functional.l1_loss(pred, clear) + 0.5 * ssim_loss(pred, clear)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += float(loss.item())
        sched.step()

        model.eval()
        val_l1 = 0.0
        n = 0
        with torch.no_grad():
            for fog, clear in val_dl:
                fog, clear = fog.to(device), clear.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(fog)
                val_l1 += float(torch.nn.functional.l1_loss(pred, clear).item()) * fog.size(0)
                n += fog.size(0)
        val_l1 /= max(1, n)
        train_loss /= max(1, len(train_dl))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_l1": val_l1})
        print(f"[unet] epoch {epoch:02d}/{args.epochs} train={train_loss:.4f} val_l1={val_l1:.4f} "
              f"({time.time()-t0:.0f}s)")

        if val_l1 < best_val:
            best_val = val_l1
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_l1": val_l1,
                        "crop": args.crop, "condition": args.condition}, args.out)

    metrics = {"condition": args.condition, "epochs": args.epochs, "best_val_l1": best_val,
               "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
               "history": history, "checkpoint": str(args.out)}
    (RESULTS / f"unet_metrics_{args.condition}.json").write_text(json.dumps(metrics, indent=2))
    print(f"[unet] best val_l1={best_val:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
