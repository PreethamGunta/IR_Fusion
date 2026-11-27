"""Unified detection data pipeline for every GATE-IR arm.

One module, one augmentation implementation, one label path -- used by the
IR-only, RGB-only and paired (IR+RGB, stacked 6-channel) experiments. Training
uses the FULL dataset; speed is controlled by the epoch count only.

Design rules
------------
* The pair is stacked BEFORE any augmentation, so every geometric transform
  (mosaic, affine, flips) applies to a single tensor: alignment is preserved
  by construction, and `close_mosaic` can never desynchronise the halves.
* Photometric jitter is per modality: HSV on the RGB half, value gain on IR.
* The augmentation set mirrors the single-stream reference recipe: mosaic with
  close_mosaic scheduling, flips, scale/translate/shear/degrees, HSV, erasing.
* Boxes are normalized xywh throughout and updated analytically (corner
  transform + clip + minimum-size filter), so labels stay consistent with the
  warped image.

Channel contract: pair -> [IR (0:3) | RGB (3:6)], labels from the IR split
(thermal GT). `ir` -> 3 channels, `rgb` -> 3 channels.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset

# Recipe defaults (match the single-stream reference tier).
DEFAULT_HYP = dict(
    mosaic=1.0, close_mosaic=10, fliplr=0.5, flipud=0.0,
    scale=0.5, translate=0.1, shear=0.0, degrees=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.4,
)

IMGSZ_DEFAULT = 640


# ---------------------------------------------------------------------------
# geometric augmentation (one tensor, analytic boxes)
# ---------------------------------------------------------------------------

def flip(img: np.ndarray, boxes: np.ndarray, horizontal: bool) -> tuple:
    """Flip image + normalized xywh boxes."""
    if horizontal:
        img = img[:, :, ::-1]
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 0] = 1.0 - boxes[:, 0]
    else:
        img = img[:, ::-1, :]
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 1] = 1.0 - boxes[:, 1]
    return np.ascontiguousarray(img), boxes


def affine(img: np.ndarray, boxes: np.ndarray, scale: float, translate: float,
           shear: float, degrees: float) -> tuple:
    """Scale/translate/shear/rotate about the centre; boxes via corner transform.

    ``img`` is [C,H,W] uint8/float. Boxes are normalized xywh.
    """
    c, h, w = img.shape
    a = math.radians(degrees)
    cos, sin = math.cos(a), math.sin(a)
    # rotation @ shear @ scale, about the normalised centre (0.5, 0.5)
    m = np.array([[cos, -sin], [sin, cos]], np.float32)
    m = m @ np.array([[1.0, shear], [0.0, 1.0]], np.float32) * scale
    tx = float(np.random.uniform(-translate, translate))
    ty = float(np.random.uniform(-translate, translate))
    centre = np.array([0.5, 0.5], np.float32)
    t = centre - m @ centre + np.array([tx, ty], np.float32)
    theta = torch.tensor([[m[0, 0], m[0, 1], t[0]],
                          [m[1, 0], m[1, 1], t[1]]], dtype=torch.float32)[None]
    grid = F.affine_grid(theta, (1, c, h, w), align_corners=False)
    warped = F.grid_sample(
        torch.from_numpy(img)[None].float(), grid,
        mode="bilinear", padding_mode="zeros", align_corners=False,
    )[0].numpy()
    if img.dtype == np.uint8:
        warped = np.clip(warped, 0, 255).round().astype(np.uint8)
    else:
        warped = np.clip(warped, 0, 255)

    if len(boxes):
        xy = boxes[:, :2]
        wh = boxes[:, 2:]
        corners = np.stack([
            np.stack([xy[:, 0] - wh[:, 0] / 2, xy[:, 1] - wh[:, 1] / 2], 1),
            np.stack([xy[:, 0] + wh[:, 0] / 2, xy[:, 1] - wh[:, 1] / 2], 1),
            np.stack([xy[:, 0] - wh[:, 0] / 2, xy[:, 1] + wh[:, 1] / 2], 1),
            np.stack([xy[:, 0] + wh[:, 0] / 2, xy[:, 1] + wh[:, 1] / 2], 1),
        ], 1)  # [n,4,2]
        moved = corners @ m.T + t
        x1, y1 = moved[..., 0].min(1), moved[..., 1].min(1)
        x2, y2 = moved[..., 0].max(1), moved[..., 1].max(1)
        x1, y1 = np.clip(x1, 0, 1), np.clip(y1, 0, 1)
        x2, y2 = np.clip(x2, 0, 1), np.clip(y2, 0, 1)
        bw, bh = x2 - x1, y2 - y1
        keep = (bw > 0.015) & (bh > 0.015)
        boxes = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, bw, bh], 1)[keep]
    else:
        keep = np.zeros(0, bool)
    return warped, boxes, keep


def mosaic4(tiles: list, imgsz: int) -> tuple:
    """Compose four letterboxed samples into one 2x imgsz canvas, random crop.

    ``tiles``: list of dicts with uint8 ``img`` [C,H,W] and normalized
    ``bboxes``. Returns (img, boxes, cls, batch_idx) for the cropped sample.
    """
    c = tiles[0]["img"].shape[0]
    canvas = np.full((c, imgsz * 2, imgsz * 2), 114, np.uint8)
    xc = int(random.uniform(imgsz * 0.5, imgsz * 1.5))
    yc = int(random.uniform(imgsz * 0.5, imgsz * 1.5))
    origins = [(yc - imgsz, xc - imgsz), (yc - imgsz, xc),
               (yc, xc - imgsz), (yc, xc)]
    out_b, out_c, out_i = [], [], []
    for idx, (tile, (y0, x0)) in enumerate(zip(tiles, origins)):
        img = tile["img"]
        h, w = img.shape[1:]
        # clip the pasted region to the canvas
        cy0, cy1 = max(0, y0), min(imgsz * 2, y0 + h)
        cx0, cx1 = max(0, x0), min(imgsz * 2, x0 + w)
        ty0, tx0 = cy0 - y0, cx0 - x0
        canvas[:, cy0:cy1, cx0:cx1] = img[:, ty0:ty0 + (cy1 - cy0),
                                           tx0:tx0 + (cx1 - cx0)]
        b = tile["bboxes"]
        if not len(b):
            continue
        px = b[:, 0] * imgsz + x0
        py = b[:, 1] * imgsz + y0
        pw = b[:, 2] * imgsz
        ph = b[:, 3] * imgsz
        out_b.append(np.stack([px, py, pw, ph], 1))
        out_c.append(tile["cls"])
        # one mosaic == ONE image: batch_idx must stay 0 (collate_fn adds the
        # sample index; per-tile ids make the loss see 4x the images it has ->
        # CUDA device-side assert in loss.preprocess)
        out_i.append(np.zeros(len(b), np.float32))
    # random crop of the canvas
    cx = int(random.uniform(0, imgsz))
    cy = int(random.uniform(0, imgsz))
    img = canvas[:, cy:cy + imgsz, cx:cx + imgsz]
    if out_b:
        b = np.concatenate(out_b, 0)
        b[:, 0] = (b[:, 0] - cx) / imgsz
        b[:, 1] = (b[:, 1] - cy) / imgsz
        b[:, 2] /= imgsz
        b[:, 3] /= imgsz
        x1 = np.clip(b[:, 0] - b[:, 2] / 2, 0, 1)
        y1 = np.clip(b[:, 1] - b[:, 3] / 2, 0, 1)
        x2 = np.clip(b[:, 0] + b[:, 2] / 2, 0, 1)
        y2 = np.clip(b[:, 1] + b[:, 3] / 2, 0, 1)
        bw, bh = x2 - x1, y2 - y1
        keep = (bw > 0.015) & (bh > 0.015)
        boxes = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, bw, bh], 1)[keep]
        cls = np.concatenate(out_c, 0)[keep]
        idxs = np.concatenate(out_i, 0)[keep]
    else:
        boxes = np.zeros((0, 4), np.float32)
        cls = np.zeros(0, np.float32)
        idxs = np.zeros(0, np.int64)
    return img, boxes, cls, idxs


# ---------------------------------------------------------------------------
# photometric augmentation (per modality)
# ---------------------------------------------------------------------------

def hsv_rgb(half: np.ndarray, hgain: float, sgain: float, vgain: float) -> np.ndarray:
    """Ultralytics-style HSV jitter on a CHW 3-channel uint8 image."""
    if hgain == 0 and sgain == 0 and vgain == 0:
        return half
    r = np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain], np.float32) + 1
    hwc = np.ascontiguousarray(half.transpose(1, 2, 0))
    hue, sat, val = cv2.split(cv2.cvtColor(hwc, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
    out = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    bgr = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    return np.ascontiguousarray(bgr.transpose(2, 0, 1))


def value_ir(half: np.ndarray, vgain: float) -> np.ndarray:
    """Brightness jitter on the IR (grayscale-replicated) half."""
    if vgain == 0:
        return half
    gain = 1.0 + float(np.random.uniform(-vgain, vgain))
    return np.clip(half.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def random_erase(img: np.ndarray, p: float, max_area: float = 0.1) -> np.ndarray:
    """Random rectangular cutout filled with uniform noise."""
    if p <= 0 or random.random() >= p:
        return img
    c, h, w = img.shape
    area = random.uniform(0.02, max_area) * h * w
    ratio = random.uniform(0.3, 3.3)
    eh = int(round(math.sqrt(area / ratio)))
    ew = int(round(math.sqrt(area * ratio)))
    if eh < 2 or ew < 2:
        return img
    y = random.randint(0, max(0, h - eh))
    x = random.randint(0, max(0, w - ew))
    img = img.copy()
    img[:, y:y + eh, x:x + ew] = np.random.randint(0, 256, (c, eh, ew), np.uint8)
    return img


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

def _no_aug_cfg(imgsz: int):
    """Base-pipeline config: letterbox/format only, no transforms.

    All augmentation is implemented in this module on the stacked tensor; the
    ultralytics chain must stay deterministic so the pair is never
    transformed twice.
    """
    from ultralytics.cfg import get_cfg
    kw = dict(task="detect", imgsz=imgsz, rect=False, cache=None,
              single_cls=False, fraction=1.0, deterministic=True,
              mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, degrees=0.0,
              translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0,
              fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, erasing=0.0,
              auto_augment=None, bgr=0.0, augmentations=[])
    return get_cfg(overrides=kw)


class UnifiedDataset(YOLODataset):
    """IR-only / RGB-only / paired dataset with one pair-safe augmentation path.

    ``modality``: ``"ir"``, ``"rgb"`` or ``"pair"``.
    Training samples are letterboxed, then (train only, mosaic permitting)
    composed into a mosaic, affinely warped, flipped, photometrically jittered
    and randomly erased. Validation samples are only letterboxed.
    """

    def __init__(self, *args, modality: str = "pair", rgb_path: str | None = None,
                 hyp: dict | None = None, **kwargs):
        if modality not in ("ir", "rgb", "pair"):
            raise ValueError(f"bad modality {modality!r}")
        if modality == "pair" and not rgb_path:
            raise ValueError("modality='pair' requires rgb_path")
        self.modality = modality
        self.train_hyp = {**DEFAULT_HYP, **(hyp or {})}
        augment = bool(kwargs.get("augment", False))
        imgsz = int(kwargs.get("imgsz", IMGSZ_DEFAULT))
        super().__init__(*args, hyp=_no_aug_cfg(imgsz), **kwargs)
        self.mosaic = augment
        if modality != "ir":
            rgb_kwargs = dict(kwargs)
            rgb_kwargs.update(img_path=rgb_path)
            self.rgb_ds = YOLODataset(*args, hyp=_no_aug_cfg(imgsz), **rgb_kwargs)
            if len(self.rgb_ds) != len(self):
                raise ValueError(f"pair length mismatch: {len(self)} vs {len(self.rgb_ds)}")
            for a, b in zip(self.im_files, self.rgb_ds.im_files):
                if Path(a).stem.replace("_PreviewData", "") != Path(b).stem.replace("_RGB", ""):
                    raise ValueError(f"pair order mismatch: {a} vs {b}")

    # -- ultralytics compatibility ----------------------------------------

    def close_mosaic(self, hyp=None):
        """Disable mosaic at the scheduled epoch WITHOUT rebuilding transforms.

        Only the mosaic switch is touched; the transform chain (and therefore
        the pair) is never rebuilt.
        """
        self.mosaic = False

    # -- loading ----------------------------------------------------------

    def _tile(self, index: int) -> dict:
        """One letterboxed sample: IR labels + the requested modality image(s)."""
        s = super().__getitem__(index)  # IR image, thermal labels, letterboxed
        ir = s["img"].numpy()
        if self.modality == "ir":
            img = ir
        else:
            r = self.rgb_ds[index]
            rgb = r["img"].numpy()
            img = np.concatenate([ir, rgb], 0) if self.modality == "pair" else rgb
        return {
            "img": np.ascontiguousarray(img),
            "bboxes": s["bboxes"].numpy(),
            "cls": s["cls"].numpy().reshape(-1),
            "batch_idx": s["batch_idx"].numpy(),
        }

    def __getitem__(self, index: int) -> dict:
        # validation / inference: letterbox only
        if not self.augment:
            s = super().__getitem__(index)
            if self.modality == "ir":
                return s
            r = self.rgb_ds[index]
            if self.modality == "rgb":
                r["bboxes"] = s["bboxes"]  # labels always from IR (thermal GT)
                r["cls"] = s["cls"]
                return r
            s["img"] = torch.cat([s["img"], r["img"]], 0)
            return s

        h = self.train_hyp
        if self.mosaic and random.random() < h["mosaic"]:
            idxs = [index] + [random.randrange(len(self)) for _ in range(3)]
            tiles = [self._tile(i) for i in idxs]
            img, boxes, cls, bidx = mosaic4(tiles, self.imgsz)
        else:
            t = self._tile(index)
            img, boxes, cls, bidx = t["img"], t["bboxes"], t["cls"], t["batch_idx"]

        if h["degrees"] or h["shear"] or h["scale"] or h["translate"]:
            sc = 1.0 + random.uniform(-h["scale"], h["scale"])
            img, boxes, keep = affine(img, boxes, sc, h["translate"],
                                      h["shear"], h["degrees"])
            cls, bidx = cls[keep], bidx[keep]

        if random.random() < h["fliplr"]:
            img, boxes = flip(img, boxes, True)
        if random.random() < h["flipud"]:
            img, boxes = flip(img, boxes, False)

        if self.modality == "pair":
            ir, rgb = img[:3], img[3:]
            ir = value_ir(ir, h["hsv_v"])
            rgb = hsv_rgb(rgb, h["hsv_h"], h["hsv_s"], h["hsv_v"])
            img = np.concatenate([ir, rgb], 0)
        elif self.modality == "ir":
            img = value_ir(img, h["hsv_v"])
        else:
            img = hsv_rgb(img, h["hsv_h"], h["hsv_s"], h["hsv_v"])

        img = random_erase(img, h["erasing"])
        return {
            "img": torch.from_numpy(np.ascontiguousarray(img)),
            "bboxes": torch.from_numpy(boxes.astype(np.float32)),
            "cls": torch.from_numpy(cls.astype(np.float32)).reshape(-1, 1),
            "batch_idx": torch.from_numpy(np.asarray(bidx, np.float32)).reshape(-1),
            "im_file": self.im_files[index],
            "ori_shape": (self.imgsz, self.imgsz),
            "resized_shape": (self.imgsz, self.imgsz),
            "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
        }


def make_dataset(data_yaml: str, split: str, modality: str = "pair",
                 mode: str = "train", imgsz: int = IMGSZ_DEFAULT,
                 batch: int = 16, cache=None, rgb_yaml: str | None = None,
                 hyp: dict | None = None) -> UnifiedDataset:
    """Build a UnifiedDataset for any arm (``mode``: train/val/test)."""
    data = check_det_dataset(data_yaml)
    rgb_path = None
    if modality == "pair":
        if not rgb_yaml:
            raise ValueError("modality='pair' requires rgb_yaml")
        rgb_path = check_det_dataset(rgb_yaml)[split]
    return UnifiedDataset(
        img_path=data[split], imgsz=imgsz, batch_size=batch,
        augment=(mode == "train"), hyp=hyp, rect=False, cache=cache,
        single_cls=False, stride=32, pad=0.0, data=data, fraction=1.0,
        task="detect", modality=modality, rgb_path=rgb_path,
    )
