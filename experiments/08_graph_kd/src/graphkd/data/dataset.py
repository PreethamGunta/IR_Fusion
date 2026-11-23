"""Deterministic datasets used for teacher caching and student KD training.

All geometric/photometric augmentation is disabled.  This is required because
teacher graphs are cached once per image; every KD arm must see exactly the
same student input as the cached teacher input.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset

NO_AUG = dict(
    mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    erasing=0.0, auto_augment=None, bgr=0.0, augmentations=[],
)


def graphkd_cfg(imgsz: int = 640, **overrides):
    kw = dict(task="detect", imgsz=imgsz, rect=False, cache=None,
              single_cls=False, fraction=1.0, deterministic=True, **NO_AUG)
    kw.update(overrides)
    return get_cfg(overrides=kw)


def make_dataset(data_yaml: str, split: str, mode: str = "train",
                 imgsz: int = 640, batch: int = 16,
                 cache=None) -> YOLODataset:
    cfg = graphkd_cfg(imgsz=imgsz, cache=cache)
    data = check_det_dataset(data_yaml)
    path = data[split]
    return build_yolo_dataset(cfg, path, batch, data, mode=mode, rect=False, stride=32)


class PairedYOLODataset(YOLODataset):
    """YOLODataset that returns a 6-channel [IR|RGB] tensor.

    The RGB dataset uses the identical deterministic transform, so the two
    3-channel tensors are pixel-aligned.  Labels come from the IR dataset (the
    aligned release shares boxes).
    """

    def __init__(self, *args, rgb_path: str, paired_augment: bool = False,
                 paired_rot90: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.paired_augment = paired_augment
        self.paired_rot90 = paired_rot90
        rgb_kwargs = copy.copy(kwargs)
        rgb_kwargs.update(img_path=rgb_path)
        self.rgb_ds = YOLODataset(*args, **rgb_kwargs)
        if len(self.rgb_ds) != len(self):
            raise ValueError(f"paired dataset length mismatch: {len(self)} vs {len(self.rgb_ds)}")
        for a, b in zip(self.im_files, self.rgb_ds.im_files):
            if Path(a).stem.replace("_PreviewData", "") != Path(b).stem.replace("_RGB", ""):
                raise ValueError(f"pair order mismatch: {a} vs {b}")

    def __getitem__(self, index: int):
        s = super().__getitem__(index)
        r = self.rgb_ds[index]
        s["img"] = torch.cat([s["img"], r["img"]], 0)
        if self.paired_augment and self.augment:
            s = self._paired_augment(s)
        return s

    def _paired_augment(self, s: dict) -> dict:
        """Geometric + photometric augmentation applied to both modalities.

        Both images are transformed by exactly the same warp, and boxes are
        updated analytically.  Used only for fused-teacher training; caches and
        student KD inputs are deterministic (paired_augment=False).
        """
        img = s["img"]                         # [6, H, W] uint8
        boxes = s["bboxes"].clone()            # [N,4] normalised xywh
        cls = s["cls"]
        bids = s["batch_idx"]

        # 1) horizontal / vertical flips
        if torch.rand(()) < 0.5:
            img = torch.flip(img, dims=[2])
            boxes[:, 0] = 1.0 - boxes[:, 0]
        if torch.rand(()) < 0.5:
            img = torch.flip(img, dims=[1])
            boxes[:, 1] = 1.0 - boxes[:, 1]

        # 2) 90-degree rotations (aerial imagery: valid; optional for ground)
        if self.paired_rot90 and torch.rand(()) < 0.75:
            k = int(torch.randint(1, 4, ()))
            img = torch.rot90(img, k, dims=[1, 2])
            for _ in range(k):
                x, y, w, h = boxes.unbind(1)
                boxes = torch.stack([y, 1.0 - x, h, w], 1)

        # 3) random scale + translation about the image centre
        sc = 0.80 + 0.45 * float(torch.rand(()))
        tx = (float(torch.rand(())) - 0.5) * 0.16
        ty = (float(torch.rand(())) - 0.5) * 0.16
        theta = torch.tensor([[[sc, 0.0, 2 * tx], [0.0, sc, 2 * ty]]],
                             dtype=torch.float32)
        grid = F.affine_grid(theta, (1, *img.shape), align_corners=False)
        img = F.grid_sample(img[None].float(), grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)[0]
        boxes[:, 0] = (boxes[:, 0] - 0.5) * sc + 0.5 + tx
        boxes[:, 1] = (boxes[:, 1] - 0.5) * sc + 0.5 + ty
        boxes[:, 2] *= sc
        boxes[:, 3] *= sc

        # 4) clip / drop boxes outside the frame
        x1 = (boxes[:, 0] - boxes[:, 2] / 2).clamp(0, 1)
        y1 = (boxes[:, 1] - boxes[:, 3] / 2).clamp(0, 1)
        x2 = (boxes[:, 0] + boxes[:, 2] / 2).clamp(0, 1)
        y2 = (boxes[:, 1] + boxes[:, 3] / 2).clamp(0, 1)
        w, h = x2 - x1, y2 - y1
        keep = (w > 0.015) & (h > 0.015)
        boxes = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, w, h], 1)[keep]
        cls = cls[keep]
        bids = bids[keep]

        # 5) independent photometric jitter per modality
        img = img.clamp(0, 255)
        ir = img[:3].float()
        rgb = img[3:].float()
        ir = ir * (0.85 + 0.30 * float(torch.rand(()))) + (float(torch.rand(())) - 0.5) * 12
        rgb = rgb * (0.75 + 0.50 * float(torch.rand(()))) + (float(torch.rand(())) - 0.5) * 18
        img = torch.cat([ir, rgb], 0).clamp(0, 255).to(torch.uint8)

        s["img"] = img
        s["bboxes"] = boxes
        s["cls"] = cls
        s["batch_idx"] = bids
        return s


def make_paired_dataset(data_yaml: str, rgb_yaml: str, split: str,
                        mode: str = "train", imgsz: int = 640,
                        batch: int = 16, cache=None,
                        paired_augment: bool = False,
                        paired_rot90: bool = False) -> PairedYOLODataset:
    cfg = graphkd_cfg(imgsz=imgsz, cache=cache)
    data = check_det_dataset(data_yaml)
    rgb_data = check_det_dataset(rgb_yaml)
    return PairedYOLODataset(
        img_path=data[split], imgsz=imgsz, batch_size=batch,
        augment=mode == "train", hyp=copy.copy(cfg), rect=False,
        cache=cache, single_cls=False, stride=32, pad=0.0,
        data=data, fraction=1.0, task="detect",
        rgb_path=rgb_data[split], paired_augment=paired_augment,
        paired_rot90=paired_rot90,
    )
