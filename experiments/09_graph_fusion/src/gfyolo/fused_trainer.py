"""Fused (6-channel) trainer for experiment 09 (E02 early fusion).

Stem adaptation adapted from experiments/08_graph_kd/src/graphkd/training/trainer.py
and the paired validator from .../training/paired_validator.py (copied, not
imported; source SHA-256 in PROVENANCE.md). Differences from 08:
- plain ultralytics ``DetectionModel`` (no KD machinery);
- the channel order [IR (0:3) | RGB (3:6)] is fixed and unit-tested in
  tests/test_fused.py — 08's earlier fused teacher trained with the two
  stem halves swapped, which E02 must not repeat.
"""
from __future__ import annotations

from copy import copy
from pathlib import Path

import torch
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from gfyolo.paired_dataset import make_paired_dataset


def adapt_first_conv(v: torch.Tensor) -> torch.Tensor:
    """Expand a 3-channel RGB stem to the paired [IR | RGB] channel order.

    The paired dataset concatenates images as [IR (3), RGB (3)] (see
    gfyolo.paired_dataset.PairedYOLODataset). The pretrained yolo12s stem is
    RGB, so channels 0:3 must be initialised from the grayscale mean (IR
    proxy) and channels 3:6 from the original RGB filters, halved to keep the
    activation scale. 08 trained an earlier fused teacher with this order
    reversed ([RGB | gray]); 09 fixes and unit-tests it.
    """
    gray = v.mean(1, keepdim=True).repeat(1, 3, 1, 1)
    return torch.cat([gray, v], 1) / 2.0


def adapt_stem(model, weights) -> None:
    """Load a 3-channel checkpoint into a 6-channel model (IR + RGB stem)."""
    if weights is None:
        return
    sd = weights.state_dict() if hasattr(weights, "state_dict") else weights
    cur = model.state_dict()
    new = {}
    for k, v in sd.items():
        if k not in cur:
            continue
        if cur[k].shape == v.shape:
            new[k] = v
        elif v.ndim == 4 and v.shape[1] == 3 and cur[k].shape[1] == 6:
            new[k] = adapt_first_conv(v)
    missing, unexpected = model.load_state_dict(new, strict=False)
    if RANK in (-1, 0):
        print(f"[fused] loaded {len(new)} tensors; missing={len(missing)} unexpected={len(unexpected)}")


class FusedDetectionTrainer(DetectionTrainer):
    """DetectionTrainer that trains a 6-ch [IR|RGB] model on paired data."""

    paired_yaml: str | None = None
    rgb_yaml: str | None = None
    paired_rot90: bool = False

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=6,
                               verbose=verbose and RANK == -1)
        if hasattr(self, "set_model_names_for_load"):
            model = self.set_model_names_for_load(model)
        if weights is not None:
            adapt_stem(model, weights)
        return model

    def get_validator(self):
        return PairedDetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args),
            _callbacks=self.callbacks,
            pair_yaml=self.paired_yaml, rgb_yaml=self.rgb_yaml,
        )

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        split = mode if mode in ("train", "val", "test") else "train"
        return make_paired_dataset(
            self.paired_yaml, self.rgb_yaml, split,
            mode=mode, imgsz=self.args.imgsz, batch=batch or self.args.batch,
            cache=self.args.cache or None,
            paired_augment=(mode == "train"),
            paired_rot90=self.paired_rot90,
        )


class PairedDetectionValidator(DetectionValidator):
    """Standard metric machinery fed with the paired 6-channel val set."""

    def __init__(self, *args, pair_yaml: str | Path | None = None,
                 rgb_yaml: str | Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_yaml = str(pair_yaml)
        self.rgb_yaml = str(rgb_yaml)

    def get_dataloader(self, dataset_path, batch_size):
        ds = make_paired_dataset(
            self.pair_yaml, self.rgb_yaml, "val", mode="val",
            imgsz=self.args.imgsz, batch=batch_size,
        )
        return build_dataloader(
            ds, batch_size, self.args.workers,
            shuffle=False, rank=-1, drop_last=False,
            pin_memory=self.training, device=self.device,
        )
