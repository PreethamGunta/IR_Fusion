"""DetectionTrainer for GF-YOLO dual-stream models (paired 6-ch data)."""
from __future__ import annotations

from copy import copy

import torch
from torch import nn
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

from gfyolo.fused_trainer import PairedDetectionValidator
from gfyolo.model import GFYOLO
from gfyolo.paired_dataset import make_paired_dataset


def sanitize_bn_buffers(model: nn.Module, warn_once: set | None = None) -> int:
    """Reset any non-finite BatchNorm running statistics on ``model``.

    Rationale (2026-11-22, E35 screening attempt 2): a single transient
    non-finite forward is absorbed silently into BN ``running_mean`` /
    ``running_var`` (train mode uses *batch* stats, so training looks healthy),
    and the very next *validation* pass — which uses the running stats —
    produces NaN features and dies. The master weights and optimizer state
    stay finite, so resetting the poisoned buffers lets them re-estimate
    from subsequent batches (~30 with momentum 0.1) and the run self-heals
    instead of crashing. Returns the number of reset layers; a no-op (and
    near-zero cost) on healthy batches.
    """
    reset = 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.modules.batchnorm._BatchNorm):
            continue
        poisoned = False
        for buf in (m.running_mean, m.running_var):
            if buf is not None and not bool(torch.isfinite(buf).all()):
                poisoned = True
        if not poisoned:
            continue
        reset += 1
        if m.running_mean is not None:
            m.running_mean.zero_()
        if m.running_var is not None:
            m.running_var.fill_(1.0)
        if m.num_batches_tracked is not None:
            m.num_batches_tracked.zero_()
        if warn_once is not None:
            key = name
            if key not in warn_once:
                warn_once.add(key)
                LOGGER.warning(f"gfyolo: reset non-finite BN running stats in "
                               f"'{name}' (transient forward overflow; stats "
                               f"will re-estimate over the next batches)")
    return reset


class GFTrainer(DetectionTrainer):
    """Trains a GFYOLO on paired [IR|RGB] data with the paired validator."""

    model_cfg = "yolo12s.yaml"
    paired_yaml: str | None = None
    rgb_yaml: str | None = None
    paired_rot90: bool = False
    fuse: str = "E10"
    fuse_kwargs: dict | None = None
    _bn_warned: set = set()

    def optimizer_step(self):
        # sanitize BEFORE super() so the EMA never absorbs a poisoned buffer
        sanitize_bn_buffers(self.model, self._bn_warned)
        super().optimizer_step()

    def validate(self):
        # sanitize the EMA's BN stats before validation: a transient
        # non-finite train forward poisons running statistics (train mode
        # uses batch stats, so training looks healthy) and the val pass
        # would otherwise clamp through garbage features on its first
        # batch (E35 attempt 2, 2026-11-22). The stats re-estimate during
        # subsequent training batches.
        if self.ema is not None:
            sanitize_bn_buffers(self.ema.ema, self._bn_warned)
        return super().validate()

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = GFYOLO(self.model_cfg, nc=self.data["nc"], ch=6,
                       fuse=self.fuse, fuse_kwargs=self.fuse_kwargs,
                       verbose=verbose)
        if hasattr(self, "set_model_names_for_load"):
            model = self.set_model_names_for_load(model)
        if weights is not None:
            model.load_pretrained(weights)
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
