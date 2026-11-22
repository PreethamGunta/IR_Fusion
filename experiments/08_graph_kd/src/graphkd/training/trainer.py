"""Custom Ultralytics trainers for graph-KD students and the fused teacher."""
from __future__ import annotations

import json
from copy import copy
from pathlib import Path

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from graphkd.data.dataset import make_paired_dataset
from graphkd.models.kd_model import KDDetectionModel
from graphkd.training.paired_validator import PairedDetectionValidator


def _adapt_first_conv(v: torch.Tensor) -> torch.Tensor:
    """Expand a 3-channel RGB stem to the paired [IR | RGB] channel order.

    The paired dataset concatenates images as [IR (3), RGB (3)]
    (``PairedYOLODataset.__getitem__``).  The pretrained yolo12s stem is RGB, so
    channels 0:3 must be initialised from the grayscale mean (IR proxy) and
    channels 3:6 from the original RGB filters.  The previous order was
    reversed, which swapped the IR/RGB initialisation.
    """
    gray = v.mean(1, keepdim=True).repeat(1, 3, 1, 1)
    return torch.cat([gray, v], 1) / 2.0


def _adapt_stem(model, weights) -> None:
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
            new[k] = _adapt_first_conv(v)
    missing, unexpected = model.load_state_dict(new, strict=False)
    if RANK in (-1, 0):
        print(f"[fused] loaded {len(new)} tensors; missing={len(missing)} unexpected={len(unexpected)}")


class GraphKDTrainer(DetectionTrainer):
    """DetectionTrainer that builds a KDDetectionModel and logs KD terms."""

    kd_cfg: dict | None = None
    paired: bool = False
    fused: bool = False
    paired_yaml: str | None = None
    rgb_yaml: str | None = None
    paired_augment: bool = False
    paired_rot90: bool = False

    def get_model(self, cfg=None, weights=None, verbose=True):
        ch = 6 if self.fused else 3
        model = self.set_model_names_for_load(
            KDDetectionModel(cfg, nc=self.data["nc"], ch=ch, verbose=verbose and RANK == -1)
        )
        if weights is not None:
            if self.fused:
                _adapt_stem(model, weights)
            else:
                model.load(weights)
        if self.kd_cfg:
            model.attach_kd(self.kd_cfg)
            if weights is not None and getattr(weights, "kd_module", None) is not None:
                try:
                    model.kd_module.load_state_dict(weights.kd_module.state_dict(), strict=False)
                except Exception as e:  # resume with a different KD config
                    print(f"[kd] could not restore KD module: {e}")
        return model

    def get_validator(self):
        if self.fused:
            # Paired 6-channel validator: real val mAP, best.pt and early stop.
            from graphkd import paths
            return PairedDetectionValidator(
                self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                _callbacks=self.callbacks,
                pair_yaml=self.paired_yaml or str(paths.CONFIGS / "paired.yaml"),
                rgb_yaml=self.rgb_yaml or str(paths.CONFIGS / "rgb.yaml"),
            )
        return super().get_validator()

    def validate(self):
        return super().validate()

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        if self.fused:
            from graphkd import paths
            paired = self.paired_yaml or str(paths.CONFIGS / "paired.yaml")
            rgb = self.rgb_yaml or str(paths.CONFIGS / "rgb.yaml")
            return make_paired_dataset(
                paired, rgb,
                split=mode if mode in ("train", "val", "test") else "train",
                mode=mode, imgsz=self.args.imgsz, batch=batch or self.args.batch,
                cache=self.args.cache or None,
                paired_augment=self.paired_augment,
                paired_rot90=self.paired_rot90,
            )
        return super().build_dataset(img_path, mode, batch)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._cb_epoch_start)
        self.add_callback("on_train_epoch_end", self._cb_epoch_end)
        self.add_callback("on_fit_epoch_end", self._cb_epoch_end)

    def _cb_epoch_start(self, trainer):
        m = unwrap_model(trainer.model)
        if getattr(m, "kd_module", None) is not None:
            m.kd_module.epoch = trainer.epoch

    def _cb_epoch_end(self, trainer):
        m = unwrap_model(trainer.model)
        if getattr(m, "kd_module", None) is None:
            return
        terms = m.kd_module.pop_terms()
        if not terms:
            return
        out = Path(trainer.save_dir) / "kd_terms.jsonl"
        with out.open("a") as f:
            f.write(json.dumps({"epoch": int(trainer.epoch), **terms}) + "\n")
