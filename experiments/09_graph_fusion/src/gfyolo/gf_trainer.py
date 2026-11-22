"""DetectionTrainer for GF-YOLO dual-stream models (paired 6-ch data)."""
from __future__ import annotations

from copy import copy

from ultralytics.models.yolo.detect.train import DetectionTrainer

from gfyolo.fused_trainer import PairedDetectionValidator
from gfyolo.model import GFYOLO
from gfyolo.paired_dataset import make_paired_dataset


class GFTrainer(DetectionTrainer):
    """Trains a GFYOLO on paired [IR|RGB] data with the paired validator."""

    model_cfg = "yolo12s.yaml"
    paired_yaml: str | None = None
    rgb_yaml: str | None = None
    paired_rot90: bool = False
    fuse: str = "E10"
    fuse_kwargs: dict | None = None

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
