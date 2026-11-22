"""Validation for the 6-channel fused RGB-IR teacher.

The stock DetectionValidator builds a 3-channel YOLODataset.  This subclass
keeps the standard metric machinery but feeds it the paired 6-channel
validation set, so the fused teacher gets a real val mAP, early stopping and
best.pt selection.
"""
from __future__ import annotations

from pathlib import Path

from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect.val import DetectionValidator

from graphkd import paths
from graphkd.data.dataset import make_paired_dataset


class PairedDetectionValidator(DetectionValidator):
    def __init__(self, *args, pair_yaml: str | Path | None = None,
                 rgb_yaml: str | Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_yaml = str(pair_yaml or (paths.CONFIGS / "paired.yaml"))
        self.rgb_yaml = str(rgb_yaml or (paths.CONFIGS / "rgb.yaml"))

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
