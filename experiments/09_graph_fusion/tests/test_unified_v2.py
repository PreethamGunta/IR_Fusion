"""Tests for the V2 unified pipeline (``src/gfdata/unified_v2.py``).

The point of this suite is to prove, not assert, the three properties the
10-epoch probe depends on:

1. **E00 by identity** -- ``modality="ir"`` produces exactly what the stock
   ``YOLODataset`` produces, sample for sample.
2. **RNG parity** -- ``modality="pair"`` consumes no extra random draws, so the
   IR half of a fused arm is bit-identical to an IR-only arm's input at the
   same index. Without this, arms cannot be compared epoch by epoch.
3. **Pair synchronisation** -- every geometric op is replayed, so transformed
   boxes and images stay pixel-aligned, and a desync raises instead of
   training silently.

Tests that need the real DroneVehicle crops are skipped when the tree is
absent, so the suite still runs on a bare checkout.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg

EXP = Path(__file__).resolve().parents[1]
DV_IR = EXP / "data" / "dronevehicle_v2" / "yolo" / "ir"
DV_RGB = EXP / "data" / "dronevehicle_v2" / "yolo" / "rgb"

HAVE_DV = (DV_IR / "images" / "train").is_dir() and (DV_RGB / "images" / "train").is_dir()
real_data = pytest.mark.skipif(not HAVE_DV, reason="DroneVehicle v2 crops not built")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cfg(imgsz: int = 640, **over):
    """A resolved Ultralytics config, as the trainer would pass it."""
    kw = dict(task="detect", imgsz=imgsz, seed=0, mosaic=1.0, fliplr=0.5,
              hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5,
              degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0,
              erasing=0.4, mixup=0.0, cutmix=0.0, copy_paste=0.0,
              close_mosaic=0, rect=False, cache=False, single_cls=False,
              fraction=1.0)
    kw.update(over)
    return get_cfg(overrides=kw)


def _write_tree(root: Path, n: int = 4, size: int = 64, invert: bool = False) -> None:
    """Write ``n`` tiny images + one label file each under a YOLO tree."""
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        if invert:
            img = 255 - img
        name = f"{i:06d}.jpg"
        cv2.imwrite(str(root / "images" / "train" / name), img)
        (root / "labels" / "train" / f"{i:06d}.txt").write_text(
            "0 0.5 0.5 0.4 0.4\n")


def _data(root: Path) -> dict:
    return {
        "path": str(root),
        "train": str(root / "images" / "train"),
        "val": str(root / "images" / "train"),
        "test": str(root / "images" / "train"),
        "names": {0: "car"},
        "nc": 1,
        "channels": 3,
    }


def _seek(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# 1. E00 by identity
# ---------------------------------------------------------------------------

@real_data
def test_ir_branch_is_the_stock_dataset():
    """``modality="ir"`` must return the stock factory's object, unwrapped."""
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    from gfdata.unified_v2 import make_dataset_v2

    cfg = _cfg()
    # Resolve through the real yaml: the label cache is keyed by the dataset
    # parameters, and a hand-written dict with the wrong ``names`` silently
    # rewrites that cache with every non-matching class treated as corrupt.
    data = check_det_dataset(str(EXP / "configs" / "dronevehicle_v2_ir.yaml"))
    assert len(data["names"]) == 5 and data["nc"] == 5
    stock = build_yolo_dataset(cfg=cfg, img_path=data["train"], batch=16,
                               data=data, mode="train", rect=False, stride=32)
    v2 = make_dataset_v2(cfg=cfg, img_path=data["train"], batch=16, data=data,
                         mode="train", rect=False, stride=32, modality="ir")

    assert type(v2) is type(stock), "V2 ir must be the stock class, not a wrapper"
    assert len(v2) == len(stock) == 17962
    assert v2.im_files == stock.im_files

    for idx in (0, 1, 7, 1234):
        for seed in (0, 5):
            _seek(seed)
            a = stock[idx]
            _seek(seed)
            b = v2[idx]
            assert torch.equal(a["img"], b["img"]), f"img differs at {idx}"
            assert torch.equal(a["bboxes"], b["bboxes"]), f"boxes differ at {idx}"
            assert torch.equal(a["cls"], b["cls"]), f"cls differs at {idx}"
            assert a["ratio_pad"] == b["ratio_pad"]


# ---------------------------------------------------------------------------
# 2. RNG parity between ir and pair
# ---------------------------------------------------------------------------

def test_pair_consumes_no_extra_random_draws(tmp_path):
    """The fused arm must draw exactly as many RNG values as IR-only."""
    from gfdata.unified_v2 import make_dataset_v2

    ir_root, rgb_root = tmp_path / "ir", tmp_path / "rgb"
    _write_tree(ir_root)
    _write_tree(rgb_root)
    cfg = _cfg(imgsz=64)
    data, rgb_data = _data(ir_root), _data(rgb_root)

    ds_ir = make_dataset_v2(cfg=cfg, img_path=data["train"], batch=2, data=data,
                            mode="train", modality="ir")
    ds_pair = make_dataset_v2(cfg=cfg, img_path=data["train"], batch=2, data=data,
                              mode="train", modality="pair",
                              rgb_data=rgb_data, rgb_img_path=rgb_data["train"])

    for idx in (0, 1, 2):
        for seed in (0, 3):
            _seek(seed)
            a = ds_ir[idx]
            state_ir = (random.getstate(), np.random.get_state())
            _seek(seed)
            b = ds_pair[idx]
            state_pair = (random.getstate(), np.random.get_state())

            assert torch.equal(a["img"], b["img"][:3]), "IR half differs from IR-only"
            assert torch.equal(a["bboxes"], b["bboxes"])
            # the proof of RNG parity: identical state after the call
            assert state_ir[0] == state_pair[0], "python RNG stream diverged"
            assert np.array_equal(state_ir[1][1], state_pair[1][1]), \
                "numpy RNG stream diverged"


def test_pair_returns_six_channels_with_three_channel_halves(tmp_path):
    from gfdata.unified_v2 import make_dataset_v2

    ir_root, rgb_root = tmp_path / "ir", tmp_path / "rgb"
    _write_tree(ir_root)
    _write_tree(rgb_root)
    cfg = _cfg(imgsz=64)
    data, rgb_data = _data(ir_root), _data(rgb_root)
    ds = make_dataset_v2(cfg=cfg, img_path=data["train"], batch=2, data=data,
                         mode="train", modality="pair", rgb_data=rgb_data,
                         rgb_img_path=rgb_data["train"])
    s = ds[0]
    assert s["img"].shape == (6, 64, 64)
    assert s["img"].dtype == torch.uint8
    assert s["bboxes"].shape[1] == 4 and s["cls"].shape[1] == 1
    assert float(s["img"].min()) >= 0 and float(s["img"].max()) <= 255


# ---------------------------------------------------------------------------
# 3. pair synchronisation
# ---------------------------------------------------------------------------

def test_identical_content_gives_identical_halves(tmp_path):
    """If the RGB pixels equal the IR pixels, the replays must be bit-identical.

    Any desynchronisation of mosaic / affine / flip would show up here.
    """
    from gfdata.unified_v2 import make_dataset_v2

    root = tmp_path / "both"
    _write_tree(root)
    cfg = _cfg(imgsz=64)
    data = _data(root)
    ds = make_dataset_v2(cfg=cfg, img_path=data["train"], batch=2, data=data,
                         mode="train", modality="pair", rgb_data=data,
                         rgb_img_path=data["train"])
    for idx in (0, 1):
        for seed in (0, 1, 2, 3, 4):
            _seek(seed)
            s = ds[idx]
            assert torch.equal(s["img"][:3], s["img"][3:]), \
                f"IR and RGB halves diverged at idx={idx} seed={seed}"


def test_different_content_keeps_boxes_aligned(tmp_path):
    """Different RGB pixels, same labels -> boxes must still match exactly."""
    from gfdata.unified_v2 import make_dataset_v2

    ir_root, rgb_root = tmp_path / "ir", tmp_path / "rgb"
    _write_tree(ir_root)
    _write_tree(rgb_root, invert=True)
    cfg = _cfg(imgsz=64)
    data, rgb_data = _data(ir_root), _data(rgb_root)

    ds_ir = make_dataset_v2(cfg=cfg, img_path=data["train"], batch=2, data=data,
                            mode="train", modality="ir")
    ds_pair = make_dataset_v2(cfg=cfg, img_path=data["train"], batch=2, data=data,
                              mode="train", modality="pair", rgb_data=rgb_data,
                              rgb_img_path=rgb_data["train"])
    for idx in (0, 1, 2, 3):
        for seed in (0, 1, 2):
            _seek(seed)
            a = ds_ir[idx]
            _seek(seed)
            b = ds_pair[idx]
            # boxes ride the IR transform; content is intentionally different
            assert torch.equal(a["bboxes"], b["bboxes"])
            assert not torch.equal(b["img"][:3], b["img"][3:])


def test_pair_order_mismatch_is_rejected(tmp_path):
    from gfdata.unified_v2 import PairSyncError, make_dataset_v2

    ir_root, rgb_root = tmp_path / "ir", tmp_path / "rgb"
    _write_tree(ir_root)
    _write_tree(rgb_root)
    # rename one RGB file -> trees no longer describe the same frames
    (rgb_root / "images/train/000002.jpg").rename(
        rgb_root / "images/train/000009.jpg")
    (rgb_root / "labels/train/000002.txt").rename(
        rgb_root / "labels/train/000009.txt")

    cfg = _cfg(imgsz=64)
    data, rgb_data = _data(ir_root), _data(rgb_root)
    with pytest.raises(PairSyncError, match="pair order mismatch"):
        make_dataset_v2(cfg=cfg, img_path=data["train"], batch=2, data=data,
                        mode="train", modality="pair", rgb_data=rgb_data,
                        rgb_img_path=rgb_data["train"])


def test_short_rgb_tree_is_rejected(tmp_path):
    from gfdata.unified_v2 import PairSyncError, make_dataset_v2

    ir_root, rgb_root = tmp_path / "ir", tmp_path / "rgb"
    _write_tree(ir_root, n=4)
    _write_tree(rgb_root, n=3)
    cfg = _cfg(imgsz=64)
    data, rgb_data = _data(ir_root), _data(rgb_root)
    with pytest.raises(PairSyncError, match="length mismatch"):
        make_dataset_v2(cfg=cfg, img_path=data["train"], batch=2, data=data,
                        mode="train", modality="pair", rgb_data=rgb_data,
                        rgb_img_path=rgb_data["train"])


# ---------------------------------------------------------------------------
# 4. mosaic schedule
# ---------------------------------------------------------------------------

def test_close_mosaic_disables_both_halves(tmp_path):
    """Mosaic must close on the IR *and* RGB chain, or the pair desyncs."""
    from ultralytics.data.augment import Mosaic

    from gfdata.unified_v2 import make_dataset_v2

    ir_root, rgb_root = tmp_path / "ir", tmp_path / "rgb"
    _write_tree(ir_root)
    _write_tree(rgb_root)
    cfg = _cfg(imgsz=64)
    data, rgb_data = _data(ir_root), _data(rgb_root)
    ds = make_dataset_v2(cfg=cfg, img_path=data["train"], batch=2, data=data,
                         mode="train", modality="pair", rgb_data=rgb_data,
                         rgb_img_path=rgb_data["train"])

    hyp = _cfg(imgsz=64, mosaic=1.0)
    ds.close_mosaic(hyp=hyp)

    assert hyp.mosaic == 0.0
    assert ds.mosaic is False
    for child in (ds._ir, ds._rgb):
        assert not any(isinstance(t, Mosaic) for t in child.transforms.transforms)
    # and the pair is still aligned after the rebuild
    for idx in (0, 1):
        _seek(0)
        s = ds[idx]
        assert torch.equal(s["img"][:3], s["img"][3:])
