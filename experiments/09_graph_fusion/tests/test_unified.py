"""Tests for the unified data pipeline (src/gfdata/unified.py)."""
import numpy as np
import pytest


def test_affine_identity_preserves_image_and_boxes():
    from gfdata.unified import affine
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)
    boxes = np.array([[0.5, 0.5, 0.3, 0.2]], np.float32)
    out, ob, keep = affine(img, boxes, 1.0, 0.0, 0.0, 0.0)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 1.0
    assert np.allclose(ob, boxes, atol=1e-4) and keep.all()


def test_affine_box_tracks_content():
    """A translated box must still cover the translated white square."""
    from gfdata.unified import affine
    img = np.zeros((3, 64, 64), np.uint8)
    img[:, 24:40, 24:40] = 255
    boxes = np.array([[0.5, 0.5, 0.25, 0.25]], np.float32)
    out, ob, keep = affine(img, boxes, 1.0, 0.05, 0.0, 0.0)
    assert keep.sum() == 1
    b = ob[0]
    x1 = int((b[0] - b[2] / 2) * 64)
    y1 = int((b[1] - b[3] / 2) * 64)
    x2 = int((b[0] + b[2] / 2) * 64)
    y2 = int((b[1] + b[3] / 2) * 64)
    assert out[:, y1:y2, x1:x2].mean() > 120


def test_flip_mirrors_boxes():
    from gfdata.unified import flip
    img = np.zeros((3, 16, 16), np.uint8)
    img[:, :, 12:] = 255
    boxes = np.array([[0.85, 0.5, 0.2, 0.2]], np.float32)
    out, ob = flip(img, boxes, True)
    assert out[:, :, :4].mean() > 200          # content moved left
    assert abs(float(ob[0, 0]) - 0.15) < 1e-6  # x mirrored


def test_mosaic4_composes_valid_sample():
    from gfdata.unified import mosaic4
    rng = np.random.default_rng(1)
    tiles = [{
        "img": rng.integers(0, 255, (6, 128, 128), dtype=np.uint8),
        "bboxes": np.array([[0.5, 0.5, 0.2, 0.2]], np.float32),
        "cls": np.array([0.0], np.float32),
        "batch_idx": np.array([0.0], np.float32),
    } for _ in range(4)]
    img, boxes, cls, bidx = mosaic4(tiles, 128)
    assert img.shape == (6, 128, 128) and img.dtype == np.uint8
    assert len(boxes) == len(cls) == len(bidx)
    # a mosaic is ONE image: per-tile batch ids would multiply the image count
    assert np.all(np.asarray(bidx) == 0)
    if len(boxes):
        assert boxes.min() >= 0.0 and boxes.max() <= 1.0
        assert (boxes[:, 2:] > 0).all()


def test_photometric_and_erase_shapes():
    from gfdata.unified import hsv_rgb, random_erase, value_ir
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)
    ir = rng.integers(0, 255, (3, 32, 32), dtype=np.uint8)
    assert hsv_rgb(rgb, 0.015, 0.7, 0.4).shape == rgb.shape
    assert value_ir(ir, 0.4).shape == ir.shape
    out = random_erase(rgb, 1.0)
    assert out.shape == rgb.shape and out.dtype == np.uint8


@pytest.mark.skipif(not __import__("pathlib").Path("configs/dronevehicle_v2_ir_screening.yaml").exists(),
                    reason="experiment configs not present")
def test_unified_dataset_modes_real_data():
    from gfdata.unified import make_dataset
    ir = make_dataset("configs/dronevehicle_v2_ir_screening.yaml", "val",
                      modality="ir", mode="val", imgsz=320, batch=4)
    s = ir[0]
    assert s["img"].shape[0] == 3 and s["bboxes"].ndim == 2

    pair = make_dataset("configs/dronevehicle_v2_paired_screening.yaml", "val",
                        modality="pair", mode="val", imgsz=320, batch=4,
                        rgb_yaml="configs/dronevehicle_v2_rgb_screening.yaml")
    s = pair[0]
    assert s["img"].shape[0] == 6

    train = make_dataset("configs/dronevehicle_v2_paired_screening.yaml", "train",
                         modality="pair", mode="train", imgsz=320, batch=4,
                         rgb_yaml="configs/dronevehicle_v2_rgb_screening.yaml")
    train.mosaic = False  # deterministic path for the test
    s = train[0]
    assert s["img"].shape[:2] == (6, 320)
    assert s["bboxes"].max() <= 1.0
    train.close_mosaic()
    assert train.mosaic is False

    # IR-only arm must survive the same train-mode path (the mosaic batch-id
    # bug hit both modalities through the loss, so exercise it explicitly)
    ir_train = make_dataset("configs/dronevehicle_v2_ir_screening.yaml", "train",
                            modality="ir", mode="train", imgsz=320, batch=4)
    ir_train.mosaic = False
    s = ir_train[0]
    assert s["img"].shape[:2] == (3, 320)
    assert bool((s["batch_idx"] == 0).all())
