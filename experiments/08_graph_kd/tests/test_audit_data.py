"""Audit tests for data conversion and fused-stem initialisation."""
import torch

from graphkd.data.build_dronevehicle import LABEL_DIRS, _parse
from graphkd.training.trainer import _adapt_first_conv


def test_adapt_first_conv_uses_ir_then_rgb_order():
    v = torch.arange(1 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 3, 2, 2)
    out = _adapt_first_conv(v)
    gray = v.mean(1, keepdim=True).repeat(1, 3, 1, 1)
    assert out.shape == (1, 6, 2, 2)
    assert torch.allclose(out[:, :3], gray / 2.0)
    assert torch.allclose(out[:, 3:], v / 2.0)


def test_dronevehicle_uses_thermal_labels_for_ir_only():
    assert LABEL_DIRS["ir"] == {"train": "trainlabelr", "val": "vallabelr", "test": "testlabelr"}
    assert LABEL_DIRS["rgb"] == {"train": "trainlabel", "val": "vallabel", "test": "testlabel"}
    for split in ("train", "val", "test"):
        assert LABEL_DIRS["ir"][split] != LABEL_DIRS["rgb"][split]


def _xml(objects):
    body = "".join(objects)
    return f"<annotation>{body}</annotation>".encode()


def _obj(name, pts):
    poly = "".join(f"<x{i+1}>{x}</x{i+1}><y{i+1}>{y}</y{i+1}>" for i, (x, y) in enumerate(pts))
    return f"<object><name>{name}</name><polygon>{poly}</polygon></object>"


def test_dronevehicle_polygon_aabb_clips_to_crop():
    # points are in the original 840x712 frame; border 100 is removed.
    xml = _xml([
        _obj("car", [(150, 200), (350, 200), (350, 400), (150, 400)]),
        _obj("truck", [(50, 50), (900, 50), (900, 700), (50, 700)]),
    ])
    out = _parse(xml)
    assert len(out) == 2
    c, x1, y1, x2, y2 = out[0]
    assert (x1, y1, x2, y2) == (50.0, 100.0, 250.0, 300.0)
    c, x1, y1, x2, y2 = out[1]
    assert (x1, y1, x2, y2) == (0.0, 0.0, 640.0, 512.0)


def test_dronevehicle_degenerate_polygon_dropped():
    xml = _xml([_obj("car", [(150, 200), (151, 200), (151, 201), (150, 201)])])
    assert _parse(xml) == []
