"""E02 fused-module tests: stem channel order, stem adaptation, paired geometry.

Covers the plan's checklist items: the [IR | RGB] channel contract (08 had
the stem halves swapped) and the paired-augmentation geometry test with
photometric aug off and IR == RGB input.
"""
import pytest
import torch

from gfyolo.fused_trainer import adapt_first_conv, adapt_stem
from gfyolo.paired_dataset import paired_geom


def test_adapt_first_conv_channel_order():
    """Channels 0:3 = grayscale mean (IR proxy), 3:6 = original RGB, halved."""
    v = torch.randn(16, 3, 3, 3)
    out = adapt_first_conv(v)
    gray = v.mean(1, keepdim=True).repeat(1, 3, 1, 1)
    assert out.shape == (16, 6, 3, 3)
    assert torch.allclose(out[:, :3], gray / 2.0, atol=1e-6)
    assert torch.allclose(out[:, 3:], v / 2.0, atol=1e-6)
    # IR half is modality-agnostic: all 3 IR channels identical per position.
    assert torch.allclose(out[:, 0], out[:, 1], atol=1e-6)
    assert torch.allclose(out[:, 1], out[:, 2], atol=1e-6)


def test_adapt_stem_loads_3ch_into_6ch():
    """A 3-ch checkpoint loads into a 6-ch model; only the stem is adapted."""
    model = torch.nn.Sequential(
        torch.nn.Conv2d(6, 8, 3, bias=False),
        torch.nn.Conv2d(8, 4, 3, bias=False),
    )
    src = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, bias=False),
        torch.nn.Conv2d(8, 4, 3, bias=False),
    )
    with torch.no_grad():
        for p in src.parameters():
            p.normal_()
    adapt_stem(model, src)
    sd = model.state_dict()
    w = src.state_dict()["0.weight"]
    gray = w.mean(1, keepdim=True).repeat(1, 3, 1, 1)
    assert torch.allclose(sd["0.weight"][:, :3], gray / 2.0, atol=1e-6)
    assert torch.allclose(sd["0.weight"][:, 3:], w / 2.0, atol=1e-6)
    assert torch.allclose(sd["1.weight"], src.state_dict()["1.weight"])


def _fixed_rand(value: float):
    def _rand(*args, **kwargs):
        # paired_geom only calls torch.rand(()); return a scalar everywhere.
        return torch.tensor(value)
    return _rand


def test_gat_runtime_caches_not_pickled():
    """Checkpoint pickles must not carry device-tagged runtime caches.

    Regression (2026-11-23, E35 exit-time eval): the pickled table cache was
    keyed (H, W, cuda:0) but reloaded to CPU by map_location, so the next
    forward mixed CPU tables with CUDA features. Caches must be dropped on
    pickle and rebuilt lazily.
    """
    import pickle
    import torch
    from testbed_gfyolo.fuse_modules import FuseGATLocal

    m = FuseGATLocal(C=4, d=8, r=1)
    m._tables(4, 4, torch.device("cpu"))
    m.last_delta = torch.zeros(1)
    m2 = pickle.loads(pickle.dumps(m))
    assert m2._table_cache == {}
    assert m2.last_delta is None
    # stale-device entries are rebuilt on demand
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        cpu_tbl = m._tables(4, 4, torch.device("cpu"))
        m._table_cache[(4, 4, dev)] = dict(cpu_tbl)
        tbl = m._tables(4, 4, dev)
        assert tbl["src"].device.type == "cuda"


def test_sanitize_bn_buffers_resets_poisoned_stats():
    """A transient inf forward must not permanently poison BN running stats.

    Regression (2026-11-22, E35 attempt 2): one non-finite train forward was
    absorbed into running_var; the next validation (which uses running stats)
    died with non-finite attention. The watchdog resets poisoned buffers so
    they re-estimate instead of killing the run.
    """
    import torch
    from torch import nn
    from gfyolo.gf_trainer import sanitize_bn_buffers

    m = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU())
    bn = m[1]
    bn.running_var[2] = float("inf")
    bn.running_mean[1] = float("nan")
    assert sanitize_bn_buffers(m) == 1
    assert bool(torch.isfinite(bn.running_var).all())
    assert bool(torch.isfinite(bn.running_mean).all())
    assert float(bn.running_var.max()) == 1.0 and float(bn.running_mean.abs().max()) == 0.0
    # healthy model: no-op, zero resets
    assert sanitize_bn_buffers(m) == 0


def test_close_mosaic_is_noop():
    """close_mosaic must not rebuild the IR transforms (pairing guard).

    Regression: the stock rebuild enabled RandomFlip/HSV on the IR half only,
    misaligning the pair from epoch 91 on and collapsing the dual-stream runs.
    """
    from ultralytics.cfg import get_cfg
    from gfyolo.paired_dataset import make_paired_dataset

    ds = make_paired_dataset("configs/dronevehicle_v2_paired_screening.yaml",
                             "configs/dronevehicle_v2_rgb_screening.yaml",
                             "train", mode="train", imgsz=640,
                             paired_augment=True, paired_rot90=True)
    before = [id(t) for t in ds.transforms.transforms]
    ds.close_mosaic(hyp=get_cfg())
    after = [id(t) for t in ds.transforms.transforms]
    assert before == after, "close_mosaic rebuilt the IR transforms"


def test_paired_geom_identical_halves(monkeypatch):
    """IR == RGB input, photometry off => geometry keeps the halves identical
    and the boxes valid (plan sec 0.5 paired augmentation test)."""
    H = W = 64
    ir = torch.randint(0, 255, (3, H, W), dtype=torch.uint8)
    img = torch.cat([ir, ir], 0)  # IR == RGB
    boxes = torch.tensor([[0.25, 0.50, 0.10, 0.20],
                          [0.75, 0.50, 0.10, 0.20]])
    cls = torch.tensor([[0.0], [1.0]])
    bids = torch.tensor([0.0, 0.0])
    monkeypatch.setattr(torch, "rand", _fixed_rand(0.9))  # no flips, no rot90
    out_img, out_boxes, out_cls, out_bids = paired_geom(
        img.clone(), boxes.clone(), cls, bids, rot90=False)
    assert out_img.shape == (6, H, W)
    assert torch.equal(out_img[:3], out_img[3:])  # halves identical
    assert (out_boxes[:, 2] > 0).all() and (out_boxes[:, 3] > 0).all()
    assert ((out_boxes[:, [0, 1]] >= 0) & (out_boxes[:, [0, 1]] <= 1)).all()
    assert len(out_cls) == len(out_boxes)


def test_paired_geom_flip_box_update(monkeypatch):
    """Forced flip mirrors box centres exactly (analytic box update)."""
    H = W = 32
    ir = torch.zeros(3, H, W, dtype=torch.uint8)
    img = torch.cat([ir, ir], 0)
    boxes = torch.tensor([[0.25, 0.50, 0.10, 0.10]])
    cls = torch.tensor([[0.0]])
    bids = torch.tensor([0.0])
    monkeypatch.setattr(torch, "rand", _fixed_rand(0.0))  # flips happen, rot90 branch off
    out_img, out_boxes, *_ = paired_geom(img.clone(), boxes.clone(), cls, bids, rot90=False)
    # flips mirror (0.25, 0.50) -> (0.75, 0.50); then the affine with
    # sc=0.80, tx=ty=-0.08 maps it to ((0.75-0.5)*0.8+0.42, (0.5-0.5)*0.8+0.42).
    cx = float(out_boxes[0, 0])
    cy = float(out_boxes[0, 1])
    assert abs(cx - 0.62) < 1e-5
    assert abs(cy - 0.42) < 1e-5
