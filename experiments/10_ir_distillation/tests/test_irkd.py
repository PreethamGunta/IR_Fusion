"""Unit & CPU smoke tests for experiment 10 (IR-only distillation).

Fast by design: 256x256 teacher forwards, one 640x512 IR forward, one backward
pass. No training, no dataset. The GPU 2-epoch pipeline smoke is stage N0 in
``scripts/run_node2_queue.sh``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

EXP10 = Path(__file__).resolve().parents[1]
EXP09 = EXP10.parent / "09_graph_fusion"
for _p in (str(EXP10 / "src"), str(EXP09 / "src"), str(EXP09)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TEACHER = EXP09 / "runs/e30_v2_100ep/weights/best.pt"


@pytest.fixture(scope="session")
def teacher():
    from irkd.teacher_hooks import E30Teacher

    if not TEACHER.exists():
        pytest.skip(f"teacher checkpoint missing: {TEACHER}")
    t = E30Teacher(TEACHER, device="cpu")
    yield t
    t.close()


@pytest.fixture(scope="session")
def student_s():
    from ultralytics.nn.tasks import DetectionModel

    return DetectionModel("yolo12s.yaml", nc=5)


def test_teacher_residual_parity(teacher):
    x = torch.randn(2, 6, 256, 256)
    ir, delta, fused = teacher(x)
    assert len(ir) == len(delta) == len(fused) == 3
    for a, d, f in zip(ir, delta, fused):
        assert a.shape == d.shape == f.shape
        torch.testing.assert_close(f, a + d, atol=1e-4, rtol=0)
    assert all(not p.requires_grad for p in teacher.model.parameters())
    assert teacher.channels == [256, 256, 512]


def test_student_anchors_match_teacher(student_s):
    from irkd.teacher_hooks import probe_channels, student_anchor_indices

    anchors = student_anchor_indices(student_s)
    assert anchors == [4, 6, 8]
    ch = probe_channels(student_s, anchors)
    assert ch == [256, 256, 512]


def test_student_purity(student_s):
    from ultralytics.nn.tasks import DetectionModel

    assert isinstance(student_s, DetectionModel)
    assert student_s.__class__.__module__.startswith("ultralytics")
    assert not any("irkd" in n for n, _ in student_s.named_modules())
    params = sum(p.numel() for p in student_s.parameters())
    assert abs(params - 9.26e6) < 6e4, f"unexpected YOLO12s size: {params}"


def test_student_ir_forward_640x512(student_s):
    student_s.eval()
    with torch.no_grad():
        out = student_s(torch.zeros(1, 3, 512, 640))
    assert isinstance(out, (tuple, list))


def test_ir_forward_rejects_six_channels(student_s):
    student_s.eval()
    with pytest.raises(Exception):
        with torch.no_grad():
            student_s(torch.zeros(1, 6, 320, 320))


def test_cos_huber_identity():
    from irkd.losses import cos_huber

    x = torch.randn(2, 16, 8, 8)
    assert float(cos_huber(x, x.clone())) < 1e-6


def test_decomposed_gradients(teacher, student_s):
    from irkd.losses import DecomposedKD
    from irkd.teacher_hooks import student_anchor_indices

    from irkd.teacher_hooks import probe_channels

    anchors = student_anchor_indices(student_s)
    s_ch = probe_channels(student_s, anchors)
    feats: dict[int, torch.Tensor] = {}
    hooks = [
        student_s.model[i].register_forward_hook(
            lambda _m, _i, o, idx=i: feats.__setitem__(idx, o)
        )
        for i in anchors
    ]
    student_s.eval()
    kd = DecomposedKD(s_ch, teacher.channels, alpha=1.0)

    x3 = torch.randn(2, 3, 256, 256)
    student_s(x3)
    s = [feats[i] for i in anchors]
    t_ir, t_delta, t_fused = teacher(torch.randn(2, 6, 256, 256))

    losses, items = kd(s, t_ir, t_delta, t_fused)
    loss = losses["kd_pres"] + losses["kd_comp"]
    assert torch.isfinite(loss)
    loss.backward()

    proj_grad = sum(float(p.grad.abs().sum()) for p in kd.parameters() if p.grad is not None)
    bb_grad = sum(float(p.grad.abs().sum()) for p in student_s.parameters() if p.grad is not None)
    assert proj_grad > 0, "projectors received no gradient"
    assert bb_grad > 0, "student backbone received no gradient"
    assert set(items) == {"kd_pres", "kd_comp"}

    for h in hooks:
        h.remove()


def test_direct_runs(student_s, teacher):
    from irkd.losses import DirectKD
    from irkd.teacher_hooks import student_anchor_indices

    from irkd.teacher_hooks import probe_channels

    anchors = student_anchor_indices(student_s)
    s_ch = probe_channels(student_s, anchors)
    feats: dict[int, torch.Tensor] = {}
    hooks = [
        student_s.model[i].register_forward_hook(
            lambda _m, _i, o, idx=i: feats.__setitem__(idx, o)
        )
        for i in anchors
    ]
    student_s.eval()
    kd = DirectKD(s_ch, teacher.channels)
    with torch.no_grad():
        student_s(torch.randn(1, 3, 256, 256))
        t_ir, t_delta, t_fused = teacher(torch.randn(1, 6, 256, 256))
        losses, items = kd([feats[i] for i in anchors], t_ir, t_delta, t_fused)
    assert torch.isfinite(losses["kd_direct"]) and "kd_direct" in items
    for h in hooks:
        h.remove()


def test_checkpoint_roundtrip_is_stock(student_s, tmp_path):
    """A plain saved DetectionModel loads in stock YOLO and runs 3-channel IR."""
    from ultralytics import YOLO

    ckpt = tmp_path / "student.pt"
    torch.save({"model": student_s.half(), "train_args": {"task": "detect"}}, ckpt)
    m = YOLO(str(ckpt))
    assert m.model.__class__.__module__.startswith("ultralytics")
    m.model.float().eval()
    with torch.no_grad():
        out = m.model(torch.zeros(1, 3, 320, 320))
    assert isinstance(out, (tuple, list))
