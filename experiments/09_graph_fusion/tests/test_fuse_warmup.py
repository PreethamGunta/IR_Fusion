"""Fuse warm-up (experiment B1): schedule, inertness at alpha=0, identity at alpha=1.

The warm-up holds the fuse contribution at exactly zero for the first K epochs
and then ramps it in. Two contracts matter and are tested here:

1. `alpha == 0` must make the fuse **completely inert** — perturbing every fuse
   parameter must not change the model output at all. Anything weaker would
   mean the "fuse-free" phase is not actually IR-only.
2. `alpha == 1` must reproduce the un-warmed forward **exactly**, so a warmed-up
   run is comparable to the current tier at matched epochs.

`alpha` is a pure function of `_fuse_epoch` (the trainer sets it on the model
*and* its EMA each epoch); these tests also pin the unknown-epoch fallback so a
loaded checkpoint can never silently run with the fuse disabled.
"""
import pytest
import torch

from gfyolo.model import GFYOLO

CFG = dict(cfg="yolo12s.yaml", nc=5, ch=6, fuse="E30",
           fuse_kwargs={"d": 128}, verbose=False)


def _model(**kw):
    torch.manual_seed(0)
    m = GFYOLO(**CFG, **kw)
    return m.eval()


def _flat(out):
    """Flatten a Detect output (dict / tuple / tensor) into a tensor list."""
    if isinstance(out, dict):
        return [t for v in out.values() for t in _flat(v)]
    if isinstance(out, (list, tuple)):
        return [t for o in out for t in _flat(o)]
    return [out] if torch.is_tensor(out) else []


def _forward(m, x):
    with torch.no_grad():
        return [t.detach().clone() for t in _flat(m(x))]


# -- schedule -------------------------------------------------------------

def _bare(warmup, ramp, epoch):
    m = GFYOLO.__new__(GFYOLO)          # schedule is a pure function
    m.fuse_warmup_epochs = warmup
    m.fuse_ramp_epochs = ramp
    m._fuse_epoch = epoch
    return m.fuse_alpha()


def test_schedule_zero_during_warmup_then_full():
    assert [_bare(5, 1, e) for e in range(5)] == [0.0] * 5
    assert _bare(5, 1, 5) == 1.0
    assert _bare(5, 1, 79) == 1.0


def test_schedule_ramp_is_linear():
    assert _bare(5, 2, 5) == pytest.approx(0.5)
    assert _bare(5, 2, 6) == 1.0
    assert _bare(5, 4, 5) == pytest.approx(0.25)
    assert _bare(5, 4, 8) == 1.0


def test_schedule_off_by_default():
    """warmup=0 is the previous behaviour: full fusion from step 0."""
    assert _bare(0, 1, 0) == 1.0
    assert _bare(0, 1, 79) == 1.0


def test_unknown_epoch_never_bypasses_the_fuse():
    """A checkpoint without `_fuse_epoch` must fail open (alpha=1), not closed."""
    m = GFYOLO.__new__(GFYOLO)
    m.fuse_warmup_epochs = 5
    m.fuse_ramp_epochs = 1
    m._fuse_epoch = None
    assert m.fuse_alpha() == 1.0


def test_legacy_checkpoint_without_warmup_attrs_fails_open():
    """Checkpoints written before B1 have no schedule attributes at all."""
    m = GFYOLO.__new__(GFYOLO)          # no fuse_warmup_epochs / _fuse_epoch
    assert m.fuse_alpha() == 1.0


# -- forward contracts ----------------------------------------------------

def test_alpha_zero_makes_the_fuse_inert():
    m = _model(fuse_warmup_epochs=5, fuse_ramp_epochs=1)
    m._fuse_epoch = 0                                  # alpha == 0
    x = torch.rand(1, 6, 256, 256)
    before = _forward(m, x)
    fuses = [mod for mod in m.modules() if getattr(mod, "is_fuse", False)]
    assert len(fuses) == 3
    with torch.no_grad():
        for f in fuses:
            for p in f.parameters():
                p.mul_(3.0)                            # corrupt the fuse
    after = _forward(m, x)
    assert before and len(before) == len(after)
    for b, a in zip(before, after):
        assert torch.equal(b, a), "fuse influenced the output at alpha=0"


def test_alpha_one_matches_the_unwarmed_forward():
    base = _model()                                    # warmup=0 -> alpha 1
    warm = _model(fuse_warmup_epochs=5, fuse_ramp_epochs=1)
    warm._fuse_epoch = 5                               # alpha == 1
    warm.load_state_dict(base.state_dict())             # identical weights
    x = torch.rand(1, 6, 256, 256)
    a, b = _forward(base, x), _forward(warm, x)
    for u, v in zip(a, b):
        assert torch.allclose(u, v, atol=1e-6, rtol=1e-6)


def test_alpha_one_is_sensitive_to_the_fuse():
    """Sanity check on the previous test: with alpha=1 the fuse does matter."""
    m = _model(fuse_warmup_epochs=5, fuse_ramp_epochs=1)
    m._fuse_epoch = 5
    x = torch.rand(1, 6, 256, 256)
    before = _forward(m, x)
    with torch.no_grad():
        for mod in m.modules():
            if getattr(mod, "is_fuse", False):
                for p in mod.parameters():
                    p.mul_(3.0)
    after = _forward(m, x)
    assert any(not torch.allclose(u, v) for u, v in zip(before, after))


# -- trainer wiring -------------------------------------------------------

class _Stub(torch.nn.Module):
    """Stands in for a GFYOLO on the attribute the trainer stamps."""

    def __init__(self, warmup=5):
        super().__init__()
        self.fuse_warmup_epochs = warmup
        self._fuse_epoch = None


def _stub_trainer(model, ema=None, epoch=3):
    from types import SimpleNamespace
    from gfyolo.gf_trainer import GFTrainer
    t = GFTrainer.__new__(GFTrainer)
    t.device = torch.device("cpu")
    t.args = SimpleNamespace(multi_scale=0.0)
    t.epoch = epoch
    t.model = model
    t.ema = SimpleNamespace(ema=ema) if ema is not None else None
    return t


def test_preprocess_batch_stamps_epoch_on_model_and_ema():
    """EMA is refreshed via state_dict only, so it must be stamped explicitly."""
    model, ema = _Stub(), _Stub()
    t = _stub_trainer(model, ema, epoch=3)
    batch = {"img": torch.zeros(1, 3, 16, 16, dtype=torch.uint8)}
    out = t.preprocess_batch(batch)
    assert model._fuse_epoch == 3
    assert ema._fuse_epoch == 3
    assert out["img"].dtype == torch.float32      # parent behaviour preserved


def test_preprocess_batch_ignores_models_without_a_schedule():
    """A plain IR-only model must not crash the shared preprocess path."""
    t = _stub_trainer(torch.nn.Conv2d(3, 3, 1), None, epoch=3)
    batch = {"img": torch.zeros(1, 3, 16, 16, dtype=torch.uint8)}
    t.preprocess_batch(batch)                    # no attribute -> no-op


# -- gradient path at alpha=0 --------------------------------------------

def test_alpha_zero_starves_the_fuse_and_the_rgb_stream():
    """Warm-up is exactly IR-only: RGB has no path to the loss.

    This documents a real consequence of the design -- during the warm-up the
    RGB backbone is frozen too, because its only consumer is the fuse whose
    output is discarded. Gradients must still reach the IR backbone.
    """
    m = _model(fuse_warmup_epochs=5, fuse_ramp_epochs=1)
    m._fuse_epoch = 0
    m.train()
    x = torch.rand(1, 6, 256, 256)
    loss = sum(t.sum() for t in _flat(m(x)))
    loss.backward()

    fuse = [mod for mod in m.modules() if getattr(mod, "is_fuse", False)]
    assert fuse, "no fuse modules found"
    for mod in fuse:
        for p in mod.parameters():
            assert p.grad is None or torch.count_nonzero(p.grad) == 0, \
                "fuse received gradient at alpha=0"

    ir_grad = sum(float(p.grad.abs().sum()) for p in m.model[0].parameters()
                  if p.grad is not None)
    assert ir_grad > 0, "IR backbone received no gradient at alpha=0"


# -- divergence watchdog --------------------------------------------------

def test_nonfinite_guard_ignores_healthy_and_transient_steps():
    from gfyolo.gf_trainer import GFTrainer

    class T:
        loss = torch.tensor(1.0)
    t = T()
    for _ in range(5):
        GFTrainer._guard_nonfinite_loss(t)        # finite: no-op
    assert getattr(t, "_nonfinite", 0) == 0
    t.loss = torch.tensor(float("nan"))
    for _ in range(5):
        GFTrainer._guard_nonfinite_loss(t)        # transient: counted, no exit
    assert t._nonfinite == 5
    t.loss = torch.tensor(2.0)                    # recovery resets the counter
    GFTrainer._guard_nonfinite_loss(t)
    assert t._nonfinite == 0


def test_nonfinite_guard_aborts_on_sustained_divergence():
    from gfyolo.gf_trainer import GFTrainer

    class T:
        loss = torch.tensor(float("nan"))
    t = T()
    for _ in range(19):
        GFTrainer._guard_nonfinite_loss(t)
    with pytest.raises(SystemExit):
        GFTrainer._guard_nonfinite_loss(t)        # 20th consecutive


