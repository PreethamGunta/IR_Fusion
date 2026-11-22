"""GF-YOLO: dual-stream YOLO12s with mid-level fusion (experiment 09).

One renumbered ModuleList so ultralytics' own machinery (_predict_once,
fuse(), stride init, v8 loss) runs unchanged:

    0-8    IR backbone  (stock yolo12s backbone layers)
    9-17   RGB backbone (fresh copy of the same layers)
    18-20  Fuse modules at P3/P4/P5 (from the reviewed testbed_gfyolo)
    21-32  stock neck + Detect head, from-indices remapped onto the fused
           anchors (backbone P3/P4/P5 -> fused P3/P4/P5)

Channel contract: input [B,6,H,W], 0:3 = IR, 3:6 = RGB
(gfyolo.paired_dataset). Init: both backbones and the neck/head load the
pretrained yolo12s weights; per the plan the IR stem filters are replaced by
their per-filter channel-mean (grayscale) copy, RGB keeps the original.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import yaml as _yaml
import yaml
from torch import nn

EXP = Path(__file__).resolve().parents[2]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))  # for the testbed_gfyolo package

from testbed_gfyolo.fuse_modules import (  # noqa: E402
    FuseConcat1x1,
    FuseGatedResidual,
    FuseBiCrossAttention,
    FuseGATLocal,
    FuseGATSemantic,
)
from ultralytics.nn.tasks import DetectionModel  # noqa: E402

FUSE_REGISTRY = {
    "E10": FuseConcat1x1,
    "E14": FuseGatedResidual,
    "E21": FuseBiCrossAttention,
    "E30": FuseGATLocal,
    "E35": FuseGATSemantic,
}


def _stock_layer_count(cfg) -> int:
    """Number of backbone layers in the stock yaml (split index)."""
    if isinstance(cfg, dict):
        return len(cfg["backbone"])
    from ultralytics.utils import ROOT
    p = Path(cfg)
    for cand in (p,
                 p.with_suffix(".yaml") if p.suffix != ".yaml" else p,
                 ROOT / "cfg" / "models" / "12" / (p.name if p.suffix else f"{p.name}.yaml"),
                 ROOT / "cfg" / "models" / "12" / f"{p.stem}.yaml"):
        if cand.exists():
            return len(_yaml.safe_load(cand.read_text())["backbone"])
    raise FileNotFoundError(f"cannot resolve model cfg {cfg}")


class GFYOLO(DetectionModel):
    """Dual-stream YOLO12s with a Fuse module at P3/P4/P5."""

    def __init__(self, cfg="yolo12s.yaml", nc=5, ch=6, fuse="E10",
                 fuse_kwargs=None, verbose=False):
        super().__init__(cfg, nc=nc, ch=3, verbose=verbose)
        # the stock model is parsed; read the split point from its own yaml
        try:
            n_bk = len(self.yaml["backbone"])
        except (KeyError, TypeError):
            from ultralytics.nn.tasks import yaml_model_load
            n_bk = len(yaml_model_load(str(cfg))["backbone"])
        base = list(self.model)

        # backbone anchors (P3/P4/P5) = backbone indices referenced by
        # neck Concat/Detect from-lists
        anchors = sorted({j for m in base[n_bk:]
                          for j in (m.f if isinstance(m.f, list) else [])
                          if isinstance(j, int) and 0 <= j < n_bk})
        assert len(anchors) == 3, f"expected 3 anchors, got {anchors}"

        # anchor output channels via a dummy forward with hooks
        self.eval()
        shapes = {}
        hooks = [base[a].register_forward_hook(
            lambda mo, i, o, a=a: shapes.__setitem__(a, tuple(o.shape)))
            for a in anchors]
        with torch.no_grad():
            self(torch.zeros(1, 3, 256, 256))
        for h in hooks:
            h.remove()
        chs = [shapes[a][1] for a in anchors]
        self.train()

        bone_ir = base[:n_bk]
        bone_rgb = [copy.deepcopy(m) for m in base[:n_bk]]

        new = nn.ModuleList()
        for i, m in enumerate(bone_ir):
            m.i, m.f = i, -1
            new.append(m)
        for i, m in enumerate(bone_rgb):
            m.i, m.f = n_bk + i, -1
            new.append(m)
        n_fuse = len(anchors)
        fk = dict(fuse_kwargs or {})
        for pos, (a, C) in enumerate(zip(anchors, chs)):
            fmod = FUSE_REGISTRY[fuse](C, **fk)
            fmod.i = 2 * n_bk + pos
            fmod.f = [a, n_bk + a]  # [IR anchor, RGB anchor]
            fmod.is_fuse = True
            new.append(fmod)

        off = 2 * n_bk + n_fuse  # neck start index
        def remap(r):
            if r in anchors:
                return 2 * n_bk + anchors.index(r)   # fused anchor
            return r + (off - n_bk)                    # neck-internal shift

        for j, m in enumerate(base[n_bk:]):
            m.i = off + j
            if isinstance(m.f, int):
                m.f = -1 if m.f == -1 else remap(m.f)
            else:
                m.f = [-1 if r == -1 else remap(r) for r in m.f]
            new.append(m)

        self.model = new
        self.save = sorted({j for m in new
                            for j in (m.f if isinstance(m.f, list) else [m.f])
                            if j != -1})
        self._rgb_entry = n_bk  # layer index where the RGB stream starts
        self._assembled = True
        # strides/bias init already ran on the stock model; the fusion does
        # not change spatial strides. Assert once, using the same trick as
        # the stock DetectionModel init (Detect.training=True -> feats):
        head = self.model[-1]
        self.model.eval()
        head.training = True
        with torch.no_grad():
            out = self(torch.zeros(1, ch, 256, 256))
        self.model.train()
        feats = out["feats"] if isinstance(out, dict) else out
        got = torch.tensor([256 / o.shape[-2] for o in feats])
        assert torch.allclose(got, self.stride.to(got)), (got, self.stride)

    def _predict_once(self, x, profile=False, embed=None):
        """Stock interpreter with two entry points (IR at 0, RGB at _rgb_entry).

        Before reassembly (`_assembled` unset) it delegates to the parent so
        the stock-model probes in __init__ keep working with 3-ch inputs.
        """
        if not getattr(self, "_assembled", False):
            return super()._predict_once(x, profile, embed)
        assert not profile and not embed, "profile/embed not supported"
        y, cur = [], None
        for m in self.model:
            i = m.i
            if i == 0:
                cur = x[:, :3].contiguous()
            elif i == self._rgb_entry:
                cur = x[:, 3:].contiguous()
            elif getattr(m, "is_fuse", False):
                cur = m(y[m.f[0]], y[m.f[1]])
                y.append(cur if i in self.save else None)
                continue
            elif m.f != -1:
                cur = y[m.f] if isinstance(m.f, int) else \
                    [cur if j == -1 else y[j] for j in m.f]
            cur = m(cur)
            y.append(cur if i in self.save else None)
        return cur

    # -- pretrained loading ------------------------------------------------

    def load_pretrained(self, weights) -> None:
        """Load stock yolo12s weights into both streams + neck/head.

        IR stem (layer 0) filters are replaced by their per-filter
        channel-mean (grayscale) copy; RGB keeps the original filters.
        """
        sd = weights.float().state_dict() if isinstance(weights, nn.Module) \
            else weights["model"].float().state_dict()
        n_bk = 9  # yolo12 backbone length; see _stock_layer_count
        off = 2 * n_bk + 3 - n_bk  # neck index shift (12)
        cur = self.state_dict()
        new = {}
        for k, v in sd.items():
            parts = k.split(".")
            if len(parts) < 2 or parts[0] != "model":
                continue
            try:
                i = int(parts[1])
            except ValueError:
                continue
            rest = ".".join(parts[2:])
            for tgt in (i, i + off if i >= n_bk else i + n_bk):
                key = f"model.{tgt}.{rest}"
                if key in cur and cur[key].shape == v.shape:
                    new[key] = v.clone()
        # IR stem = grayscale mean of the RGB stem filters
        stem = new.get("model.0.conv.weight", None)
        if stem is not None:
            gray = stem.mean(1, keepdim=True).repeat(1, 3, 1, 1)
            if "model.0.conv.weight" in cur and \
                    cur["model.0.conv.weight"].shape == gray.shape:
                new["model.0.conv.weight"] = gray
        missing, unexpected = self.load_state_dict(new, strict=False)
        print(f"[gfyolo] loaded {len(new)} pretrained tensors "
              f"(stem grayscale-adapted); missing={len(missing)} "
              f"unexpected={len(unexpected)}")
