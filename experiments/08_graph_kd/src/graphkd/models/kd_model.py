"""Graph-KD model and loss module.

The detector is an ordinary Ultralytics DetectionModel; the only change is that
`loss()` adds the graph distillation terms.  Teacher graphs are read from a
precomputed sharded cache, so the frozen teacher is never run during student
training.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from graphkd import paths
from graphkd.distillation.condensation import select, subset
from graphkd.distillation.consensus import consensus
from graphkd.distillation.gt_anchored import gt_anchored_kd
from graphkd.distillation.losses import (edge_kd, feature_kd, node_kd, pred_kd,
                                         struct_kd)
from graphkd.distillation.reliability import load_temperature, reliability
from graphkd.graph.cache import get_cache
from graphkd.graph.candidates import decode_predictions
from graphkd.matching import match
from graphkd.matching.cost import node_cost


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (paths.EXP_ROOT / p)


def neck_channels(model, ch: int = 3, imgsz: int = 640) -> tuple[list[int], list[int]]:
    head = model.model[-1]
    feats = []
    hook = head.register_forward_hook(lambda m, i, o: feats.append(i[0]))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, ch, imgsz, imgsz))
    hook.remove()
    model.train(was_training)
    return [f.shape[1] for f in feats[0]], [int(s) for s in model.stride.tolist()]


class KDModule(nn.Module):
    def __init__(self, cfg: dict, model):
        super().__init__()
        self.cfg = cfg
        self.nc = int(cfg.get("nc", 4))
        self.k = int(cfg.get("k", 100))
        self.conf = float(cfg.get("conf", 0.02))
        self.method = cfg.get("method", "node")
        self.matcher = cfg.get("matcher", "hungarian")
        base = ["kd_cls", "kd_box", "kd_obj", "kd_feat"]
        if self.method in ("node_edge", "sinkhorn", "fgw", "partial", "reliability",
                           "condense", "condensation", "decomposed"):
            base.append("kd_edge")
        if self.method == "fgw":
            base.append("kd_struct")
        if self.method == "decomposed":
            base.append("kd_if")
        if self.method in ("gt_anchor", "gt"):
            base = ["kd_gt", "kd_gt_n"]
        if self.method == "pred":
            base = ["kd_pred"]
        if self.method == "three_teacher":
            base.append("kd_consistency")
        self.term_keys = base
        self.weight = float(cfg.get("kd_weight", 1.0))
        self.warmup = int(cfg.get("warmup_epochs", 0))
        self.kd_every = max(1, int(cfg.get("kd_every", 1)))
        self.step = 0
        self.epoch = 0
        self.temp = load_temperature(_resolve(cfg["temperature_file"]) if cfg.get("temperature_file") else None)
        self.teachers = cfg.get("teachers", [])
        self.feat_out = int(cfg.get("feat_dim", 128))
        self.use_adapter = self.method in ("decomposed", "decomposed_light", "f")
        self.terms: list[dict] = []

        tin, strides = neck_channels(model, ch=int(cfg.get("student_ch", 3)))
        self.strides = torch.tensor(strides)
        self.student_feat_dim = sum(tin)
        head = model.model[-1]
        self.reg_max = int(getattr(head, "reg_max", 16))

        dims = []
        for t in self.teachers:
            meta = json.loads((_resolve(t["cache"]) / "meta.json").read_text())
            dims.append(int(meta["feat_dim"]))
        self._dims = dims
        self.use_feat = bool(cfg.get("use_feature_kd", False))
        if self.use_feat:
            self.proj_s = nn.Linear(self.student_feat_dim, self.feat_out, bias=False)
            self.proj_t = nn.ModuleList([nn.Linear(d, self.feat_out, bias=False) for d in dims])
        else:
            self.proj_s = self.proj_t = None

        if self.use_adapter:
            self.adapter = nn.ModuleList([nn.Conv2d(c, c, 1) for c in tin])
            for a in self.adapter:
                nn.init.zeros_(a.weight)
                nn.init.zeros_(a.bias)
        else:
            self.adapter = None

    def _load(self, stem: str, tconf: dict) -> dict | None:
        cache = get_cache(str(_resolve(tconf["cache"])))
        rec = cache.get(stem)
        if rec is None:
            return None
        nodes = {}
        for k in ("boxes", "cls", "obj", "feat", "anchor"):
            if k in rec:
                nodes[k] = rec[k].float() if k != "anchor" else rec[k]
        dense = None
        if "dense_obj" in rec:
            dense = {"obj": rec["dense_obj"].float(), "cls": rec["dense_cls"].float(),
                     "box": rec["dense_box"].float()}
        return {"nodes": nodes, "dense": dense, "weight": float(tconf.get("weight", 1.0))}

    def _epoch_scale(self) -> float:
        if self.warmup <= 0:
            return 1.0
        return min(1.0, (self.epoch + 1) / self.warmup)

    def forward(self, batch, preds, feats, gt_loss_items=None):
        self.step += 1
        if self.kd_every > 1 and (self.step - 1) % self.kd_every != 0:
            z = feats[0].new_zeros(())
            return z, {k: z for k in self.term_keys}
        scale = self._epoch_scale() * self.kd_every
        ref = feats[0]
        zero = ref.new_zeros(())
        names = [Path(f).stem.replace("_PreviewData", "").replace("_RGB", "") for f in batch["im_file"]]
        s_nodes, s_dense = decode_predictions(
            preds, feats, self.strides.to(ref.device), self.nc,
            k=self.k, conf=self.conf, dense=True,
        )
        if scale <= 0:
            return zero, {k: zero for k in self.term_keys}

        total = zero
        acc: dict[str, float] = {}
        n_img = 0
        for bi, name in enumerate(names):
            s = {k: (v.to(ref.device) if torch.is_tensor(v) else v) for k, v in s_nodes[bi].items()}
            gtb = batch["bboxes"][batch["batch_idx"] == bi].to(ref.device).float()
            gtc = batch["cls"][batch["batch_idx"] == bi].to(ref.device).long().flatten()
            loaded = []
            for ti, tconf in enumerate(self.teachers):
                rec = self._load(name, tconf)
                if rec is not None:
                    rec["_idx"] = ti
                    rec["nodes"] = {k: (v.to(ref.device) if torch.is_tensor(v) else v)
                                    for k, v in rec["nodes"].items()}
                    if rec["dense"] is not None:
                        rec["dense"] = {k: v.to(ref.device) for k, v in rec["dense"].items()}
                    loaded.append(rec)
            if not loaded:
                continue
            n_img += 1

            if self.method == "pred":
                for rec in loaded:
                    if rec["dense"] is None:
                        continue
                    sd = {k: s_dense[k][bi] for k in s_dense}
                    l, terms = pred_kd(sd, rec["dense"], self.cfg)
                    total = total + rec["weight"] * l
                    for k, v in terms.items():
                        acc[k] = acc.get(k, 0.0) + float(v)
                continue

            if self.method in ("gt_anchor", "gt"):
                if gtb.numel() == 0:
                    continue
                for rec in loaded:
                    l, terms = gt_anchored_kd(s, rec["nodes"], gtb, gtc, self.cfg)
                    total = total + rec["weight"] * l
                    for k, v in terms.items():
                        if k.startswith("kd_"):
                            acc[k] = acc.get(k, 0.0) + float(v)
                continue

            if self.method in ("consensus", "multi_teacher", "three_teacher"):
                aligned = []
                for rec in loaded:
                    t = rec["nodes"]
                    cost = node_cost(s, t, alpha=tuple(self.cfg.get("cost_alpha", (1, 1, .5, .5))))
                    p = match(self.matcher, s, t, cost, self.cfg)
                    rel = reliability(t, self.nc, self.temp).to(p.device) if self.cfg.get("reliability") else None
                    aligned.append((t, p, rel))
                pseudo, valid = consensus(s, aligned)
                if valid.sum() == 0:
                    continue
                ns = s["boxes"].shape[0]
                p = torch.eye(ns, device=s["boxes"].device, dtype=s["boxes"].dtype)
                l, terms = node_kd(s, pseudo, p, self.cfg, node_weight=valid)
                total = total + self.cfg.get("lambda_node", 1.0) * l
                for k, v in terms.items():
                    acc[k] = acc.get(k, 0.0) + float(v)
                if self.method == "three_teacher" and len(aligned) > 1:
                    cons = 0.0
                    for i in range(len(aligned)):
                        for j in range(i + 1, len(aligned)):
                            pi = aligned[i][1] / aligned[i][1].sum(0, keepdim=True).clamp_min(1e-6)
                            pj = aligned[j][1] / aligned[j][1].sum(0, keepdim=True).clamp_min(1e-6)
                            cons = cons + (pi * (pi.clamp_min(1e-8).log() - pj.clamp_min(1e-8).log())).sum()
                    l = cons / max(1, len(aligned)) * float(self.cfg.get("consistency_weight", 0.05))
                    total = total + l
                    acc["kd_consistency"] = acc.get("kd_consistency", 0.0) + float(l)
                continue

            rec = loaded[0]
            t = rec["nodes"]
            if self.method in ("condense", "condensation"):
                ccfg = self.cfg.get("condensation", {})
                idx = select(t, ccfg.get("strategy", "centrality"), int(ccfg.get("k", 30)), gtb, gtc)
                t = subset(t, idx)
                if t["boxes"].shape[0] == 0:
                    continue
            cost = node_cost(s, t, alpha=tuple(self.cfg.get("cost_alpha", (1, 1, .5, .5))))
            p = match(self.matcher, s, t, cost, self.cfg)
            nw = None
            if self.cfg.get("reliability") or self.method == "reliability":
                nw = reliability(t, self.nc, self.temp).to(p.device)
                if self.cfg.get("reliability_power"):
                    nw = nw.pow(float(self.cfg["reliability_power"]))
            l, terms = node_kd(s, t, p, self.cfg, node_weight=nw)
            total = total + rec["weight"] * self.cfg.get("lambda_node", 1.0) * l
            for k, v in terms.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            if self.method in ("node_edge", "sinkhorn", "fgw", "partial", "reliability",
                               "condense", "condensation", "decomposed"):
                le, te = edge_kd(s, t, p, self.cfg, node_weight=nw)
                total = total + rec["weight"] * self.cfg.get("lambda_edge", 0.5) * le
                acc.update({k: acc.get(k, 0.0) + float(v) for k, v in te.items()})
            if self.method == "fgw":
                ls, ts = struct_kd(s, t, p, self.cfg)
                total = total + rec["weight"] * self.cfg.get("lambda_struct", 0.1) * ls
                acc.update({k: acc.get(k, 0.0) + float(v) for k, v in ts.items()})
            if self.use_feat and self.proj_s is not None:
                lf, tf = feature_kd(s, t, p, self.cfg, self.proj_s, self.proj_t[rec["_idx"]])
                total = total + rec["weight"] * self.cfg.get("lambda_feat", 0.1) * lf
                acc.update({k: acc.get(k, 0.0) + float(v) for k, v in tf.items()})

        if n_img:
            total = total / n_img * scale * self.weight
            acc = {k: v / n_img for k, v in acc.items()}
        total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
        out = {k: torch.tensor(float(acc.get(k, 0.0)), device=ref.device) for k in self.term_keys}
        self.terms.append({k: float(v) for k, v in out.items()})
        return total, out

    def pop_terms(self) -> dict:
        if not self.terms:
            return {}
        keys = set().union(*[t.keys() for t in self.terms])
        avg = {k: sum(t.get(k, 0.0) for t in self.terms) / len(self.terms) for k in keys}
        self.terms = []
        return avg


from ultralytics.nn.tasks import DetectionModel


class KDDetectionModel(DetectionModel):

    def attach_kd(self, cfg: dict, **kw):
        self.kd_module = KDModule(cfg, self)

    def predict(self, x, *args, **kwargs):
        # The decomposed arm's residual adapter is part of the student and
        # must also run at inference/export, not only inside the KD loss.
        mod = getattr(self, "kd_module", None)
        if mod is None or mod.adapter is None:
            return super().predict(x, *args, **kwargs)
        head = self.model[-1]

        def _pre(m, args):
            feats = args[0]
            out = [f + a(f) for f, a in zip(feats, mod.adapter)]
            return (out,)

        pre = head.register_forward_pre_hook(_pre)
        try:
            return super().predict(x, *args, **kwargs)
        finally:
            pre.remove()

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        head = self.model[-1]
        captured = []

        def _hook(m, inp, out):
            captured.append(inp[0])

        hook = head.register_forward_hook(_hook)
        try:
            if preds is None:
                preds = self.forward(batch["img"])
        finally:
            hook.remove()
        det_loss, det_items = self.criterion(preds, batch)
        if getattr(self, "kd_module", None) is None or not getattr(self, "kd_enabled", True):
            return det_loss, det_items
        feats = preds["feats"] if isinstance(preds, dict) and "feats" in preds else (
            captured[0] if captured else None)
        if feats is None:
            return det_loss, det_items
        kd_loss, kd_items = self.kd_module(batch, preds, feats, det_items)
        items = dict(det_items)
        items.update(kd_items)
        return det_loss + kd_loss, items
