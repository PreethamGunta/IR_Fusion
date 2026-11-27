"""Ultralytics DetectionTrainer with feature distillation from the frozen E30 teacher.

Contract of this trainer
- Training batches come from the V2 **paired** loader (6-channel [IR|RGB]).
- The student receives only the first three channels (true IR input).
- The frozen teacher receives all six and returns per-level (IR, Delta, fused).
- Validation uses the V2 **IR-only** loader, so val AP is IR-only AP.
- Projectors and the teacher live on the trainer; Ultralytics checkpoints store
  only the EMA of the student (``"model": None``), so ``best.pt`` is a pure
  stock IR-only detector.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER

from .losses import build_kd
from .teacher_hooks import E30Teacher, probe_channels, student_anchor_indices


class KDDetectionTrainer(DetectionTrainer):
    """DetectionTrainer subclass; configure via class attributes.

    The launcher creates a dynamic subclass with these attributes set, mirroring
    the pattern used by ``09_graph_fusion`` (Ultralytics rejects unknown CLI
    keys, so they must not travel through ``args``).
    """

    kd_mode: str = "none"                 # none | direct | decomposed
    teacher_ckpt: str | None = None
    rgb_yaml: str | None = None           # RGB yaml (required for kd_mode != none)
    lambda_pres: float = 0.25
    lambda_comp: float = 0.50
    lambda_direct: float = 1.0
    kd_warmup_epochs: int = 3
    kd_alpha: float = 1.0
    strict_sync: bool = True
    verbose_kd: bool = True

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.teacher: E30Teacher | None = None
        self.kd_module: nn.Module | None = None
        self._student_anchors: list[int] | None = None
        self._student_channels: list[int] | None = None
        self._s_hooks: list = []
        self._s_feats: dict[int, torch.Tensor] = {}
        self._paired_img: torch.Tensor | None = None
        self._stock_loss = None
        self._rgb_data = None
        super().__init__(cfg, overrides, _callbacks)

    # ------------------------------------------------------------------ setup

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build a **3-channel** student regardless of the data yaml's channels.

        The paired yaml declares ``channels: 6``; letting the stock builder read
        it would give the student a 6-channel stem and break IR-only inference.
        """
        from ultralytics.nn.tasks import DetectionModel

        model = DetectionModel(
            cfg or self.args.model, nc=self.data["nc"], ch=3, verbose=verbose
        )
        if weights is not None:
            model.load(weights)
        self._ensure_kd_modules(model)
        return model

    def _ensure_kd_modules(self, model) -> None:
        if self.kd_mode == "none":
            return
        if self.teacher is None:
            if not self.teacher_ckpt:
                raise ValueError("teacher_ckpt is required for kd_mode != 'none'")
            self.teacher = E30Teacher(self.teacher_ckpt, device="cpu")
        if self._student_anchors is None:
            self._student_anchors = student_anchor_indices(model)
            self._student_channels = probe_channels(model, self._student_anchors)
        if self.kd_module is None:
            self.kd_module = build_kd(
                self.kd_mode, self._student_channels, self.teacher.channels,
                alpha=self.kd_alpha,
            )
            n = sum(p.numel() for p in self.kd_module.parameters())
            LOGGER.info(
                f"[irkd] student anchors {self._student_anchors} "
                f"C={self._student_channels}; teacher C={self.teacher.channels}; "
                f"projectors {n/1e3:.1f}k params"
            )

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9,
                        decay=1e-5, iterations=1e5):
        opt = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        if self.kd_module is not None:
            params = [p for p in self.kd_module.parameters() if p.requires_grad]
            opt.add_param_group({
                "params": params, "lr": lr, "momentum": momentum,
                "weight_decay": decay,
            })
            LOGGER.info(f"[irkd] optimizer gained a projector group "
                        f"({sum(p.numel() for p in params)} params)")
        return opt

    def _setup_train(self):
        super()._setup_train()
        self._activate_kd()

    def _activate_kd(self) -> None:
        if self.kd_mode == "none" or self.kd_module is None:
            return
        self.teacher.model.to(self.device)
        self.kd_module.to(self.device)
        self.kd_module.train()

        side = Path(self.save_dir) / "kd_proj.pt"
        if self.args.resume and side.exists():
            self.kd_module.load_state_dict(
                torch.load(side, map_location="cpu", weights_only=True)
            )
            LOGGER.info(f"[irkd] resumed projectors from {side}")

        self._register_student_hooks()
        self._stock_loss = self.model.loss
        self.model.loss = self._kd_loss
        LOGGER.info(
            f"[irkd] ACTIVE mode={self.kd_mode} teacher={self.teacher_ckpt} "
            f"λ_pres={self.lambda_pres} λ_comp={self.lambda_comp} "
            f"λ_dir={self.lambda_direct} warmup={self.kd_warmup_epochs} α={self.kd_alpha}"
        )

    def _register_student_hooks(self) -> None:
        self._s_hooks = [
            self.model.model[i].register_forward_hook(self._make_hook(i))
            for i in self._student_anchors
        ]

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            self._s_feats[idx] = output

        return hook

    # ------------------------------------------------------------- data / loss

    def build_dataset(self, img_path, mode="train", batch=None):
        """Paired training data; IR-only validation data."""
        from gfdata.unified_v2 import make_dataset_v2
        from ultralytics.data.utils import check_det_dataset

        stride = int(max(self.model.stride).item())
        bs = batch or self.args.batch
        if self.kd_mode != "none" and mode == "train":
            if self._rgb_data is None:
                self._rgb_data = check_det_dataset(self.rgb_yaml)
            return make_dataset_v2(
                self.args, img_path, bs, self.data, mode=mode, rect=False,
                stride=stride, modality="pair", rgb_data=self._rgb_data,
                rgb_img_path=self._rgb_data[mode], strict_sync=self.strict_sync,
            )
        return make_dataset_v2(
            self.args, img_path, bs, self.data, mode=mode, rect=False,
            stride=stride, modality="ir",
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        img = batch["img"]
        if img.ndim == 4 and img.shape[1] == 6:
            # keep the paired tensor for the teacher, hand IR to the student
            self._paired_img = img
            batch["img"] = img[:, :3].contiguous()
        return batch

    def _kd_loss(self, batch, preds=None):
        loss, items = self._stock_loss(batch, preds)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("[irkd] non-finite detection loss; aborting")
        kd, kd_items = self._compute_kd()
        if kd is None:
            return loss, items
        return loss + kd.to(loss.dtype), {**items, **kd_items}

    def _compute_kd(self):
        if self._paired_img is None or self._paired_img.shape[1] != 6:
            raise RuntimeError(
                "[irkd] KD needs a paired 6-channel training batch; "
                "preprocess_batch did not see one"
            )
        feats = [self._s_feats[i] for i in self._student_anchors]

        # Teacher and projectors stay fp32 even under the trainer's AMP autocast.
        with torch.autocast(device_type=self.device.type, enabled=False):
            t_ir, t_delta, t_fused = self.teacher(self._paired_img.float())
            losses, items = self.kd_module(feats, t_ir, t_delta, t_fused)
            if self.kd_mode == "direct":
                total = self.lambda_direct * losses["kd_direct"]
            else:
                total = (self.lambda_pres * losses["kd_pres"]
                         + self.lambda_comp * losses["kd_comp"])
            warm = min(1.0, (float(self.epoch) + 1.0) / max(1, self.kd_warmup_epochs))
            total = total * warm

        if not torch.isfinite(total).all():
            raise FloatingPointError("[irkd] non-finite KD loss; aborting")
        # keep tensor values (validator does zeros_like on loss_items); CSV
        # logging converts them with float() downstream
        logged = {k: v.detach() for k, v in items.items()}
        logged["kd_w"] = torch.tensor(warm, device=total.device, dtype=total.dtype)
        return total, logged

    # ------------------------------------------------------------- maintenance

    def validate(self):
        # a transient non-finite forward can poison BN running stats (09 lesson)
        from gfyolo.gf_trainer import sanitize_bn_buffers

        sanitize_bn_buffers(self.model)
        return super().validate()

    def save_model(self):
        super().save_model()
        if self.kd_module is not None:
            tmp = Path(self.save_dir) / "kd_proj.pt.tmp"
            torch.save(self.kd_module.state_dict(), tmp)
            tmp.replace(Path(self.save_dir) / "kd_proj.pt")
