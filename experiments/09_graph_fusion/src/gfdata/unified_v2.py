"""Unified paired (IR+RGB) data pipeline, V2.

Why V2 exists
-------------
``gfdata.unified`` (V1) re-implemented mosaic, affine, flip, HSV/value jitter
and erasing **by hand**, on tiles that had already been letterboxed and
normalised by Ultralytics' ``Format`` transform. The IR-only arm trained
through V1 reached 0.1829 val mAP50-95 where the native-recipe reference E00
reaches 0.6306, so the data path is a prime suspect (postmortem §2E-3).

V2 takes the opposite approach: it contains **no augmentation code at all**.

* ``modality="ir"``
    Returns the *stock* ``YOLODataset`` object that Ultralytics'
    ``build_yolo_dataset`` would have built for E00 -- unwrapped. E00's IR
    loading is therefore reproduced by **identity**, not by imitation.

* ``modality="rgb"`` / ``modality="pair"``
    The same stock transform chain is evaluated *twice per sample*: once for
    the IR image and once for the RGB image, with the Python/NumPy RNG state
    restored in between. Every geometric draw (mosaic tile indices, mosaic
    centre, perspective matrix, flip decisions) is therefore **replayed**
    rather than re-randomised, so the pair stays pixel-aligned without any
    operation being re-implemented.

* Supervision always comes from the IR (thermal) labels. The RGB dataset
  shares the IR label objects, which makes the replay check meaningful: its
  transformed boxes must be numerically identical to the IR ones, and that is
  asserted on every training sample (``strict_sync``).

Channel contract: ``pair`` -> ``[IR (0:3) | RGB (3:6)]``. Values stay ``uint8``
0-255 in CHW; the trainer's ``preprocess_batch`` converts to float and divides
by 255, exactly as it does for E00.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset

MODALITIES = ("ir", "rgb", "pair")

#: Numerical tolerance for the replay check. Both branches execute the same
#: code on the same labels, so the expected difference is exactly zero; the
#: tolerance only absorbs last-bit differences from parallel reductions.
SYNC_ATOL = 1e-6


class PairSyncError(RuntimeError):
    """The RGB replay diverged from the IR sample (pair would be misaligned)."""


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------

def rng_state() -> tuple:
    """Snapshot both RNGs used by the Ultralytics augmentation chain."""
    return random.getstate(), np.random.get_state()


def rng_restore(state: tuple) -> None:
    """Restore a snapshot taken with :func:`rng_state`."""
    py_state, np_state = state
    random.setstate(py_state)
    np.random.set_state(np_state)


# ---------------------------------------------------------------------------
# pairing helpers
# ---------------------------------------------------------------------------

def _names(ds) -> list[str]:
    return [Path(f).name for f in ds.im_files]


def validate_pairing(ir_files, rgb_files) -> None:
    """Assert the two trees describe the same frames in the same order.

    DroneVehicle v2 keeps identical file names in the IR and RGB trees
    (``00001.jpg`` in both), so pairing is a strict name equality check -- no
    fuzzy matching, no silent reordering.
    """
    a, b = list(ir_files), list(rgb_files)
    if len(a) != len(b):
        raise PairSyncError(
            f"pair length mismatch: IR={len(a)} RGB={len(b)}")
    for i, (x, y) in enumerate(zip(a, b)):
        if Path(x).name != Path(y).name:
            raise PairSyncError(
                f"pair order mismatch at {i}: {Path(x).name!r} vs "
                f"{Path(y).name!r} -- the RGB tree is not aligned to the IR "
                f"tree; refusing to build a mislabelled pair dataset")


# ---------------------------------------------------------------------------
# dataset wrapper
# ---------------------------------------------------------------------------

class UnifiedV2Dataset:
    """Pair-capable dataset built from one or two *stock* ``YOLODataset``.

    The wrapper delegates every attribute it does not define to the IR
    dataset, so Ultralytics sees a normal dataset (``rect``, ``batch_shapes``,
    ``im_files``, ``labels``, ``mosaic``, ``collate_fn``, ...).
    """

    #: ``build_dataloader`` looks this up with ``getattr(ds, "collate_fn")``.
    collate_fn = staticmethod(YOLODataset.collate_fn)

    def __init__(self, ir_ds: YOLODataset, rgb_ds: YOLODataset | None = None,
                 modality: str = "ir", strict_sync: bool = True) -> None:
        if modality not in MODALITIES:
            raise ValueError(f"bad modality {modality!r}; expected {MODALITIES}")
        if modality != "ir" and rgb_ds is None:
            raise ValueError(f"modality={modality!r} requires an RGB dataset")
        object.__setattr__(self, "_ir", ir_ds)
        object.__setattr__(self, "_rgb", rgb_ds)
        object.__setattr__(self, "_modality", modality)
        object.__setattr__(self, "_strict", bool(strict_sync))
        object.__setattr__(self, "_sync_checked", 0)
        if rgb_ds is not None:
            validate_pairing(ir_ds.im_files, rgb_ds.im_files)
            # Supervision is thermal GT for every arm: make the RGB dataset
            # carry the *same* labels as the IR one. This is what turns the
            # replay check into a real proof -- if the geometric replay ever
            # diverged, the transformed boxes would differ and we would fail
            # loudly instead of training on misaligned pairs.
            rgb_ds.labels = ir_ds.labels

    # -- delegation ------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal attribute lookup fails. Guard the private
        # names so an unset ``_ir`` (e.g. during unpickling) cannot recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_ir"), name)

    def __getstate__(self) -> dict:
        """Pickle only the owned state (DataLoader workers copy the dataset)."""
        return {
            "ir": object.__getattribute__(self, "_ir"),
            "rgb": object.__getattribute__(self, "_rgb"),
            "modality": object.__getattribute__(self, "_modality"),
            "strict": object.__getattribute__(self, "_strict"),
            "checked": object.__getattribute__(self, "_sync_checked"),
        }

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "_ir", state["ir"])
        object.__setattr__(self, "_rgb", state["rgb"])
        object.__setattr__(self, "_modality", state["modality"])
        object.__setattr__(self, "_strict", state["strict"])
        object.__setattr__(self, "_sync_checked", state.get("checked", 0))

    def __len__(self) -> int:
        return len(self._ir)

    @property
    def modality(self) -> str:
        return self._modality

    @property
    def mosaic(self) -> bool:
        return bool(getattr(self._ir, "mosaic", False))

    @mosaic.setter
    def mosaic(self, value: bool) -> None:
        # The trainer sets ``dataset.mosaic = False`` directly before calling
        # ``close_mosaic``. Keep both halves in step.
        for ds in self._children():
            if hasattr(ds, "mosaic"):
                ds.mosaic = value

    @property
    def epoch(self) -> int:
        return int(getattr(self._ir, "epoch", -1))

    @epoch.setter
    def epoch(self, value: int) -> None:
        for ds in self._children():
            ds.epoch = value

    def _children(self) -> tuple:
        rgb = object.__getattribute__(self, "_rgb")
        return (object.__getattribute__(self, "_ir"),) if rgb is None else (
            object.__getattribute__(self, "_ir"), rgb)

    # -- mosaic schedule -------------------------------------------------

    def close_mosaic(self, hyp=None) -> None:
        """Rebuild both transform chains with mosaic disabled.

        Both halves receive the *same* ``hyp`` object so the rebuilt chains are
        identical; the RGB chain is rebuilt second, after ``hyp.mosaic`` has
        already been zeroed, which is harmless because the rebuild is pure.
        """
        if hyp is None:
            raise ValueError("close_mosaic requires the training hyp")
        self._ir.close_mosaic(hyp)
        rgb = object.__getattribute__(self, "_rgb")
        if rgb is not None:
            rgb.close_mosaic(hyp)

    # -- loading ---------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        if self._modality == "ir":
            # E00, by identity: the stock chain, untouched.
            return self._ir[index]

        state = rng_state()
        ir = self._ir[index]
        rng_restore(state)          # replay, do not re-randomise
        rgb = object.__getattribute__(self, "_rgb")[index]

        if self._strict:
            self._assert_sync(ir, rgb, index)

        out = dict(ir)              # labels/metadata always from IR
        if self._modality == "rgb":
            out["img"] = rgb["img"]
            out["im_file"] = rgb["im_file"]
        else:
            out["img"] = torch.cat([ir["img"], rgb["img"]], 0)
        return out

    # -- replay verification ---------------------------------------------

    def _assert_sync(self, ir: dict, rgb: dict, index: int) -> None:
        """Fail loudly if the RGB replay did not follow the IR transform."""
        problems = []
        if tuple(ir["img"].shape) != tuple(rgb["img"].shape):
            problems.append(f"img shape {tuple(ir['img'].shape)} vs "
                            f"{tuple(rgb['img'].shape)}")
        if tuple(ir["bboxes"].shape) != tuple(rgb["bboxes"].shape):
            problems.append(f"bbox count {tuple(ir['bboxes'].shape)} vs "
                            f"{tuple(rgb['bboxes'].shape)}")
        elif ir["bboxes"].numel() and not torch.allclose(
                ir["bboxes"].float(), rgb["bboxes"].float(),
                atol=SYNC_ATOL, rtol=0):
            problems.append("bbox geometry diverged (max |d| = "
                            f"{float((ir['bboxes'] - rgb['bboxes']).abs().max()):.3g})")
        if tuple(ir["cls"].shape) != tuple(rgb["cls"].shape) or (
                ir["cls"].numel() and not torch.equal(ir["cls"], rgb["cls"])):
            problems.append("cls diverged")
        if ir.get("ratio_pad") != rgb.get("ratio_pad"):
            problems.append(f"ratio_pad {ir.get('ratio_pad')} vs "
                            f"{rgb.get('ratio_pad')}")
        if problems:
            raise PairSyncError(
                f"IR/RGB desynchronisation at index {index}: "
                + "; ".join(problems))
        object.__setattr__(self, "_sync_checked",
                           int(object.__getattribute__(self, "_sync_checked")) + 1)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def make_dataset_v2(cfg, img_path: str, batch: int, data: dict,
                    mode: str = "train", rect: bool = False, stride: int = 32,
                    modality: str = "ir", rgb_data: dict | None = None,
                    rgb_img_path: str | None = None, fraction: float | None = None,
                    strict_sync: bool = True):
    """Build a dataset for any arm.

    The IR dataset is produced by Ultralytics' own ``build_yolo_dataset``, so
    every argument (``pad``, ``rect``, ``fraction``, ``prefix``, ``hyp``) is
    resolved exactly as it is for a native run. ``modality="ir"`` returns that
    object directly; the other modalities wrap it together with an RGB dataset
    built by the same factory.

    Args:
        cfg: the trainer's resolved args (``self.args``).
        img_path: absolute IR image directory for this split.
        batch: batch size (used for rect sizing by the factory).
        data: resolved IR dataset dict from ``check_det_dataset``.
        mode: ``train`` / ``val`` / ``test``.
        rect: rectangular-inference flag from the trainer.
        stride: model stride.
        modality: ``ir``, ``rgb`` or ``pair``.
        rgb_data: resolved RGB dataset dict (required unless ``modality="ir"``).
        rgb_img_path: absolute RGB image directory for this split.
        fraction: dataset fraction override (``None`` = Ultralytics default).
        strict_sync: assert the IR/RGB replay on every sample.
    """
    from ultralytics.data.build import build_yolo_dataset

    ir_ds = build_yolo_dataset(
        cfg=cfg, img_path=img_path, batch=batch, data=data, mode=mode,
        rect=rect, stride=stride, fraction=fraction,
    )
    if modality == "ir":
        return ir_ds

    if rgb_data is None or rgb_img_path is None:
        raise ValueError(f"modality={modality!r} requires rgb_data/rgb_img_path")

    rgb_ds = build_yolo_dataset(
        cfg=cfg, img_path=rgb_img_path, batch=batch, data=rgb_data, mode=mode,
        rect=rect, stride=stride, fraction=fraction,
    )
    return UnifiedV2Dataset(ir_ds, rgb_ds, modality=modality,
                            strict_sync=strict_sync)
