"""Unpaired image pairs for CycleGAN.

The two domains are deliberately *not* the same size and are shuffled
independently, which is the whole point of unpaired translation: a M3FD clear
frame is never supposed to correspond to any particular DSERT fog frame.

Thermal only. The project decided against RGB, and M3FD evaluation is IR-only:
every eval YAML points at IR image paths, so the `rgb_fog` / `rgb_rain`
generators that used to live here trained to completion without being able to
move a single number on the books. They are gone, along with the `rgb` key of
M3FD_SOURCE, so nothing can ask for a source domain that does not exist.

Two things in here are load-bearing rather than stylistic:

1. M3FD is globbed with an explicit *.png filter.  split/IR/images/<split>
   holds one .npy sibling per .png -- they are Ultralytics' own `--cache disk`
   decoded-image cache, written next to the images by every training run, and
   they are byte-identical to the PNGs (`npy (768,1024,3) uint8 ==
   cv2.imread('00000.png')` returns True).  A bare glob("*") silently loads
   twice as many files, half of which cannot be decoded.  Same failure class as
   the YOLO silent-label bug.

2. There is no colour jitter, and never should be.  Thermal images are stored
   pseudo-coloured (blue=cold, red=warm); hue/saturation jitter destroys the
   palette's meaning, which is why hsv_h=0.0 and hsv_s=0.0 are pinned in
   train_m3fd.py.  Spatial augmentation only.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DSERT_ROOT = REPO / "data" / "dsert_roll"
DSERT_MANIFEST = HERE / "dsert_files.json"

M3FD_SPLIT = REPO / "data" / "m3fd" / "split"
M3FD_SOURCE = {
    "thermal": M3FD_SPLIT / "IR" / "images" / "train",
}

# Both domains are brought to M3FD's native 1024x768 before cropping.
#
# This is not cosmetic. inspect_dsert.py found the target domain is *smaller
# than the crop*: DSERT thermal is 384x480 while --crop is 512, so the
# original augment() would have reflection-padded every target image and
# trained the discriminator on padding borders. A generator and a
# discriminator that can both key on a padding edge is a generator that
# produces padding edges.
#
# It also removes the train/inference scale mismatch. Because the target is
# resampled to the source's native geometry, a 512 crop covers the same
# fraction of the scene in both domains, so the only thing the discriminator
# can separate them on is weather rather than field of view or sharpness.
CANON = (768, 1024)   # (h, w) -- M3FD native

# Frames in the two domains. Deliberately not equal.
CACHE_SIZE = 2048


def list_m3fd_source(modality):
    """Clear M3FD images for one modality. *.png only -- see module docstring."""
    d = M3FD_SOURCE[modality]
    if not d.is_dir():
        raise FileNotFoundError(f"M3FD source dir missing: {d}")
    files = sorted(d.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"no *.png under {d} (did the label-path bug bite?)")
    return files


def list_dsert_target(modality, conditions, stride=1):
    """Adverse DSERT-RoLL images, read from the download manifest.

    The manifest is used in preference to globbing because it carries the
    condition, sequence and frame index per file, which is what makes a
    temporal stride possible.  DSERT fog is 32 sequences of consecutive
    driving frames and consecutive frames are near-duplicates -- the same
    redundancy that forced sequence-disjoint splits for UNIRI-TID -- so
    `stride` lets you thin the target domain without losing condition
    diversity.
    """
    if not DSERT_MANIFEST.exists():
        raise FileNotFoundError(
            f"{DSERT_MANIFEST} missing. Run download_dsert.py first.")
    blob = json.loads(DSERT_MANIFEST.read_text())
    want = set(conditions)
    picked = []
    for f in blob["files"]:
        if f["modality"] != modality or f["condition"] not in want:
            continue
        picked.append((f["condition"], f["sequence"], f["frame"], DSERT_ROOT / f["path"]))

    if stride > 1:
        # thin within each sequence so every sequence keeps some frames, rather
        # than dropping sequences entirely and losing location diversity
        by_seq = {}
        for c, s, fr, p in picked:
            by_seq.setdefault((c, s), []).append((fr, p))
        picked = []
        for (c, s), items in sorted(by_seq.items()):
            items.sort()
            picked += [(c, s, fr, p) for fr, p in items[::stride]]

    files = [p for _, _, _, p in picked if p.exists()]
    if not files:
        raise FileNotFoundError(
            f"no on-disk DSERT {modality} files for {sorted(want)} under "
            f"{DSERT_ROOT}. Run download_dsert.py.")
    return files


def to_canonical(img, canon=CANON):
    """Centre-crop to the canonical aspect ratio, then resample to `canon`.

    Crop-then-resize rather than resize-then-crop so that the full canonical
    field of view is filled: resizing first and cropping after would throw
    away resolution we just paid to create.
    """
    ch, cw = canon
    h, w = img.shape[:2]
    if (h, w) == (ch, cw):
        return img

    target_ar = cw / ch
    ar = w / h
    # Guard each crop with the correct axis. An earlier version used max(ch,..)
    # / max(cw,..) -- the wrong axis -- which for DSERT thermal (384x480, too
    # tall) computed nh = max(1024, 360) = 1024 on a 384-row image, giving
    # y0 = -320 and a negative slice. The resize then happened to restore the
    # right shape, so the bug was invisible except as a silent loss of the top
    # 64 rows of every target image.
    if ar > target_ar:                      # too wide: trim the sides
        nw = min(w, max(1, int(round(h * target_ar))))
        x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]
    else:                                   # too tall: trim top and bottom
        nh = min(h, max(1, int(round(w / target_ar))))
        y0 = (h - nh) // 2
        img = img[y0:y0 + nh, :]

    ih, iw = img.shape[:2]
    shrinking = (cw <= iw and ch <= ih)
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(img, (cw, ch), interpolation=interp)


def _imread(path, canonical=True):
    """Read BGR -> RGB uint8. Handles the extension/format mismatch class of
    bug by letting OpenCV sniff content rather than trusting the suffix.

    Both thermal sets were measured at 8-bit, 3-channel, unsaturated
    (Fog: values confined to ~[73,165], std 22; M3FD IR: ~[3,228], std 48),
    so the global /255 mapping below is lossless and no per-image rescaling
    is wanted. The width difference is the signal -- fog lifts the black
    floor and halves contrast -- so a per-image contrast stretch here would
    erase the exact effect the generator is being trained to produce.

    Reads UNCHANGED once rather than IMREAD_COLOR, so the bit depth is
    actually visible: IMREAD_COLOR silently rescales >8-bit input and would
    discard dynamic range with nothing to indicate it had done so. Doing the
    channel fixup here costs one cvtColor that IMREAD_COLOR was doing anyway,
    and keeps this to a single decode -- reading twice to check the depth
    would double per-image I/O on the hot path.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"cv2 could not decode {path}")
    if raw.dtype.itemsize > 1:
        raise ValueError(
            f"{path} is {raw.dtype.itemsize * 8}-bit; reducing it needs "
            f"per-image normalisation to 8-bit (bit depth only -- do NOT "
            f"contrast-stretch, see above).")
    if raw.ndim == 2:
        # greyscale: replicate to 3 so CycleGAN's 3-channel generators and
        # discriminators see the expected layout.
        img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    else:
        img = raw[:, :, :3]          # drops alpha if the file carries any
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return to_canonical(img) if canonical else img


def augment(img, crop, hflip, rng):
    """Spatial-only augmentation, then to tensor in [-1, 1].

    Pads rather than skipping when the image is smaller than the crop: all
    four domains here are 1024x768 or larger so this never fires in practice,
    but a silent skip would quietly shrink the target domain instead of
    reporting it.
    """
    h, w = img.shape[:2]
    ph, pw = max(0, crop - h), max(0, crop - w)
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
        h, w = img.shape[:2]

    # random.randint is INCLUSIVE of its upper bound, so the bound is
    # h - crop and not h - crop + 1. With the +1 this can return 257 for a
    # 768-row image, and img[257:769] silently yields 511 rows instead of 512
    # -- a 1-in-257 crash per axis that would land at a random iteration of a
    # 24h run rather than during the smoke test.
    y = rng.randint(0, h - crop) if crop < h else 0
    x = rng.randint(0, w - crop) if crop < w else 0
    img = img[y:y + crop, x:x + crop]

    if hflip and rng.random() < 0.5:
        img = img[:, ::-1]

    arr = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0)


class UnpairedDataset(Dataset):
    """Yields one image from each domain, independently shuffled.

    The shorter domain is the one that cycles, so a step count is meaningful in
    terms of the *source* (M3FD clear) domain, which is the one the generator
    will be applied to and therefore the one whose diversity bounds the result.
    """

    def __init__(self, a_files, b_files, crop=512, hflip=True, length=None,
                 seed=0, canonical=True):
        assert len(a_files) > 0 and len(b_files) > 0, "empty domain"
        self.a_files = list(a_files)
        self.b_files = list(b_files)
        self.crop = crop
        self.hflip = hflip
        self.seed = seed
        self.epoch = 0
        self.canonical = canonical
        self.length = length or len(self.a_files)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        """Both domains reshuffle every epoch.

        Without this the source is consumed as a_files[0..N-1] on every pass,
        i.e. the same sorted order forever -- only the random crops vary. The
        epoch is folded into the seed string rather than into a RNG object, so
        that a resumed run continues on the schedule it was already on instead
        of silently reshuffling to a different one.
        """
        self.epoch = epoch

    def _paths(self, idx):
        # random.Random seeded from a *string*: string seeding hashes with
        # sha512 internally and is stable across processes, whereas
        # tuple.__hash__() embeds PYTHONHASHSEED randomisation for the str
        # member and would hand a resumed run a different pairing schedule
        # than it had before the restart.
        ra = random.Random(f"{self.seed}:{self.epoch}:{idx}:a")
        rb = random.Random(f"{self.seed}:{self.epoch}:{idx}:b")
        return (self.a_files[ra.randrange(len(self.a_files))],
                self.b_files[rb.randrange(len(self.b_files))])

    def __getitem__(self, idx):
        pa, pb = self._paths(idx)
        rng = random.Random(f"{self.seed}:{self.epoch}:{idx}:c")
        a = augment(_imread(pa, self.canonical), self.crop, self.hflip, rng)
        b = augment(_imread(pb, self.canonical), self.crop, self.hflip, rng)
        # Belt and braces. A wrong-sized crop makes the cycle and identity L1
        # losses broadcast against each other, which PyTorch allows with only
        # a warning and which silently trains on garbage. Catching it here
        # names the two files, which is a far better error than a stack trace
        # 10,000 iterations into a 24h job.
        want = (3, self.crop, self.crop)
        if tuple(a.shape) != want or tuple(b.shape) != want:
            raise ValueError(
                f"crop {self.crop} produced a{tuple(a.shape)} for {pa.name} and "
                f"b{tuple(b.shape)} for {pb.name}; expected {want}. "
                f"Check augment()'s randint bounds and to_canonical().")
        return (a, b, pa.name, pb.name)


def build_pairs(run, crop=512, target_stride=1, seed=0, hflip=True,
                max_source=None):
    """Build the (source, target) file lists for one of the four runs."""
    modality, condition = run["modality"], run["target_conditions"]
    src = list_m3fd_source(modality)
    if max_source:
        src = src[:max_source]
    tgt = list_dsert_target(modality, condition, stride=target_stride)
    return src, tgt


def dsert_manifest_count(modality, conditions):
    """How many files the manifest says exist for this domain. Used to tell a
    complete download from a partial one."""
    if not DSERT_MANIFEST.exists():
        return 0
    blob = json.loads(DSERT_MANIFEST.read_text())
    want = set(conditions)
    return sum(1 for f in blob["files"]
               if f["modality"] == modality and f["condition"] in want)


def describe(run, src, tgt, crop=512, min_fraction=0.9):
    print(f"  source ({run['modality']:7s}) : {len(src):6,d}  {src[0].parent}")
    print(f"  target ({'+'.join(run['target_conditions']):12s}): {len(tgt):6,d}  "
          f"{tgt[0].parents[2].name}/{tgt[0].parents[1].name}/{tgt[0].parent.name}")
    print(f"  both domains resampled to {CANON[1]}x{CANON[0]} before a "
          f"{crop}-crop")

    expected = dsert_manifest_count(run["modality"], run["target_conditions"])
    have = len(tgt)
    if expected and have < expected * min_fraction:
        print(f"  INCOMPLETE target: {have:,d}/{expected:,d} frames on disk "
              f"({100 * have / expected:.0f}%, need {min_fraction:.0%}).")
        print("  Training on a partial target domain gives a generator that "
              "has seen only a")
        print("  slice of the weather conditions. Finish download_dsert.py "
              "first.")
        return False
    print(f"  target complete: {have:,d}/{expected:,d} frames")
    return True
