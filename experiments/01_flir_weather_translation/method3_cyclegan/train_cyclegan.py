#!/usr/bin/env python3
"""Method 3 — CycleGAN unpaired translation for FLIR ADAS v2.

Trains CycleGAN to translate clear thermal IR -> foggy/rainy/snowy using
DSERT-RoLL (Fog, Heavy_Rain) and Heavy_Snow as target domains.

Adapted from the M3FD Method 3 implementation in m3fd_experiments/, with
FLIR ADAS v2-specific data paths and provenance tracking.

Usage:
    python method3_cyclegan/train_cyclegan.py --config fog --epochs 50
    python method3_cyclegan/train_cyclegan.py --config rain --epochs 50
    python method3_cyclegan/train_cyclegan.py --config snow --epochs 50

Key properties:
  RESUME IS THE POINT: checkpoints every 100 iters with full state (G, D, opts,
    schedulers, scaler, RNG states). Atomic writes via temp+rename.
  SOURCE.json: records the frame pool hash so a resumed/requeued run cannot
    silently reuse a generator trained on a different split.
  DONE marker: only written when the full epoch budget was met; a partial run
    resumes rather than being skipped.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import cv2
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # release root

# Reuse the proven CycleGAN architecture from M3FD
sys.path.insert(0, str(REPO / "experiments" / "03_m3fd_weather" / "method3_cyclegan"))
from models import ResnetGenerator, NLayerDiscriminator, GANLoss
from models import get_norm_layer, ImagePool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# All paths are relative to the worktree root; override via --source-root etc.
M3FD_M3_DIR = REPO / "experiments" / "03_m3fd_weather" / "method3_cyclegan"
FLIR_DATA = REPO / "data" / "flir_adas_v2" / "FLIR_ADAS_v2"

# DSERT-RoLL target domains are in the main repo checkout (gitignored data).
# The worktree is under .kilo/worktrees/<name>; main repo is 3 levels up.
MAIN_REPO = REPO
DSERT_ROOT = REPO / "data" / "dsert_roll"
# Heavy_Snow is downloaded into the worktree's own data dir (already contains Heavy_Snow/)
SNOW_ROOT = REPO / "data" / "dsert_roll"

# CycleGAN hyperparameters (from the paper / M3FD reference)
DEFAULTS = dict(
    crop=256, epochs=50, batch=1, lr=2e-4, betas=(0.5, 0.999),
    n_blocks=9, ngf=64, ndf=64, pool=50,
    lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5,
    ckpt_every=5000, sample_every=500,
)

CONFIGS = {
    "fog": {
        "target_conditions": ["Fog"],
        "target_modalities": ["rectified_THERMAL_L"],
        "target_root": DSERT_ROOT,
    },
    "rain": {
        "target_conditions": ["Heavy_Rain"],
        "target_modalities": ["rectified_THERMAL_L"],
        "target_root": DSERT_ROOT,
    },
    "snow": {
        "target_conditions": ["Heavy_Snow"],
        "target_modalities": ["rectified_THERMAL_L"],
        "target_root": SNOW_ROOT,
    },
}


def log(msg, f=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if f:
        f.write(line + "\n")
        f.flush()


def linear_lr(n_epochs):
    """Original CycleGAN schedule: hold for n_epochs, then linear to zero."""
    return lambda e: max(0.0, 1.0 - e / float(n_epochs))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def list_flir_clear(flir_root, modality="thermal", split="train"):
    """List FLIR ADAS v2 clear thermal (or RGB) images from the train split."""
    sub = "images_thermal_train" if modality == "thermal" else "images_rgb_train"
    d = flir_root / sub / "data"
    if not d.is_dir():
        raise FileNotFoundError(f"FLIR source dir missing: {d}")
    files = sorted(d.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"no *.jpg under {d}")
    return files


def list_dsert_target(condition, dsert_root, snow_root=None, modality="thermal"):
    """List DSERT-RoLL target images for one weather condition.

    DSERT structure:
      DSERT-RoLL/<Condition>/<sequence>/<rectified_THERMAL_L>/*.png
      DSERT-RoLL/<Condition>/<sequence>/<rectified_crop_RGB_L>/*.jpg
    Heavy_Snow is downloaded into the worktree at a slightly different root.
    """
    # Heavy_Snow is under the snow_root which already contains Heavy_Snow/
    if condition == "Heavy_Snow" and snow_root:
        root = snow_root
    else:
        root = dsert_root

    cond_dir = root / condition
    if not cond_dir.is_dir():
        raise FileNotFoundError(
            f"DSERT condition dir missing: {cond_dir}. "
            f"Download from HuggingFace first.")

    # Map modality to subdirectory name
    mod_dir = "rectified_THERMAL_L" if modality == "thermal" else "rectified_crop_RGB_L"

    files = []
    for seq_dir in sorted(cond_dir.iterdir()):
        if not seq_dir.is_dir():
            continue
        mod_path = seq_dir / mod_dir
        if not mod_path.is_dir():
            continue
        files.extend(sorted(mod_path.glob("*.png")))
        files.extend(sorted(mod_path.glob("*.jpg")))

    if not files:
        raise FileNotFoundError(
            f"no target images for {condition} under {cond_dir}/{mod_dir}")
    return files


def read_image(path, canonical_size=None):
    """Read image as RGB float32 tensor in [-1, 1].

    Handles grayscale -> 3-channel replication for thermal images.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise OSError(f"cv2 could not decode {path}")

    if raw.dtype.itemsize > 1:
        raw = (raw.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)

    if raw.ndim == 2:
        img = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    if canonical_size is not None:
        h, w = img.shape[:2]
        cw, ch = canonical_size
        if (h, w) != (ch, cw):
            img = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_LINEAR)

    arr = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return torch.from_numpy(arr * 2.0 - 1.0)


class FlirUnpairedDataset(torch.utils.data.Dataset):
    """Unpaired clear source / adverse target dataset for FLIR ADAS v2.

    Source: FLIR ADAS v2 thermal train split (clear).
    Target: DSERT-RoLL thermal for the specified conditions.

    Both domains are resampled to a canonical size before random cropping, so
    a crop covers the same field of view in both domains.
    """

    def __init__(self, src_files, tgt_files, crop=256, hflip=True,
                 length=None, seed=0, canonical=(512, 512)):
        assert len(src_files) > 0, "empty source domain"
        assert len(tgt_files) > 0, "empty target domain"
        self.src_files = list(src_files)
        self.tgt_files = list(tgt_files)
        self.crop = crop
        self.hflip = hflip
        self.seed = seed
        self.epoch = 0
        self.canonical = canonical  # (h, w)
        self.length = length or len(self.src_files)

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _paths(self, idx):
        ra = random.Random(f"{self.seed}:{self.epoch}:{idx}:a")
        rb = random.Random(f"{self.seed}:{self.epoch}:{idx}:b")
        return (self.src_files[ra.randrange(len(self.src_files))],
                self.tgt_files[rb.randrange(len(self.tgt_files))])

    def __getitem__(self, idx):
        ps, pt = self._paths(idx)
        rng = random.Random(f"{self.seed}:{self.epoch}:{idx}:c")

        img_s = read_image(ps, self.canonical)
        img_t = read_image(pt, self.canonical)

        # Random crop
        _, h, w = img_s.shape
        ph, pw = max(0, self.crop - h), max(0, self.crop - w)
        if ph or pw:
            img_s = torch.nn.functional.pad(img_s, (0, pw, 0, ph), mode='reflect')
            img_t = torch.nn.functional.pad(img_t, (0, pw, 0, ph), mode='reflect')
            h, w = img_s.shape[1], img_s.shape[2]

        y = rng.randint(0, max(0, h - self.crop))
        x = rng.randint(0, max(0, w - self.crop))
        img_s = img_s[:, y:y + self.crop, x:x + self.crop]
        img_t = img_t[:, y:y + self.crop, x:x + self.crop]

        if self.hflip and rng.random() < 0.5:
            img_s = img_s.flip(2)
            img_t = img_t.flip(2)

        return img_s, img_t, ps.name, pt.name


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_ckpt(state, ckpt_dir, keep=3):
    """Atomic checkpoint write; rotating snapshots for crash recovery."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    tmp = ckpt_dir / f".latest.pt.{os.getpid()}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, latest)

    snaps = sorted(ckpt_dir.glob("ckpt_step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    snap = ckpt_dir / f"ckpt_step_{state['iters_done']:07d}.pt"
    t2 = ckpt_dir / f".snap.{os.getpid()}.tmp"
    torch.save(state, t2)
    os.replace(t2, snap)
    for old in snaps[-(keep + 1):]:
        old.unlink(missing_ok=True)
    return latest


def load_ckpt(ckpt_dir, G_A, G_B, D_A, D_B, optG, optD, schG, schD, scaler, logf):
    latest = ckpt_dir / "latest.pt"
    if not latest.exists():
        return 0, 0
    try:
        st = torch.load(latest, map_location="cpu", weights_only=False)
    except Exception as exc:
        log(f"[resume] {latest} unreadable ({type(exc).__name__}: {exc}) -- starting clean", logf)
        return 0, 0
    try:
        G_A.load_state_dict(st["G_A2B"])
        G_B.load_state_dict(st["G_B2A"])
        D_A.load_state_dict(st["D_A"])
        D_B.load_state_dict(st["D_B"])
        optG.load_state_dict(st["optG"])
        optD.load_state_dict(st["optD"])
        schG.load_state_dict(st["schedG"])
        schD.load_state_dict(st["schedD"])
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["numpy_rng"])
        random.setstate(st["random_rng"])
        log(f"[resume] epoch {st['epoch']}, {st['iters_done']} iters already done -- continuing", logf)
        return st["epoch"], st["iters_done"]
    except Exception as exc:
        log(f"[resume] incompatible checkpoint ({type(exc).__name__}: {exc}) -- starting clean", logf)
        return 0, 0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def source_fingerprint(src_files, tgt_files, spec, args):
    """Identity of the frame pool this generator was trained on."""
    src_stems = sorted(p.stem for p in src_files)
    tgt_stems = sorted(p.stem for p in tgt_files)
    return {
        "run": args.config,
        "modality": "thermal",
        "source_dir": str(src_files[0].parent),
        "n_source": len(src_stems),
        "source_hash": hashlib.sha256(
            json.dumps(src_stems, sort_keys=True).encode()).hexdigest()[:16],
        "n_target": len(tgt_stems),
        "target_hash": hashlib.sha256(
            json.dumps(tgt_stems, sort_keys=True).encode()).hexdigest()[:16],
        "target_conditions": spec["target_conditions"],
        "target_root": str(spec["target_root"]),
        "seed": args.seed,
        "crop": args.crop,
        "epochs": args.epochs,
    }


def check_source_unchanged(run_dir, fingerprint, logf):
    """Refuse to reuse a generator trained on a different frame pool."""
    reusable = [p for p in (run_dir / "DONE", run_dir / "G_AB.pt",
                            run_dir / "ckpt") if p.exists()] \
        if run_dir.exists() else []
    if not reusable:
        log(f"[source] {run_dir} holds no weights or DONE marker -- training from scratch", logf)
        return None
    src_json = run_dir / "SOURCE.json"
    if not src_json.exists():
        log(f"[source] {src_json} missing -- cannot verify frame pool. "
            f"Re-run with --no-resume to retrain.", logf)
        raise SystemExit(1)
    recorded = json.loads(src_json.read_text())
    if recorded.get("source_hash") == fingerprint["source_hash"]:
        return recorded
    raise SystemExit(
        f"REFUSING TO REUSE {fingerprint['run']}: trained on a different frame pool.\n"
        f"  recorded: n={recorded.get('n_source')} hash={recorded.get('source_hash')}\n"
        f"  on disk:  n={fingerprint['n_source']} hash={fingerprint['source_hash']}\n"
        f"Retrain with --no-resume, or use a different output directory.")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def sample_grid(G_A, G_B, real_a, real_b, out_path):
    import torchvision.utils as vutils
    with torch.no_grad():
        fake_b = G_A(real_a)
        recon_a = G_B(fake_b)
    grid = torch.cat([real_a.cpu(), fake_b.cpu().float(),
                      real_b.cpu(), recon_a.cpu().float()], 0)
    try:
        vutils.save_image(grid * 0.5 + 0.5, out_path, nrow=min(len(real_a), 4))
    except Exception:
        pass
    return out_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Method 3: CycleGAN for FLIR ADAS v2")
    ap.add_argument("--config", required=True, choices=["fog", "rain", "snow"],
                    help="weather condition to synthesize")
    ap.add_argument("--out", type=Path, default=HERE / "runs",
                    help="output directory for runs")
    ap.add_argument("--source-root", type=Path, default=FLIR_DATA,
                    help="FLIR ADAS v2 root")
    ap.add_argument("--dsert-root", type=Path, default=DSERT_ROOT,
                    help="DSERT-RoLL root (containing Fog/Heavy_Rain/Light_Rain)")
    ap.add_argument("--snow-root", type=Path, default=SNOW_ROOT,
                    help="Heavy_Snow download root")
    ap.add_argument("--crop", type=int, default=DEFAULTS["crop"])
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--n-blocks", type=int, default=DEFAULTS["n_blocks"])
    ap.add_argument("--norm", default="instance", choices=("instance", "batch"))
    ap.add_argument("--pool", type=int, default=DEFAULTS["pool"])
    ap.add_argument("--lambda-a", type=float, default=DEFAULTS["lambda_A"])
    ap.add_argument("--lambda-b", type=float, default=DEFAULTS["lambda_B"])
    ap.add_argument("--lambda-identity", type=float,
                    default=DEFAULTS["lambda_identity"])
    ap.add_argument("--ckpt-every", type=int, default=DEFAULTS["ckpt_every"])
    ap.add_argument("--sample-every", type=int, default=DEFAULTS["sample_every"])
    ap.add_argument("--max-source", type=int, default=None)
    ap.add_argument("--limit-iters", type=int, default=None,
                    help="stop after N iterations (smoke test)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-weights", type=Path, default=None)
    ap.add_argument("--calibrate", type=int, default=0, metavar="N",
                    help="time N real iterations, report rate, then train")
    ap.add_argument("--no-resume", action="store_true",
                    help="retrain from scratch even if DONE or checkpoint exists")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--check-data", action="store_true",
                    help="verify both domains are present and non-empty, then exit")
    ap.add_argument("--print-fingerprint", action="store_true",
                    help="print source fingerprint as JSON and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="check data availability, then exit without training")
    args = ap.parse_args()

    spec = CONFIGS[args.config].copy()
    # Allow overriding the target root
    if args.config == "snow":
        spec["target_root"] = args.snow_root
    else:
        spec["target_root"] = args.dsert_root

    # ---- Provenance / data check ----
    flir_root = args.source_root          # already points to .../FLIR_ADAS_v2/FLIR_ADAS_v2
    dsert_root = args.dsert_root
    snow_root = args.snow_root

    try:
        src_files = list_flir_clear(flir_root, "thermal", "train")
    except FileNotFoundError as exc:
        print(f"UNAVAILABLE  {args.config}: {exc}")
        return 1

    try:
        tgt_files = []
        for cond in spec["target_conditions"]:
            tgt_files.extend(list_dsert_target(cond, dsert_root, snow_root, "thermal"))
    except FileNotFoundError as exc:
        print(f"UNAVAILABLE  {args.config}: {exc}")
        return 1

    fingerprint = source_fingerprint(src_files, tgt_files, spec, args)

    if args.print_fingerprint:
        print(json.dumps(fingerprint, indent=2))
        return 0

    if args.check_data or args.dry_run:
        print(f"OK          {args.config}")
        print(f"  source: {len(src_files)} files -> {src_files[0].parent}")
        print(f"  target: {len(tgt_files)} files (conditions={spec['target_conditions']})")
        return 0

    # ---- Setup ----
    run_dir = args.out / args.config
    run_dir.mkdir(parents=True, exist_ok=True)

    logf = open(run_dir / "train.log", "a")
    logf.write(f"\n{'=' * 70}\n=== {args.config} started {time.ctime()}\n{'=' * 70}\n")

    if args.no_resume:
        log(f"[source] --no-resume: retraining against {fingerprint['n_source']} "
            f"source frames, {fingerprint['n_target']} target frames", logf)
    else:
        check_source_unchanged(run_dir, fingerprint, logf)
    (run_dir / "SOURCE.json").write_text(json.dumps(fingerprint, indent=2))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        log("WARNING: no CUDA -- this will be extremely slow", logf)

    torch.backends.cudnn.benchmark = True

    log(f"run={args.config}  device={dev}  crop={args.crop}  epochs={args.epochs}  "
        f"batch={args.batch}  norm={args.norm}  seed={args.seed}", logf)
    if dev == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}", logf)
    log(f"  source: {len(src_files)} files", logf)
    log(f"  target: {len(tgt_files)} files ({spec['target_conditions']})", logf)
    log("", logf)

    # ---- Data ----
    if args.max_source:
        src_files = src_files[:args.max_source]

    # Thin target domain by stride to reduce redundancy
    tgt_files_sub = tgt_files[::max(1, len(tgt_files) // len(src_files))] if len(tgt_files) > len(src_files) * 2 else tgt_files

    dataset = FlirUnpairedDataset(
        src_files, tgt_files_sub,
        crop=args.crop, seed=args.seed,
        canonical=(512, 512))

    n_workers = args.workers
    if n_workers is None:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
        n_workers = max(0, min(cpus - 2, 12))

    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False,
        num_workers=n_workers, pin_memory=True, drop_last=True,
        persistent_workers=n_workers > 0,
    )

    # ---- Networks ----
    norm_layer = get_norm_layer(args.norm)
    G_A = ResnetGenerator(3, 3, DEFAULTS["ngf"], norm_layer, n_blocks=args.n_blocks).to(dev)
    G_B = ResnetGenerator(3, 3, DEFAULTS["ngf"], norm_layer, n_blocks=args.n_blocks).to(dev)
    D_A = NLayerDiscriminator(3, DEFAULTS["ndf"], n_layers=3, norm_layer=norm_layer).to(dev)
    D_B = NLayerDiscriminator(3, DEFAULTS["ndf"], n_layers=3, norm_layer=norm_layer).to(dev)

    if args.init_weights:
        try:
            blob = torch.load(args.init_weights, map_location="cpu", weights_only=False)
            sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
            sd = {re.sub(r"^module\.", "", k): v for k, v in sd.items()}
            matched = {k: v for k, v in sd.items()
                       if k in G_A.state_dict() and G_A.state_dict()[k].shape == v.shape}
            if matched:
                G_A.load_state_dict(matched, strict=False)
                log(f"[init] loaded {len(matched)}/{len(G_A.state_dict())} tensors from {args.init_weights.name}", logf)
        except Exception as exc:
            log(f"[init] {args.init_weights} failed ({exc}) -- training from scratch", logf)

    # Size-preservation check: crop must be multiple of 4 for resnet generator
    with torch.no_grad():
        probe = torch.zeros(1, 3, args.crop, args.crop, device=dev)
        out = G_A(probe)
    if tuple(out.shape) != tuple(probe.shape):
        log(f"FATAL: generator is not size-preserving at --crop {args.crop}: "
            f"{tuple(probe.shape)} -> {tuple(out.shape)}. Use a multiple of 4.", logf)
        return 1
    log(f"generator size-preservation check passed at {args.crop}px", logf)

    optG = torch.optim.Adam(list(G_A.parameters()) + list(G_B.parameters()),
                            lr=args.lr, betas=DEFAULTS["betas"])
    optD = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                            lr=args.lr, betas=DEFAULTS["betas"])
    schG = torch.optim.lr_scheduler.LambdaLR(optG, lr_lambda=linear_lr(args.epochs))
    schD = torch.optim.lr_scheduler.LambdaLR(optD, lr_lambda=linear_lr(args.epochs))
    scaler = None if args.no_amp else torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    loss_fn = GANLoss("lsgan").to(dev)
    cyc_fn = nn.L1Loss()
    pool_A, pool_B = ImagePool(args.pool), ImagePool(args.pool)

    # ---- Resume ----
    if (run_dir / "DONE").exists() and not args.no_resume:
        log(f"[resume] DONE marker exists -- nothing to do", logf)
        logf.close()
        return 0
    if not args.no_resume:
        start_epoch, iters_done = load_ckpt(
            run_dir / "ckpt", G_A, G_B, D_A, D_B, optG, optD, schG, schD, scaler, logf)
    else:
        start_epoch, iters_done = 0, 0

    # ---- Calibration ----
    if args.calibrate:
        log(f"calibrating: timing {args.calibrate} iterations after warmup...", logf)
        _train_chunk(loader, G_A, G_B, D_A, D_B, optG, optD, schG, schD,
                     loss_fn, cyc_fn, pool_A, pool_B, scaler, dev, args,
                     iters_done, steps=args.calibrate + 10, warmup=10,
                     logf=logf, count_iters=False)

    # ---- Train ----
    total_iters = len(src_files) * args.epochs
    log(f"training {total_iters:,d} iterations ({args.epochs} epochs x {len(src_files):,d})", logf)

    t_start = time.time()
    epochs_done = start_epoch
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        iters_done = _train_chunk(
            loader, G_A, G_B, D_A, D_B, optG, optD, schG, schD,
            loss_fn, cyc_fn, pool_A, pool_B, scaler, dev, args,
            iters_done, steps=None, warmup=0, logf=logf,
            sample_every=args.sample_every, run_dir=run_dir,
            ckpt_every=args.ckpt_every, total=total_iters, t_start=t_start,
            limit_iters=args.limit_iters, epoch=epoch)
        epochs_done = epoch + 1
        schG.step()
        schD.step()
        save_ckpt(_ckpt_state(G_A, G_B, D_A, D_B, optG, optD, schG, schD, scaler,
                              epochs_done, iters_done, args), run_dir / "ckpt")
        log(f"  [epoch] {epoch + 1}/{args.epochs} done, {iters_done:,d} iters, "
            f"{(time.time() - t_start) / 3600:.2f} h elapsed", logf)
        if args.limit_iters and iters_done >= args.limit_iters:
            log(f"stopping at --limit-iters {args.limit_iters}", logf)
            break

    # ---- Final save ----
    save_ckpt(_ckpt_state(G_A, G_B, D_A, D_B, optG, optD, schG, schD, scaler,
                          epochs_done, iters_done, args), run_dir / "ckpt")
    torch.save(G_A.state_dict(), run_dir / "G_AB.pt")
    torch.save(G_B.state_dict(), run_dir / "G_BA.pt")

    complete = epochs_done >= args.epochs
    if complete:
        (run_dir / "DONE").write_text(json.dumps(dict(
            config=args.config, iters_done=iters_done, epochs=args.epochs,
            crop=args.crop, finished=time.ctime(),
            source_hash=fingerprint["source_hash"],
            target_hash=fingerprint["target_hash"],
            n_source=fingerprint["n_source"],
            n_target=fingerprint["n_target"]), indent=2))
        log(f"DONE {args.config}: {iters_done} iters in {(time.time() - t_start) / 3600:.2f} h", logf)
    else:
        (run_dir / "DONE").unlink(missing_ok=True)
        log(f"PARTIAL {args.config}: stopped at epoch {epochs_done}/{args.epochs}, "
            f"{iters_done} iters -- NOT marked done, resubmit to resume", logf)
    log(f"generator -> {run_dir / 'G_AB.pt'}", logf)
    logf.close()
    return 0


def _ckpt_state(G_A, G_B, D_A, D_B, optG, optD, schG, schD, scaler, epoch, iters, args):
    return {
        "G_A2B": G_A.state_dict(), "G_B2A": G_B.state_dict(),
        "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
        "optG": optG.state_dict(), "optD": optD.state_dict(),
        "schedG": schG.state_dict(), "schedD": schD.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch, "iters_done": iters,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "random_rng": random.getstate(),
        "args": vars(args),
    }


def _train_chunk(loader, G_A, G_B, D_A, D_B, optG, optD, schG, schD,
                 loss_fn, cyc_fn, pool_A, pool_B, scaler, dev, args, iters_done,
                 steps, warmup, logf, sample_every=0, run_dir=None, ckpt_every=0,
                 total=0, t_start=0.0, count_iters=True, limit_iters=None, epoch=0):
    """One pass (or `steps` iterations) of the standard CycleGAN update."""
    G_A.train(); G_B.train(); D_A.train(); D_B.train()
    accum = dict(g_ab=0.0, g_ba=0.0, d_a=0.0, d_b=0.0, cyc=0.0, idt=0.0, n=0)
    done = 0
    measured = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for real_a, real_b, _, _ in pbar:
        if steps is not None and done >= steps:
            break
        if limit_iters and iters_done >= limit_iters:
            break

        t_iter = time.time() if (steps is not None and done >= warmup) else None

        real_a = real_a.to(dev, non_blocking=True)
        real_b = real_b.to(dev, non_blocking=True)
        amp_on = scaler is not None and scaler.is_enabled()

        with torch.amp.autocast("cuda", enabled=amp_on):
            fake_b = G_A(real_a)
            rec_a = G_B(fake_b)
            fake_a = G_B(real_b)
            rec_b = G_A(fake_a)
            idt_a = G_A(real_a)
            idt_b = G_B(real_b)

            loss_cyc = cyc_fn(rec_a, real_a) + cyc_fn(rec_b, real_b)
            loss_idt = cyc_fn(idt_a, real_a) + cyc_fn(idt_b, real_b)
            loss_g_ab = (loss_fn(D_B(fake_b), True) + args.lambda_b * loss_cyc
                         + args.lambda_identity * loss_idt)
            loss_g_ba = (loss_fn(D_A(fake_a), True) + args.lambda_a * loss_cyc
                         + args.lambda_identity * loss_idt)

        optG.zero_grad(set_to_none=True)
        if amp_on:
            scaler.scale(loss_g_ab + loss_g_ba).backward()
            scaler.step(optG)
            scaler.update()
        else:
            (loss_g_ab + loss_g_ba).backward()
            optG.step()

        with torch.amp.autocast("cuda", enabled=amp_on):
            fake_b_d = pool_B.query(fake_b.detach())
            fake_a_d = pool_A.query(fake_a.detach())
            loss_d_b = 0.5 * (loss_fn(D_B(real_b), True) + loss_fn(D_B(fake_b_d), False))
            loss_d_a = 0.5 * (loss_fn(D_A(real_a), True) + loss_fn(D_A(fake_a_d), False))

        optD.zero_grad(set_to_none=True)
        if amp_on:
            scaler.scale(loss_d_a + loss_d_b).backward()
            scaler.step(optD)
            scaler.update()
        else:
            (loss_d_a + loss_d_b).backward()
            optD.step()

        if t_iter is not None:
            measured += time.time() - t_iter

        if count_iters:
            iters_done += 1
        done += 1
        accum["g_ab"] += loss_g_ab.detach().item()
        accum["g_ba"] += loss_g_ba.detach().item()
        accum["d_a"] += loss_d_a.detach().item()
        accum["d_b"] += loss_d_b.detach().item()
        accum["cyc"] += loss_cyc.detach().item()
        accum["idt"] += loss_idt.detach().item()
        accum["n"] += 1

        if steps is not None:
            continue

        if sample_every and iters_done % sample_every == 0 and run_dir is not None:
            try:
                p = run_dir / "samples" / f"{args.config}_{iters_done:07d}.png"
                p.parent.mkdir(exist_ok=True, parents=True)
                sample_grid(G_A, G_B, real_a, real_b, p)
                n = max(accum["n"], 1)
                log(f"  [diag] {iters_done:7,d}/{total:,d} "
                    f"g_ab={accum['g_ab']/n:.3f} g_ba={accum['g_ba']/n:.3f} "
                    f"d_a={accum['d_a']/n:.3f} d_b={accum['d_b']/n:.3f} "
                    f"cyc={accum['cyc']/n:.3f} idt={accum['idt']/n:.3f} "
                    f"-> {p.name}", logf)
            except Exception as exc:
                log(f"  [diag] grid failed ({type(exc).__name__}: {exc}) -- continuing", logf)
            accum = dict(g_ab=0.0, g_ba=0.0, d_a=0.0, d_b=0.0, cyc=0.0, idt=0.0, n=0)

        if ckpt_every and iters_done % ckpt_every == 0 and run_dir is not None:
            save_ckpt(_ckpt_state(G_A, G_B, D_A, D_B, optG, optD, schG, schD,
                                  scaler, epoch, iters_done, args),
                      run_dir / "ckpt")
            el = time.time() - t_start
            log(f"  [ckpt] {iters_done:7,d}/{total:,d} "
                f"({100 * iters_done / max(total, 1):5.1f}%)  {el/3600:.2f} h elapsed", logf)

        if steps is not None:
            continue

    rate = 0.0
    if measured > 0:
        n_measured = max(done - warmup, 1)
        rate = n_measured / measured
    return iters_done


if __name__ == "__main__":
    sys.exit(main())
