"""Train one CycleGAN for Method 3.

Two runs, IR only.  thermal_fog is first because thermal fog is exactly what
Method 2 failed on (gap -0.002 under Protocol 3B) and because every M3FD metric
is thermal-only.

    thermal_fog   M3FD IR   ->  DSERT Fog
    thermal_rain  M3FD IR   ->  DSERT Heavy_Rain + Light_Rain

Four properties this file is built around, three of them fixes:

RESUME IS THE POINT.  Two 512px generators take ~4 h each and a 24h slot also has
to fit every YOLO run, so the job will be cut off mid-run, possibly more than
once, and Slurm nodes fail outright.  A checkpoint is written every --ckpt-every
iterations -- not per epoch -- and it carries both generators, both
discriminators, both optimizers, both schedulers, the AMP scaler, the
iteration/epoch counters and all three RNG states.  Writes go to a temp file and
are renamed, so a kill during the write cannot leave a truncated checkpoint that
fails to load on resume.

A GENERATOR IS PINNED TO THE FRAME POOL IT WAS TRAINED ON.  This is the fix for
the contamination that survived the previous resplit.  `data.py` sources M3FD
frames from `data/split/IR/images/train`, and the two generators on disk had
been trained when that directory held the OLD random split -- which turned out to
overlap the new test set on 432 of 616 frames (70.1%).  `translate.py` then fed
those exact frames through a generator that had already seen them as source
samples, so Protocol 3A/3B would have measured a generator's memory of its own
training data.  Five documents asserted the opposite ("unaffected by the split
contamination"), and no `train.log` recorded which frames were used, so there
was no way to check after the fact.

So every run writes `SOURCE.json` -- the source directory, the sorted frame
stems, their sha256, and the split hash -- next to its weights, and both the
`DONE` fast path and the checkpoint resume path refuse to proceed when that
record does not match the pool currently on disk.  Reusing a generator across a
resplit now requires an explicit `--no-resume`, which is the point: it has to be
a decision rather than an accident.

MEASURE, DO NOT GUESS.  --calibrate times real iterations after a warmup and
reports the measured rate plus an estimate of the full run, warning if that
overruns the allocation.  It is advisory: it never resizes the schedule and
never stops training, because an estimate is not a reason to throw away a
half-trained generator.

DIAGNOSTICS NEVER GATE.  The sample grid includes G_B2A reconstruction of a DSERT
fog frame, which is the cheap check on whether this generator learned *weather*
or merely *sensor identity* -- DSERT's thermal imager is a different camera
from M3FD's, and if reconstruction does not return a DSERT fog frame to
something M3FD-clear-like, then any gap this generator later produces is
measuring the camera.  That is a serious finding and worth seeing early, but it
is a judgement about a downstream experiment that has not run yet, so it is
logged and never acted on.  No diagnostic raises, pauses, or exits.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import data as data_mod
import provenance
from models import ImagePool, define_D, define_G, GANLoss

# IR only: the RGB generators trained to completion but cannot move any M3FD
# number, since every eval YAML points at IR image paths.
RUNS = {
    "thermal_fog":  {"modality": "thermal", "target_conditions": ["Fog"]},
    "thermal_rain": {"modality": "thermal",
                     "target_conditions": ["Heavy_Rain", "Light_Rain"]},
}
RUN_ORDER = list(RUNS)

# hyperparameters pinned to the CycleGAN paper's own settings, so the result is
# attributable to the data rather than to a recipe we tuned on it
DEFAULTS = dict(crop=512, epochs=50, batch=1, lr=2e-4, betas=(0.5, 0.999),
                n_blocks=9, ngf=64, ndf=64, pool=50,
                lambda_A=10.0, lambda_B=10.0, lambda_identity=0.5)


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
# pretrained init
# ---------------------------------------------------------------------------

def load_pretrained(G, path, logf):
    """Load hgupta01 clear2foggy/clear2rainy weights into G_AB if we have them.

    Never fatal.  A missing or mismatched checkpoint costs us a fine-tuning
    starting point, not a run, and the whole point of this is to be able to
    drop the file in later without touching any code.

    The subtlety: hgupta01's define_G() defaults to norm='batch' while
    CycleGAN's convention is norm='instance'.  These differ in state_dict keys
    -- BatchNorm carries running_mean/running_var, InstanceNorm has no
    buffers at all -- so we sniff the checkpoint's own key set to decide which
    one it was trained with, rather than guessing.
    """
    if not path:
        return False
    p = Path(path)
    if not p.exists():
        log(f"[init] {p} not found -- training from scratch", logf)
        return False
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:
        log(f"[init] cannot read {p} ({type(exc).__name__}: {exc}) -- scratch", logf)
        return False

    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    sd = {re.sub(r"^module\.", "", k): v for k, v in sd.items()}

    keys = set(sd)
    has_bn = any(k.endswith("running_mean") for k in keys)
    cur_is_instance = not any(
        isinstance(m, nn.BatchNorm2d) for m in G.modules())
    if has_bn and cur_is_instance:
        log("[init] checkpoint has BatchNorm buffers but the model is built "
            "instance-norm; key sets cannot match -- rebuild with --norm batch",
            logf)
        return False

    tgt = G.state_dict()
    matched = {k: v for k, v in sd.items()
               if k in tgt and tgt[k].shape == v.shape}
    if not matched:
        log(f"[init] no shape-compatible keys in {p.name} -- scratch", logf)
        return False

    G.load_state_dict(matched, strict=False)
    log(f"[init] loaded {len(matched)}/{len(tgt)} tensors from {p.name}"
        f" ({'batch' if has_bn else 'instance'}-norm detected)", logf)
    if len(matched) < len(tgt):
        log(f"[init] {len(tgt) - len(matched)} tensors left at init", logf)
    return True


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def save_ckpt(state, ckpt_dir, keep=3):
    """Atomic, so a kill mid-write cannot corrupt the resume point."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    tmp = ckpt_dir / f".latest.pt.{os.getpid()}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, latest)          # atomic within a filesystem

    # rotating step snapshots, so a mid-run corruption is recoverable
    snaps = sorted(ckpt_dir.glob("ckpt_step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    snap = ckpt_dir / f"ckpt_step_{state['iters_done']:07d}.pt"
    t2 = ckpt_dir / f".snap.{os.getpid()}.tmp"
    torch.save(state, t2)
    os.replace(t2, snap)
    for old in snaps[-(keep + 1):]:
        old.unlink(missing_ok=True)
    return latest


def source_fingerprint(src_files, spec, args):
    """Identity of the frame pool a generator is being trained on.

    The fingerprint is over the sorted frame *names*, not file contents: the
    names are what determines which images the generator sees, and hashing 2 400
    PNGs on every launch would cost more than it is worth.
    """
    stems = sorted(p.stem for p in src_files)
    return {
        "run": args.run,
        "modality": spec["modality"],
        "source_dir": str(data_mod.M3FD_SOURCE[spec["modality"]]),
        "n_source": len(stems),
        "source_hash": hashlib.sha256(
            json.dumps(stems, sort_keys=True).encode()).hexdigest()[:16],
        "split_hash": provenance.split_hash_for_dataset(
            data_mod.M3FD_SPLIT / "m3fd_clear.yaml"),
        "target_conditions": spec["target_conditions"],
        "target_stride": args.target_stride,
        "max_source": args.max_source,
        "seed": args.seed,
        "crop": args.crop,
        "epochs": args.epochs,
    }


def check_source_unchanged(run_dir, fingerprint, logf):
    """Refuse to reuse a generator trained on a different frame pool.

    The previous DONE check was `if (run_dir / "DONE").exists(): return 0` with
    no record of what the run had seen, so a generator trained before a resplit
    was silently reused against the new split's test frames -- 70.1% overlap in
    the case that actually happened.
    """
    # Reusable artefacts, not just an existing directory: train.log is opened
    # before this point, so the directory is never empty by now and "does the dir
    # exist" cannot distinguish a fresh start from an aborted attempt.
    reusable = [p for p in (run_dir / "DONE", run_dir / "G_AB.pt",
                            run_dir / "ckpt") if p.exists()] \
        if run_dir.exists() else []
    if not reusable:
        # Nothing to reuse, so nothing to refuse. The runner archives an
        # unverifiable run and then calls this to train a replacement; treating
        # the missing directory the same as a directory whose provenance is
        # missing made a clean retrain impossible, because a directory that does
        # not exist trivially has no SOURCE.json.
        log(f"[source] {run_dir} holds no weights or DONE marker -- training "
            f"from scratch", logf)
        return None
    src_json = run_dir / "SOURCE.json"
    if not src_json.exists():
        log(f"[source] {src_json} missing -- this run predates frame-pool "
            f"provenance and cannot be verified. Re-run with --no-resume to "
            f"retrain against the current split.", logf)
        raise SystemExit(1)
    recorded = json.loads(src_json.read_text())
    if recorded.get("source_hash") == fingerprint["source_hash"]:
        return recorded
    raise SystemExit(
        f"REFUSING TO REUSE {fingerprint['run']}: it was trained on a different "
        f"frame pool.\n  recorded: n={recorded.get('n_source')} "
        f"hash={recorded.get('source_hash')} dir={recorded.get('source_dir')}\n"
        f"  on disk: n={fingerprint['n_source']} hash={fingerprint['source_hash']} "
        f"dir={fingerprint['source_dir']}\n"
        f"Translating the new test split through this generator would score its "
        f"memory of its own training frames. Retrain with --no-resume, or point "
        f"--run at a different output directory.")


def load_ckpt(ckpt_dir, G_A, G_B, D_A, D_B, optG, optD, schG, schD, scaler, logf):
    latest = ckpt_dir / "latest.pt"
    if not latest.exists():
        return 0, 0
    try:
        st = torch.load(latest, map_location="cpu", weights_only=False)
    except Exception as exc:
        log(f"[resume] {latest} unreadable ({type(exc).__name__}: {exc}) -- "
            f"starting this run clean", logf)
        return 0, 0
    try:
        G_A.load_state_dict(st["G_A2B"]); G_B.load_state_dict(st["G_B2A"])
        D_A.load_state_dict(st["D_A"]);   D_B.load_state_dict(st["D_B"])
        optG.load_state_dict(st["optG"]); optD.load_state_dict(st["optD"])
        schG.load_state_dict(st["schedG"]); schD.load_state_dict(st["schedD"])
        if scaler is not None and st.get("scaler"):
            scaler.load_state_dict(st["scaler"])
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["numpy_rng"])
        random.setstate(st["random_rng"])
        log(f"[resume] epoch {st['epoch']}, {st['iters_done']} iters already done "
            f"-- continuing", logf)
        return st["epoch"], st["iters_done"]
    except Exception as exc:
        log(f"[resume] incompatible checkpoint ({type(exc).__name__}: {exc}) -- "
            f"starting clean", logf)
        return 0, 0


# ---------------------------------------------------------------------------
# diagnostics -- logged, never acted on
# ---------------------------------------------------------------------------

def sample_grid(G_A, G_B, real_a, real_b, out_path):
    """Row 1: M3FD clear -> G_A2B (the translation we will actually use).
    Row 2: DSERT target real -> G_B2A reconstruction (the cycle-consistency
    read on whether weather or sensor identity was learned)."""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=RUN_ORDER)
    ap.add_argument("--out", type=Path, default=HERE / "runs")
    ap.add_argument("--data-root", type=Path, default=HERE / "data")
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
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--target-stride", type=int, default=1)
    ap.add_argument("--no-canonical", action="store_true",
                    help="skip resampling both domains to M3FD native geometry. "
                         "Only useful for A/B-ing the resolution fix; at --crop "
                         "512 the raw DSERT thermal (384x480) would be padded.")
    ap.add_argument("--max-source", type=int, default=None)
    ap.add_argument("--limit-iters", type=int, default=None,
                    help="stop after N iterations (smoke test)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-weights", type=Path, default=None)
    ap.add_argument("--calibrate", type=int, default=0, metavar="N",
                    help="time N real iterations, report the rate, then train")
    ap.add_argument("--no-resume", action="store_true",
                    help="retrain from scratch even if this run is DONE or has a "
                         "checkpoint. Also the only way to reuse a run directory "
                         "after the M3FD split changes.")
    # Off by default; see the long note in main() for why. For debugging only.
    ap.add_argument("--deterministic", action="store_true",
                    help="opt-in deterministic kernels (slower; for debugging)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--check-data", action="store_true",
                    help="verify both domains for this run are present and "
                         "non-empty, print the pairing, then exit. Lets the "
                         "runner decide skip-vs-run without launching a job "
                         "that would fail on the first batch.")
    ap.add_argument("--print-fingerprint", action="store_true",
                    help="print this run's source fingerprint as JSON and exit, "
                         "without training. The pipeline runner uses it to "
                         "decide whether an existing run directory is still "
                         "valid for the current split.")
    args = ap.parse_args()

    spec = RUNS[args.run]

    if args.print_fingerprint:
        try:
            src, _ = data_mod.build_pairs(spec, crop=args.crop,
                                          target_stride=args.target_stride,
                                          seed=args.seed, max_source=args.max_source)
        except FileNotFoundError as exc:
            print(f"UNAVAILABLE  {args.run}: {exc}")
            return 1
        print(json.dumps(source_fingerprint(src, spec, args), indent=2))
        return 0

    if args.check_data:
        try:
            src, tgt = data_mod.build_pairs(spec, crop=args.crop,
                                            target_stride=args.target_stride,
                                            seed=args.seed,
                                            max_source=args.max_source)
        except FileNotFoundError as exc:
            print(f"UNAVAILABLE  {args.run}: {exc}")
            return 1
        complete = data_mod.describe(spec, src, tgt, crop=args.crop)
        if not complete:
            print(f"UNAVAILABLE  {args.run}: target domain download incomplete")
            return 1
        print(f"OK          {args.run}")
        return 0

    run_dir = args.out / args.run
    run_dir.mkdir(parents=True, exist_ok=True)

    # The frame pool has to be known before anything else, because both the DONE
    # fast path and the checkpoint resume depend on whether it has changed.
    try:
        _src_probe, _ = data_mod.build_pairs(spec, crop=args.crop,
                                             target_stride=args.target_stride,
                                             seed=args.seed,
                                             max_source=args.max_source)
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}")
        return 1

    logf = open(run_dir / "train.log", "a")
    logf.write(f"\n{'=' * 70}\n=== {args.run} started {time.ctime()}\n{'=' * 70}\n")

    fingerprint = source_fingerprint(_src_probe, spec, args)
    if args.no_resume:
        log(f"[source] --no-resume: retraining against {fingerprint['n_source']} "
            f"frames (hash {fingerprint['source_hash']}, "
            f"split {fingerprint['split_hash']})", logf)
    else:
        check_source_unchanged(run_dir, fingerprint, logf)
    (run_dir / "SOURCE.json").write_text(json.dumps(fingerprint, indent=2))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        log("WARNING: no CUDA -- this will be extremely slow", logf)

    # Determinism is opt-in, and deliberately OFF for the 24h job.
    #
    # It is not a correctness requirement here. Model/optimizer/scheduler/
    # scaler/RNG state all round-trip exactly through the checkpoint, and the
    # data pipeline is deterministic per item (data.py seeds a private
    # random.Random from "seed:epoch:idx", and the loader uses shuffle=False),
    # so a resumed run takes the same batches as an uninterrupted one. The
    # only residual divergence is floating-point kernel scheduling -- which
    # makes two uninterrupted runs differ from each other just as much, so
    # there is no "reference" trajectory to diverge from in the first place.
    #
    # Turning it on for production is the wrong trade: deterministic cuDNN
    # algorithms are measurably slower, and the two thermal runs already consume
    # ~9h of a 24h slot that also has to fit five YOLO trainings.
    # use_deterministic_algorithms(strict=True) also *raises* on any op lacking a
    # deterministic kernel, which would kill the run at an arbitrary iteration
    # -- exactly the failure mode the resume logic exists to survive. Hence
    # warn_only, and hence opt-in.
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log("deterministic mode on (warn_only: unsupported ops will warn, "
            "not raise)", logf)
    else:
        # benchmark=True lets cuDNN autotune per shape. Every batch here is
        # the same shape, so the one-off autotune pays for itself immediately.
        torch.backends.cudnn.benchmark = True

    log(f"run={args.run}  device={dev}  crop={args.crop}  epochs={args.epochs}  "
        f"batch={args.batch}  norm={args.norm}  seed={args.seed}", logf)
    if dev == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}", logf)

    # ---- data ------------------------------------------------------------
    try:
        src, tgt = data_mod.build_pairs(spec, crop=args.crop,
                                        target_stride=args.target_stride,
                                        seed=args.seed,
                                        max_source=args.max_source)
    except FileNotFoundError as exc:
        log(f"FATAL: {exc}", logf)
        return 1
    data_mod.describe(spec, src, tgt, crop=args.crop)
    log("", logf)
    n_workers = args.workers
    if n_workers is None:
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
        n_workers = max(0, min(cpus - 2, 12))
    dataset = data_mod.UnpairedDataset(src, tgt, crop=args.crop, seed=args.seed,
                                       canonical=not args.no_canonical)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False,   # pairing is per-index
        num_workers=n_workers, pin_memory=True, drop_last=True,
        persistent_workers=n_workers > 0,
    )

    # ---- nets ------------------------------------------------------------
    G_A = define_G(3, 3, DEFAULTS["ngf"], args.n_blocks, args.norm).to(dev)
    G_B = define_G(3, 3, DEFAULTS["ngf"], args.n_blocks, args.norm).to(dev)
    D_A = define_D(3, DEFAULTS["ndf"], 3, args.norm).to(dev)
    D_B = define_D(3, DEFAULTS["ndf"], 3, args.norm).to(dev)
    # No G_A.eval()/G_B.eval() dance: InstanceNorm2d(track_running_stats=False)
    # has no buffers, so train/eval mode changes nothing for the generators.

    if args.init_weights:
        load_pretrained(G_A, args.init_weights, logf)

    # Size-preservation self-check. The resnet generator is fully convolutional
    # but only exactly size-preserving for input sizes that are multiples of 4
    # (two stride-2 downsamplings, then matched upsampling). A crop size that
    # is not a multiple of 4 makes the output 1-3px larger than the input, and
    # the cycle and identity L1 losses then compare mismatched tensors -- which
    # PyTorch resolves by broadcasting, silently training on garbage. Checked
    # here so it fails at iteration 0 rather than 10,000.
    with torch.no_grad():
        probe = torch.zeros(1, 3, args.crop, args.crop, device=dev)
        out = G_A(probe)
    if tuple(out.shape) != tuple(probe.shape):
        log(f"FATAL: generator is not size-preserving at --crop {args.crop}: "
            f"{tuple(probe.shape)} -> {tuple(out.shape)}. Use a multiple of 4.",
            logf)
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

    start_epoch, iters_done = 0, 0
    if (run_dir / "DONE").exists() and not args.no_resume:
        log(f"[resume] {run_dir / 'DONE'} exists -- this run already finished. "
            f"Nothing to do (pass --no-resume to train it again from scratch).",
            logf)
        logf.close()
        return 0
    if not args.no_resume:
        start_epoch, iters_done = load_ckpt(
            run_dir / "ckpt", G_A, G_B, D_A, D_B, optG, optD, schG, schD, scaler, logf)
    else:
        log("[resume] disabled by --no-resume", logf)

    # ---- optional calibration --------------------------------------------
    if args.calibrate:
        log(f"calibrating: timing {args.calibrate} iterations after a 10-iter "
            f"warmup ...", logf)
        it = _train_chunk(loader, G_A, G_B, D_A, D_B, optG, optD, schG, schD,
                          loss_fn, cyc_fn, pool_A, pool_B, scaler, dev, args,
                          iters_done, steps=args.calibrate + 10, warmup=10,
                          logf=logf, count_iters=False)
        rate = it["rate"]           # iterations per SECOND
        if rate <= 0:
            log("CALIBRATION: could not measure (rate 0) -- training anyway", logf)
        else:
            sec_per_iter = 1.0 / rate
            iters = len(src) * args.epochs
            per_run = sec_per_iter * iters
            log(f"CALIBRATION: {sec_per_iter * 1000:.0f} ms/iter "
                f"({rate:.1f} it/s)", logf)
            log(f"CALIBRATION: {args.epochs} epochs x {len(src)} iters "
                f"= {iters:,d} iterations = {per_run / 3600:.2f} h for "
                f"{args.run} at {args.crop}px", logf)
            (run_dir / "calibration.json").write_text(json.dumps(dict(
                run=args.run, crop=args.crop, batch=args.batch,
                ms_per_iter=round(sec_per_iter * 1000, 1),
                iters_per_sec=round(rate, 3),
                iters_per_epoch=len(src), epochs=args.epochs,
                est_hours=round(per_run / 3600, 2)), indent=2))
            if per_run > 20 * 3600:
                log("CALIBRATION: WARNING -- this run alone exceeds ~20h, so "
                    "all four will not fit one 24h slot. Expected; resume "
                    "covers the remainder. Not stopping.", logf)

    # ---- train -----------------------------------------------------------
    total_iters = len(src) * args.epochs
    log(f"training {total_iters:,d} iterations "
        f"({args.epochs} epochs x {len(src):,d})", logf)

    t_start = time.time()
    epochs_done = start_epoch
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        stats = _train_chunk(loader, G_A, G_B, D_A, D_B, optG, optD, schG, schD,
                             loss_fn, cyc_fn, pool_A, pool_B, scaler, dev, args,
                             iters_done, steps=None, warmup=0, logf=logf,
                             sample_every=args.sample_every,
                             run_dir=run_dir, ckpt_every=args.ckpt_every,
                             total=total_iters, t_start=t_start,
                             limit_iters=args.limit_iters, epoch=epoch)
        iters_done = stats["iters_done"]
        epochs_done = epoch + 1
        # schedulers step at the *end* of the epoch, after the optimizers have
        # moved. Stepping first (as the original CycleGAN repo does) trips a
        # PyTorch warning and silently drops the first LR value.
        schG.step(); schD.step()
        # end-of-epoch checkpoint: a walltime kill usually lands between
        # epochs as often as mid-epoch, and 100-iter granularity is only
        # guaranteed *within* one
        save_ckpt(_ckpt_state(G_A, G_B, D_A, D_B, optG, optD, schG, schD, scaler,
                              epochs_done, iters_done, args), run_dir / "ckpt")
        log(f"  [epoch] {epoch + 1}/{args.epochs} done, {iters_done:,d} iters, "
            f"{(time.time() - t_start) / 3600:.2f} h elapsed", logf)
        if args.limit_iters and iters_done >= args.limit_iters:
            log(f"stopping at --limit-iters {args.limit_iters}", logf)
            break

    # ---- final -----------------------------------------------------------
    # epochs_done, not args.epochs: this save also runs when the loop exits
    # early, and recording args.epochs there would make a resumed run believe
    # it had already finished its epoch budget.
    save_ckpt(_ckpt_state(G_A, G_B, D_A, D_B, optG, optD, schG, schD, scaler,
                          epochs_done, iters_done, args), run_dir / "ckpt")
    torch.save(G_A.state_dict(), run_dir / "G_AB.pt")
    torch.save(G_B.state_dict(), run_dir / "G_BA.pt")

    # Completion marker, so run_cyclegan.sh can skip a finished run on a
    # requeue instead of resuming a run that already reached its epoch count.
    # Only written when the full epoch budget was actually met -- otherwise a
    # run stopped early (--limit-iters, or a future early-exit path) would
    # leave a DONE marker that makes the runner skip a half-trained generator.
    complete = epochs_done >= args.epochs
    if complete:
        (run_dir / "DONE").write_text(json.dumps(dict(
            run=args.run, iters_done=iters_done, epochs=args.epochs,
            crop=args.crop, finished=time.ctime(),
            source_hash=fingerprint["source_hash"],
            n_source=fingerprint["n_source"],
            split_hash=fingerprint["split_hash"]), indent=2))
        log(f"DONE {args.run}: {iters_done} iters in "
            f"{(time.time() - t_start) / 3600:.2f} h", logf)
    else:
        (run_dir / "DONE").unlink(missing_ok=True)
        log(f"PARTIAL {args.run}: stopped at epoch {epochs_done}/{args.epochs}, "
            f"{iters_done} iters -- NOT marked done, resubmit to resume", logf)
    log(f"generators -> {run_dir / 'G_AB.pt'}  {run_dir / 'G_BA.pt'}", logf)
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
                 total=0, t_start=0.0, count_iters=True, limit_iters=None,
                 epoch=0):
    """One pass (or `steps` iterations) of the standard CycleGAN update.

    count_iters=False is the calibration path: it runs real forward/backward
    steps so the timing is honest, but does not advance iters_done, so
    measuring the cost of a run never eats into that run's budget.
    """
    G_A.train(); G_B.train(); D_A.train(); D_B.train()
    accum = dict(g_a=0.0, g_b=0.0, d_a=0.0, d_b=0.0, cyc=0.0, idt=0.0, n=0)
    done = 0
    measured = 0

    for real_a, real_b, _, _ in loader:
        if steps is not None and done >= steps:
            break
        # --limit-iters has to be honoured inside the epoch: at 512px a single
        # epoch is ~2940 iterations, so a limit checked only at the epoch
        # boundary would overshoot by orders of magnitude
        if limit_iters and iters_done >= limit_iters:
            break

        t_iter = time.time() if (steps is not None and done >= warmup) else None

        real_a = real_a.to(dev, non_blocking=True)
        real_b = real_b.to(dev, non_blocking=True)
        amp_on = scaler is not None and scaler.is_enabled()

        # ---- generators ----
        with torch.amp.autocast("cuda", enabled=amp_on):
            fake_b = G_A(real_a)                       # clear -> adverse
            rec_a = G_B(fake_b)                        # cycle back to clear
            fake_a = G_B(real_b)                       # adverse -> clear
            rec_b = G_A(fake_a)                        # cycle back to adverse
            idt_a = G_A(real_a)                        # adverse->clear on a clear image
            idt_b = G_B(real_b)                        # clear->adverse on an adverse image

            loss_cyc = cyc_fn(rec_a, real_a) + cyc_fn(rec_b, real_b)
            loss_idt = cyc_fn(idt_a, real_a) + cyc_fn(idt_b, real_b)
            # --lambda-a / --lambda-b become lambda_a / lambda_b on the
            # namespace; the two cases weight the cycle term from each
            # generator's own direction
            loss_g_a = (loss_fn(D_B(fake_b), True) + args.lambda_b * loss_cyc
                        + args.lambda_identity * loss_idt)
            loss_g_b = (loss_fn(D_A(fake_a), True) + args.lambda_a * loss_cyc
                        + args.lambda_identity * loss_idt)

        optG.zero_grad(set_to_none=True)
        if amp_on:
            scaler.scale(loss_g_a + loss_g_b).backward()
            scaler.step(optG)
            scaler.update()
        else:
            (loss_g_a + loss_g_b).backward()
            optG.step()

        # ---- discriminators ----
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
        # .detach().item(), not float(tensor): float() on a grad-tracking
        # tensor works but warns and is a footgun if the graph is ever kept
        accum["g_a"] += loss_g_a.detach().item()
        accum["g_b"] += loss_g_b.detach().item()
        accum["d_a"] += loss_d_a.detach().item()
        accum["d_b"] += loss_d_b.detach().item()
        accum["cyc"] += loss_cyc.detach().item()
        accum["idt"] += loss_idt.detach().item()
        accum["n"] += 1

        if steps is not None:
            continue

        # ---- periodic: diagnostics + checkpoint ----
        # Everything in this block is best-effort. A failure is logged and
        # training continues; nothing here can stop a run.
        if sample_every and iters_done % sample_every == 0 and run_dir is not None:
            try:
                p = run_dir / "samples" / f"{args.run}_{iters_done:07d}.png"
                p.parent.mkdir(exist_ok=True, parents=True)
                sample_grid(G_A, G_B, real_a, real_b, p)
                n = max(accum["n"], 1)
                log(f"  [diag] {iters_done:7,d}/{total:,d} "
                    f"g_a={accum['g_a']/n:.3f} g_b={accum['g_b']/n:.3f} "
                    f"d_a={accum['d_a']/n:.3f} d_b={accum['d_b']/n:.3f} "
                    f"cyc={accum['cyc']/n:.3f} idt={accum['idt']/n:.3f} "
                    f"-> {p.name}", logf)
            except Exception as exc:
                log(f"  [diag] grid failed ({type(exc).__name__}: {exc}) "
                    f"-- continuing", logf)
            accum = dict(g_a=0.0, g_b=0.0, d_a=0.0, d_b=0.0, cyc=0.0, idt=0.0, n=0)

        if ckpt_every and iters_done % ckpt_every == 0 and run_dir is not None:
            # The epoch recorded here must be the epoch actually in progress,
            # NOT args.epochs. Writing args.epochs meant every mid-epoch
            # checkpoint claimed the run was on the final epoch, so a walltime
            # kill at iteration 50,000 of 147,000 would resume into
            # range(50, 50) -- an empty loop that immediately "finished" the
            # run and threw away 34 hours of work without a single error.
            save_ckpt(_ckpt_state(G_A, G_B, D_A, D_B, optG, optD, schG, schD,
                                  scaler, epoch, iters_done, args),
                      run_dir / "ckpt")
            el = time.time() - t_start
            log(f"  [ckpt] {iters_done:7,d}/{total:,d} "
                f"({100 * iters_done / total:5.1f}%)  {el/3600:.2f} h elapsed", logf)

    rate = 0.0
    if measured > 0:
        n_measured = max(done - warmup, 1)
        rate = n_measured / measured          # iterations per second
        log(f"  measured {1.0 / rate * 1000:.0f} ms/iter over "
            f"{n_measured} iters", logf)
    return dict(iters_done=iters_done, rate=rate)


if __name__ == "__main__":
    sys.exit(main())
