# Method 3 — CycleGAN weather synthesis

Learned unpaired translation, clear → adverse, for M3FD. This exists because
Method 2 failed.

Method 2 (Koschmieder + monocular depth) passed Protocol 3A with a gap of
0.154 mAP@50-95 but **collapsed to −0.002 under Protocol 3B**
(`../method2_depth/report.md:106`). The reason is structural: a Koschmieder
filter is a deterministic, invertible function of the depth map, so a detector
shown clear and fog during training learns to invert it. No amount of
exposure exposes it, because there is nothing left to expose.

A CycleGAN hypothesises the opposite. Its output depends on unpaired target
images and on adversarial pressure, so the transform is not invertible from a
depth map and carries texture variation a model has to actually learn. **The
bar is a 3B gap ≥ 0.10, not a 3A gap** — Method 2 proved that a large 3A gap
says nothing.

## Datasets

Two, and only two.

| | source (clear) | target (adverse) |
|---|---|---|
| dataset | M3FD | DSERT-RoLL (CVPR 2026) |
| thermal | `../data/split/IR/images/train` (2,940) | `Fog/` or `Heavy_Rain`+`Light_Rain/`, `rectified_THERMAL_L` |
| rgb | `../data/split/RGB/images/train` (2,940) | same conditions, `rectified_crop_RGB_L` |
| size | 1024×768 8-bit | thermal 384×480 8-bit · rgb 1200×1920 8-bit |

The target domain is real weather, never synthetic. Method 2's fog as a
training target for a generator would be circular, and UNIRI-TID's fog blocks
are one site and one camera.

## The four runs

`thermal_fog` → `thermal_rain` → `rgb_fog` → `rgb_rain`, in that order.
`thermal_fog` is first because thermal fog is exactly what Method 2 failed on,
and because every M3FD metric is IR-only — the RGB generators cannot move any
number currently on the books. If the job is cut off, the furthest-progressed
run is the one that mattered.

## Running it

```bash
PY=$HOME/.conda/envs/gateir/bin/python
M3=m3fd_experiments/method3_cyclegan

# 1. download the target domain -- ONCE, FROM THE LOGIN NODE
$PY $M3/download_dsert.py --dry-run  # what it would fetch, downloads nothing
$PY $M3/download_dsert.py            # ~3.7 GB, resumable
# export HF_TOKEN=$(cat HF_TOKEN) first if you have one -- the anonymous rate
# limit is ~2 files/s, authenticated with 4 workers is far quicker. Do NOT use
# --workers above 4; the Hub returns 429s and backs off.
# Run it detached (setsid) or a killed foreground command takes the download
# with it.

# 2. verify the format before spending any GPU time
$PY $M3/inspect_dsert.py

# 3. smoke test: 4 runs x 20 iters, and an explicit resume check
bash $M3/scripts/debug_cyclegan.sh

# 4. the real thing
sbatch $M3/scripts/submit_cyclegan.sbatch
```

## Measured cost

From `--calibrate 40` on an RTX 6000 Ada at 512px, batch 1, 6 dataloader
workers, with AMP:

```
109 ms/iter  (9.1 it/s)
50 epochs x 2,940 iters = 147,000 iterations = 4.47 h per run
```

All four runs therefore land near **18 h and fit in a single 24h slot**, which
is why `submit_cyclegan.sbatch` asks for `--time=24:00:00` rather than the
usual 6h. Re-measure on the first run of any future change; `calibration.json`
is written per run.

If the four runs ever do not fit, the order is the fallback: `thermal_fog`
completes first because it is the one that tests Method 2's actual failure.

### Continuous debugging inside a live allocation

The GPU queue here runs 7+ deep, so waiting a day to discover a typo is not
acceptable. Secure the allocation once, then iterate inside it:

```bash
tmux new-session -d -s gateir "salloc --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=24:00:00"

srun --jobid=<JOBID>.1 --overlap bash $M3/scripts/debug_cyclegan.sh
# kill a step, never the job:
scancel <JOBID>.1
```

`--overlap` lets the smoke test share the GPU with the real job rather than
queue behind it, so you can iterate while training continues.

> **Never `scancel` the allocation itself.** The queue is far too long to give
> up a held GPU. `scancel <JOBID>.<step>` is safe; `scancel <JOBID>` is not.

## Resume

This is the load-bearing part of the design. Four resnet-9 generators at 512px
do **not** fit in one 24h slot, and nodes on this cluster fail outright
(`../../bench/scripts/sweep.sbatch:13-19` lost a sweep that way). So:

- a checkpoint is written every 100 iterations, carrying both generators, both
  discriminators, both optimizers, both schedulers, the AMP scaler, the
  iteration and epoch counters, and all three RNG states
- it is written to a temp file and `os.replace`d, so a kill during the write
  cannot leave a truncated checkpoint that fails to load
- a run with a `DONE` marker is skipped; a run with a checkpoint continues
- `--requeue` is set on the sbatch, so a node failure restarts and the restart
  *continues* rather than repeating

Resubmit the same `submit_cyclegan.sbatch` as many times as needed. Worst case
is losing 100 iterations of one run.

## Things that will bite you

- **The M3FD split dirs contain 2,940 stray `.npy` files** sitting beside the
  2,940 `.png`. `split_dataset.py` only ever copies `.png`/`.jpg` and nothing
  in the repo writes `.npy`, so they are leftovers from an earlier ad-hoc
  depth pass. Always glob `*.png` explicitly. Same failure class as the YOLO
  silent-label bug (`../../docs/memory.md:196-204`).
- **M3FD is not uniformly 1024×768.** Scanning all 2,940 IR training images
  finds 9 distinct geometries — 2,745 at 1024×768, and 195 spread across
  360×480, 480×800, 320×440, 280×440, 280×400, 420×780, 400×740 and 520×880.
  Anything that assumes a fixed input size (a resize target, a model with a
  fixed bottleneck, a batching assumption) will hit 195 exceptions. This is
  also why the size-preservation self-check and the per-item shape assertion
  in `data.py` exist.
- **`snapshot_download(allow_patterns=...)` hangs on this repo.** It still walks
  the repository root, which holds ~10⁶ files. `download_dsert.py` walks only
  the branches it needs (3 conditions × N sequences × 2 directories, ~40 s) and
  caches the result in `dsert_files.json`.
- **DSERT thermal is 384×480, smaller than the 512 crop.** Without
  normalisation every target image gets reflection-padded and the
  discriminator learns padding borders. Both domains are resampled to M3FD
  native 1024×768 before cropping (`data.py:to_canonical`). This also makes a
  512 crop cover the same fraction of the scene in both domains, so the only
  thing the discriminator can separate them on is weather. `--no-canonical`
  exists to A/B it.
- **M3FD IR and DSERT thermal are both greyscale stored in 3 identical
  channels** (verified: `b == g == r` exactly). There is no colour palette in
  play, so no palette-shift confound — but the real reason this is worth
  knowing is that any colour jitter would desaturate a greyscale image into
  nonsense. There is no colour jitter. Same reason `hsv_h=0.0, hsv_s=0.0` is
  pinned in `train_m3fd.py`.
- **Do not use the VGG weather classifiers** in the `hgupta01` repo alongside
  the CycleGAN weights. Deriving weather labels from a weather classifier is
  the circularity trap at `../../docs/memory.md:89-94`.
- **Generation happens at native 1024×768**, via the fully-convolutional
  generator, with no resize in between. A resize would perturb object scale and
  YOLO would score that as a weather gap — the same trap as the
  label-filtering artifact at `../method2_depth/report.md:146-154`.

## The diagnostic, and why it never gates

Every 500 iterations each run writes a sample grid to `runs/<run>/samples/`:

```
row 1:  M3FD clear        |  G_A2B(clear)      <- the translation we will use
row 2:  DSERT target real |  G_B2A(DSERT fog)   <- cycle reconstruction
```

Row 2 is the one that matters. DSERT's thermal imager is a different camera
from M3FD's, and a CycleGAN can plausibly learn to transfer **sensor identity**
rather than weather. If reconstruction does not return a DSERT fog frame to
something M3FD-clear-like, then any gap this generator later produces is
measuring the camera, not the fog — which would undercut the paper's claim.

It is worth seeing early. It is **not** a reason to stop training. These grids
are written and logged; nothing raises, pauses, or exits on them. A judgement
about a downstream experiment that has not run yet does not get to throw away
a half-trained generator.

An encouraging early signal: DSERT fog thermal has std 0.077 against M3FD
clear's 0.135 — roughly 43% less contrast, with a higher mean. That is the
expected LWIR fog signature (path radiance raises the floor, contrast
collapses), so the target domain really does carry degradation.

## After training: generating the test sets

Out of scope for this directory — no Protocol 3A/3B here, no detector
training. `translate.py` exists so that work can start:

```bash
$PY $M3/translate.py --run thermal_fog --split test --check-shift
```

`--check-shift` reports the mean displacement of edge energy between input and
output. Real fog lowers edge contrast without moving edges, so a systematically
non-zero shift means the inherited labels no longer line up and the mAP from
that split cannot be trusted. It flags its own ±3px search boundary, because
an undertrained generator produces noise and the metric is then uninformative
rather than reassuring.

## Layout

| file | |
|---|---|
| `download_dsert.py` | selective Hub download, cached manifest |
| `inspect_dsert.py` | pre-training format gate; prints resolution, writes `dsert_inspect.json` |
| `models.py` | resnet-9 generator + 70×70 PatchGAN, **verbatim** from the reference (required for the pretrained weights to load) |
| `data.py` | unpaired pairing, canonicalisation, 512 crops, no colour jitter |
| `train_cyclegan.py` | training loop, resume, calibration, diagnostics |
| `translate.py` | apply a generator at native resolution, with the edge check |
| `scripts/run_cyclegan.sh` | orchestrator, idempotent per run |
| `scripts/submit_cyclegan.sbatch` | 24h, `--requeue` |
| `scripts/debug_cyclegan.sh` | smoke test incl. the resume check |

### Pretrained initialisation

The handoff named `Weather_Effect_Generator` weights. They are at
`github.com/hgupta01/Weather_Effect_Generator` — `clear2foggy.pth`,
`clear2rainy.pth`, `clear2snowy.pth`, on Google Drive, and **RGB only**. There
are no pretrained thermal weather generators anywhere, so the thermal runs
train from scratch.

`models.py` is a verbatim copy of the reference architecture precisely so those
weights load unchanged; `--init-weights` will use them if you retrieve them.
It is non-fatal by design — a missing or mismatched checkpoint costs a starting
point, not a run. Note the ambiguity it has to resolve: that repo's `define_G`
defaults to `norm='batch'` while CycleGAN's convention is `norm='instance'`,
and those produce different `state_dict` keys. The loader sniffs the
checkpoint's own key set to decide.

## Hyperparameters

CycleGAN's own settings, so the result is attributable to the data rather than
to a recipe tuned on it. Resnet-9 generator `ngf=64` instance-norm, 70×70
PatchGAN `ndf=64`, LSGAN, Adam 2e-4 betas (0.5, 0.999), λ_A = λ_B = 10,
identity λ = 0.5, image pool 50, batch 1, 512px crops, 50 epochs (not the
200-epoch figure — `../../docs/dataset_generation_plan.md:126-128`).

`--calibrate N` times N real iterations after a warmup and reports the measured
rate and full-run estimate. It is advisory only: it warns, and never resizes
the schedule or stops training. It exists because the repo already
measure-don't-guess (`../../bench/scripts/estimate_runtime.py`).
