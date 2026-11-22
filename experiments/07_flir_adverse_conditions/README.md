# 07 — FLIR adverse-condition screen and retraining

## Overview

This workstream extends GATE-IR's synthetic-fog question to **eight additional
adverse conditions** on FLIR ADAS v2 thermal frames. It first asks whether a
clear-trained detector is actually hurt by a condition, and only if it is, asks
whether adding that condition to training closes the gap.

The protocol is deliberately staged so compute is not spent retraining on
conditions the detector already handles:

```
C_Tr (clear train) ──► train CM ──► C_T (clear test) ──► mAP_clear
                                      │
                                      ├─ A_T (condition test) ──► mAP_adverse
                                      │
                              drop = (mAP_clear − mAP_adverse) / mAP_clear
                                      │
                         drop > 5 % ? ─┴─ no  ──► record "not selected"
                                      │
                                     yes
                                      ▼
                    A_Tr (condition train) + C_Tr ──► train Mix_M
                                      │
                                      ├─ C_T  ──► mAP_clear
                                      └─ A_T  ──► mAP_adverse
                                      gap = mAP_clear − mAP_adverse
```

All arms use the same detector, recipe and seed. The primary metric is
**mAP50-95**; mAP50 is reported alongside it. Severity is **medium** by default
(the screen is a single point on the severity ladder, not a sweep).

## Conditions

| Condition | Family | Physical implementation | Uses labels | Edits labels |
|---|---|---|---|---|
| `fog` | CycleGAN-suitable | Atmospheric scattering `I = I0·t + A·(1−t)`, `t = exp(−β·d)`; depth map optional, uniform transmission otherwise | no | no |
| `rain` | CycleGAN-suitable | Global contrast attenuation + motion-blurred cool droplet streaks | no | no |
| `snow` | CycleGAN-suitable | Cold flake particles, ground accumulation, contrast loss | no | no |
| `crossover` | CycleGAN-suitable | Region temperature blending `T' = (1−γ)T + γT_bg` inside each labelled target | yes | no |
| `sensor` | CycleGAN-suitable | Gaussian + fixed-pattern noise, calibration drift, vignetting, dirty-lens smudges | no | no |
| `small_target` | physics-only | Target crop → `k×k` pixels → upsample → blur → blend back | yes | no |
| `defocus` | physics-only | Gaussian PSF or disk PSF convolution | no | no |
| `occlusion` | physics-only | Procedural fence/foliage alpha mask; boxes below 25 % visibility removed | yes | yes |

The distinction matters: CycleGAN can plausibly learn the first five
appearance changes, but it cannot reliably shrink targets, produce a
mathematically uniform defocus, or occlude a target without hallucinating
structure. Those three are generated directly from the physics/geometry.

### Scope caveat

FLIR ADAS v2 has no real adverse weather, no depth maps and no per-frame
temperature metadata. Every condition here is a **simulation on 8-bit thermal
JPEGs**:

- `fog` defaults to a uniform transmission because no depth is shipped; pass
  `--depth-dir` to use per-frame depth sidecars if they become available.
- `crossover` uses the local ring around each box as a proxy for the
  sun-heated background; it is not a radiometric temperature model.
- `sensor` uses a fixed-pattern seed so the same sensor artefacts appear on
  every frame, as they would on a real camera.
- `occlusion` is procedural unless `--mask-dir` supplies real masks.

## Directory contents

```
07_flir_adverse_conditions/
├── src/
│   ├── paths.py             paths derived from the file location
│   ├── severity.py          severity ladders + measured degradation
│   ├── labels.py            YOLO read/write, box pixels, visibility
│   ├── conditions.py        the eight generators
│   ├── generate.py          CLI: generate one condition over a split
│   ├── build_dataset.py     YOLO YAMLs (b0/b1/b2 + matched evals)
│   ├── run_protocol.py      resumable screen → retrain → report
│   ├── final_report.py      polished results table from state + eval JSONs
│   ├── train_condition.py   one detector arm
│   ├── eval_condition.py    one matched evaluation + provenance JSON
│   └── sample_sheet.py      qualitative source/generated contact sheet
├── scripts/
│   ├── generate.sh          one condition inside an allocation
│   ├── generate_all.sh      all conditions
│   ├── run_protocol.sh      one-shot protocol inside an allocation
│   └── supervise.sh         allocation-aware outer loop
├── configs/                 generated YAMLs and manifests
├── data/generated/          generated frames (regenerable, gitignored)
├── results/                 eval JSONs, protocol state/report, contact sheets
├── runs/                    Ultralytics runs (weights, results.csv)
└── logs/                    generation, training, eval and supervisor logs
```

Output layout per condition:

```
data/generated/<condition>/<severity>/
├── train/images/*.jpg   train/labels/*.txt
├── test/images/*.jpg    test/labels/*.txt
└── generation.json      params, counts, measured contrast/PSNR, failures
```

Ultralytics resolves `.../images` to the sibling `.../labels` directory, so the
labels sit where YOLO expects them.

## Execution

Every heavy step must run inside a live `salloc` allocation as an `srun` step.
The current cluster rule is 16 CPUs / 48–64 GB / 24 h per allocation; never
cancel an allocation (see the repository `rules.md`).

```bash
conda activate gateir            # or set PY=/path/to/gateir/python
cd experiments/07_flir_adverse_conditions

# 1. generate one condition (test + train)
bash scripts/generate.sh rain medium

# 2. build its YAMLs
python -m src.build_dataset --condition rain --severity medium

# 3. run the whole staged protocol (resumable)
bash scripts/run_protocol.sh
```

### Autonomous run across allocations

`scripts/supervise.sh` finds whichever `gateir` allocation is currently
running, launches the protocol with `srun`, and when the allocation ends it
waits and resumes on the next one. Start it in its own tmux session from the
login node:

```bash
tmux new-session -d -s exp07_run \
  'PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python \
   bash experiments/07_flir_adverse_conditions/scripts/supervise.sh'
```

State lives in `results/protocol_state.json`; every stage skips work that
already exists (generated images, YAMLs, checkpoints, eval JSONs). If an
allocation expires mid-training, the next run resumes from
`runs/<arm>/weights/last.pt`. A terminal blocker for one condition is written
to `results/protocol_report.md` and the remaining conditions continue.

Progress can be checked with:

```bash
tail -f experiments/07_flir_adverse_conditions/logs/supervisor.log
cat experiments/07_flir_adverse_conditions/results/protocol_report.md
```

## Dataset YAMLs

For condition `<cond>` at severity `<sev>`:

| YAML | Train | Test |
|---|---|---|
| `b0_<cond>_<sev>_train.yaml` | clear | condition |
| `b1_<cond>_<sev>_train.yaml` | clear + condition | condition |
| `b2_<cond>_<sev>_train.yaml` | condition only | condition |
| `b1_<cond>_<sev>_eval_clear.yaml` | clear + condition | clear |
| `b1_<cond>_<sev>_eval_<cond>.yaml` | clear + condition | condition |
| `b0_clear_train.yaml` | clear | clear |

The clear test tree is a relative-symlink farm at
`data/clear_test/{images,labels}/test` so the 3,749 clear frames have adjacent
labels without duplicating data.

## Results

The authoritative write-up is [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md),
with a machine-readable table in `results/protocol_summary.csv`.

Baseline CM on C_T: **mAP50-95 0.14966 / mAP50 0.27546** (3,749 frames).

| condition | family | CM C_T | CM A_T | drop | selected | Mix C_T | Mix A_T | Mix gap | gap reduction |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| fog | cyclegan | 0.14966 | 0.14645 | 2.14 % | no | — | — | — | — |
| rain | cyclegan | 0.14966 | 0.14521 | 2.97 % | no | — | — | — | — |
| snow | cyclegan | 0.14966 | 0.12652 | 15.46 % | yes | 0.16306 | 0.15806 | 0.0050 | 78.4 % |
| crossover | cyclegan | 0.14966 | 0.14301 | 4.44 % | no | — | — | — | — |
| sensor | cyclegan | 0.14966 | 0.09470 | 36.72 % | yes | 0.15057 | 0.12825 | 0.0223 | 59.4 % |
| small_target | physics-only | 0.14966 | 0.12675 | 15.31 % | yes | 0.14323 | 0.37717 | −0.2339 | n/a (negative) |
| defocus | physics-only | 0.14966 | 0.05373 | 64.10 % | yes | 0.14706 | 0.10070 | 0.0464 | 51.7 % |
| occlusion | physics-only | 0.14966 | 0.11775 | 21.32 % | yes | 0.14796 | 0.13521 | 0.0128 | 60.0 % |

Fog, rain and crossover fall below the 5 % screen and are not retrained
under this protocol. The small-target mixed model is *better* on A_T than on
C_T: the 8×8 pixelation is a strong learnable cue, so the model specialises to
it. That negative gap is a real domain-cue effect, not a label error (the clear
and small-target test labels are identical: 61,596 instances).

One incident: the first `Mix_sensor_yolo12s` attempt was OOM-killed while
RAM-caching 18,102 images under the 48 GB allocation. It was retried with
`cache=false` and 12 workers and completed; the table uses the completed run.
No unresolved blockers.

## Known limitations

- Single seed (0) and a single medium severity per condition; no variance.
- The screen threshold is a relative mAP50-95 drop > 5 %, as requested. A
  condition just below it is recorded as `skipped_below_threshold`, not as
  "harmless".
- `occlusion` is the only condition that edits ground truth; it removes boxes
  whose alpha-weighted visibility is below 25 %. Without that, a fully hidden
  target would be scored as a miss.
- `small_target` does not move box geometry; it degrades the target's pixels
  within the existing box, which simulates low resolution rather than a
  different range geometry.
- The generated data is ~10–15 GB at full train/test scale and is regenerable;
  it is gitignored.
