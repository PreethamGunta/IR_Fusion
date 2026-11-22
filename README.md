# GATE-IR — consolidated research release

**Gated Adaptive Thermal Enhancement for Infrared Detection.**
Snapshot packaged on **2026-11-10** from the working tree at `Gate_IR/`. This
folder is a self-contained, move-friendly copy of the project: datasets, code,
configs, trained weights, logs, results and documentation.

> **No experiment was re-run while this release was assembled.** Every number
> below comes from result files and logs produced by the earlier runs, which are
> preserved unchanged under `experiments/*/results/` and `experiments/*/runs/`.
> See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) for the caveats.

---

## 1. Project overview

GATE-IR studies whether a two-stage pipeline helps thermal object detection in
adverse weather:

```
thermal frame ──► gate (clear / fog) ──┬─► clear  ─────────────────────► YOLO detector
                                       └─► fog    ──► learned dehazer ──► YOLO detector
```

The central comparison is **baseline vs filtered**:

| | Baseline (B1) | Filtered / gated (G1) |
|---|---|---|
| Clear frames | raw | raw |
| Fog frames | raw synthetic fog | U-Net-dehazed fog |
| Training set | clear + raw fog | clear + filtered fog |
| Inference | detector only | gate routes fog through the U-Net first |
| Detector | YOLOv12s | YOLOv12s (same recipe) |

All final FLIR comparisons use the **same 3,749 clear and 3,749 fog test-set
frames** with the same 15-class labels. The test set is the video-thermal-test
split; result files retain the historical `_vtt` suffix.

The repository is organised around eight workstreams. Sections 2–3 explain how
they are laid out and how to find the code, configs, runs and results for each
experimental line.

## 2. Repository guide

### 2.1 Repository at a glance

```
gate_ir_release/
├── experiments/   eight workstreams: code, configs, runs, results, logs
├── data/          datasets, labels and dataset YAMLs (see Section 5)
├── docs/          consolidated report, design notes, known issues
├── results/       cross-experiment index; authoritative files stay in experiments/
├── models/        pretrained initialisation weights only
├── configs/       pointers to generated and hand-written config locations
├── scripts/       relocation, validation and maintenance scripts
├── environment/   conda and pip environment definitions
└── logs/          cross-experiment log index
```

The experiment directories are the centre of the project; everything else either
supplies their inputs (`data/`, `models/`, `configs/`) or indexes their outputs
(`results/`, `docs/`, `logs/`).

### 2.2 Workstreams

| # | Directory | Scope | Status |
|---|---|---|---|
| 01 | [`experiments/01_flir_weather_translation/`](experiments/01_flir_weather_translation/README.md) | FLIR weather synthesis (CycleGAN, CycleDiff, classical, depth) and the original YOLOv8s detector runs | CycleGAN and CycleDiff-75 fog translations and detectors complete; rain detector runs stopped after 1 epoch; snow translation only; classical/depth code only |
| 02 | [`experiments/02_flir_gate_ir/`](experiments/02_flir_gate_ir/README.md) | **GATE-IR**: gate, gamma/CLAHE/DCP/U-Net filters, 60-epoch filter screen, 100-epoch main comparison, CycleDiff-75 transfer, cross-generator study, ablations | CycleGAN 100-epoch B1/G1 complete; CycleDiff-75 baseline complete; CycleDiff-75 filtered arm stopped at epoch 27/100 and resumable |
| 03 | [`experiments/03_m3fd_weather/`](experiments/03_m3fd_weather/README.md) | M3FD synthetic weather: classical, depth-aware, CycleGAN; YOLOv12s under clear-only and mixed-weather training | Runs complete; Protocol-3B training images are not shipped |
| 04 | [`experiments/04_uniri_gate/`](experiments/04_uniri_gate/README.md) | UNIRI-TID gate/preprocessing variant scripts | Scripts and logs only; no completed result JSONs |
| 05 | [`experiments/05_detector_bench/`](experiments/05_detector_bench/README.md) | UNIRI-TID detector bench: YOLOv8/10/11/12, P2, weather augmentation, RT-DETR-l | Complete except RT-DETR-l, which is invalid (29 epochs) |
| 06 | [`experiments/06_gate_ir_scaffolding/`](experiments/06_gate_ir_scaffolding/README.md) | Early GATE-IR package scaffolding | Demo only; no trained weights or evaluation pipeline |
| 07 | [`experiments/07_flir_adverse_conditions/`](experiments/07_flir_adverse_conditions/README.md) | Eight additional FLIR thermal conditions (fog, rain, snow, crossover, sensor, small target, defocus, obstruction): baseline screen, then retrain on conditions that hurt | Complete: 5/8 crossed the 5 % screen; final table in `results/FINAL_REPORT.md` |
| 08 | [`experiments/08_graph_kd/`](experiments/08_graph_kd/README.md) | Graph-structured cross-modal knowledge distillation into a compact IR-only student on aligned FLIR RGB/IR pairs: node/edge/FGW/partial-OT matchers, multi-teacher consensus, GT anchoring, condensation, decomposition | Running: see `results/SUMMARY.md` |

Each workstream README is the detailed reference for that experiment: scope,
data requirements, directory contents, commands, results and caveats.

### 2.3 Anatomy of an experiment directory

Most workstreams follow a common layout, so once you know one it is easy to
find the equivalent pieces in another. Not every workstream has every
subdirectory---04 has no runs or result JSONs, and 06 is a code sketch---but
the conventions below hold wherever a directory exists:

```
experiments/<id>_<name>/
├── README.md          scope, data, commands, results, caveats
├── src/ or *.py       implementation (dataset builders, training, evaluation)
├── scripts/           runnable entry points and cluster wrappers
├── configs/           dataset YAMLs (generated in experiment 02)
├── runs/              Ultralytics run directories
│   └── <run>/
│       ├── args.yaml  frozen training configuration
│       ├── results.csv training-time validation curve
│       └── weights/   best.pt, last.pt
├── results/           evaluation JSONs, summaries, gate models
├── logs/              training / evaluation / orchestration logs
└── data/              experiment-local intermediates (regenerable)
```

Two distinctions matter when looking for numbers:

- `runs/<run>/results.csv` is the **training-time validation** curve. In
  experiment 02 it is computed on the clear-only validation split (1,635
  frames); it drives `best.pt` selection and early stopping, but it is not the
  reported test-set metric.
- `results/*.json` is the **final evaluation** on the 3,749-frame test set and
  carries the weights path, dataset YAML, image count and split hash. Use these
  for reported numbers.

Trained weights stay with their experiment under `runs/<run>/weights/`.
`models/pretrained/` contains only the pretrained initialisation weights.

### 2.4 Naming conventions

The `B*` and `G1` labels come from workstream 02. Workstreams 01 and 03 also
use `3a`/`3b` protocol labels; those are local to their experiments and do not
mean the same thing everywhere, so both usages are listed separately.

| Label | Meaning |
|---|---|
| **B0** | Clear-only training. |
| **B1** | Clear + raw adverse weather (the baseline). |
| **B2** | Adverse-only training. |
| **B3** | Always-on filtering: filtered clear + filtered adverse at training and inference. |
| **G1** | Gated inference: clear is passed raw; adverse is filtered before the detector. |
| `3a_*` | Workstream 01 (FLIR): historical detector runs trained on **mixed clear + synthetic adverse** data and evaluated on both clear and synthetic-adverse test sets. |
| `3b_*` | Workstream 01 (FLIR): an evaluation mode, not a training run---the clear-only baseline is scored on a translated adverse test set. Workstream 03 (M3FD): `3b_*` run names denote Protocol-B **mixed clear + synthetic training**. |
| `*_train.yaml` | Training dataset description. |
| `*_eval_clear.yaml` | Evaluation of a trained model on the clear test set. |
| `*_eval_fog.yaml` | Evaluation of a trained model on the fog test set. |
| `b1_yolo12s_100` | Pipeline + detector + epoch budget for a run. |
| `g1_unet_yolov8s_60` | Gated pipeline, U-Net filter, YOLOv8s detector, 60-epoch budget. |
| `-2`, `-3`, `-8` | Sequential repeat attempts of the same line, not new methods. |
| `_vtt` | Legacy suffix on result files for the 3,749-frame test set. |
| `<run>_clear_vtt.json`, `<run>_fog_vtt.json` | Final evaluation of that run on the matched clear and fog test sets. |
| `_adv` | Historical 01 suffix for evaluation on a synthetic-adverse test set. |

### 2.5 Where training, evaluation and results live

- **Training entry points:** `experiments/01_flir_weather_translation/train_detector.py`
  (historical FLIR runs), `experiments/02_flir_gate_ir/src/train_detector.py`
  (GATE-IR), `experiments/05_detector_bench/train_detector.py` (UNIRI bench),
  `experiments/03_m3fd_weather/train_m3fd.py`, and the translation trainers
  `experiments/02_flir_gate_ir/src/train_unet.py`,
  `experiments/01_flir_weather_translation/method3_cyclegan/train_cyclegan.py`
  and the CycleDiff scripts under `method4_cyclediff/scripts/`.
- **Evaluation entry points:** `experiments/01_flir_weather_translation/eval_detector.py`,
  `experiments/02_flir_gate_ir/src/eval_detector.py`,
  `experiments/03_m3fd_weather/eval_m3fd.py` and
  `experiments/05_detector_bench/eval/detect_eval.py`.
- **Generated FLIR/GATE-IR configs** live in
  `experiments/02_flir_gate_ir/configs/`. They are generated, not hand-edited;
  see Section 6.
- **Final metrics** are the JSON files in each experiment's `results/`
  directory. The consolidated Markdown tables are in
  `experiments/02_flir_gate_ir/results/REPORT.md` and
  `experiments/02_flir_gate_ir/results/all_results.md`.
- **Logs** are in each experiment's `logs/` directory; `logs/README.md` gives a
  cross-experiment index.

## 3. Experiment index

### 3.1 FLIR weather translation and original detectors (01)

This is the experimental line that produced the synthetic FLIR weather and the
first single-generator detector baselines. Paths below are relative to
`experiments/01_flir_weather_translation/`.

| Line | What is trained or evaluated | Key runs | Key results | Details |
|---|---|---|---|---|
| CycleGAN fog detector | YOLOv8s on clear + CycleGAN fog; evaluated on clear and fog test sets | `runs/3a_cyclegan_fog-3` | `results/3a_cyclegan_fog_clear.json`, `results/3a_cyclegan_fog_adv.json` | [Results](experiments/01_flir_weather_translation/README.md#results) |
| CycleDiff-75 detector | YOLOv8s on clear + 75 % diffusion-blend fog; evaluated on clear and fog test sets | `runs/3a_cyclediff_fog-8` | `results/3a_cyclediff_fog_clear.json`, `results/3a_cyclediff_fog_75_adv.json` | [Results](experiments/01_flir_weather_translation/README.md#results) |
| Full CycleDiff (100 %) | YOLOv8s on clear + full diffusion fog; evaluated on the fog test set; the translation was unusable | `runs/3a_cyclediff_fog-3` | `results/3a_cyclediff_fog_adv.json` (0.0) | [Results](experiments/01_flir_weather_translation/README.md#results) |
| CycleGAN rain | YOLOv8s on clear + CycleGAN rain; detector run stopped after 1 epoch | `runs/3a_cyclegan_rain-2`, `runs/3a_cyclegan_rain-3` | none | [Notes](experiments/01_flir_weather_translation/README.md#reproducibility-notes) |
| CycleGAN snow | Translation exists; no detector run | `runs/3a_cyclegan_snow` (provenance only) | none | [Notes](experiments/01_flir_weather_translation/README.md#reproducibility-notes) |
| Translation models | CycleGAN and CycleDiff training / inference code and checkpoints | `method3_cyclegan/runs/{fog,rain,snow}`, `method4_cyclediff/{results,sample_results,configs}` | translation logs | [Directory](experiments/01_flir_weather_translation/README.md#directory-contents) |
| Classical and depth synthesis | Image-generation code for methods 1 and 2 | `method1_classical/`, `method2_depth/` | none for FLIR | [Directory](experiments/01_flir_weather_translation/README.md#directory-contents) |

The synthetic images are under `data/flir_adas_v2/synthetic/`: `cyclegan/fog`,
`cyclegan/rain`, `cyclegan/snow`, `cyclediff_75fog/fog` and `cyclediff/fog`.
The last one is the full 100 % diffusion set; its test images are symlinks to
the final CycleDiff model's `sample_results` directory.

### 3.2 GATE-IR on FLIR synthetic fog (02)

This is the main experimental line: filter choice, gating, and the controlled
100-epoch comparison. Paths below are relative to
`experiments/02_flir_gate_ir/`.

| Line | Training / inference setup | Key runs | Key results | Details |
|---|---|---|---|---|
| Filter screen (60 epochs, YOLOv8s) | Each filter is a separate detector trained on clear + that filter's fog; evaluated on raw clear + filtered fog | `runs/b1_yolov8s_60`, `runs/g1_<filter>_yolov8s_60` (filters: gamma, clahe, dcp, unet, gamma_clahe) | `results/b1_yolov8s_60_*`, `results/g1_<filter>_*_vtt.json` | [Filter screen](experiments/02_flir_gate_ir/README.md#final-results-matched-test-set-frames) |
| Main comparison (100 epochs, YOLOv12s) | Baseline: clear + raw CycleGAN fog. Filtered: clear + U-Net-dehazed CycleGAN fog | `runs/b1_yolo12s_100`, `runs/g1_unet_yolo12s_100` | `results/b1_yolo12s_100_*`, `results/g1_unet_100_*` | [Main results](experiments/02_flir_gate_ir/README.md#final-results-matched-test-set-frames) |
| Preliminary 40-epoch comparison | Same pipelines at a 40-epoch budget | `runs/b1_yolo12s_40`, `runs/g1_unet_yolo12s_40` | `results/b1_yolo12s_40_*`, `results/g1_unet_40_*` | [Preliminary](experiments/02_flir_gate_ir/README.md#preliminary-40-epoch-yolov12s-cyclegan) |
| CycleDiff-75, 100 epochs | Baseline: clear + raw CycleDiff-75 fog. Filtered: clear + target-generator U-Net fog | `runs/b1_cyclediff75_yolo12s_100`, `runs/g1_unet_cd75_yolo12s_100` (partial) | `results/b1_cyclediff75_yolo12s_100_*`; filtered arm has no final JSON | [RUN_STATUS](experiments/02_flir_gate_ir/results/RUN_STATUS.md) |
| Cross-generator, 40 epochs | Same CycleDiff-75 detector data; filter trained on CycleGAN vs on CycleDiff-75 | `runs/b1_cyclediff75_yolov8s_40`, `runs/g1_unet_cyclediff75_yolov8s_40`, `runs/g1_unet_cd75_yolov8s_40` | `results/b1_cyclediff75_yolov8s_40_*`, `results/g1_unet_cyclediff75_yolov8s_40_*`, `results/g1_unet_cd75_40_*` | [Cross-generator](experiments/02_flir_gate_ir/README.md#cross-generator-transfer-negative-result) |
| Ablations | Clear-only (B0), fog-only (B2), always-on filtered (B3) | `runs/b0_yolov8s_40`, `runs/b2_yolov8s_40`, `runs/b2_cyclediff75_yolov8s_40`, `runs/b3_unet_yolov8s_40` | `results/b0_40_*`, `results/b2_40_*`, `results/b2_cyclediff75_40_*`, `results/b3_unet_40_*` | [README](experiments/02_flir_gate_ir/README.md) |
| Gate | Binary clear/fog classifier on five image features | no Ultralytics run; model in `results/gate_cyclegan.pkl` | `results/gate_cyclegan.json` | [Gate metrics](experiments/02_flir_gate_ir/README.md#gate-metrics) |
| Config generation | Builders for every dataset-YAML variant | `src/build_direct.py`, `src/build_screen.py`, `src/build_cyclediff.py`, `src/build_ablations.py`, `src/build_datasets.py` | `configs/*.yaml` | [§6](#6-configurations-and-generated-files) |

`results/RUN_STATUS.md` records the exact paused state of the incomplete
CycleDiff-75 filtered arm and the U-Net regeneration steps.

### 3.3 M3FD synthetic weather (03)

`experiments/03_m3fd_weather/` tests whether synthetic weather can be learned
on the paired M3FD infrared/visible dataset. The three synthesis methods live in
`method1_classical/`, `method2_depth/` and `method3_cyclegan/`; the shared
detector wrapper is `train_m3fd.py` and the evaluator is `eval_m3fd.py`.

- **Clear baseline:** `runs/v12s_m3fd_clear`, result
  `results/v12s_m3fd_clear.json`.
- **Mixed-weather runs:** `runs/v12s_m3fd_3b_cgan_fog`,
  `runs/v12s_m3fd_3b_cgan_rain`, `runs/v12s_m3fd_3b_fog_heavy`; results in
  `results/v12s_m3fd_3b_*.json`.
- **Method evaluations:** `method1_classical/results_*.json`,
  `method2_depth/results_*.json`, `method3_cyclegan/results/3a_*_eval.json` and
  `3b_*_eval.json`.
- **Data:** `data/m3fd/{M3FD,split,synthetic,synthetic_depth,3b_fog_heavy}`.

The 03 README has the full protocol table. Under Protocol B (mixed training),
the largest residual gap is 0.049 (CycleGAN fog); deterministic fog is
effectively free (gap $-0.001$). The Protocol-3B training images are not shipped; only the YAML and
split manifest are present.

### 3.4 UNIRI-TID (04 and 05)

UNIRI-TID is the project's real adverse-weather detection dataset (DSERT-RoLL
is used as a translation target rather than a detection benchmark). It supports
two lines:

- **Detector bench (05):** `experiments/05_detector_bench/` contains
  `train_detector.py`, the shared scorer `eval/detect_eval.py`, runs
  `runs/{v8s,v10s,v11s,v12s,v8s_p2,v8s_weather,rtdetrl}`, result JSONs
  `results/eval_*`, and qualitative `contact_sheets/`.
- **Gate / preprocessing variants (04):** `experiments/04_uniri_gate/`
  contains `src/build_variants.py` and the scripts
  `scripts/run_shortlist.sh`, `scripts/run_variants.sh` and
  `scripts/wait_then_run.sh`. It produced logs but no completed result JSONs.

The 05 README has the detector table; the 04 README documents the gate
variants and the external-baseline blockers.

### 3.5 Scaffolding (06)

`experiments/06_gate_ir_scaffolding/` is the early package sketch. It has
module interfaces and a demo, but no dataloader, training loop, mAP evaluation
or trained weights. The working implementation is experiment 02.

### 3.6 Additional FLIR adverse conditions (07)

`experiments/07_flir_adverse_conditions/` extends the FLIR line beyond fog with
eight conditions: fog/mist, rain, snow, thermal crossover, sensor degradation,
small targets, defocus and static obstruction. It runs a staged protocol:
train a clear baseline `CM`, screen it on each condition's test set, and only
if the relative mAP50-95 drop exceeds 5 % retrain on clear + that condition
(`Mix_M`) and report the remaining clear/adverse gap. In the completed run,
snow, sensor degradation, small targets, defocus and obstruction crossed the
screen and were retrained; fog, rain and thermal crossover fell below 5 %.
The final table is [`results/FINAL_REPORT.md`](experiments/07_flir_adverse_conditions/results/FINAL_REPORT.md);
generation is deterministic and the generated images are regenerable under
`data/generated/`.

## 4. Results at a glance

The authoritative detailed tables are in
[`experiments/02_flir_gate_ir/results/REPORT.md`](experiments/02_flir_gate_ir/results/REPORT.md)
and every result JSON is tabulated in
[`all_results.md`](experiments/02_flir_gate_ir/results/all_results.md). A
narrative 16-page report is at [`docs/report/main.pdf`](docs/report/main.pdf).

### 4.1 CycleGAN fog — YOLOv12s, 100-epoch budget, matched test set

Source: `experiments/02_flir_gate_ir/results/{b1_yolo12s_100,g1_unet_100}_*_vtt.json`

| Arm | Clear mAP50 | Clear mAP50-95 | Fog mAP50 | Fog mAP50-95 | Gap (50-95) |
|---|---:|---:|---:|---:|---:|
| B1 — clear + unfiltered fog | 0.2849 | 0.1525 | 0.2430 | 0.1266 | 0.0259 |
| **G1 — clear + U-Net-dehazed fog** | **0.3063** | **0.1683** | **0.2856** | **0.1535** | **0.0148** |
| Δ | +0.0214 | +0.0158 | **+0.0426 (+17.5 %)** | **+0.0269 (+21.3 %)** | **−42.9 %** |

Both arms used `epochs=100`, `patience=25`, seed 0; they early-stopped at epochs
55 (B1, best 30) and 57 (G1, best 32). The filter also improves **clear** mAP,
so the gain is not a clear/fog trade-off.

### 4.2 CycleDiff-75 fog — YOLOv12s, 100-epoch budget (baseline complete)

Source: `experiments/02_flir_gate_ir/results/b1_cyclediff75_yolo12s_100_*_vtt.json`

| Arm | Clear mAP50 | Clear mAP50-95 | Fog mAP50 | Fog mAP50-95 | Gap (50-95) |
|---|---:|---:|---:|---:|---:|
| B1 | 0.2746 | 0.1491 | 0.2747 | 0.1495 | −0.0004 |

CycleDiff-75 fog is effectively free for this detector: fog mAP is marginally
above clear. The matching **G1 (target-generator U-Net)** was stopped at
**epoch 27/100** and is resumable; see `results/RUN_STATUS.md`.

### 4.3 Historical FLIR detectors (YOLOv8s)

Sources: CycleGAN clear/fog rows from
`experiments/02_flir_gate_ir/results/b1_cyclegan_{clear,fog}_vtt.json`;
CycleDiff-75 fog row from
`experiments/01_flir_weather_translation/results/3a_cyclediff_fog_75_adv.json`.

| Result | n | mAP50-95 | mAP50 |
|---|---:|---:|---:|
| CycleGAN fog — clear test set | 3,749 | 0.16686 | 0.30548 |
| CycleGAN fog — fog test set | 3,749 | 0.14448 | 0.27182 |
| CycleDiff-75 — fog test set | 3,749 | 0.16221 | 0.30371 |

The older 1,200-frame in-domain clear-evaluation rows are omitted because they
are not comparable to the 3,749-frame test set. The full 100 % diffusion model
scored 0.0000 and is unusable; the 75 % blend is used.

### 4.4 M3FD synthetic weather (YOLOv12s, split hash `effb9b3090ad2fcb`)

Clear baseline: **mAP50-95 0.148, mAP50 0.268**. The largest residual
mixed-training gap is 0.049 (CycleGAN fog); deterministic fog is effectively
free (gap $-0.001$). Full tables:
[`experiments/03_m3fd_weather/README.md`](experiments/03_m3fd_weather/README.md)
and [`docs/report/main.pdf`](docs/report/main.pdf).

### 4.5 UNIRI-TID detector bench (test set)

| Model | mAP50-95 | clear−fog APm |
|---|---:|---:|
| YOLOv8s | 0.6204 | 0.128 |
| YOLO11s | 0.6194 | 0.132 |
| **YOLOv12s** | **0.6296** | **0.094** |
| YOLOv8s + aug | 0.6278 | 0.096 |
| YOLOv10s | 0.6063 | 0.114 |
| YOLOv8s-P2 | 0.6071 | 0.153 |
| RT-DETR-l | 0.2778 | invalid (29 epochs) |

Full table with APm per condition:
[`experiments/05_detector_bench/README.md`](experiments/05_detector_bench/README.md).

### 4.6 FLIR adverse conditions (07)

A clear-trained YOLOv12s (`CM`, C_T **0.14966** mAP50-95) was screened on eight
simulated conditions at medium severity. Five crossed the 5 % relative-drop
threshold and were retrained on clear + condition:

| Condition | CM drop | Mix gap | Gap reduction |
|---|---:|---:|---:|
| Snow | 15.5 % | 0.0050 | 78 % |
| Sensor degradation | 36.7 % | 0.0223 | 59 % |
| Small targets | 15.3 % | −0.2339 | negative gap (pixelation cue) |
| Defocus | 64.1 % | 0.0464 | 52 % |
| Obstruction | 21.3 % | 0.0128 | 60 % |

Fog (2.1 %), rain (3.0 %) and thermal crossover (4.4 %) fell below the screen
and were not retrained. Full protocol, per-arm JSONs and caveats:
[`experiments/07_flir_adverse_conditions/results/FINAL_REPORT.md`](experiments/07_flir_adverse_conditions/results/FINAL_REPORT.md).

## 5. Data

All datasets live under `data/`; [`data/README.md`](data/README.md) documents the
layouts in detail.

| Path | What it contains |
|---|---|
| `data/flir_adas_v2/FLIR_ADAS_v2/` | Raw FLIR thermal/RGB frames: `train/clear`, `test/clear`, `video_thermal_test/data`, `images_thermal_{train,val}/data` |
| `data/flir_adas_v2/yolo/thermal/` | Thermal labels (`labels_all`, `labels_vtt_test`) |
| `data/flir_adas_v2/yolo/train_3a/` | Sequence-disjoint clear train/val/test splits and the `train_adv` synthetic-fog training images |
| `data/flir_adas_v2/synthetic/cyclegan/{fog,rain,snow}/` | CycleGAN images and labels, split into `train`/`test` |
| `data/flir_adas_v2/synthetic/cyclediff_75fog/fog/` | 75 % diffusion-blend fog images and labels |
| `data/flir_adas_v2/synthetic/cyclediff/fog/` | Full 100 % diffusion blend (test images symlinked to the final model's `sample_results`) |
| `data/m3fd/{M3FD,split,synthetic,synthetic_depth,3b_fog_heavy}` | M3FD raw data, temporal split, classical/depth synthetic weather and the Protocol-3B YAML plus split manifest (3B training images are not shipped) |
| `data/uniri_tid/{raw,annotations,yolo,yolo_debug}` | UNIRI-TID raw sequences, annotations and YOLO splits |
| `data/dsert_roll/{Fog,Heavy_Rain,Light_Rain,Heavy_Snow,Light_Snow,train,test}` | DSERT-RoLL target domains and processed splits |
| `data/coco8/` | Small Ultralytics sanity dataset |
| `data/legacy_flir_synthetic_diff/` | The seven non-identical files preserved when the duplicate legacy synthetic tree was removed |

`data/flir_adas_v2/README.txt` and `data/flir_adas_v2/md5sums_full.txt` are the original FLIR
release notes and checksums.

## 6. Configurations and generated files

The GATE-IR configs are **generated**, not hand-maintained. The naming pattern
describes the pipeline and the evaluation condition:

| Pattern | Meaning |
|---|---|
| `b1_direct_train.yaml` | Baseline training: clear + raw CycleGAN fog. |
| `b1_direct_eval_clear.yaml` | Baseline evaluated on the 3,749 clear frames. |
| `b1_direct_eval_fog.yaml` | Baseline evaluated on the 3,749 fog frames. |
| `g1_unet_direct_*` | Filtered/gated pipeline with the U-Net dehazer. |
| `b1_cyclediff75_direct_*`, `g1_unet_cd75_direct_*` | The same patterns for CycleDiff-75. |
| `b0_*`, `b2_*`, `b3_*` | Clear-only, fog-only and always-on-filtered ablations. |

Regenerate them from `experiments/02_flir_gate_ir/`:

```bash
python -m src.build_direct init       # b0/b1/b2/b3/g1 direct YAMLs
python -m src.build_datasets          # matched symlink trees + YAMLs
python -m src.build_screen            # filter-screen YAMLs
python -m src.build_cyclediff         # CycleDiff-75 YAMLs
python -m src.build_ablations         # ablation YAMLs
```

Generated YAMLs contain absolute paths (an Ultralytics requirement); the
generating code derives them from `src/paths.py`. After moving the release, run
`scripts/relocate_paths.py` or regenerate. The hand-written CycleDiff configs in
`experiments/01_flir_weather_translation/method4_cyclediff/configs/` and the
M3FD dataset YAMLs under `data/m3fd/` also contain absolute paths. See
[`configs/README.md`](configs/README.md).

## 7. Environment

Tested with the `gateir` conda environment:

- Python 3.11.16
- torch 2.14.0+cu126, torchvision 0.29.0+cu126
- ultralytics 8.4.138
- numpy 2.4.6, opencv-python 5.0.0.93, scikit-learn 1.9.0, scipy 1.17.1
- pillow 12.3.0, matplotlib 3.10.9, pandas 3.0.5, pyyaml 6.0.3, tqdm 4.70.0
- timm 1.0.30, einops 0.8.2, pytorch-lightning 1.9.3 (CycleDiff only)
- huggingface_hub 0.17.3 (dataset/model downloads)

Setup:

```bash
conda env create -f environment/environment.yml
conda activate gateir
# or: pip install -r environment/requirements.txt
```

Disk after cleanup: data ≈54 GB, experiments ≈47 GB, models ≈143 MB; home quota
≈195 / 220 GB used at packaging time.

SLURM is required for the training scripts. **Every job must be launched as an
`srun` step inside an `salloc` allocation**; on this cluster, processes started
directly by `salloc` were SIGTERM'd after ~6–7 minutes (2026-11-09 incident).
`salloc` allocations must not be cancelled between runs.

## 8. Reproducing and resuming

### GATE-IR (02) — the headline comparison

```bash
cd experiments/02_flir_gate_ir

# 1. rebuild the generated dataset YAMLs for this checkout
python -m src.build_direct init

# 2. inside a live salloc allocation
srun --jobid="$SLURM_JOB_ID" --overlap --cpus-per-task=16 --gres=gpu:1 \
     bash scripts/run_main_100.sh            # CycleGAN B1 + G1
srun --jobid="$SLURM_JOB_ID" --overlap --cpus-per-task=16 --gres=gpu:1 \
     bash scripts/run_main_100_cyclediff.sh  # CycleDiff-75 B1 + G1
```

`scripts/supervise_main100.sh` retries the chain and resumes each arm from
`runs/<arm>/weights/last.pt`. The **pending CycleDiff G1** arm is at epoch 27;
run `scripts/run_main_100_cyclediff.sh` (or the supervisor) to finish it.
The full recipe, including the U-Net filter regeneration, is in
`experiments/02_flir_gate_ir/results/RUN_STATUS.md`.

### Other workstreams

Each experiment README has its own execution section:

- [01 — FLIR weather translation](experiments/01_flir_weather_translation/README.md#execution)
- [02 — GATE-IR](experiments/02_flir_gate_ir/README.md#execution)
- [03 — M3FD](experiments/03_m3fd_weather/README.md#execution)
- [04 — UNIRI variants](experiments/04_uniri_gate/README.md#execution)
- [05 — UNIRI detector bench](experiments/05_detector_bench/README.md#execution)
- [06 — scaffolding](experiments/06_gate_ir_scaffolding/README.md)

## 9. Documentation and report index

| Document | Purpose |
|---|---|
| [`docs/report/main.pdf`](docs/report/main.pdf) | 16-page consolidated narrative report (source: [`main.tex`](docs/report/main.tex)) |
| [`experiments/02_flir_gate_ir/results/REPORT.md`](experiments/02_flir_gate_ir/results/REPORT.md) | Main GATE-IR result tables |
| [`experiments/02_flir_gate_ir/results/all_results.md`](experiments/02_flir_gate_ir/results/all_results.md) | Every GATE-IR result JSON as a row |
| [`experiments/02_flir_gate_ir/results/RUN_STATUS.md`](experiments/02_flir_gate_ir/results/RUN_STATUS.md) | Paused state of the incomplete CycleDiff-75 filtered arm and its resume recipe |
| [`experiments/01_flir_weather_translation/report/report.pdf`](experiments/01_flir_weather_translation/report/report.pdf) | FLIR weather-translation report |
| [`experiments/03_m3fd_weather/methods_report.tex`](experiments/03_m3fd_weather/methods_report.tex) | M3FD methods report source |
| [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) | Defects, caveats, missing artefacts and incomplete runs |
| [`docs/README_HISTORICAL.md`](docs/README_HISTORICAL.md) | Full working README as it stood before packaging; useful for the older experimental narrative and raw tables |
| [`docs/README.md`](docs/README.md) | Documentation index and reading order |
| [`docs/project_notes/`](docs/project_notes/) | Design notes, devlog, handoff and dataset-generation plans |
| [`docs/baseline_survey_and_experiment_matrix.md`](docs/baseline_survey_and_experiment_matrix.md) | Baseline survey and planned experiment matrix |
| [`docs/REORG_PLAN.md`](docs/REORG_PLAN.md) | Audit and move plan used to assemble this release |
| [`results/README.md`](results/README.md) | Cross-experiment results index |
| [`data/README.md`](data/README.md) | Dataset layouts, sizes and provenance |
| [`models/README.md`](models/README.md) | Pretrained weights and where trained weights live |
| [`configs/README.md`](configs/README.md) | Config locations and regeneration |
| [`scripts/README.md`](scripts/README.md) | Portability and validation tools |
| [`environment/README.md`](environment/README.md) | Environment details |
| [`logs/README.md`](logs/README.md) | Log locations and what to look for |

## 10. Moving and validating

All symlinks are relative and all source code derives paths from
`Path(__file__)`. Absolute paths remain only as embedded provenance or in
**generated** dataset descriptions: Ultralytics `args.yaml` and
`configs/*.yaml`, the M3FD dataset YAMLs, and the `weights` field of result
JSONs. The YAMLs are the ones that affect execution---regenerate them, or run:

```bash
python3 scripts/relocate_paths.py --from /old/path --to /new/path
```

Then validate the layout:

```bash
python3 scripts/verify_release.py
```

## 11. Known issues and limitations

See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md). The main ones:

- Results are **single-seed**; "100 epochs" means up to 100 with
  `patience=25` early stopping.
- CycleDiff-75 **filtered arm is incomplete** (epoch 27/100) and resumable.
- Symlinks are all relative and resolve (**0 broken**); the 282 dangling
  `cyclediff/fog/images/train` links were re-pointed to `train_val`.
- The gate/filter YAMLs point at `data/filtered/…`, which is intermediate data
  regenerated by the pipeline (deleted after the runs to save quota).
- The 40-epoch always-on vs gated routing comparison uses different detector
  sizes (YOLOv8s vs YOLOv12s); it is indicative, not a controlled comparison.
  The controlled comparison is the 100-epoch B1 vs filtered table in §4.1.
- M3FD Protocol-3B training images are not shipped; only the YAML and split
  manifest remain. The result JSONs are preserved and the data is regenerable.
- Historical scripts in workstream 01 are archived; several referenced the old
  `.kilo/worktrees/recover-session-12` checkout, which was audited and its
  unique content consolidated (DSERT Light_Snow, `_old/`, samples, newer docs).
- **2026-11-10 cleanup:** removed 4.7 GB of fully duplicate legacy synthetic
  data (7 non-identical files preserved in `data/legacy_flir_synthetic_diff/`),
  6 byte-identical method-4 checkpoint copies (15.48 GB) and 2 duplicate rain
  `last.pt` files. Kept method-4 finals were renamed `final_model.pt.bak`.
  Total reclaimed ≈20 GB.
- **2026-11-19 cleanup:** removed 8.2 GB of regenerable
  `experiments/07_flir_adverse_conditions/data/generated` images (rebuild with
  `scripts/generate_all.sh`), ~19 GB of method-4 intermediate checkpoints (finals
  kept), and all Light_Snow LiDAR/radar `.npy` and event `.npz` files (16 GB →
  2 GB; RGB/thermal images and `label.pkl` annotations kept). Total reclaimed ≈40 GB.
- Five design docs have two snapshots; the newer ones are in
  `docs/project_notes/worktree_recover-session-12/`.
- RT-DETR-l in the bench is invalid (29 epochs).
- Secrets (`HF_TOKEN`, `KAGGLE_TOKEN`) are **not** included.
