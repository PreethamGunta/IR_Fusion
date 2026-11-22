# `models/` — pretrained weights

`models/pretrained/` contains the pretrained checkpoints used to initialise
every detector run in the release. They are the only weights stored outside an
experiment `runs/` directory.

| File | Size | Used by |
|---|---:|---|
| `yolov8s.pt` | 22 MB | workstream 02 filter screen, workstream 01 baselines |
| `yolo12s.pt` | 19 MB | workstream 02 main comparison, workstream 03 |
| `yolo11s.pt` | 19 MB | workstream 05 |
| `yolov10s.pt` | 16 MB | workstream 05 |
| `yolo26n.pt` | 5.3 MB | workstream 05 |
| `rtdetr-l.pt` | 64 MB | workstream 05 (RT-DETR-l) |

## Where trained weights live

Trained weights are **not** duplicated here. Each experiment keeps its runs:

- `experiments/02_flir_gate_ir/runs/<arm>/weights/{best,last}.pt`
  — the B1/G1 models (CycleGAN and CycleDiff-75).
- `experiments/01_flir_weather_translation/runs/<run>/weights/best.pt`
  — the original YOLOv8s detectors; CycleGAN/CycleDiff translators under
  `method3_cyclegan/runs/` and `method4_cyclediff/results/`.
- `experiments/03_m3fd_weather/runs/<run>/weights/best.pt`
  — M3FD YOLOv12s baselines.
- `experiments/05_detector_bench/runs/<run>/weights/best.pt`
  — UNIRI-TID detector bench.

`experiments/02_flir_gate_ir/results/` also holds the learned filters:
`unet_dehazer.pt`, `unet_dehazer_cyclediff75.pt`, and `gate_cyclegan.pkl`.

Total trained checkpoints in the release: **138 files, ≈54 GB**.
