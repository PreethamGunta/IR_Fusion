# `configs/` — configuration index

Generated and hand-written configs live with their experiment. This directory
is an index and a place for future cross-experiment configs.

| Experiment | Config directory | Notes |
|---|---|---|
| 01 FLIR translation | `experiments/01_flir_weather_translation/method4_cyclediff/configs/` | CycleDiff VAE/LDM/translator YAMLs; **absolute dataset paths** |
| 02 GATE-IR | `experiments/02_flir_gate_ir/configs/` | 105 generated Ultralytics dataset YAMLs (`b0/b1/b2/b3/g1_*`) |
| 03 M3FD | `experiments/03_m3fd_weather/method3_cyclegan/datasets/` + `data/m3fd/**/*.yaml` | 3B and variant dataset YAMLs |
| 05 detector bench | `experiments/05_detector_bench/runs/*/args.yaml` | frozen Ultralytics arguments per run |

## Regenerating the GATE-IR configs

`experiments/02_flir_gate_ir/src/*.py` build every dataset YAML from
`src/paths.py`:

```bash
cd experiments/02_flir_gate_ir
python -m src.build_direct init      # b0/b1/b2/b3/g1 direct YAMLs
python -m src.build_datasets         # matched symlink trees + YAMLs
python -m src.build_screen           # filter-screen YAMLs
python -m src.build_cyclediff        # CycleDiff-75 YAMLs
python -m src.build_ablations        # B0/B2/B3 YAMLs
```

The generated YAMLs contain **absolute** paths (Ultralytics requirement); the
generating code does not hardcode them. After moving the release, run
`python3 scripts/relocate_paths.py --from <old> --to <new>`.
