# Reorganisation plan — GATE-IR unified release

Status: written 2026-11-10, before any file move.

## 1. Audit summary

### Workstreams found

| # | Workstream | Source | Size | Notes |
|---|---|---|---|---|
| 01 | FLIR ADAS v2 weather translation + detectors | `flir_adas_experiments/flir_adras_v2` | 53 GB | CycleGAN (method3), CycleDiff (method4), classical/depth methods, YOLO runs, report |
| 02 | GATE-IR on FLIR synthetic fog | `flir_gate_experiments` | 2.6 GB | gate, filters, U-Net dehazer, YOLOv8s/YOLOv12s B1/G1, 100-epoch results |
| 03 | M3FD synthetic weather | `m3fd_experiments` | 21 GB | methods 1/2/3, YOLOv12s, results, report |
| 04 | UNIRI-TID gate / variants | `uniri_gate_experiments` | 1.1 MB | variant builder + logs |
| 05 | Detector bench | `bench` | 758 MB | YOLOv8/10/11/12/RT-DETR L baselines, contact sheets |
| 06 | GATE-IR architecture scaffolding | `GATE-IR` | 560 KB | demo-only package (see its README) |

### Data found

| Dataset | Location | Size |
|---|---|---|
| FLIR ADAS v2 raw + video test | `flir_adas_experiments/data/flir_adas_v2/FLIR_ADAS_v2` | 13 GB |
| FLIR YOLO splits (thermal labels, train_3a) | `.../data/flir_adas_v2/yolo` | 341 MB |
| FLIR synthetic fog/rain/snow | `.../data/flir_adas_v2/synthetic` | 5.9 GB |
| Pre-reorg synthetic copy | `flir_adas_experiments/data/flir_adras_v2/synthetic` | 4.7 GB |
| M3FD raw + 3B splits | `m3fd_experiments/data` | 12 GB |
| UNIRI-TID | `data/UNIRI-TID` | 610 MB |
| DSERT-RoLL | `flir_adas_experiments/flir_adras_v2/data/DSERT-RoLL` | 391 MB |
| COCO8 sanity set | `datasets/coco8` | 544 KB |

### Artefacts found

- 138 checkpoints (`.pt`/`.pth`/`.ckpt`), **54.3 GB** total.
- Result JSONs: 5 (FLIR report), 42 (GATE-IR), 5 (M3FD), 20 (bench).
- Reports: `README.md` (44 KB), `flir_gate_experiments/results/REPORT.md`, `all_results.md`, `RUN_STATUS.md`, FLIR `report.pdf/tex`, M3FD `methods_report.tex`.
- Logs: 141 MB (GATE-IR) + 6 MB (M3FD) + 7.7 MB (bench) + FLIR logs + 1 MB UNIRI.
- 198,558 absolute symlinks: 185,964 into `flir_adas_experiments`, 12,588 into `data`, 1 into `m3fd_experiments`.
- 634 absolute-path occurrences in text files; only **78** in `.py/.sh/.sbatch` (rest are generated YAMLs).

### Excluded from the release (kept untouched in `Gate_IR/`)

| Item | Size | Reason |
|---|---|---|
| `.git/` | 4 GB | VCS history, not a data artefact; stays with the source tree |
| `.kilo/` | 21 GB (before audit) | agent worktrees; **`recover-session-12` was audited and its unique content consolidated** (see Post-audit correction below) |
| `_trash/` | 124 KB | explicitly discarded material |
| `.agents/`, `opencode.json` | small | local tool configuration |
| `HF_TOKEN`, `KAGGLE_TOKEN` | small | **secrets — must never be shared** |
| `flir_gate_experiments/SHELL_TEST*` | 0 B | empty test files |

## 2. Target structure

```
gate_ir_release/
├── README.md
├── rules.md
├── data/
│   ├── flir_adas_v2/{FLIR_ADAS_v2,yolo,synthetic}
│   ├── legacy_flir_synthetic/        # superseded pre-reorg copy (documented)
│   ├── m3fd/
│   ├── uniri_tid/
│   ├── dsert_roll/
│   └── coco8/
├── experiments/
│   ├── 01_flir_weather_translation/{src,scripts,configs,runs,results,logs,report}
│   ├── 02_flir_gate_ir/{src,scripts,configs,runs,results,logs,data}
│   ├── 03_m3fd_weather/{...}
│   ├── 04_uniri_gate/{...}
│   ├── 05_detector_bench/{...}
│   └── 06_gate_ir_scaffolding/{...}
├── models/pretrained/                # yolo*.pt, rtdetr-l.pt
├── results/{reports,metrics,figures}
├── logs/
├── configs/
├── docs/{project_notes,references}
├── environment/{requirements.txt,environment.yml}
└── scripts/{relocate_paths.py,rewrite_symlinks.py,verify_release.py}
```

## 3. Move mapping

| Old | New |
|---|---|
| `flir_adas_experiments/data/flir_adas_v2` | `data/flir_adas_v2` |
| `flir_adas_experiments/data/flir_adras_v2/synthetic` | `data/legacy_flir_synthetic` |
| `flir_adas_experiments/flir_adras_v2/data/DSERT-RoLL` | `data/dsert_roll` |
| `m3fd_experiments/data` | `data/m3fd` |
| `data/UNIRI-TID` | `data/uniri_tid` |
| `datasets/coco8` | `data/coco8` |
| `flir_adas_experiments/flir_adras_v2` | `experiments/01_flir_weather_translation` |
| `flir_adas_experiments/reports` | `experiments/01_flir_weather_translation/report` |
| `flir_gate_experiments` | `experiments/02_flir_gate_ir` |
| `m3fd_experiments` | `experiments/03_m3fd_weather` |
| `uniri_gate_experiments` | `experiments/04_uniri_gate` |
| `bench` | `experiments/05_detector_bench` |
| `GATE-IR` | `experiments/06_gate_ir_scaffolding` |
| `weights` | `models/pretrained` |
| `docs/*` | `docs/project_notes/` (except UNIRI harness doc → experiment 04) |
| `refs/*` | `docs/references/` |
| `baseline_links.md`, `baseline_survey_and_experiment_matrix.md`, `cluster_usage_instructions` | `docs/` |
| `README.md` | `docs/README_HISTORICAL.md` (new root README written) |

No file is deleted. All moves are same-filesystem renames and are logged.

## 4. Path work

1. Rewrite every absolute symlink to a **relative** symlink.
2. Rewrite every `/home2/home/preetham_g/...` string in text files to the new root, using the longest-prefix mapping.
3. Make `flir_gate_experiments/src/paths.py` derive from its new depth (`parents[3]`).
4. Leave generated dataset YAMLs absolute but provide `scripts/relocate_paths.py` and regenerate-on-build; the generating code derives from the project root.
5. Replace the machine-specific conda interpreter path with `python` and a documented env.

## 5. Validation plan (static only — no compute will be run)

- Compare file/symlink counts and byte totals before vs after.
- Check every `path:` in every YAML exists.
- Check every symlink resolves.
- Grep for remaining `/home2/home/preetham_g` in `.py/.sh/.sbatch/.yaml`.
- Confirm every README exists and every documented path exists.
- No training/evaluation will be started (user instruction: do not run anything).


## Post-audit correction (2026-11-10, after initial packaging)

The initial plan excluded `.kilo/worktrees/recover-session-12` wholesale. A
file-by-file audit showed that was wrong: the worktree contained unique data and
newer documents. The following were consolidated:

- `data/DSERT-RoLL` (20 GB) merged into `data/dsert_roll`: added Light_Snow
  (16 GB), the full Heavy_Snow sequences (1,333 files per sequence vs 296 in
  the old copy), Fog/Heavy_Rain/Light_Rain, and processed train/test splits.
- `m3fd_experiments/_old/` (187 MB) -> `experiments/03_m3fd_weather/archive/_old/`.
- `sample_images/` (30 PNGs) -> `experiments/01_flir_weather_translation/figures/sample_images/`.
- `docs/gap_establishment.md`, two session transcripts, report build logs,
  the original method-1 code, LPIPS weights, and `run_fog_vae_sbatch.sh`.
- Newer versions of `devlog.md`, `MASTER_PLAN.md`, `dataset_generation_plan.md`,
  `handoff.md`, `memory.md` -> `docs/project_notes/worktree_recover-session-12/`.
- Alternative versions of 20 `bench/` files and 14 `m3fd_experiments/` files ->
  `experiments/*/archive/worktree_recover-session-12/`.

The other two worktrees (`abstracted-side`, pool `beaded-pisces`) were compared
too; they contained no unique content beyond a `.gitignore`.

### Cleanup performed after the audit (2026-11-10)

Byte-identical copies were removed with explicit user confirmation:

- `data/legacy_flir_synthetic/` (4.7 GB, 86,782/86,789 files identical to
  `data/flir_adas_v2/synthetic/`); the 7 non-identical files were preserved in
  `data/legacy_flir_synthetic_diff/`.
- Six method-4 checkpoint copies (15.48 GB): the periodic `model-N.pt` that was
  identical to `final_model.pt`, plus one stale `final_model.pt.bak`.
- Two `last.pt` files identical to `best.pt` in the one-epoch rain runs.
- Kept `final_model.pt` files in the six affected run directories were renamed
  to `final_model.pt.bak` as requested.
- The 282 dangling CycleDiff train symlinks were re-pointed to `train_val`;
  release-wide broken symlinks are now 0.
