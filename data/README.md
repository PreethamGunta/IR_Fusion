# `data/` — unified dataset root

All datasets live once in this directory. Experiments reference them by
relative path and keep only derived labels, splits and caches locally.

| Path | Size | Contents | Used by |
|---|---:|---|---|
| `flir_adas_v2/` | 19 GB | FLIR ADAS v2 raw thermal/RGB frames, YOLO splits, synthetic fog/rain/snow | 01, 02, 07 |
| `m3fd/` | 12 GB | M3FD raw IR/RGB images, temporal split, classical/depth synthetic weather, 3B splits | 03 |
| `uniri_tid/` | 610 MB | UNIRI-TID images, annotations, YOLO splits, diagnostic CSVs | 04, 05 |
| `dsert_roll/` | 24 GB | DSERT-RoLL weather domains (Fog, Heavy/Light Rain, Heavy/Light Snow) and processed splits | 01, 03 |
| `flir_aligned/` | 1.5 GB | Zhang et al. aligned FLIR RGB/IR pairs (5,142 pairs, 4 classes) and generated YOLO trees | 08, 09 |
| `dronevehicle/` | ~20 GB | Paired aerial RGB/IR frames (28.4k pairs, 5 classes) and 640×512 YOLO crops | 08, 09 |
| `coco8/` | 544 KB | Ultralytics COCO8 sanity dataset | smoke tests |

Experiment-local derived data (label trees, split lists, warped images) is
described in the corresponding experiment README.

## `flir_adas_v2/` layout

```
flir_adas_v2/
├── FLIR_ADAS_v2/           # original FLIR release
│   ├── images_thermal_{train,val}/data/
│   ├── video_thermal_test/data/      # 3,749 frames
│   ├── train/clear/  test/clear/
│   └── ...
├── yolo/
│   ├── thermal/            # thermal labels, labels_all, labels_vtt_test
│   └── train_3a/           # train_clear, train_adv, val, test splits
├── synthetic/
│   ├── cyclegan/{fog,rain,snow}/
│   ├── cyclediff/fog/      # 100 % diffusion blend
│   └── cyclediff_75fog/fog/ # 75 % blend
├── README.txt              # original FLIR notes
├── md5sums_full.txt        # original checksums
└── rgb_to_thermal_vid_map.json
```

## Symlinks

The dataset uses relative symlinks (~198k) that point into
`FLIR_ADAS_v2/video_thermal_test/data/` and `images_thermal_{train,val}/data/`.
This lets the YOLO split trees reuse image bytes without copying them. All
symlinks resolve; `scripts/verify_release.py` checks this.

## Split construction

- **FLIR:** sequence-disjoint train/val/test splits; synthetic fog/rain/snow are
  generated per training and test condition (experiment 01).
- **DroneVehicle:** official train/val/test lists; paired crops built to 640×512
  with thermal (`*labelr`) and visible (`*label`) labels (experiment 09).
- **M3FD:** chunk-disjoint temporal split with a split hash recorded in every
  result file (experiment 03).
- **UNIRI-TID:** clear/fog/rain test conditions from the annotated sequences
  (experiments 04 and 05).
- **FLIR-aligned:** 4,129 train / 404 val (paper split, val carved by frame-id
  blocks) / 609 test for experiment 08; 1,013-frame paper test and two-block
  validation for the 3-class experiment-09 variant.
