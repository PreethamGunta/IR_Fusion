# Experiment 09 — Phase 0/1 Audit (data, labels, leakage, evaluator)

Evidence labels: [VERIFIED] ran it and saw output; [CODE-READ] read code/config;
[REPORT-ONLY] only in a report; [INFERRED] indirect.

## 1. External-baseline protocol findings

### F1. IV-PAFNet uses visible annotations for the IR stream too — [VERIFIED]

The released COCO annotations (`DroneVehicle-COCO-annotations.tar`, SHA-256
`1e41468f…`) match the **visible** XML labels, not the thermal labels:

| Split | Released COCO anns | Our visible `*label` | Our thermal `*labelr` |
|---|---:|---:|---:|
| train | 286,792 | 286,789 (+3 unknown-class) | 316,408 |
| val | 22,462 | 22,462 | 24,490 |
| test | 143,314 | 143,313 (+1 duplicate) | 159,613 |

The config uses one `ann_file` for both `img_folder` (visible) and
`img_folder_ir` (IR). Therefore IV-PAFNet's “IR” stream is supervised with
visible boxes. This is the same label-policy defect found in experiment 08.
Our primary protocol uses `*labelr` for IR and `*label` for RGB.

### F2. IV-PAFNet retains the 100 px border and squashes 840×712 → 640×640 — [VERIFIED]

- Sample XML declares `width=840, height=712`; bboxes reach x=840.
- The released COCO images are `840×712`.
- `configs/rtdetr/include/dataloader.yml` applies `Resize [640, 640]` to both
  streams; there is no crop.
- Our protocol removes the border to 640×512, so the two protocols are not
  directly comparable.

### F3. IV-PAFNet evaluates and selects on the validation set — [CODE-READ]

`configs/rtdetr/paper_73e/common_73e.yml` defines a val dataloader and no test
loader; README §7.1 evaluates the DroneVehicle validation set. The best
checkpoint is selected on the same set that is reported. Our gate reports val
selection and a locked test separately.

### F4. ICAFusion's FLIR “val” is the test directory — [VERIFIED]

`external/ICAFusion/data/multispectral/FLIR-align-3class.yaml`:

```
train_rgb: .../visible/train
val_rgb:   .../visible/test
train_ir:  .../infrared/train
val_ir:    .../infrared/test
```

The reported FLIR numbers are selected on the test split. We do not reuse this
protocol; 09 selects on a held-out val block.

### F5. ICAFusion has no DroneVehicle loader — [VERIFIED]

A README DroneVehicle table exists, but the repo has no DV data YAML or loader.
A DV adapter would have to be written for Tier A.

## 2. DroneVehicle v2

### F6. Split and label provenance — [VERIFIED]

`src/data/build_dv_v2.py` built:

| Split | Pairs | IR boxes (`*labelr`) | RGB boxes (`*label`) | Round-trip mismatches |
|---|---:|---:|---:|---:|
| train | 17,962 | 315,826 | 286,217 | 0 |
| val | 1,469 | 24,490 | 22,462 | 0 |
| test | 8,980 | 159,613 | 143,313 | 0 |

All frames have both modalities; `ir_only=0`, `rgb_only=0`. Hardlinks point to
the read-only 08 crops. IR labels match `*labelr` on 100% of frames; RGB labels
match `*label` on 100%.

### F7. The official 17,990 vs our 17,962 — [VERIFIED]

The raw archive has 17,990 label XMLs but only 17,962 complete RGB/IR pairs:

- 11 IR images have no RGB counterpart: 02221–02223, 04896–04898, 05599,
  09013–09014, 12901–12902.
- 17 RGB images have no IR counterpart: 04992–04994, 05002–05003,
  06464–06467, 14036–14038, 14049–14050, 16010–16012.

Policy: **keep the 17,962 complete pairs, do not invent the 28 missing images**;
all downstream counts use 17,962. The original zips were removed after the 08
extraction, so repair from the archive is blocked.

### F8. Polygon → AABB edge cases — [VERIFIED]

Test split: 159,616 parsed objects; 16 no-point, 3 degenerate dropped →
159,613 written; 22,166 (13.9%) edge-clipped; 149,975 four-point polygons and
9,641 two-point `bndbox` objects. Edge clipping is expected at the 640×512
crop boundary; the AABB conversion is a protocol deviation from oriented-box
mAP and is documented.

## 2b. Dataset duplication status — [VERIFIED]

| Pair | Status | Action |
|---|---|---|
| `scratch/flir_aligned/aligned.zip` ↔ `data/flir_aligned/raw` | exact archive copy | **deleted** with user approval |
| `data/dronevehicle/raw/extracted` (840×712 + XML) ↔ `data/dronevehicle/yolo` (640×512 crops) | same frames, different preprocessing; raw XML is unique | not deleted; raw images are the only exact copy of the 840×712 source |
| `experiments/09/data/dronevehicle_v2` images ↔ `08/data/dronevehicle/yolo` images | same inodes (hardlinks) | no extra quota blocks; 56,822 links |
| `experiments/09/data/flir3` images ↔ `08/data/flir_aligned` | hardlinks; warped RGB and 3-class labels are new | 71,263 links total; only **0.29 GB unique** in 09 |
| `data/flir_adas_v2` ↔ `data/flir_aligned` | **not duplicates** | original FLIR ADAS v2 vs Zhang aligned release |
| `data/m3fd`, `data/dsert_roll`, `data/uniri_tid` | separate datasets | used by experiments 03/07 |
| `experiments/02_flir_gate_ir/data/filtered/unet_cd75` | derived synthetic/filtered outputs (25,602 files) | not a copy of `FLIR_ADAS_v2` |
| `Gate_IR` ↔ `gate_ir_release` | separate old project tree | not verified duplicate; keep |

## 3. FLIR 3-class and alignment

### F9. The 667 letterboxed RGB frames are concentrated in 5,500–8,859 — [VERIFIED]

A single tail holdout (frame ≥ 5,268) was 79% cropped and left 64 val frames
under the drop policy. We replaced it with two contiguous blocks:

- 3,900–4,400 (clean) and 7,000–7,120 (letterboxed cluster), ±10-frame buffer.

Result (warp policy): train 3,752 (cropped 16.3%), val 365 (cropped 13.2%),
test 1,013 (cropped 0%). Val `flagged>2px` = 38.9%; train = 63.0%; test =
51.5%. The cropped fraction matches, but the raw shift flag rate differs; the
policy comparison is therefore reported explicitly rather than assumed.

### F10. Raw phase correlation is unreliable on low-texture/night frames — [VERIFIED]

`audit/flir3_shifts.csv`: response can be negative; many val frames produced
spurious 20+ px shifts. `src/audit/robust_shifts.py` marks measurements valid
only when response ≥ 0.05 and |shift| ≤ 10 px, fills invalid frames from a
±7-frame local median, and applies a 5-frame rolling median. A 20-pair
brute-force check (±12 px edge overlap) on val gave 2–4 px raw shifts and 0–2 px
after warp, so the warp policy is directionally correct; the phase-correlation
residual metric is not reliable for these frames. [INFERRED for the population]

### F11. Two alignment policies — [VERIFIED]

| Policy | train | val | test | test_clean | dog dropped |
|---|---:|---:|---:|---:|---:|
| drop | 1,387 | 223 | 1,013 | 491 | 45 |
| warp | 3,752 | 365 | 1,013 | 491 | 120 |

`drop` removes flagged train/val pairs and keeps `test_all` plus `test_clean`.
`warp` keeps everything and translates flagged RGB frames by the estimated
shift. Test remains the complete 1,013-frame paper list for comparability.

### F12. Class counts (3 classes) — [VERIFIED]

| Policy / split | person | car | bicycle |
|---|---:|---:|---:|
| warp train | 7,835 | 19,249 | 2,501 |
| warp val | 1,081 | 1,348 | 65 |
| warp test | 4,107 | 4,124 | 360 |
| drop train | 2,675 | 6,864 | 927 |
| drop val | 793 | 904 | 35 |
| drop test | 4,107 | 4,124 | 360 |

Bicycle is scarce in val (35–65 boxes); per-class bicycle claims will be weak.

### F13. Leakage and normalization — [VERIFIED]

- Frame-id overlap: train/val/test = 0 for all pairs.
- Pair integrity: RGB/IR base names equal in every split.
- Exact dHash duplicates across splits: 0.
- dHash nearest-neighbour median: test→train 16, val→train 16, train→train 13;
  test is not closer to train than train is to itself, so no near-duplicate
  leakage was detected by this probe.
- Train-only normalization means/stds and SHA-256 are in
  `audit/flir3_leakage_{drop,warp}.json`.

## 4. Evaluator and lock

- `tests/test_eval.py`: 6 passed — GT-as-pred AP=1.0, empty preds, category
  names, letterbox round trip, known-shift IoU, hflip. [VERIFIED]
- `scripts/check_leakage.py --policy {drop,warp}`: PASS. [VERIFIED]
- `scripts/final_eval.py` with a dummy lock: first run AP=1.0; second run
  refused; `--override` logged. [VERIFIED]
- Agreement with stock Ultralytics on **identical predictions** is pending until
  a 09 model exists; the 08 lesson is that square-letterbox vs rect predictions,
  not COCOeval itself, caused the 0.008–0.009 gap.

## 5. Blockers

### B1. GPU / sigma_seed — [RESOLVED 2026-11-21 ~15:25 UTC]

Slurm job `10133` (`GIRKD`) was `RUNNING` experiment 08 with 32 CPUs and 1 GPU;
new steps failed:

```
srun --jobid=10133 --overlap --ntasks=1 --cpus-per-task=1 ...
srun: error: Unable to create step for job 10133: Job/step already completing or completed
```

Audit of the 08 arms showed they all distil into **IR-only students** —
`dv_a2_node`/`dv_a3_node_edge` from the fused early-fusion teacher (itself
lower than IR-only), `dv_b_consensus` from an IR+RGB consensus — so they do not
serve 09's goal of building a good fused model. With explicit user approval,
**step 10133.1 was cancelled** (`scancel 10133.1`); the allocation itself was
NOT cancelled and remains usable. 08 progress is preserved and resumable:
`dv_a2_node` DONE ep77 (best 0.4317), `dv_a3_node_edge` paused ep40 (best
0.44139), `dv_b_consensus` paused ep17 (best 0.41937); rerunning the
orchestrator skips DONE arms and auto-resumes the rest from `last.pt`.
Phase 0.6 and the E-ladder are now unblocked pending the first 09 srun step.

### B2. Environment policy — [VERIFIED]

The user instructed: **use `gateir` for all and delete the new envs.** The
`gf09`/`ivpaf` clones and the failed `icaf` prefix were removed. All 09 commands
now use `/home2/home/preetham_g/.conda/envs/gateir/bin/python`. Tier A IV-PAFNet
needs torch 2.7.1/Python 3.10, so that native reproduction is blocked until the
user authorises recreating a separate env. Tier B (our re-implementation) does
not depend on it.

### B3. Disk quota — [VERIFIED]

The user quota peaked at 230,686,720/230,686,720 blocks. After the approved
deletions (temporary envs, caches, VS Code binaries, kilo DB, 09 invalid builds,
08 caches/smoke/invalid runs, aligned zip), usage is **198.2M blocks**
(~31 GB headroom). Checkpoints must still be planned carefully.

## 6. Gate A status

**Not passed.** Phase 0.1–0.5 are complete (with B2 caveats). B1 is resolved:
step 10133.1 was cancelled with user approval and GPU0 is free inside the still
running allocation. Phase 0.6 (GPU timing and 3-seed noise) is the immediate
next action. No baseline has been trained or reproduced yet.

Ordered next steps:

1. Phase 0.6 timing/memory and 3-seed E00/E10 noise; set `sigma_seed` and the
   promotion threshold before the ladder.
2. CPU-only: Phase 1.2 complementarity oracle; GF-YOLO testbed/unit tests
   (channel contract, zero-modality, gate init, graph edge bounds).
3. Train E00–E03 on DroneVehicle v2, then E10–E14 and Gate B.
4. 08 resumption is deferred and safe (DONE markers + `last.pt` preserved).
5. Tier A external reproduction remains blocked on weights/env.
