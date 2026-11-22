# Adversarial Audit — Experiment 08 (Graph-Structured RGB/IR KD)

**Audit date:** 2026-11-21, 10:44–11:30 UTC
**Scope:** `experiments/08_graph_kd/` (code, configs, scripts, logs, runs, results)
**Live job:** Slurm `GIRKD` (10133) was `RUNNING` throughout; all checks below were
CPU-only or read-only. No step was cancelled, no cache was deleted, no run
directory was modified.
**Snapshot:** `audit/manifest_baseline.sha256` + `audit/experiment08_src_20261121T104413Z.tar.gz`
(taken before any analysis).

**Environment (exact):**

| Item | Value |
|---|---|
| Python | `/home2/home/preetham_g/.conda/envs/gateir/bin/python` 3.11.16 |
| torch | 2.14.0+cu126 |
| torchvision | 0.29.0+cu126 |
| ultralytics | 8.4.138 |
| numpy | 2.4.6 |
| opencv | 4.11.0 |
| pytest | 9.2.0 |
| platform | Linux 5.15.0-198-generic x86_64 |

---

## 1. Verdict

1. The graph-KD code stack is real and mostly functional; **28/28 tests pass** after the audit additions.
2. The corrected FLIR test numbers (A0 0.2518, A3 0.2586, A5 0.2569) are reproducible, but the A3/A5 gains are **not significant at 95%** in a paired image bootstrap.
3. The report’s FLIR fused-teacher result **0.206 / 0.598 / AP75 0.078 is invalid** (pre-fix evaluator); the current checkpoints score **0.376–0.378 val / 0.282–0.284 test**, making fused the second-best teacher, not a “dead end”.
4. **DroneVehicle IR train/val use visible labels, not thermal labels** (14,860/17,962 train and 1,218/1,469 val frames differ); every DV IR/fused result is label-policy-contaminated and must be retrained.
5. FLIR val and test are **two temporal halves of the same validation sequence** (8864–9499 vs 9512–10227), so the “locked test” is not sequence-disjoint.
6. The completed FLIR matrix is **IR→IR KD**, not cross-modal KD; the project’s cross-modal headline is untested on FLIR.
7. The custom evaluator is ~0.008–0.009 **optimistic vs the stock validator**; the report mixes stock-val and custom-test numbers.
8. The “fused early fusion is weak” conclusion is unsupported; the correct statement is “fused is 0.033 below IR but 0.083 above RGB on FLIR val”.

---

## 2. Claim ledger

| # | Report claim | Evidence file(s) | Evidence label | Verdict |
|---|---|---|---|---|
| 1 | FLIR teachers: IR 0.4168, RGB 0.2994 | `runs/ir_teacher/results.csv` (ep 67), `runs/rgb_teacher/results.csv` (ep 28) | [VERIFIED] | Correct |
| 2 | FLIR fused teacher 0.2060 / AP50 0.5980 / AP75 0.078 | `logs/fused_teacher_val_cpu.log` (mtime 2026-11-20 01:38) | [VERIFIED] | **Invalid**: log predates the 06:48 evaluator fix; current best/last score 0.376/0.378 val |
| 3 | FLIR test A0 0.2518, A1 0.2409, A2 0.2498, A3 0.2586, A5 0.2569 | `results/*_test.json` | [VERIFIED] | Correct under the custom evaluator; +0.0068/+0.0051 are point estimates only |
| 4 | FLIR A0 val 0.3370 | `runs/a0_gt/results.csv` (ep 12) | [VERIFIED] | Correct (trainer val), but not the custom evaluator |
| 5 | FLIR val/test gap ~0.085 | stock val 0.3360 vs custom test 0.2429/0.2518 | [VERIFIED] | Gap is real (~0.092 same-pipeline); report mixes pipelines |
| 6 | DV IR teacher 0.5484 / RGB 0.5882 / fused 0.4865 | `runs/dv_*_teacher/results.csv`; `logs/fused_teacher_paired_val2.log` | [VERIFIED] | Numbers reproduce, **but IR/fused use visible train/val labels** |
| 7 | DV students A0 0.4348, A1 0.4407, A2_ir 0.4383 | `runs/dv_*/results.csv` | [VERIFIED] | Numbers reproduce on visible labels; not valid thermal IR results |
| 8 | DV test labels rebuilt from `testlabelr`, 159,613 boxes | `data/dronevehicle/yolo/ir/labels/test` | [VERIFIED] | Correct; per-class counts now recorded (see §5) |
| 9 | DV train/val labels are nearly identical across modalities | report §5.2 S3 | [VERIFIED] | **False**: 14,860/17,962 train and 1,218/1,469 val frames differ |
| 10 | FLIR val/test are sequence-disjoint | `build_dataset.py`, `align_*.txt` | [VERIFIED] | **False** for val/test (same sequence, two temporal halves); train/val boundary is plausible but not proven |
| 11 | Cache uses top-60 FLIR nodes / top-40 DV | `data/cache/*/meta.json` | [VERIFIED] | **False**: both are k=40 |
| 12 | FLIR fused trained without paired augmentation | `trainer.py`, `train.py`, `logs/fused_teacher.log` | [CODE-READ] | **False**: `paired_augment = bool(args.fused)` enables paired flips/scale/translate/photometric jitter |
| 13 | Full graph-KD stack exists; 7 tests pass | `tests/test_matching.py`, pytest | [VERIFIED] | Correct; now 28 tests pass |
| 14 | M1–M12 fixed | code + logs | mixed | Most fixes exist; M5 was only a data patch (builder still wrong); M10/M11 are process-only |
| 15 | No seeds recorded per run | `runs/*/args.yaml` | [VERIFIED] | **False**: every run records `seed: 0` |
| 16 | Fused teacher “best epoch 100” | `runs/fused_teacher/results.csv` | [VERIFIED] | Misleading: `val=False`; best/mAP column is 0.0002, not 0.206 |
| 17 | A4 “started” and later “dropped” | `runs/dv_a4_sinkhorn/`, logs | [VERIFIED] | Not a contradiction; started then dropped (no `results.csv`) |
| 18 | DGX/temp caches are node-local | `/tmp/.cache_8f2c91/` | [VERIFIED] | Correct; caches disappear if the node changes |
| 19 | Evaluator is correct on identical GT/pred | new `tests/test_audit_eval.py` | [VERIFIED] | AP = 1.0; top-left/letterbox tests pass |
| 20 | `--device cpu` works | `evaluate.py --device cpu` | [VERIFIED] | **Was broken** (`cuda:cpu`); fixed and tested |

---

## 3. Findings

Severity: **Critical** invalidates a headline result/comparison; **High** could
change a conclusion; **Medium** weakens rigor/reproducibility; **Low** cleanup.
Evidence labels are exactly one of [VERIFIED], [CODE-READ], [REPORT-ONLY],
[INFERRED].

| ID | Severity | Evidence | Impact | Concrete fix | Status |
|---|---|---|---|---|---|
| F1 | **Critical** | [VERIFIED] `audit/dv_label_audit.log`: train label 286,789 boxes vs labelr 316,408; 3,108/17,990 frames equal; YOLO IR labels match `label` 17,962/17,962 and `labelr` 0/17,962. Val: 22,462 vs 24,490; 251/1,469 equal; YOLO matches `label` 1,469/1,469. Test: YOLO matches `labelr` 8,980/8,980. Thermal-val re-eval: IR teacher 0.5263 vs 0.5484 (−0.0221); A0 student 0.4686 vs 0.4348 (+0.0338). | Every DV IR teacher/student and the fused teacher were trained and model-selected on **visible** annotations; test is scored on **thermal** annotations. They are not valid thermal-modality benchmarks, the policy effect is model-dependent (−0.022 teacher, +0.034 student), and no constant correction exists. | Rebuild train/val IR labels from `*labelr` (builder now fixed), invalidate `*.cache`, retrain IR and fused teachers + all arms, then evaluate. | Code fixed; data/retrain **open** |
| F2 | **Critical** | [VERIFIED] `logs/fused_teacher_val_cpu.log` mtime 01:38 < `evaluate.py` mtime 06:48. Controlled re-eval: best val AP 0.37645/AP50 0.70015/AP75 0.34150; last val 0.37827/0.70099/0.34488; best test 0.28172; last test 0.28398. | The “fused 0.206 / AP75 0.078” negative result and the “go/no-go failed” decision are based on a stale pre-fix evaluator. The current fused teacher is second-best, not catastrophic. | Re-evaluate fused with the corrected pipeline; replace the 0.206 row; re-run the go/no-go on 0.378 vs IR 0.412. | Corrected numbers measured; report correction **proposed** |
| F3 | **High** | [VERIFIED] frame lists: val 8864–9499 (404), test 9512–10227 (609), same `FLIR_*` sequence, 12-frame hole; train max 8859, val min 8864. | The “locked test” is the second half of the same sequence used for model selection; test is not an independent sequence estimate. Absolute test AP and val→test gap are optimistic/biased. Arm ranking remains internally comparable. | Use a truly held-out sequence, or relabel this as a temporal test and stop claiming sequence disjointness. | **Open** |
| F4 | **High** | [VERIFIED] `audit/eval_compare.log`: custom vs stock A0 val 0.34406 vs 0.33599 (+0.0081), test 0.25185 vs 0.24291 (+0.0089). | Report tables mix stock trainer val (0.337) with custom test (0.252); absolute numbers differ by ~0.008–0.009. All custom test arms remain comparable to each other. | Use one evaluator for all reported numbers; state geometry (square letterbox vs rectangular) and NMS settings. | **Open** (proposed) |
| F5 | **High** | [VERIFIED] 60 cropped vs 60 non-cropped pairs: cropped median |dx|/|dy| = 4/6 px, 93.3% >2 px, mean dy −3.93; non-cropped 2/2 px, 53.3% >2 px; 1/30 cropped still has a residual border. | The 667 RGB “repairs” are systematically misaligned; 13% of paired training data is degraded, which can explain part of the fused gap. The report’s aggregate 2 px median hides this. | Improve crop/border detection or exclude cropped pairs; re-audit stratified; retrain fused. | **Open** |
| F6 | **High** | [VERIFIED] paired bootstrap (500 resamples): A3−A0 point +0.0073, 95% CI [−0.0008, +0.0160], P(>0)=0.968; A5−A0 point +0.0058, 95% CI [−0.0018, +0.0186], P(>0)=0.932. | “A3 and A5 beat A0” is a point estimate; both 95% CIs include 0. Should be downgraded to “no detectable difference / weak evidence”. | Report CIs; run ≥3 seeds; do not claim significance from one seed. | **Open** |
| F7 | **High** | [VERIFIED] args.yaml: all KD/A0 students use mosaic/HSV/flip = 0; teachers use mosaic 1.0, HSV, randaugment. Best student epochs are 11–16 (early stop at 41–43); teachers best at 54–84. | KD gains may be a regularization effect on an unaugmented, early-stopped nano student, not privileged-information transfer. Capacity (n vs s) and augmentation are confounded with KD. | Add an augmented student control, a longer schedule, and a label-smoothing/weight-decay control; report the KD effect over those controls. | **Open** |
| F8 | **High** | [VERIFIED] factor table in §5. | A1_pred vs A2_ir differs in method **and** teacher; A5 vs A3 differs in matcher **and** `kd_every` (6 vs 4); DV A3 vs B differs in method and teacher count. These comparisons are not attributable. | Run matched controls: A3 with IR teacher on DV; A5 with `kd_every=4`; A1 with IR teacher. | **Open** |
| F9 | **High** | [VERIFIED] FLIR completed arms: A1_pred_ir, A2_ir_control, A3_node_edge_ir, A5_fgw_ir all use `data/cache/ir`; A1_pred and A2_node (fused) are invalid/stopped. | The project’s headline “RGB/IR cross-modal KD” is not tested by the completed FLIR matrix; it is IR→IR KD. | Run B/C or a fused-teacher arm with the corrected fused teacher; frame the FLIR result as IR-teacher graph KD until then. | **Open** |
| F10 | **High** | [VERIFIED] DV fused teacher 0.4865 < IR 0.5484; `configs/kd_dronevehicle/a1_pred.json`, `a2_node.json`, `a3_node_edge.json` use the fused cache. | DV A1/A2_node/A3 test a **weaker teacher**, so “privileged RGB information” is not being tested; they test whether KD from a weaker fused model helps. | Compare against A2_ir; add a fused teacher that beats IR before claiming privileged-information gains. | **Open** |
| F11 | Medium | [VERIFIED] code: `_adapt_stem` built `[v, gray]` while the dataset builds `[IR, RGB]`; channels 0–2 (IR) received RGB filters and 3–5 (RGB) received grayscale filters. | The 6-channel fused teacher was initialised with swapped modality filters, a plausible contributor to the fused gap. Existing checkpoints are unaffected; future runs benefit. | Swapped to `[gray, v]`; added `_adapt_first_conv` + regression test. | **Fixed + verified** (28 tests) |
| F12 | Medium | [VERIFIED] `build_dronevehicle.py` used `testlabel` for IR test; current test files were corrected out-of-band and match `testlabelr`. | Running the builder would reintroduce the wrong test labels; the M5 “fix” was not in the builder. | Added `LABEL_DIRS` with thermal labels for IR and visible for RGB; per-modality label writing; regression test. | **Fixed + verified** |
| F13 | Medium | [VERIFIED] `evaluate.py` accepted `--device cpu` then built `cuda:cpu` and crashed; after the fix, `--device cpu` val gives AP 0.344062, matching the CPU-env run. | CPU-only audits needed `CUDA_VISIBLE_DEVICES=""`; the documented flag was broken. | Added `resolve_device`; `--device cpu` and no-CUDA map to CPU; test added. | **Fixed + verified** |
| F14 | Medium | [CODE-READ] `TeacherCache.__init__` reads `meta.json` but never checks `weights_sha256`, `k`, `imgsz`, or preprocessing against the config that references the cache. | A stale cache (old weights/K/preprocessing) is silently accepted; the current caches happen to match (hashes verified). | Add an expected-meta argument and refuse mismatches; include a cache schema version. | **Proposed** |
| F15 | Medium | [VERIFIED] `hungarian(NaN)` silently falls back to identity; `sinkhorn(NaN)` returns all-NaN; `KDModule` then `nan_to_num`s the total to 0. | A matching failure can silently disable KD for a batch instead of raising, masking a real defect. | Guard non-finite costs at the matcher boundary (raise or fall back to Hungarian with a counter); log the event. | **Proposed** |
| F16 | Medium | [VERIFIED] cropped RGB alignment (F5); [CODE-READ] resize path. | A fused teacher trained on 13% misaligned pairs cannot be compared fairly to single-modality teachers. | Same as F5. | **Open** |
| F17 | Medium | [VERIFIED] `logs/fused_teacher.log` tail validates 404 images with AP 0.376/0.703; `results.csv` best is 0.0002 because `val=False`. | “Best epoch 100” is not a validation-selected checkpoint; the report’s table is ambiguous. | Report the actual paired-val checkpoint and value; use `best.pt` consistently. | **Open** |
| F18 | Low | [VERIFIED] `args.yaml` for all runs has `seed: 0`; report S5 says seeds are not recorded. | Reproducibility is better than reported, but only one seed exists. | Correct the report; add seeds 1/2 for the headline comparison. | **Open** |
| F19 | Low | [VERIFIED] cache meta `k=40` for FLIR and DV; report says 60/40. | Reproducibility/documentation error only; student and cache both use 40, so no mismatch. | Correct the report. | **Open** |
| F20 | Low | [VERIFIED] `train_teachers.sh` / `evaluate_all.sh` use `PY=python`; orchestrator/relaunch use absolute `$PY`/`sys.executable`. | A shell without the conda env reproduces M9. | Use the absolute interpreter everywhere. | **Open** |
| F21 | Low | [VERIFIED] FLIR val class counts have **0 dog**; test has 13. | Model selection never sees dog; per-class dog AP is not a meaningful comparison. | State it; exclude dog from claims or add dog frames to val. | **Open** |
| F22 | Low | [VERIFIED] `results/dv_a0_gt_test_WRONG_GT.json`; old invalids in report §4.7. | Already quarantined; naming is clear. | Keep; do not cite. | **Closed** |
| F23 | Low | [VERIFIED] no git; manifest/tarball created. | Provenance relies on files/hashes. | Keep `audit/manifest_*.sha256`; add a post-audit manifest. | **Mitigated** |
| F24 | Low | [CODE-READ] `build_dataset._write_label` divides by fixed 640×512; [VERIFIED] all FLIR images are 640×512. | No current defect; a future resize would break labels. | Assert image size before writing. | **Proposed** |

### What is sound (no issue found)

- **FLIR image sizes**: all 4,129/404/609 images are 640×512 [VERIFIED].
- **FLIR label counts**: 32,256 / 3,646 / 4,958 boxes match `dataset_audit.json` and the label files [VERIFIED]; only 238/43 out-of-range boxes (0.7–0.9%), 1–2 duplicates per split [VERIFIED].
- **DV test label conversion**: 159,616 parsed objects, 16 with no points, 3 degenerate dropped → 159,613 boxes; 22,166/159,616 (13.9%) edge-clipped, which is expected for objects at the crop boundary [VERIFIED].
- **DV test per-class counts** (closed S4): car 137,145; truck 8,657; bus 4,467; van 4,282; freight_car 5,062 [VERIFIED].
- **Category IDs/names**: FLIR 0–3 (`person,bicycle,car,dog`) and DV 0–4 (`car,truck,bus,van,freight_car`) agree across YAML, cache meta, and COCO category IDs [VERIFIED by code/output].
- **Evaluator unit behavior**: identical GT/pred on one image gives AP=1.0; top-left COCO bbox and letterbox mapping tests pass [VERIFIED].
- **Gradient path**: node/edge/struct/feature/GT-anchor/consensus losses all produce finite non-zero student gradients; dense maps keep autograd; teacher cache tensors are detached [VERIFIED].
- **M1**: relaunch uses `srun --ntasks=1` [VERIFIED].
- **M2**: dense maps have no `.detach()`; test covers it [VERIFIED].
- **M3/M4**: letterbox + top-left fixes present; tests pass [VERIFIED].
- **M7**: OOM artifacts exist in `logs/status_3h.txt`, `parallel3.log`, `parallel5.log`; DV runs use `cache=none`/`disk` [VERIFIED].
- **M12**: `tempfile.mkstemp` and `dev.type` guard present; CPU evaluation wrote JSONs successfully [VERIFIED].

---

## 4. Corrections made

All edits were made only after reproducing a defect. No existing result JSON
was overwritten; no `runs/` or cache was modified; no training data was
modified.

### C1. Fused-stem channel order (`src/graphkd/training/trainer.py`)

**Defect:** `PairedYOLODataset` concatenates `[IR, RGB]`, but `_adapt_stem`
built `[v, gray]`, giving the IR channels the RGB-pretrained filters and the
RGB channels grayscale filters.
**Change:** added `_adapt_first_conv(v)` returning `cat([gray, v]) / 2` and used
it in `_adapt_stem`.
**Verification:** `tests/test_audit_data.py::test_adapt_first_conv_uses_ir_then_rgb_order` passes; the full suite is 28/28.

### C2. DroneVehicle per-modality labels (`src/graphkd/data/build_dronevehicle.py`)

**Defect:** the builder used `trainlabel`/`vallabel`/`testlabel` for **both**
modalities, so a rebuild would put visible labels on IR (M5 was only a data
patch).
**Change:** added `LABEL_DIRS = {ir: *labelr, rgb: *label}`; the loop now
requires both XMLs, parses IR and RGB separately, writes separate label files,
and counts classes from the IR labels.
**Verification:** `tests/test_audit_data.py::test_dronevehicle_uses_thermal_labels_for_ir_only`,
`test_dronevehicle_polygon_aabb_clips_to_crop`, and
`test_dronevehicle_degenerate_polygon_dropped` pass. Existing data was **not**
rebuilt (the live job depends on it); the fix is for the required re-run.

### C3. CPU device selection (`src/graphkd/training/evaluate.py`)

**Defect:** `--device cpu` produced `cuda:cpu` and crashed with
`RuntimeError: Invalid device string: 'cuda:cpu'` (reproduced).
**Change:** added `resolve_device(arg, cuda_available=None)` and used it.
**Verification:** `tests/test_audit_eval.py::test_resolve_device_cpu_on_gpu_node` passes; an end-to-end `--device cpu` run on FLIR val wrote `results/audit_device_cpu_val_val.json` with AP 0.344062 / AP50 0.640501 / AP75 0.319008, exactly matching the `CUDA_VISIBLE_DEVICES=""` run.

### C4. Audit-only test suite

Added:
- `tests/test_audit_eval.py` (AP=1.0, per-class names, empty preds, letterbox, hflip, device).
- `tests/test_audit_kd.py` (loss gradients, dense autograd, cache detachment, matcher empty/ties/marginals/FGW fallback, empty graphs).
- `tests/test_audit_data.py` (channel order, label-dir mapping, polygon clipping/degenerate).

**Result:** 28 passed, 1 warning.

### C5. Documentation

- Added a prominent audit banner to `PROJECT_REPORT.md` with the invalidated
  claims and a pointer to this file.
- Renamed `results/a0_gt_valcheck_val.json` →
  `results/a0_gt_valcheck_val_INVALID_pre_fix_evaluator.json` (AP ≈ 1e-5).
- Added `audit/` scripts and logs (no effect on experiment results).

### C6. Snapshot / diff

`audit/manifest_baseline.sha256` covers `src/ configs/ scripts/ results/`.
After the changes, `sha256sum -c` reports only:
`build_dronevehicle.py`, `evaluate.py`, `trainer.py`, the renamed valcheck, and
`__pycache__`. No `runs/`, `results/*_test.json`, or cache was changed.

---

## 5. Tests and validation actually run

### 5.1 Pytest

```
cd experiments/08_graph_kd/src
PYTHONPATH=. /home2/.../gateir/bin/python -m pytest ../tests -q
28 passed, 1 warning in 2.65s
```

### 5.2 Custom evaluator vs stock Ultralytics validator (CPU, same weights/split)

```
A0 test: custom AP 0.251848  vs stock 0.242906  (+0.008942)
A0 val : custom AP 0.344062  vs stock 0.335985  (+0.008077)
```

Both use square 640 letterbox in the custom path; the stock validator’s default
`rect` is the likely source of the small gap. All arms in the report’s test
table use the custom path, so relative deltas remain valid.

### 5.3 FLIR teachers on one pipeline (custom evaluator, CPU)

| Model | Split | AP | AP50 | AP75 |
|---|---|---:|---:|---:|
| IR teacher | val | 0.4115 | 0.7296 | 0.3878 |
| IR teacher | test | 0.3143 | 0.5938 | 0.2758 |
| RGB teacher | val | 0.2950 | 0.5784 | 0.2537 |
| RGB teacher | test | 0.2251 | 0.4667 | 0.1888 |
| Fused best | val | 0.3765 | 0.7001 | 0.3415 |
| Fused best | test | 0.2817 | 0.5511 | 0.2475 |
| Fused last | val | 0.3783 | 0.7010 | 0.3449 |
| Fused last | test | 0.2840 | 0.5547 | 0.2491 |

**Corrected conclusion:** IR > fused > RGB on both splits; fused is 0.033
below IR and 0.083 above RGB on val. The old 0.206/0.078 row is invalid.

### 5.4 Paired bootstrap (FLIR test, 609 images)

Paired image resampling; fast COCO-style AP validates against pycocotools to
≤0.0008 on all five arms. Point AP:

| Arm | Point AP | Bootstrap mean | 95% CI |
|---|---:|---:|---:|
| A0 | 0.2510 | 0.2518 | [0.2425, 0.2621] |
| A3 | 0.2584 | 0.2597 | [0.2480, 0.2718] |
| A5 | 0.2568 | 0.2592 | [0.2473, 0.2721] |

500 paired resamples of the 609 test images:

```
A3 - A0: point +0.0073, 95% CI [-0.0008, +0.0160], P(>0)=0.968
A5 - A0: point +0.0058, 95% CI [-0.0018, +0.0186], P(>0)=0.932
```

Both 95% CIs include zero. **Do not claim a significant graph-KD gain.**

### 5.5 DroneVehicle label audit (all frames)

| Split | `*label` boxes | `*labelr` boxes | Frames equal | YOLO IR matches |
|---|---:|---:|---:|---|
| train | 286,789 | 316,408 | 3,108/17,990 | `label` 17,962/17,962 |
| val | 22,462 | 24,490 | 251/1,469 | `label` 1,469/1,469 |
| test | 143,313 | 159,613 | 1,545/8,980 | `labelr` 8,980/8,980 |

This is the single most consequential finding: train/val are visible, test is
thermal.

### 5.6 FLIR split audit

| Split | Frames | Range | Gaps >1 |
|---|---|---|---:|
| train | 4,129 | 2–8,859 | 787 |
| val | 404 | 8,864–9,499 | 57 |
| test | 609 | 9,512–10,227 | 52 |

Val and test are the two halves of the paper-validation sequence
(8864–10227); test→val nearest distance is 13 frames at the boundary, median
373 frames. All images are 640×512.

### 5.7 Cropped-RGB alignment (stratified)

| Subset | n | median |dx| | median |dy| | frac >2 px |
|---|---:|---:|---:|---:|
| cropped (667) | 60 | 4 px | 6 px | 93.3% |
| not cropped (4,475) | 60 | 2 px | 2 px | 53.3% |

1/30 cropped images still had a residual non-content border. The crop repair is
not alignment-neutral.

### 5.8 DroneVehicle test label edge cases

159,616 parsed objects; 16 with no points and 3 degenerate dropped → 159,613
written; 22,166 (13.9%) edge-clipped; 149,975 four-point polygons and 9,641
two-point `bndbox` objects.

### 5.9 DV label-policy impact (audit-only, thermal labels)

`audit/dv_ir_labelr_val/` uses hardlinked images and `vallabelr` labels
(24,490 boxes). The earlier symlink attempt silently reused the visible labels
(22,462 boxes), which is why it is preserved as
`*_VISIBLE_SYMLINK_COPY.json`; the hardlink run is the valid one.

| Model | Visible-label val (reported) | Thermal-label val (audit) | Δ |
|---|---:|---:|---:|
| `dv_ir_teacher` | 0.5484 | **0.5263** (AP50 0.7420, AP75 0.6249) | −0.0221 |
| `dv_a0_gt` | 0.4348 | **0.4686** (AP50 0.6803, AP75 0.5520) | +0.0338 |

Stock-validator calibration on the visible-label val: `dv_ir_teacher`
0.548145 / AP50 0.799791 / AP75 0.641249, matching its `results.csv` best
(0.5484 / 0.8000). The reported teacher val is therefore a valid stock
visible-label number; it is simply the wrong label policy for an IR benchmark
relative to the thermal test set.

The label-policy effect is **model-dependent and not a constant offset**: the
teacher drops ~0.022, the student rises ~0.034. The thermal labels contain more
objects (24,490 vs 22,462), so a model trained on the sparser visible labels can
be credited for detections that were unannotated in the visible set. This is
exactly why the mixed protocol cannot be corrected by a calibration factor:
the correct fix is to train and select on the same thermal annotations used for
the test set. (`dv_a0_gt`: visible trainer val 0.4348 → thermal custom val
0.4686; `dv_ir_teacher`: 0.5484 → 0.5263.)

### 5.10 Reproduced matcher defects

```
hungarian([[nan,1],[2,3]]) -> identity [[1,0],[0,1]]
sinkhorn([[nan,1],[2,3]])   -> all NaN
hungarian([[inf,1],[2,3]])  -> [[0,1],[1,0]]
```

`KDModule` then applies `nan_to_num`, so a NaN matcher result silently zeros the
KD loss for that batch.

### 5.11 Fix verification M1–M12

| Fix | Current evidence | Regression coverage |
|---|---|---|
| M1 `--ntasks=1` | `scripts/relaunch_orchestrator.sh:10` | none (process) |
| M2 dense autograd | `candidates.py` no detach; test | `test_dense_maps_keep_autograd` |
| M3 letterbox GT | `evaluate.py:168–176` | `test_unletterbox_640x512`, `test_letterbox_roundtrip_smaller_image` |
| M4 top-left COCO | `evaluate.py:177–183` | `test_gt_as_pred_single_image_ap_is_one` |
| M5 `testlabelr` | current test labels match labelr | builder fixed; `test_dronevehicle_uses_thermal_labels_for_ir_only` |
| M6 fused val | `train.py:75`, paired validator | no automated trainer test; paired-val log |
| M7 OOM | `cache=none`/`disk` in args; OOM logs | none |
| M8 DONE | `parallel_orchestrator.py:129–135` | none |
| M9 absolute Python | `relaunch_orchestrator.sh` `$PY`; `sys.executable` | none |
| M10 cancellation | `slurmstepd ... CANCELLED` logs; no code | none |
| M11 expiry watcher | no code; report-only | none |
| M12 temp/CUDA | `tempfile.mkstemp`; `dev.type` guard; CPU JSONs | CPU audit runs |

---

## 6. Results to quarantine or re-run

### Quarantine (do not cite)

| Artifact | Reason |
|---|---|
| `logs/fused_teacher_val_cpu.log` (0.206/0.598/0.078) | pre-fix evaluator; not reproducible |
| `results/a0_gt_valcheck_val_INVALID_pre_fix_evaluator.json` | AP ≈ 1e-5; already renamed |
| `results/dv_a0_gt_test_WRONG_GT.json` | wrong GT; already named |
| `results/audit_*` | audit-only; not headline results |
| `runs/a1_pred` | dense maps detached; invalid |
| `runs/a1_pred_fix` | 1 epoch; abandoned |
| `runs/a2_node` (FLIR) | 1 epoch; stopped |
| `runs/dv_fused_teacher_v2` | `val=False`; zero metrics; stopped |
| Future `results/dv_{ir_teacher,a0_gt,a1_pred,a2_ir_control,a2_node,a3_node_edge,b_consensus,fused_teacher}_test.json` | trained on visible labels, tested on thermal labels; invalid until retrained |

`dv_rgb_teacher_test.json` will be valid **on visible test labels**; it is the
only DV test result that is internally consistent.

### Minimum re-run

1. Rebuild DV `yolo/ir/labels/{train,val}` from `*labelr`; keep RGB labels from
   `*label`; delete `*.cache`; verify `yolo` IR labels match labelr on all
   frames.
2. Regenerate DV teacher caches; retrain `dv_ir_teacher` and
   `dv_fused_teacher` (RGB teacher labels are unchanged, but rebuilding is
   cheap to keep provenance consistent).
3. Retrain all DV IR student arms (A0, A1, A2_ir, A2_node, A3, B) and re-run
   the corrected test evaluation.
4. For FLIR, keep the corrected A0/A1/A2/A3/A5 test numbers but add the CIs and
   a sequence-held-out test; re-evaluate the fused teacher (done) and rerun the
   go/no-go on 0.378 vs 0.412.
5. Add ≥3 seeds and the controls in F7/F8 before any significance claim.

---

## 7. Unresolved items and inaccessible information

- **Exact command/weights behind the old 0.206 log are not recorded.** The
  mtime (01:38) precedes the evaluator fix (06:48), and the current checkpoints
  do not reproduce it; this is the best-supported explanation, but the command
  itself is unavailable.
- **Slurm job history for 10090/10096** is not available; only logs remain.
  M10/M11 are therefore [REPORT-ONLY] as process narratives.
- **Train/val sequence identity on FLIR is not provable from the release
  metadata.** The 5-frame gap is consistent with a sequence boundary; val/test
  is confirmed to be one sequence.
- **DV labelr-val and stock-validator calibration** were still running at the
  time of writing; numbers are in `audit/dv_labelr_val_eval.log` when complete.
- **B consensus, A2_node, A3** were still running in job 10133; no final DV
  test evaluation exists.
- **No GPU allocation** was used for this audit; retraining and seed runs are
  proposed, not executed.
- **Visual overlay inspection** was not performed (no image viewer); alignment
  was quantified with Canny edge shifts, not human inspection.
- **The two-stream fusion teacher** mentioned in the report’s future work does
  not exist in the codebase; no file was found.
- **No git history** exists; diffs are relative to the audit manifest/tarball.
- **The old `a1_pred` identical-to-A0 curve** is explained by the detached dense
  maps; the artifact is retained but not repaired.

---

## 8. Ordered next steps (cheapest and most decisive first)

1. **Rebuild DV IR train/val labels from `*labelr` and retrain** (F1). This is
   the only way to make any DV IR/fused number meaningful; it is also the most
   expensive item, so start it before further modeling.
2. **Adopt the corrected FLIR fused numbers** (F2): 0.376–0.378 val /
   0.282–0.284 test; re-run the go/no-go on 0.378 vs IR 0.412 and stop calling
   early fusion a localization failure.
3. **Fix the FLIR evaluation protocol** (F3/F4): either hold out a real sequence
   or relabel the current test as temporal; report all numbers from one
   evaluator with CIs.
4. **Add controls and seeds** (F6/F7/F8): augmented A0, A3 with the IR teacher
   on DV, A5 with `kd_every=4`, A1 with the IR teacher, seeds 1/2.
5. **Repair or exclude the 667 cropped RGB pairs** (F5/F16), then retrain the
   fused teacher; re-check the 6-channel stem order (fixed) in the new run.
6. **Harden caches/matchers** (F14/F15): verify meta on read; raise/log on
   non-finite costs instead of silently zeroing KD.
7. **Finish the live trimmed baseline** (A2_node, A3, B) only as a
   visible-label reference; do not promote its test numbers to IR claims.
8. **Update `PROJECT_REPORT.md` and `results/SUMMARY.md`** with the corrected
   tables, CIs, and the quarantine list; keep the historical numbers clearly
   marked as superseded.

---

## Appendix A — Audit artifacts

| Path | Contents |
|---|---|
| `audit/manifest_baseline.sha256` | pre-audit hashes of src/configs/scripts/results |
| `audit/experiment08_src_*.tar.gz` | pre-audit source/results tarball |
| `audit/eval_compare.log` | custom vs stock A0 val/test |
| `audit/fused_eval.log` | fused best/last val/test |
| `audit/teacher_eval.log` | IR/RGB teacher val/test |
| `audit/dv_label_audit.log` | full DV label-source and per-class audit |
| `audit/bootstrap_flir.log` | paired bootstrap point AP + CIs |
| `audit/dump_preds/` | per-image GT/detections for A0/A1/A2/A3/A5 test |
| `audit/dv_labelr_val/` | hardlinked images + thermal val labels (audit only) |
| `audit/dv_labelr_val_eval.log` | thermal-val and stock-validator calibration |
| `audit/run_eval_compare.sh`, `run_fused_eval.sh`, `run_dv_labelr.sh` | reproducible commands |

## Appendix B — Key code references

- `src/graphkd/training/evaluate.py`: `resolve_device`, `unletterbox`, `coco_eval`.
- `src/graphkd/training/trainer.py`: `_adapt_first_conv`, `_adapt_stem`, `GraphKDTrainer.get_validator`.
- `src/graphkd/data/build_dronevehicle.py`: `LABEL_DIRS`, `_parse`, per-modality labels.
- `src/graphkd/graph/cache.py`: `TeacherCache` (no meta validation).
- `src/graphkd/matching/{hungarian,sinkhorn,fused_gw}.py`: matcher edge behavior.
- `tests/test_audit_*.py`: 21 new regression tests.
