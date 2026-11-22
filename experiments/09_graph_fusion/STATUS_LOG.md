# STATUS LOG — Experiment 09 (running log for autonomous operation)

Dated, append-only. Blockers get recorded here (with a one-line impact and
the workaround/next-step taken), then work moves on. Live job state also
lives in README.md "Current state"; this file is the chronological record.

## 2026-11-21

- **~14:55 UTC** — 08 audit: all running arms distil into IR-only students
  (fused teacher / IR+RGB consensus); user approved stopping them.
- **15:32 UTC** — `scancel 10133.1` (step only, allocation 10133 kept).
  08 paused resumable: `dv_a2_node` DONE ep77 0.4317, `dv_a3_node_edge` ep40
  0.44139, `dv_b_consensus` ep17 0.41937. No files deleted.
- **16:04 UTC** — E00 `e00_ir` + E01 `e01_rgb` launched (step 10133.2,
  `scripts/run_refs.py`), YOLO12s/640/b16/seed0/patience30, two-at-a-time on
  GPU0.
- **16:17 UTC** — Phase 0.6 measured from the running refs [VERIFIED]:
  ~4 min/epoch each (concurrent), ~7.8 GB/run, 9,255,071 params, 1,123
  iters/epoch → `audit/phase06_timing.json`. sigma_seed NOT measured
  (single-seed minimal pass; 0.5 pp substitute rule per plan §5.4).
- **16:08→16:25 UTC** — E02 launched; **BLOCKER (resolved, no action needed)**:
  the second GPU srun step does not start while step 10133.2 holds the
  allocation's GPU GRES (this Slurm does not run two overlapping GPU steps;
  08's three arms shared ONE step). The E02 srun client stays queued and will
  auto-start when the refs finish. GPU0 compute is already ~99% with two
  runs, so concurrency would not add throughput anyway.
- **16:30 UTC** — E03 pipeline built and tested [VERIFIED]: COCO GT for
  DV v2 val/test × ir/rgb (`results/gt/`, counts match the builder audit
  exactly), `scripts/predict_export.py`, `src/gfyolo/wbf.py`,
  `scripts/late_fusion_e03.py`; 16/16 unit tests pass. E03 calibration will
  run on VAL only once E00/E01 finish; test scoring only via lock.
- **16:35 UTC** — GF-YOLO testbed modules (E10/E14/E21/E30/E35) delegated to
  the approved `deepseek-v4.1-flash` subagent in the dedicated
  `testbed_gfyolo/` workspace (pure torch + unit tests, no ultralytics
  integration); review pending on completion, integration by main agent.
- **20:45 UTC** — E00 + E01 **DONE** [VERIFIED]:
  - E00 `e00_ir`: 85 epochs (patience), **best ep55 val AP50-95 0.63055,
    AP50 0.84234** (thermal GT), wall 4.14 h, `best.pt` saved.
  - E01 `e01_rgb`: 100 epochs, best ep84 val AP50-95 0.58822, AP50 0.81318
    (visible-GT training val; thermal-GT grading happens at final eval).
- **20:45 UTC** — E02 restart (experiment management): the first launch
  (step 10133.3) was dataloader-bound at ~9.5 min/epoch (8 workers, GPU 55%)
  → would have overrun the allocation wall. Renamed the 3-epoch partial to
  `runs/e02_fused_PARTIAL_w8` (preserved, not deleted), relaunched as step
  10133.4 with 16 workers → **10.3 it/s ≈ 1.8 min/epoch** (GPU 65%), ETA
  ~23:00 UTC.
- **00:08 UTC (Nov 22)** — E02 **DONE** [VERIFIED]: 100 epochs (no plateau —
  still climbing at ep100, recipe-capped like E01), best ep100 val AP50-95
  **0.50471**, AP50 0.73375 (paired validator, thermal GT). Early fusion
  well below IR-only (0.505 vs 0.631), consistent with 08's finding. The
  queued CPU predict step auto-fired and exported E00/E01 val predictions.
- **00:10 UTC** — E03 WBF calibration launched (step 10133.6, 12-config grid
  on VAL only). Test locks created BEFORE any test access for the three
  finished runs [VERIFIED]: `LOCK_e00_ir` (best ep55, 0.63055),
  `LOCK_e01_rgb` (best ep84, 0.58822, visible-GT val; thermal grading at
  final eval), `LOCK_e02_fused` (ep100, 0.50471). E03's lock follows its
  val calibration. `scripts/predict_export_paired.py` written for E02's
  6-channel test export (uint8-range tensors; ultralytics normalises
  internally).
- **20:47 UTC** — discovered: this Slurm **serializes steps** within an
  allocation (CPUs as well as the GPU gres; `--overlap` does not grant
  concurrent steps). The E03 val-prediction CPU step is therefore QUEUED in
  `srun_s` and will auto-start when E02's step ends. Monitor set to alert
  when E02 DONE + preds exist (then E03 val calibration runs).
- **16:45 UTC** — screening subset built [VERIFIED]: 6,000/17,962 train
  frames (33.4%) in 15 contiguous ON=400/OFF=800 blocks;
  `data/screening/dv_v2_train_on400_off800.txt` + 3 screening yamls (ir/rgb/
  paired) + `screening_audit.json`. List entries verified to point only at
  existing `images/train/*` files (DV stems are per-split indices, so stem
  collisions across splits are different files — checked inodes). First build
  had a block-logic bug (kept 100%); fixed and regenerated before any use.
- **17:00 UTC** — subagent testbed REVIEWED and ACCEPTED [VERIFIED]:
  77/77 tests re-run by me; 6 independent probes passed (E14 identity at
  init 1.2e-7, gradient flow, E30 corner 3/4 vs interior 8/9 border counts,
  semantic ids in range + deterministic + border validity, E21/E35 fp16
  finite with proper `module.half()`); boundary check: nothing outside
  `testbed_gfyolo/` modified, 08 tree untouched. Accepted caveats: gate
  identity uses -4.0 bias + tiny output init (documented); PyG parity test
  blocked (PyG not installed, no new packages allowed); E30 leaves the
  intra-RGB grid to a later rung; graph modules are CPU-slow at P3 sizes
  (fine on GPU). Integration into the dual-stream GF-YOLO model is the next
  queued step, done by the main agent, not the subagent.

- **01:10 UTC** — E03 WBF calibration **DONE** [VERIFIED] (VAL, thermal GT,
  our locked evaluator): E00 0.58757; E01 graded on thermal GT 0.43578;
  best WBF (iou 0.6, w 1.5:1, members-scaling) **0.58833 — only +0.08 pp
  over E00**, i.e. free late fusion has **no detectable gain** over IR-only
  on DV val. `results/e03_val_report.json`; E03 lock created
  (`results/e03_config.json` manifest + `LOCK_e03_late_fusion.json`).
- **Protocol caveat recorded**: E00's ultralytics training val (0.63055,
  rect letterbox, its NMS/eval) differs from our locked export protocol
  (0.58757, square letterbox + pycocotools). All cross-model comparisons
  and final numbers use the locked protocol only; per-model ultralytics val
  is just early stopping/checkpoint selection.
- **01:02 UTC** — **BLOCKER (fixed)**: first test-export step OOM'd on GPU.
  Root cause [VERIFIED]: ultralytics predict warmup forwards the ENTIRE
  source list as one batch — `[8980,3,512,640]` ≈ 35 GB. The CPU val export
  only survived because 5.7 GB of host RAM absorbed the same warmup.
  Fix: `predict_export.py` now chunks the split into per-`--batch` predict
  calls (identical results, small warmups) and forces fp32 (`half=False`)
  to match the CPU-produced val predictions' protocol. Added a clean
  `scripts/score_preds.py` sanity scorer; relaunched as step 10133.8.

- **01:06 UTC** — second exporter bug found and fixed [VERIFIED]: E02 paired
  val sanity AP was 0.002 → root cause: ultralytics **tensor-source** predict
  expects already-normalised 0–1 tensors (its preprocess does "no division"
  for tensors, unlike the file path), so the 6-ch uint8 inputs were fed
  255× overscaled. Fixed (`predict_export_paired.py` divides by 255; tensor
  sources also skip letterbox, and the native 640×512 is already
  stride-multiple so geometry matches training). Invalid output renamed
  `results/preds/e02_fused_val_INVALID_uint255.json` (kept, not deleted).
  Relaunched as step 10133.9: val sanity score gates the test export (`&&`).

- **01:15 UTC** — deeper diagnosis of the E02 exporter [VERIFIED]: the
  0.002 AP was NOT a scaling bug — ultralytics auto-divides raw tensors by
  255 (both runs had identical, correct model inputs, hence identical
  outputs). Real bug: the paired dataset letterboxes to a SQUARE 640×640
  (64 px pads) and ultralytics reports tensor-source boxes in that padded
  frame, while the GT is in the original 640×512 frame → every box
  misaligned by the pad → AP 0.002. Fix: `predict_export_paired.py`
  un-letterboxes boxes via `eval.geometry.unletterbox` to each sample's
  `ori_shape`; verified on-image (mean best-IoU 0.845 vs GT at conf 0.25).
  The invalid val+test outputs are renamed
  `*_INVALID_lbframe.json` (kept). NOTE: one invalid test export did touch
  the test split before the gate existed — nothing was selected on it;
  regenerated under the new `--min-ap 0.30` sanity gate
  (`score_preds.py --min-ap` exits nonzero so the `&&` chain stops before
  the test export). Relaunched as step 10133.10.

- **01:15 UTC** — E02 paired exporter FIXED and verified [VERIFIED]: val
  sanity AP **0.46847** (locked protocol; consistent with the paired
  validator's 0.50471 minus the same ~4 pp protocol gap as E00). E02 test
  export completed under the `--min-ap 0.30` gate.
- **01:17 UTC** — final leg launched (step 10133.11): E03 test merge
  (val-calibrated config) + the four official locked test evaluations
  (e00_ir, e01_rgb thermal-graded, e02_fused, e03_late_fusion) via
  `scripts/final_eval.py` against `dv_v2_ir_test.json` (thermal GT).

- **02:23 UTC** — **REFERENCE TIER COMPLETE** [VERIFIED]. All four locked
  test evals passed the locks and finished (thermal GT, DV v2 test, 8,980
  frames):
  - E00 IR-only: **AP50-95 0.57828** (AP50 0.76281) — the reference.
  - E01 RGB-only (thermal-graded): 0.41264 — weak; visible/thermal label
    disagreement costs ~15 pp vs its visible-GT val.
  - E02 early fusion: 0.47619 — **10.2 pp below IR-only**; still climbing at
    its recipe-capped ep100, so partly under-training, but directionally
    confirms 08's finding.
  - E03 WBF late fusion: **0.57939** — +0.11 pp AP50-95 over E00 (below the
    0.5 pp line = "no detectable gain") but +2.40 pp AP50 / +1.00 pp AP75
    and +1.6 pp on truck; bootstrap pending before any stronger claim.
  Full table + per-class APs in RESULTS.md; ledger EXPERIMENTS.csv updated.
  Remaining in this allocation (ends 08:13): Phase 1.2 oracle (CPU),
  E00-vs-E03 paired bootstrap, GF-YOLO integration work.
- **03:00-03:13 UTC** — **GF-YOLO integrated by the main agent** [VERIFIED]
  (`src/gfyolo/model.py`, `gf_trainer.py`, `scripts/train_gfyolo.py`): one
  renumbered ModuleList (IR backbone 0-8, RGB 9-17, Fuse 18-20, neck 21-32
  with remapped from-indices) so ultralytics' own forward/fuse/loss/stride
  machinery runs unchanged; dual-entry `_predict_once` (IR at 0, RGB at 9,
  fuse modules called with two positional inputs); both backbones +
  neck/head load pretrained yolo12s (IR stem grayscale-adapted per plan).
  CPU smoke: E10 15.49M / E14 16.28M params, forward+backward clean,
  strides asserted [8,16,32]. Two launch bugs fixed immediately (fuse kwarg
  `d` passed to E10; screening txt needed absolute paths per modality).
  **E10 screening training live** (step 10133.14) with E14 chained; both
  resume-safe from `last.pt`. ETA: E10 ~2 h, E14 ~2.2 h — the allocation
  wall (08:13) may cut E14; auto-resume in the next allocation.
- **02:45 UTC** — Phase 1.2 complementarity oracle RUN [VERIFIED]
  (`results/oracle_dv_val.json`): oracle gain **+3.18 pp** overall (IR
  recall 0.918, RGB 0.798, best-of-either 0.950), **+6.73 pp on small
  objects**, day +4.34 / night +2.25. RGB matches IR in day (0.913 vs
  0.916), collapses at night (0.707). Real fusion headroom exists; E03
  captured it only at AP50 (+2.40 pp), E02 lost 10 pp. Bootstrap for the
  +0.11 pp E00-vs-E03 AP50-95 delta is NOT needed (already below the
  pre-registered no-gain line); it is reserved for the ladder finalists.

## Queue (autonomous)

1. Step 10133.7: E00/E01 test exports + E02 paired val export (sanity-scored).
2. If E02 val export is sane: E02 test export → E03 `--emit-test` →
   `final_eval` × 4 (official locked test numbers).
3. RESULTS.md reference table (val + locked test) + README/STATUS updates.
4. Phase 1.2 complementarity oracle from the existing E00/E01 val preds (CPU).
5. GF-YOLO integration into `src/gfyolo/` (next allocation; main agent).

## Blockers / constraints (active)

- 10157 (`girkd`) pending (Priority), est. start 2026-11-25 22:20 — the E10+
  ladder waits for it; only E00–E03 fit in allocation 10133 (ends
  2026-11-22 08:13).
- Tier A externals blocked (no accessible weights; gateir-only policy).
- Single overlapping GPU step per allocation (see 16:08 entry).
