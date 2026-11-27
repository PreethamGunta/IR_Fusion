# Experiment 10 — IR-only distillation from the E30 RGB–IR teacher

**Status:** code complete, CPU smoke-tested on the dev node; GPU smoke (`N0`) runs
on the target node before any long job. Teacher: `09_graph_fusion` E30
(`runs/e30_v2_100ep/weights/best.pt`). Students: stock YOLO12s / YOLO12n.

This experiment distils the **fused** teacher's knowledge into a **pure IR-only
student**: zero RGB input, zero graph/teacher overhead at inference. The student
checkpoint is a stock Ultralytics `DetectionModel`, loadable with
`YOLO(ckpt)` and exportable with `09_graph_fusion/scripts/predict_export.py
--modality ir`.

---

## 1. Why this can work

The teacher's fused feature at level `l` is an **IR-anchored correction**:

```
F_fused_l = F_ir_l + Δ_l           (FuseGATLocal, verified to 1e-4)
```

`F_ir_l` is what an IR-only backbone can represent; `Δ_l` is the cross-modal
information the graph added. Decomposed KD therefore has two separately
supervisable targets:

1. **preservation** — keep the student's IR features close to the teacher's IR
   features (the part the student can already represent);
2. **compensation** — make the student's features *predict the correction* the
   teacher would have written, so the student learns to hallucinate the useful
   part of the missing modality **from IR alone**.

Direct KD (align student to `F_fused`) mixes the two and, empirically, tends to
pull the student away from its strong IR prior. The decomposed arm is the core
claim; direct is the ablation.

---

## 2. Losses

For scale `l ∈ {P3, P4, P5}`, student feature `S_l` (C_s channels), teacher
outputs `T_ir_l`, `Δ_l`, `T_fused_l = T_ir_l + Δ_l` (C_t channels), and
train-only 1×1 projectors `Proj(·)`:

```
cos_huber(A, B) = 0.5 * mean(1 - cos(A, B)) + 0.5 * SmoothL1(normalize(A), normalize(B))
```

where the cosine is per spatial location over channels.

**`none`**

```
L = L_det
```

**`direct`**

```
L = L_det + w_kd * λ_dir * mean_l cos_huber(Proj_l(S_l), T_fused_l)
```

**`decomposed`** (default)

```
L_pres = mean_l cos_huber(Proj_P_l(S_l), T_ir_l)

m_l    = ||Δ_l||_2                       (per spatial location)
w_l    = 1 + α * m_l / mean(m_l)         (α = 1.0)

L_comp = mean_l  Σ w_l * [0.5(1-cos(Proj_C_l(S_l), Δ_l)) + 0.5 SmoothL1(...)]
                 ───────────────────────────────────────────────────────────
                                Σ w_l

L = L_det + w_kd * (λ_pres * L_pres + λ_comp * L_comp)
```

Defaults: `λ_pres = 0.25`, `λ_comp = 0.50`, `λ_dir = 1.0`, `α = 1.0`,
`w_kd = min(1, epoch / 3)` (0 → 1 over the first three epochs).

`L_det` stays dominant by construction. The spatial weight `w_l` makes the
residual loss stronger exactly where the graph made a larger RGB→IR correction
(typically cluttered/small-object regions), instead of averaging over
background where `Δ ≈ 0`.

---

## 3. Architecture & checkpoint purity

```
training batch (6ch [IR|RGB], V2 paired loader)
   ├─ teacher (E30, frozen, eval): 6ch ─► T_ir, Δ, T_fused       no_grad
   └─ student (stock YOLO12s/n):   IR ─► S_P3/S_P4/S_P5 ─► det head
                                             │
                      Proj_P / Proj_C (trainer-owned, never in the model)
```

- Projectors live on the **trainer**, not on `student.model`.
- The KD loss is attached by **replacing the trainer's loss callable**
  (`model.loss`) after `ModelEMA` has been created; Ultralytics saves `"model":
  None` and serialises only the EMA, so neither the projectors, the hooks nor
  the teacher can enter `best.pt` / `last.pt`.
- Validation uses the **IR-only** dataset (`modality="ir"`), so val AP is true
  IR-only AP, not fused AP.
- The projector state is mirrored to `<run>/kd_proj.pt` every epoch (and loaded
  on resume) so an interrupted KD run continues with its projectors.

---

## 4. Data & recipe

- Dataset: `scripts/build_dv_from_raw.py` rebuilds the exact audited v2 tree
  from `data/dronevehicle/raw` using the committed stem manifest
  (`assets/dv_v2_splits.json`) and the verified crop + JPEG q92 recipe
  (bit-exact against the 09 tree); the queue runs it automatically when
  `DV_ROOT` is missing. The 5.5 GB prebuilt tree can also be copied instead.
- Loader: `09_graph_fusion/src/gfdata/unified_v2.make_dataset_v2`
  (`*labelr` thermal labels; the V2 loader is the one verified bit-identical to
  stock IR training). **Never** the 08 dataset.
- Train: `modality="pair"` (6ch), val: `modality="ir"` (3ch).
- Recipe identical to E00/E30: MuSGD, `lr0=0.01`, `lrf=0.01`, momentum 0.937,
  weight decay 5e-4, warmup 3 ep (`warmup_bias_lr=0.1`, momentum 0.8), batch 16,
  640 px, `close_mosaic=10`, patience 30, seed 0, deterministic, workers 8,
  `cache=False` (never RAM-cache 18k images — 08 M7).

---

## 5. CLI

```bash
cd experiments/10_ir_distillation
PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python   # node-specific

# smoke (2 epochs)
$PY scripts/train_student_kd.py --name smoke_12s_decomp \
  --student-cfg /path/models/pretrained/yolo12s.pt \
  --teacher-ckpt ../09_graph_fusion/runs/e30_v2_100ep/weights/best.pt \
  --kd-mode decomposed --epochs 2 --close-mosaic 0 --patience 2

# full run
$PY scripts/train_student_kd.py --name student_12s_decomp \
  --student-cfg /path/models/pretrained/yolo12s.pt \
  --teacher-ckpt ../09_graph_fusion/runs/e30_v2_100ep/weights/best.pt \
  --kd-mode decomposed --epochs 100 --close-mosaic 10
```

`--resume` is automatic when `<project>/<name>/weights/last.pt` exists; pass
`--fresh` to start over.

---

## 6. Queue (target node)

```bash
cd experiments/10_ir_distillation
# target node: point PY at that node's conda env, and DV_ROOT if the dataset
# is not at experiments/09_graph_fusion/data/dronevehicle_v2/yolo
PY=/path/to/env/bin/python DV_ROOT=/path/to/dronevehicle_v2/yolo \
  bash scripts/run_node2_queue.sh
# optional: --srun-jobid 12345 (only if launching through srun)
```

The queue derives data yamls from `DV_ROOT` at start-up
(`scripts/make_data_configs.py`), so no absolute path is baked into the code.

| Stage | Run | Student | KD | Est. |
|---|---|---|---|---:|
| N0 | tests + 2-epoch smoke | 12s | decomposed | 0.25 h |
| N1 | `student_12s_decomp` | 12s | decomposed | ~6 h |
| N2 | `student_12n_gt` | 12n | none | ~3.5 h |
| N3 | `student_12n_decomp` | 12n | decomposed | ~4 h |
| N4 | `student_12s_direct` | 12s | direct | ~6 h |
| N5 | eval + profiling | — | — | ~1 h |
| N6 | `student_12s_decomp_seed1` | 12s | decomposed | ~6 h |

`N0–N5 ≈ 20.8 h` fits a 22 h window only with early stopping; N6 is spillover
and is cut-safe (`last.pt` + `kd_proj.pt` resume). The KD runs are slower than
plain E00 because the frozen teacher adds a second forward per step (~25–35 %).

---

## 7. Evaluation protocol

All reported numbers use the locked 8,980-image DroneVehicle test split and the
`09_graph_fusion` evaluator, **without modification**:

```
scripts/export_ir.py (self-contained; conf 0.001, NMS 0.6, maxDet 100)
  → 09 make_lock.py (val-selected best.pt; checkpoint + config SHA-256)
  → 09 final_eval.py (pycocotools)
  → 09 paired_bootstrap.py vs e00_ir and vs e30_v2_100ep
```

Experiment 10 depends only on **tracked** 09 modules (`gfdata.unified_v2`,
`gfyolo.gf_trainer`, `eval.coco_eval`, the three eval scripts). It does not
require any uncommitted 09 change, so it can be pushed on its own while the 09
queue keeps running.

Results land in `results/`; the table template is in §9.

---

## 8. 08-graph-kd mistakes this design must not repeat

| ID | Failure | Guard here |
|---|---|---|
| M1 | parallel `srun` duplicated a run | queue runs `--ntasks=1` when srun is used; direct mode otherwise |
| M2 | detached student maps → no KD gradient | unit test asserts non-zero grads on projectors **and** backbone |
| M3/M4 | letterbox/frame confusion in GT/preds | reuse the audited 09 exporters/evaluator; no new coordinate code |
| M5 | visible labels on IR test | V2 loader (`*labelr`) only |
| M6 | fused teacher without validation | teacher is frozen and audited; students validate IR-only every epoch |
| M7 | RAM-cache OOM | `cache=False` everywhere |
| M8 | DONE markers conflated with process success | queue separates train success (`rc==0`) from eval artifacts |
| M9 | wrong interpreter | queue pins `PY` and prints `sys.executable` |
| M10 | broad `pkill` killed the wrong step | no process killing in the queue |
| M11 | watcher sampled before expiry | no watcher needed; direct execution |
| M12 | fixed `/tmp` GT path race | 09 evaluator uses `tempfile.mkstemp` |

---

## 9. Results table (fill after N5)

| Run | Student | KD | Params | Latency (ms) | Test AP | AP50 | Δ vs E00 | Δ vs E30 |
|---|---|---|---:|---:|---:|---:|---|---|
| E00 (IR-only) | 12s | — | 9.26 M | 6.97 | 0.5783 | 0.7628 | — | −2.08 |
| E30 (teacher) | 12s+graph | — | 15.35 M | 14.28 | 0.5988 | 0.7819 | +2.08 | — |
| `student_12s_decomp` | 12s | decomposed | 9.26 M | 6.97 | | | | |
| `student_12s_direct` | 12s | direct | 9.26 M | 6.97 | | | | |
| `student_12n_gt` | 12n | none | 2.6 M | ~2.5 | | | | |
| `student_12n_decomp` | 12n | decomposed | 2.6 M | ~2.5 | | | | |

**Claim to test:** `student_12s_decomp` > `e00_ir` at identical inference cost,
and `student_12s_decomp` > `student_12s_direct` (decomposed > naive).


---

## 10. Implementation notes (verified on CPU)

- **Backbone anchors are not the neck widths.** For YOLO12s the anchors are
  layers `[4, 6, 8]` with `[256, 256, 512]` channels; for YOLO12n they are
  `[128, 128, 256]`. The teacher fuses at exactly these anchors, so the
  projectors map `student_anchor → teacher_anchor` per level. The code probes
  both at runtime; nothing is hardcoded.
- **Student stem is forced to 3 channels.** The paired data yaml declares
  `channels: 6`; Ultralytics' stock `get_model` would build a 6-channel stem.
  `KDDetectionTrainer.get_model` builds `DetectionModel(..., ch=3)` explicitly.
- **Checkpoint purity is structural.** Ultralytics 8.4.138 saves
  `"model": None` and serialises only `ModelEMA`. The EMA is created before the
  loss is patched and before hooks are attached, so `best.pt` / `last.pt` are
  stock `ultralytics.nn.tasks.DetectionModel` objects: verified 9,255,071
  params (12s) / 2,569,023 (12n), first conv `(32,3,3,3)` / `(16,3,3,3)`, no
  `irkd` modules, `YOLO(ckpt)` loads and runs 3-channel input.
- **Resume.** Each epoch writes `<run>/kd_proj.pt` next to `last.pt`; a resumed
  run restores the projectors (`[irkd] resumed projectors` in the log) and the
  optimizer/scheduler/EMA from `last.pt`.
- **Completed runs are stripped** (`epoch=-1`, optimizer/EMA moved to `model`);
  such a checkpoint cannot be resumed — start a new directory instead.
- **Validation columns.** Ultralytics initialises the validator's loss dict with
  `zeros_like(loss_items)`, so `val/kd_*` columns are always 0; only
  `train/kd_pres`, `train/kd_comp`, `train/kd_w` are meaningful.
- **`--no-val` still validates once** at the final epoch (Ultralytics
  behaviour); it is only used for CPU smoke tests.
- **Run plots are on by default** (`plots=True`, parity with the 09 runs):
  `results.png`, `Box{P,R,F1,PR}_curve.png`, `confusion_matrix[_normalized].png`,
  `labels.jpg`, `train_batch{0,1,2}.jpg`, `val_batch{0,1,2}_{labels,pred}.jpg`.
  `--no-plots` disables them for quick smokes.

## 11. What must be copied to the target node (not in git)

`runs/`, `results/`, `logs/` and `*.pt` are gitignored. Before `N0`:

```
models/pretrained/yolo12s.pt
models/pretrained/yolo12n.pt
experiments/09_graph_fusion/runs/e30_v2_100ep/weights/best.pt     # teacher
experiments/09_graph_fusion/results/gt/dv_v2_ir_test.json         # locked eval
experiments/09_graph_fusion/results/preds/e00_ir_test.json        # bootstrap ref
experiments/09_graph_fusion/results/preds/e30_v2_100ep_test.json  # bootstrap ref
```

The queue's `preflight` aborts with a clear message if any is missing. See
`ASSETS.md` and use the one-command sync:

```bash
bash scripts/sync_assets.sh --check
bash scripts/sync_assets.sh --to yash_gupta@<node>:/abs/path/to/IR_Fusion
```

The DroneVehicle v2 YOLO trees must be at
`experiments/09_graph_fusion/data/dronevehicle_v2/yolo` or pointed to by
`DV_ROOT`.


---

## 12. Paper figures

`paper/make_figures.py` reads whatever exists and writes `paper/figures/*.pdf|png`:

| Figure | Content |
|---|---|
| `fig_test_ap` | locked-test AP bars (E00, E30, students) with 95 % CI |
| `fig_kd_ablation` | 12s (E00 / decomposed / direct) and 12n (E00 / no-KD / decomposed) |
| `fig_pareto` | AP vs A100 latency, bubble = params |
| `fig_deltas` | paired-bootstrap ΔAP vs E00 forest plot |
| `fig_val_curves` | val mAP50-95 curves |

It runs automatically at the end of `N5`. Missing runs are skipped, so it is
safe to call before the queue finishes.
