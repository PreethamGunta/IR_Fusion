# EXPERIMENT 09: GRAPH-BASED RGB-IR FUSION — METHOD + EXECUTION PLAN
Extends the earlier "Direction 2" prompt, whose data-integrity list, residual
gate and deliverables still apply where not superseded here.

> **Active plan = the Minimal First Pass (section 5), single seed.** Phase 0
> (section 3) is complete and its checks gate everything below.
> **This file is a plan, not a status report:** live status (allocations,
> running steps, paused 08 state, quota) is maintained in `README.md`
> ("Current state") and `RESULTS.md`, never here.

## 0. STANDING RULES

- **One environment: `gateir` only.** All 09 work uses
  `/home2/home/preetham_g/.conda/envs/gateir/bin/python`. No separate envs are
  created (the temporary `gf09`/`ivpaf`/`icaf` envs were deleted; recreating
  any env needs explicit user authorisation). Tier A native reproductions stay
  blocked on this policy.
- **Never modify `experiments/08_graph_kd/`.** Copy (do not import) any 08
  module needed into `09/src/` and record its SHA-256 in `PROVENANCE.md`.
  08 is paused, not dead: its checkpoints stay untouched and it is resumable
  (section 12).
- **Shared Dataset Rule.** Large datasets live once under repo-root `data/`;
  experiments reference them by relative paths derived from the repo root.
  Experiment-local data only for derived labels/splits/warped images/caches.
  No new per-experiment hardlink dataset trees.
- **SLURM.** Allocation names are lowercase, unidentifiable acronyms
  (e.g. `girkd`). Never cancel an allocation; stop work by cancelling **steps**
  only, with explicit approval. Launch via `srun --jobid=<alloc> --overlap`
  from the existing `srun_s` tmux session. Check `squeue` and GPU memory
  before launching. Heavy work only inside an allocation.
- Never overwrite checkpoints, datasets or results. Invalid results are
  renamed `*_INVALID_<reason>`, never deleted (deletions only from the
  user-approved list).
- Tag every finding [VERIFIED] / [CODE-READ] / [REPORT-ONLY] / [INFERRED].
  Never write "tested", "fixed" or "reproduced" unless [VERIFIED].
- One architectural change per experiment. Same backbone, head, recipe, data
  and seed; only the fusion module changes.
- **The test split is locked.** Selection on VAL only; test numbers only via
  `scripts/final_eval.py` with its LOCK mechanism (`results/LOCK_<exp>.json`,
  refusal on second run without logged `--override`).
- If something is blocked (weights, network, GPU), log the blocker and
  continue with independent work.

## 1. STRATEGY IN ONE PAGE

Question: does a graph formulation of cross-modal interaction beat (a) simple
fusion, (b) attention fusion (ICAFusion/DMFF-style) and (c) IV-PAFNet-style
gated/aligned fusion, under matched capacity — and does it genuinely use RGB
rather than just adding parameters?

Three comparison tiers, reported separately (never mixed in one table):
- **Tier A**, external as-is: official repos retrained under our protocol,
  scored by OUR evaluator. *Blocked: no accessible weights + gateir-only
  policy. Deferred, not part of the minimal pass.*
- **Tier B**, controlled module swap (the scientific core): one common
  detector (dual-stream YOLO12s); only the fusion module changes.
- **Tier C**, final head-to-head (optional): drop our best module into
  IV-PAFNet's "modules-off" RT-DETR baseline config. Deferred.

Fusion happens mid-level on P3/P4/P5 backbone features (80×64 / 40×32 / 20×16
at 640×512). Expect small deltas (~0.5–1.5 pp AP50-95); design for detecting
~1 pp via paired image bootstrap, matched-capacity controls, and — for the
finalists only, if time allows — extra seeds.

## 2. FACTS ABOUT THE TWO EXTERNAL BASELINES (for the deferred Tier A)

IV-PAFNet (Yuan et al., PLOS ONE 2026; MIT code): RT-DETR dual-stream
"FusionDETR", PResNet-50; ISA (high-pass IR purification + visible-guided
deformable resampling) and MSFA (channel + spatial softmax gates). DV only,
640×640, 73 epochs, batch 8, seeds 42/43/44; paper 62.81±0.70 mAP50-95.
Audited deviations: visible GT for both streams, 100 px border retained,
840×712 squashed to 640×640, selects/reports on VAL. No released weights.

ICAFusion (Shen et al., PR 2023; AGPL-3.0): YOLOv5 two-stream, DMFF
bidirectional cross-attention. FLIR numbers are 3 classes. Weights on Baidu
only; its FLIR config points `val_*` at the test directories; no DV loader.
AGPL: re-implement DMFF from the paper in our own code (E21), never paste its
code into `src/`.

Both repos stay pinned under `external/` (SHAs in `PROVENANCE.md`).

## 3. PHASE 0 — EXECUTED (details in AUDIT_REPORT.md)

| Step | Outcome |
|---|---|
| 0.1 skeleton/envs | gateir-only per user instruction; temp envs deleted |
| 0.2 DroneVehicle v2 | 17,962/1,469/8,980 pairs; IR labels `*labelr`, RGB `*label`; round-trip asserts pass; 28 missing official frames documented in `audit/dv_v2_audit.json`; thermal GT is primary |
| 0.3 FLIR 3-class | person/car/bicycle; `warp` 3,752/365/1,013 and `drop` 1,387/223/1,013; paper 1,013-frame val list = locked TEST; 2 contiguous blocks = VAL; policy chosen later on val evidence |
| 0.4 evaluator | `src/eval/coco_eval.py`, pycocotools-only; 6/6 unit tests PASS (`PYTHONPATH=src`) |
| 0.5 leakage + lock | `scripts/check_leakage.py` PASS for both policies; frame-id disjoint, pair integrity, no exact-hash duplicates, train-only normalization; `final_eval.py` lock tested |
| 0.6 measurement | folded into the first E00/E01 runs (epoch time, memory, throughput, plateau epoch) → `audit/phase06_timing.json`. **`sigma_seed` is NOT measured in the minimal pass** — replaced by the 5.4 substitute |

## 4. PHASE 1 FOUNDATIONS — MEASUREMENT POLICY

- **Phase 0.6 is folded into the first reference runs**, not a separate smoke
  pass: epoch time, GPU memory, dataloader throughput and the plateau epoch
  are read from E00/E01's own results files, and recorded in
  `audit/phase06_timing.json`.
- **`sigma_seed` is not measured in the minimal pass.** The 3-seed E00/E10
  noise protocol is replaced by the pre-registered substitute in 5.4;
  3-seed measurement is the planned upgrade for finalists.
- References are trained on the full DV v2 splits; ladder rungs E10+ are
  screened on a train-only subset (5.5).

## 5. MINIMAL FIRST PASS — THE ACTIVE PLAN (single seed)

### 5.1 Runs kept, in order

| # | Run | What |
|---|---|---|
| 1 | **E00** | IR-only YOLO12s on DV v2 (thermal GT) — the reference every fusion must beat |
| 2 | **E01** | RGB-only YOLO12s, trained on visible labels, graded against thermal GT at final eval; feeds the oracle |
| 3 | **E02** | 6-channel early fusion; stem channel order fixed + unit-tested (08 had IR/RGB swapped) |
| 4 | **E03** | late fusion (WBF) of E00+E01 detections; no training |
| 5 | **E10** | dual-stream, concat + 1×1 at P3–P5 (ladder base) |
| 6 | **E14** | IR-anchored residual `f_ir + sigmoid(g)·conv([f_ir,f_rgb])`, gate bias negative → starts as E00; Direction-3-compatible form |
| 7 | **E21** | bidirectional cross-attention, DMFF-style own implementation |
| 8 | **E22** | matched-capacity intra-modal self-attention control — **only if E21 > E14** |
| 9 | **E30** | GAT on grid + cross-modal same-location edges (IR←RGB); parent E14 |
| 10 | **E35** | + semantic-kNN cross-modal edges (top-k by embedding similarity in a radius) — the novel rung |

~9 trained runs instead of 26 rungs × 3 seeds ≈ 78. Rungs 5–10 are **screened
on a ⅓-train subset** (contiguous train-only blocks with gaps, to be built);
finalists rerun on full DV. FLIR3 runs after the DV ladder, `drop` vs `warp`
chosen on val.

### 5.2 Skipped (documented; revisitable as future work)

E11, E12 (historical curiosity); E13 (MSFA port — only for a final Tier-B
head-to-head table); E15 (optional alignment); E20 (superseded by E21);
E31–E33 (operator swaps — only if E30/E35 show the graph idea works);
E34, E36, E37 (refinements of E35); E38, E40, E41 (tuning sweeps — last,
finalists only); E39 (only if the best graph model's output form is in
question); E50 (speculative object-level OT graph); EXT-IVPAF / EXT-ICAF
(blocked: no accessible weights + gateir-only); Phase 4 (defined only after
Gate B).

### 5.3 Recipe (identical across all runs except the fusion module)

YOLO12s, 640 px, batch 16, seed 0, patience 30, val-only selection on the
official DV val; loss/assigner unchanged from the 08 recipe; per-run
`runs/<name>/` with DONE markers, `logs/ref_*.log` / ladder logs; resume from
`last.pt` supported everywhere.

### 5.4 Single-seed statistics (pre-registered substitute)

- `sigma_seed` unknown → **treat any delta < 0.5 pp AP50-95 as "no detectable
  gain"** (the promotion rule becomes `delta >= max(0.5 pp, paired-bootstrap
  CI excluding 0)`).
- The headline claim (best fusion vs E00 / E03) is decided by the **paired
  image-level bootstrap on the locked test set** through `final_eval.py`.
- Small deltas are reported as "suggestive", never "proven".
- 3-seed reruns of finalists remain the planned upgrade if compute allows.

### 5.5 Compute budget

| Batch of work | Est. GPU-h | Window |
|---|---|---|
| E00 + E01 (concurrent, full DV) → E02 → E03 | ~11–13 | allocation day 1 |
| E10, E14, E21, E30, E35 (screening, two at a time); E22 if triggered | ~25–32 | allocation day 2 (±spill into day 3) |
| Finalist reruns (full DV + FLIR, 3 seeds if compute allows) | remaining | allocation day 3 |

Total ≈ **40–45 GPU-h ≈ 2–3 × 24 h allocations**, vs several hundred for the
full 26-rung × 3-seed matrix. GPU-heavy runs go at most two-at-a-time on one
GPU; CPU-side work (5.6) proceeds while references train.

### 5.6 Code queue (CPU work; runs alongside GPU training)

1. **E02 stem**: 6-ch first conv, channels 0–2 = IR, 3–5 = RGB, IR stem from
   mean RGB filters; unit test for the order (the 08 bug). Copy the needed 08
   module, don't import 08.
2. **E03 script**: run E00+E01, weighted-box-fusion merge, score with
   `src/eval/coco_eval.py`; thresholds fixed on val only.
3. **GF-YOLO testbed**: dual-stream YOLO12s, fusion at P3/P4/P5 into one
   shared neck + Detect head; `Fuse(f_ir, f_rgb) -> f_fused` interface
   (residual modules also expose Delta); input contract [B,6,H,W] with
   zeroed-modality test; per-scale (H_IR, H_MM, Delta) hooks. Implements E10,
   E14, E21, E30, E35. Log params/FLOPs/latency per module.
4. **Screening subset builder**: ~⅓ of DV train, contiguous blocks with gaps,
   train-only, selection on official val.
5. **Phase 1.2 complementarity oracle**: per-GT-object detected by IR only /
   RGB only / both / neither, by day/night and size, from E00/E01 val
   predictions. Oracle gain < ~1.5 pp → expect fusion gains within noise.

### 5.7 Gates under the minimal pass

- **Gate A (substitute):** protocol/leakage checks 0.2–0.5 all PASS
  [VERIFIED] + E00/E01/E02/E03 sane against the 08 reference numbers. The
  formal Gate A (sigma_seed + external reproductions) is explicitly waived
  with the single-seed caveat in 5.4.
- **Gate B:** best simple/attention rung beats E00 by ≥ 0.5 pp AP50-95 with
  paired-bootstrap CI excluding 0 → climb to graph rungs; otherwise STOP,
  diagnose (residual misalignment, label policy, modality imbalance,
  pretrained-LR, normalization) and report — "fusion does not help beyond
  noise on DV v2" is itself a result.
- **Gate C:** a graph rung (E30/E35) beats the best non-graph rung at matched
  params/FLOPs by ≥ 0.5 pp with the paired bootstrap on the locked test set.

## 6. FULL LADDER — REFERENCE FOR DEFERRED RUNGS (spec unchanged)

Node/edge spec (per scale P3/P4/P5; P3 may pool to stride 2): node = one
spatial feature vector per position per modality, projected to d≈128. Edge
types: 8-neighbour grid within each modality (E_ii/E_rr); cross-modal window
E_ri (radius r=1); semantic E_sem = top-k (k=4–8) RGB nodes by cosine
similarity within radius R (~4/3/2 cells at P3/P4/P5); cross-scale E_scale.
Edge features: (dx,dy), cosine similarity, per-node quality q. Message:
`alpha_ij = softmax_j(LeakyReLU(a^T[W h_i ‖ W h_j ‖ W_e e_ij]))`. Update:
`h_i' = h_i + g_i·Σ_j alpha_ij W_v h_j`, gate g_i negative-bias init, output
in E14 residual form unless stated. Fixed neighbour-index tensors + gather;
PyG only in a unit test (hand-rolled GAT must match GATConv with copied
weights).

- E11 elementwise add; E12 channel-softmax gate; E13 channel+spatial gates
  (MSFA-style, MIT attribution); E15 E13 + deformable alignment (ISA-lite).
- E20 dense one-directional cross-attention; E21 bidirectional (DMFF-style);
  E22 CONTROL: intra-modal self-attention, parameter-matched to E21.
- E31 GCN; E32 GraphSAGE; E33 GATv2 + edge features; E34 bidirectional;
  E35 + E_sem (misregistration-robust edges — the core novelty); E36 +
  E_scale (cite TopoGraph-Fusion); E37 typed edges; E38 connectivity sweep
  (r, R, k, sparse vs dense); E39 residual vs direct output; E40 gate-type
  sweep; E41 placement study (P3/P4/P5/all/post-neck); E50 object-level OT
  graph (Sinkhorn/Hungarian from 08/matching; subject to the 0.5 leakage
  rule if trained post hoc).

## 7. PROBES (run on every fusion model; answers "capacity or complementarity?")

(a) normal, (b) RGB zeroed, (c) IR zeroed, (d) RGB swapped with another
image's RGB, (e) RGB shifted ±4/±8 px, (f) day/night split, (g) recall by
oracle category (IR-only / RGB-only / both objects), (h) mean gate/quality by
day vs night. A model that truly uses RGB shows a clear drop in (b)/(d),
robustness in (e) for graph edges, and gates shifting toward IR at night.
Every learned module is compared against E22 (same params, no cross-modal
path) when E22 is run.

## 8. PHASE 3 HEAD-TO-HEAD (deferred)

Tier B: E13, E21, best graph, best simple — 3 seeds each on full DV (thermal
GT) and FLIR; accuracy vs params/FLOPs/latency scatter. Claim rules: "beats X"
requires paired-bootstrap 95% CI excluding 0 on the locked test, same
evaluator and split, means (not best seed); published numbers only where
protocols match, else "not comparable". If we do not beat them: say so, keep
the matched-capacity module study as the contribution, and diagnose via 7.

## 9. PHASE 4 RESIDUAL PREDICTABILITY GATE (prerequisite for Direction 3)

Only after Gate B, on the best residual-form fusion model; extract
(H_IR, H_MM, Delta) at P3–P5. Proxies IR-only (P1 conv/MLP, P2 attention, P3
graph, P4 sparse graph-query); targets from TRAIN only; block-level
out-of-fold folds (K=3–5); controls: shuffled targets, random Delta of equal
norm, constant mean Delta. Pre-register the threshold BEFORE running
(suggested: ≥30% of teacher gain recovered, CI excluding 0, above the
constant-Delta control) with a timestamp in this file.

## 10. IMPLEMENTATION BUG CHECKLIST (tests before any full training)

Channel contract (0-based IR, 3-based RGB; BGR/RGB order; 1-ch IR replication
consistent); token indexing round-trip (y,x) at every scale; graph edge indices
in bounds, no unintended self-loops, neighbour counts at borders, E_sem
restricted to its window, no reads from the wrong modality; attention softmax
over valid neighbours only, no NaN under AMP (log and raise, never silent
nan_to_num); after one backward every trainable parameter has a finite
nonzero gradient; gate init: residual output equals f_ir within 1e-6 at
step 0; determinism (neighbour tables, seeds, worker seeding); evaluator +
leakage tests pass before every launch; 16-pair overfit test and a one-epoch
smoke run with loss curves inspected before any long run; checkpoint
selection by VAL only, test never touched before the lock.

## 11. TRACKING AND DELIVERABLES

`results/EXPERIMENTS.csv`: id, tier, parent, hypothesis, change, dataset,
split, seed, epochs, params, FLOPs, latency, AP50-95, AP50, AP75, per-class
AP, val/test flag, lock id, decision (promote / no gain / invalid), notes,
run dir. Documents updated continuously: README.md (state + next steps),
this file (hypotheses, gates, pre-registered thresholds), RESULTS.md (tables,
probes), AUDIT_REPORT.md (data, leakage, evaluator, verified bugs).
Final answer must address: strongest baseline; best graph variant; whether we
beat each baseline (per tier, CIs); which components matter (E22, E30 vs E20,
E35 probes); credibility of gains; RGB incremental value; residual
predictability; suitability as a Direction 3 teacher.

## 12. EXPERIMENT 08 — DEFERRED, RESUMABLE

08 is paused by user decision (its arms distil into IR-only students from the
fused early-fusion teacher or IR+RGB consensus — none builds a better fused
model). Nothing under `experiments/08_graph_kd/` was modified or deleted; the
orchestrator is resume-safe: rerunning
`experiments/08_graph_kd/scripts/parallel_orchestrator.py` in a new srun step
skips DONE arms, auto-resumes partial arms from `last.pt`, then runs the
remaining evals. The live paused-state numbers (arms, epochs, best APs) are
recorded in `README.md`/`RESULTS.md`, not here.

## 13. SOURCES

IV-PAFNet: PLOS ONE 21(10):e0361401 (doi 10.1371/journal.pone.0361401);
code github.com/iv-pafnet/IV-PAFNet; dataset notes
github.com/iv-pafnet/IV-PAFNet_dataset. ICAFusion: Shen et al., Pattern
Recognition 2023 (arXiv 2308.07504); code github.com/chanchanchan97/ICAFusion
(AGPL-3.0; re-implemented, never copied). Related: TopoGraph-Fusion
(Symmetry 18(8):1272, 2026); YOLOv11-RGBT (arXiv 2506.14696); IGNet (arXiv
2308.03256); WaveMamba (arXiv 2507.18173); MS2Fusion (arXiv 2507.14643);
IRDFusion (arXiv 2509.09085). FLIR-aligned: Zhang et al., Multispectral
Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks. DroneVehicle:
Sun et al., IEEE TCSVT 2022 (UA-CMDet).
