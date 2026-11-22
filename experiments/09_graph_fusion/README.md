# Experiment 09 — Graph-Based RGB-IR Fusion

Status: **Phase 0 executed; GPU freed for 09.** No 09 model has been trained
yet; Phase 0.6 and the E-ladder are now unblocked.

## Current state (2026-11-21 ~15:30 UTC)

| Item | State |
|---|---|
| 10133 (job `GIRKD`) | **RUNNING, empty.** Step 10133.1 (08 orchestrator) was cancelled with user approval; GPU0 and the 32 CPUs are free. The `salloc` shell is intact; end time 2026-11-22 08:13 UTC |
| 08 paused state | `dv_a2_node` **DONE** ep77 best 0.4317; `dv_a3_node_edge` paused ep40 best 0.44139; `dv_b_consensus` paused ep17 best 0.41937. All `best.pt`/`last.pt` preserved; the orchestrator auto-resumes via `--resume`, so 08 can be continued later |
| Why 08 was stopped | Its arms distill into **IR-only students** from the fused early-fusion teacher (a2/a3) or IR+RGB consensus (b); none improves the fused model, so they do not serve 09's goal of getting a good fused model first |
| 10157 (job `girkd`) | still **PENDING**, reason `Priority`; irrelevant while 10133 is usable |
| Environment | all 09 work uses `gateir`; temporary envs were deleted |
| Quota | 198.2M / 230.7M blocks, ~31 GB free |
| 09 verification | leakage gate PASS (`drop` and `warp`); evaluator tests 6/6 PASS (`PYTHONPATH=src`); builders derive paths from the repo root, no hardcoded user paths in code |
| 09 models | **E00 DONE** (best ep55 val AP50-95 0.63055, AP50 0.84234, thermal GT, 4.14 h) and **E01 DONE** (best ep84 val AP50-95 0.58822 on visible-GT training val) [VERIFIED]. **E02 training** in step 10133.4 (restarted with 16 workers → 10.3 it/s ≈ 1.8 min/epoch, ETA ~23:00 UTC); the 8-worker partial is preserved as `runs/e02_fused_PARTIAL_w8`. E03 val predictions queued as a CPU step (auto-starts when E02's step ends — this Slurm serializes steps) |

## What exists

- `data/dronevehicle_v2/` — 17,962/1,469/8,980 paired images; IR labels from
  `*labelr`, RGB from `*label`; 28 missing-image frames documented.
- `data/flir3/{drop,warp}/` — 3-class FLIR-aligned; 1,013-frame paper test;
  2-block temporal val; two alignment policies; leakage/normalization audits.
- `src/eval/` — pycocotools-only evaluator and geometry helpers.
- `src/data/` — DroneVehicle v2 and FLIR 3-class builders.
- `src/audit/` — robust shift estimation and leakage audit.
- `scripts/` — leakage gate, test lock, locked final evaluation.
- `external/` — pinned IV-PAFNet, IV-PAFNet_dataset, ICAFusion clones.

## Quick checks

```bash
PY=/home2/home/preetham_g/.conda/envs/gateir/bin/python
PYTHONPATH=src $PY -m pytest tests -q   # 6 passed
$PY scripts/check_leakage.py --policy warp
$PY src/data/build_flir3.py --help
```

## Data YAMLs

- `configs/dronevehicle_v2_{ir,rgb,paired}.yaml`
- `configs/flir3_{drop,warp}_{ir,rgb,paired}.yaml`

**Data policy:** large datasets live once under the repository-root `data/`.
Experiment-local data is only derived labels/splits/warped images/caches; the
existing 09 image trees are legacy hardlinks (same inodes, ~0.29 GB unique).
See the Shared Dataset Rule in `rules.md`.

## Gate A status

Phase 0.1–0.5 are complete (with the external-env caveats). Phase 0.6 (epoch
time, GPU memory, sigma_seed) and the E-ladder are **now unblocked**: step
10133.1 was cancelled with user approval on 2026-11-21 (~15:25 UTC) and GPU0 is
free inside allocation 10133, which itself was NOT cancelled. See
`AUDIT_REPORT.md` §5 and `audit/PHASE0_BLOCKERS.md`.

## Next steps

1. **Phase 0.6 (first GPU work):** measure epoch time, GPU memory, dataloader
   throughput on DroneVehicle v2, and run 3-seed E00/E10 noise; set `sigma_seed`
   and the promotion threshold before any ladder run.
2. **Baselines:** E00–E03 on DroneVehicle v2 (thermal GT), then the E10–E14
   simple/attention ladder and Gate B. Keep the test split locked.
3. **CPU-only in parallel:** Phase 1.2 complementarity oracle (IR-only vs
   RGB-only detections by day/night and size), and the GF-YOLO testbed/unit
   tests (channel contract, zero-modality, gate init, graph edge bounds).
4. **FLIR policy choice:** pick `drop` vs `warp` from validation evidence.
5. **08 resumption (deferred):** rerun `scripts/parallel_orchestrator.py` in a
   new `srun --jobid=10133` step; DONE arms are skipped and paused arms resume
   from `last.pt` automatically.
6. **Tier A:** blocked on missing IV-PAFNet weights/env and ICAFusion
   Baidu-only weights; revisit only with user authorisation.

## Documents

- `STATUS_LOG.md` — dated running log: autonomous progress, blockers and the
  workarounds taken (check here first when catching up).

- `EXPERIMENT_PLAN.md` — the full plan (saved verbatim).
- `AUDIT_REPORT.md` — Phase 0 data/protocol findings with evidence tags.
- `RESULTS.md` — Phase 0 dataset tables; no model results yet.
- `EXPERIMENTS.csv` — experiment ledger (planned/blocked).
- `PROVENANCE.md` — repos, SHAs, licenses, environments, data policy.
