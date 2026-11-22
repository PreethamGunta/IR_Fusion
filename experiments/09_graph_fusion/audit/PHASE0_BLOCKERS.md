# Phase 0 blockers (2026-11-21)

## GPU / allocation — RESOLVED 2026-11-21 ~15:25 UTC

- Job **10133 (GIRKD)** held 32 CPUs + GPU0 for experiment 08 while step
  10133.1 ran `parallel_orchestrator.py`; new srun steps failed with
  `Job/step already completing or completed`.
- Audit of the running arms showed all of them distil into **IR-only
  students**: `dv_a2_node`/`dv_a3_node_edge` from the fused early-fusion
  teacher (which is below IR-only), `dv_b_consensus` from IR+RGB teachers.
  They do not advance 09's goal of a good fused model.
- With explicit user approval, **step 10133.1 was cancelled**
  (`scancel 10133.1`). The allocation was NOT cancelled and remains usable
  until 2026-11-22 08:13. GPU0 is free for 09 work.
- 08 progress preserved and resumable: `dv_a2_node` DONE ep77 best 0.4317;
  `dv_a3_node_edge` paused ep40 best 0.44139; `dv_b_consensus` paused ep17
  best 0.41937. Rerunning the orchestrator in a new step skips DONE arms and
  auto-resumes the rest from `last.pt`.
- A second request **`girkd`** (10157) remains PENDING (`Priority`,
  estimated 2026-11-25 22:20); it is irrelevant while 10133 is usable.
- Consequence: Phase 0.6 (epoch time, GPU memory, dataloader throughput) and
  all 09 training are **unblocked**; `sigma_seed` still needs to be measured.

## External baselines

- IV-PAFNet: no released weights; Tier A requires training from PResNet-50.
  Protocol differences recorded (visible GT, 840×712 border retained, 640×640
  squash, val selection).
- ICAFusion: weights/data on Baidu; no accessible checkpoint. FLIR YAML uses
  the test directory as val; DroneVehicle has no loader.
- Per user instruction, all work uses the existing `gateir` environment; the
  temporary `gf09`/`ivpaf` clones and the failed `icaf` prefix were deleted.
  Tier A IV-PAFNet (needs torch 2.7.1/Python 3.10) and ICAFusion (no accessible
  env/weights) remain blocked until the user authorises a separate env.

## Disk quota

- Quota peaked at 230,686,720/230,686,720 blocks. After deleting the temporary
  envs, caches and the user-approved artifacts, usage is **198.2M blocks**
  (~31 GB headroom).
- Training checkpoints must still be planned carefully. Do not launch multiple
  full-data runs without a quota plan.

## What is unblocked

- **GPU work in allocation 10133** via new `srun --jobid=10133` steps:
  Phase 0.6 timing/memory and the E-ladder.
- CPU-only Phase 1.2 complementarity oracle.
- Code/test work for the GF-YOLO testbed (no training).
- Documentation and data audits.
