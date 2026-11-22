# Run status — 2026-11-09 14:45 UTC (paused)

SLURM job **9743** was released deliberately. No training is running.

## Completed and evaluated (matched 3,749 VTT frames)

| arm | run dir | clear mAP50 / 50-95 | fog mAP50 / 50-95 |
|---|---|---|---|
| CycleGAN B1 | `runs/b1_yolo12s_100` | 0.2849 / 0.1525 | 0.2430 / 0.1266 |
| CycleGAN G1 U-Net | `runs/g1_unet_yolo12s_100` | 0.3063 / 0.1683 | 0.2856 / 0.1535 |
| CycleDiff-75 B1 | `runs/b1_cyclediff75_yolo12s_100` | 0.2746 / 0.1491 | 0.2747 / 0.1495 |

All six result JSONs are in `results/` and included in `REPORT.md` / `all_results.md`.

## Pending

`g1_unet_cd75_yolo12s_100` — CycleDiff-75 G1 using the CycleDiff-trained U-Net.

- Stopped cleanly at **epoch 27/100** (early stop rule = patience 25).
- Checkpoints: `runs/g1_unet_cd75_yolo12s_100/weights/{last,best}.pt`
- Filtered fog input still materialised: `data/filtered/unet_cd75` (1.2 GB).
- U-Net weights: `results/unet_dehazer_cyclediff75.pt`.
- Expected remaining time: ~2 h at ~95 s/epoch, plus two ~2.5 min evals.

## Resume recipe

**Always launch through `srun` inside an allocation.** On 2026-11-09, processes started directly by `salloc` were SIGTERM'd after ~6–7 minutes; the same jobs ran fine as `srun` steps.

```bash
cd /home2/home/preetham_g/gate_ir_release/experiments/02_flir_gate_ir

# 1. get an allocation (12 h was enough for the whole chain)
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=48G \
       --time=12:00:00 -J gateir_resume

# 2. inside it, resume + evaluate + rebuild reports
srun --jobid="$SLURM_JOB_ID" --overlap --cpus-per-task=16 --gres=gpu:1 bash -c '
  PY=$HOME/.conda/envs/gateir/bin/python
  $PY -m src.train_detector --dataset g1_unet_cd75_direct \
      --name g1_unet_cd75_yolo12s_100 --resume
  $PY -m src.eval_detector --weights runs/g1_unet_cd75_yolo12s_100/weights/best.pt \
      --data configs/g1_unet_cd75_direct_eval_clear.yaml \
      --name g1_unet_cd75_100_clear_vtt --force
  $PY -m src.eval_detector --weights runs/g1_unet_cd75_yolo12s_100/weights/best.pt \
      --data configs/g1_unet_cd75_direct_eval_fog.yaml \
      --name g1_unet_cd75_100_fog_vtt --force
  $PY -m src.make_report
  $PY -m src.summarize_results
'
```

If `data/filtered/unet_cd75` is ever missing, rebuild it first:

```bash
$PY -m src.apply_unet --weights results/unet_dehazer_cyclediff75.pt \
    --src $PWD/../flir_adas_experiments/data/flir_adas_v2/synthetic/cyclediff_75fog/fog/images/train \
    --dst data/filtered/unet_cd75/cyclediff75/train --batch 8
$PY -m src.apply_unet --weights results/unet_dehazer_cyclediff75.pt \
    --src $PWD/../flir_adas_experiments/data/flir_adas_v2/synthetic/cyclediff_75fog/fog/images/test \
    --dst data/filtered/unet_cd75/cyclediff75/test --batch 8
```

## Notes

- `scripts/supervise_main100.sh` skips arms with a `runs/<name>/DONE` marker and resumes partial ones.
- Every arm used `epochs=100`, `patience=25`, `batch=16`, `imgsz=640`, `cache=ram`, `close_mosaic=10`, seed 0.
- Training-time `val` is clear-only (`yolo/train_3a/images/val`, 1,635 frames); it drives `best.pt` and early stopping but is **not** the reported metric.
