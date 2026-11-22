# Environment

The project runs in a conda environment named `gateir` (Python 3.11.16, CUDA
12.6 wheels). The versions below are the ones the reported results were produced
with; they are a record of the environment rather than a guarantee that every
version is installable from the public index.

## Create the environment

```bash
conda env create -f environment/environment.yml
conda activate gateir
```

or install into an existing environment:

```bash
# CUDA 12.6 wheels
pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.14.0+cu126 torchvision==0.29.0+cu126
pip install -r environment/requirements.txt
```

## Notes

- **Ultralytics settings.** `experiments/05_detector_bench/train_detector.py`
  sets `YOLO_CONFIG_DIR` inside the experiment to avoid writing to `$HOME`.
  Other entry points use the default Ultralytics config directory.
- **CPU/GPU.** Training scripts assume one CUDA GPU (`--device 0`). The gate,
  filters and evaluation can run on CPU, but the U-Net and detectors are
  GPU-bound in practice.
- **GPU jobs.** Training scripts assume one CUDA GPU (`--device 0`). On this
  cluster they are launched as `srun` steps inside a live `salloc` allocation.
- **Disk.** Data ≈54 GB, experiments ≈47 GB (≈54 GB of checkpoints), models
  ≈143 MB.

## Packages actually used

| Package | Version | Used for |
|---|---|---|
| torch | 2.14.0+cu126 | all training |
| torchvision | 0.29.0+cu126 | transforms, CycleGAN |
| ultralytics | 8.4.138 | YOLO training/eval |
| numpy | 2.4.6 | numerics |
| opencv-python | 5.0.0.93 | image IO, filters |
| scikit-learn | 1.9.0 | gate MLP, metrics |
| scipy | 1.17.1 | numerics |
| pillow | 12.3.0 | image IO |
| matplotlib | 3.10.9 | plots |
| pandas | 3.0.5 | CSV handling |
| PyYAML | 6.0.3 | configs |
| tqdm | 4.70.0 | progress |
| huggingface_hub | 0.17.3 | dataset/model download |
| timm | 1.0.30 | CycleDiff backbones |
| einops | 0.8.2 | CycleDiff |
| pytorch-lightning | 1.9.3 | CycleDiff training |
| torchmetrics | 1.9.0 | CycleDiff metrics |
| scikit-image | 0.26.0 | image metrics |
| torch-fidelity | 0.4.0 | FID in CycleDiff |
| ema-pytorch | 0.3.0 | CycleDiff EMA |
