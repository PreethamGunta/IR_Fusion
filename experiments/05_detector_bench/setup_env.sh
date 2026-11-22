#!/usr/bin/env bash
# Build the `gateir` conda env. Runs unattended; log to bench/logs/setup_env.log
set -euo pipefail

ENV_NAME=gateir
PY=3.11
# /opt/miniforge/envs is not writable by us; keep envs under $HOME
ENV_PREFIX="${HOME}/.conda/envs/${ENV_NAME}"

source $(conda info --base)/etc/profile.d/conda.sh

if [ ! -d "${ENV_PREFIX}" ]; then
    echo "[setup] creating env at ${ENV_PREFIX} (python ${PY})"
    mamba create -y -p "${ENV_PREFIX}" "python=${PY}" pip
else
    echo "[setup] env ${ENV_PREFIX} already exists, reusing"
fi

conda activate "${ENV_PREFIX}"

echo "[setup] installing torch (cu12)"
pip install --no-input torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo "[setup] installing the rest"
pip install --no-input \
    ultralytics \
    opencv-python-headless \
    scikit-image \
    scikit-learn \
    pandas \
    matplotlib \
    thop \
    pycocotools \
    huggingface_hub \
    tqdm

echo "[setup] versions:"
python - <<'PY'
import torch, torchvision, ultralytics, cv2, sklearn, numpy
print("  torch       ", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("  torchvision ", torchvision.__version__)
print("  ultralytics ", ultralytics.__version__)
print("  cv2         ", cv2.__version__)
print("  sklearn     ", sklearn.__version__)
print("  numpy       ", numpy.__version__)
PY

echo "[setup] DONE"
