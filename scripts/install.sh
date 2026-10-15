#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools==80.9.0' wheel ninja cython
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/runtime.txt
python -m pip install --no-build-isolation mmcv==1.7.2
MAX_JOBS=4 python -m pip install --no-build-isolation ./code/ACPV-Net/kernels/selective_scan
python -c 'import torch, selective_scan, mmcv; print(torch.__version__, mmcv.__version__)'
