# Environment

Validated experiment: Linux x86_64, Python3.12, PyTorch2.7.1+cu128, torchvision0.22.1+cu128, CUDA toolkit12.8, timm0.9.16, mmcv1.7.2. Full captured package versions are in `requirements/`; local wheel paths have been replaced with version pins.
`scripts/install.sh` creates a standalone virtual environment and builds the selective_scan CUDA extension for the selected GPU. A CUDA toolkit with nvcc and a matching NVIDIA driver are required; do not copy a virtualenv or prebuilt .so from the experiment machine.

The original staged run used one GPU, batch8, BF16, up to32 CPU cores, and approximately21.93GiB peak PyTorch-allocated GPU memory. This is an observed allocation, not a universal minimum or total driver memory. Runtime guard limits remain130GB workspace,64GiB host/GPU and one visible GPU. The free-filesystem reserve defaults to5GB and can be increased with ACPV_MIN_FREE_BYTES; the shared experiment used500GB.

Default generated files stay below this checkout. ACPV_WORKSPACE selects another workspace; ACPV_DATA_ROOT and ACPV_RAW_ROOT select existing data without copying it. Keep configs/train.json, configs/wandb_monitor.json, storage_guard.py and storage_budget.json under an external workspace when using that option. Repository source/config defaults are resolved relative to the installed checkout.

W&B is disabled by default. Set enabled=true, mode=online, and your own entity in configs/wandb_monitor.json, then authenticate locally. No account credentials are included.
