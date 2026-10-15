"""Portable paths. Only the selected workspace receives generated files."""
import os,sys,shutil
from pathlib import Path
REPOSITORY=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('ACPV_WORKSPACE',str(REPOSITORY))).expanduser().resolve()
SOURCE=REPOSITORY/'code/ACPV-Net'
DATA=Path(os.environ.get('ACPV_DATA_ROOT',str(ROOT/'data/irsamap_acpv8_corrected'))).expanduser().resolve()
RAW=Path(os.environ.get('ACPV_RAW_ROOT',str(ROOT/'sources/IRSAMap_V2'))).expanduser().resolve()
sys.path.insert(0,str(SOURCE))
for name in ['tmp','qa','qa/label_correction','configs','outputs/irsamap8','cache/triton','cache/cuda','cache/matplotlib']:
    (ROOT/name).mkdir(parents=True,exist_ok=True)
if ROOT != REPOSITORY:
    for relative in ['configs/train.json','configs/wandb_monitor.json','storage_guard.py','storage_budget.json']:
        destination=ROOT/relative
        if not destination.exists():shutil.copy2(REPOSITORY/relative,destination)
for key,rel in [('TMPDIR','tmp'),('TRITON_CACHE_DIR','cache/triton'),('CUDA_CACHE_PATH','cache/cuda'),('MPLCONFIGDIR','cache/matplotlib')]:
    os.environ.setdefault(key,str(ROOT/rel))
