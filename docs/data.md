# Data contract

Version: `official-rvwsb-semantic-v3`. Official source: https://github.com/ucas-dlg/IRSAMap .
Original split counts: train 4617 / val 340 / test 912; four non-overlapping 512x512 crops per original yield 18468 / 1360 / 3648.

Each split contains `images/*.png`, `masks/*.png`, `vertices/*.json`, `annotations/*.geojson`, `metadata/*.json`, and `manifest.json`.
Masks are uint8 IDs 0..7. See `configs/classes.json`; ID 0 is farmland, not ignore. River/lake/sea including package-specific code24 are merged into water. Source residual0 and bareland40 are mapped to bareland7; this is a documented project mapping convention. The package README says code24 was excluded in the paper; this project retains it as coarse water, so our evaluation is not the paper benchmark.

Semantic masks exactly equal mapped official `SegLabel_rvwsb` pixels. Shared simplified geometry is separately required to reach >=99% raster agreement and >=95% per-present-class IoU; derived geometry does not alter the supervision mask. Coordinates are pixel edges, not georeferenced GIS coordinates.

Optional `heatmap_augmented_latent_kl-4/<augmentation>/z/*.pt` contains float32 [3,128,128] tensors. Train uses all eight D4 directions; validation/test only rot0. Total: 152752 latent files. Per-item deterministic CPU posterior seeds and VAE scale0.086164 are in `program/prepare_dataset.py`.

Regenerate directly from official data (the deleted old processed dataset is not required):

```bash
python program/fetch_sources.py
python program/prepare_dataset.py convert --workers 32
python program/prepare_dataset.py validate --without-latents
CUDA_VISIBLE_DEVICES=0 python program/prepare_dataset.py encode --batch-size 4
python program/prepare_dataset.py validate
```

Alternatively download the prepared core archives and optional latent archives through `program/download_artifacts.py`. Files restore the same folder layout; train/val/test remain separate. Archives contain regular files and can be restored without the original hardlink source.
