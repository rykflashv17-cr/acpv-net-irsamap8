"""Compare all eight new-loader augmentations against the official loader."""
import sys
from pathlib import Path
import numpy as np
import torch
from project_paths import ROOT, SOURCE, DATA, RAW
sys.path.insert(0,str(SOURCE))
import irsamap8_runtime as runtime
from topomapper.dataset.latent_vertexheatmap_dataset import LatentVertexHeatmapTrainDataset
from topomapper.dataset.transforms import Compose,ResizeImage,ToTensor,Normalize

transforms=Compose([ResizeImage(512,512),ToTensor(),Normalize([109.730,103.832,98.681],[22.275,22.124,23.229],True)])
original_names=list(runtime.D4); ours=runtime.IRSData('train',augment=True,latent=True)
official=LatentVertexHeatmapTrainDataset(str(runtime.DATA),transform=transforms)
index=official.image_list.index(ours.ids[0]+'.png')
maximum=0.
for name in original_names:
    runtime.D4=[name]; official.augmentations=[name]
    a=ours[0]; image,ann=official[index]
    assert torch.equal(a['mask'],ann['mask'].long()), name
    assert torch.equal(a['latent'],ann['vertex_heatmap_latent']), name
    error=float((a['image']-image).abs().max()); maximum=max(maximum,error)
    assert torch.allclose(a['image'],image,atol=3e-6,rtol=1e-6), (name,error)
runtime.D4=original_names
runtime.atomic_json(ROOT/'qa/irsamap8_loader_compatibility.json',dict(passed=True,augmentations=original_names,
    image_normalization_max_error=maximum,masks_and_latents_equal=True))
print('LOADER_COMPATIBILITY_OK',maximum)
