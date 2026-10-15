#!/usr/bin/env python3
"""Predict eight-class masks and pixel-coordinate polygons for one RGB tile."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
import torch

from project_paths import ROOT, SOURCE, DATA, RAW
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
for name,rel in [('TMPDIR','tmp'),('TRITON_CACHE_DIR','cache/triton'),('CUDA_CACHE_PATH','cache/cuda'),('MPLCONFIGDIR','cache/matplotlib')]:
    os.environ[name]=str(ROOT/rel)
sys.path.insert(0,str(SOURCE))
from irsamap8_runtime import configure_runtime,make_model,set_phase,ResourceGuard,atomic_json,CLASSES,MEAN,STD
from evaluate_irsamap8_vectors import polygon_job
from frozen_vae import FrozenVAE
sys.path.insert(0,str(SOURCE/'tools'))
from extract_vertices_from_heatmap import extract_vertices

@torch.inference_mode()
def main(args):
    configure_runtime(); output=args.output or ROOT/'outputs/irsamap8/inference'/args.image.stem
    if not output.resolve().is_relative_to(ROOT/'outputs/irsamap8'): raise ValueError('Outputs must remain in the workspace outputs/irsamap8 directory')
    guard=ResourceGuard(output); guard.check(64_000_000)
    for folder in ['polygons','masks']: (output/folder).mkdir(parents=True,exist_ok=True)
    with Image.open(args.image) as im: image=np.asarray(im.convert('RGB'))
    if image.shape!=(512,512,3): raise ValueError('Provide one 512 x 512 RGB tile')
    doc=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    checkpoint_step=doc.get('global_step'); model=make_model(); model.load_state_dict(doc['model'],strict=True); del doc
    model.cuda(); set_phase(model,'joint'); model.eval(); vae=FrozenVAE().cuda(); vae.use_sdpa()
    x=torch.from_numpy((image.transpose(2,0,1).astype(np.float32)-MEAN)/STD).unsqueeze(0).cuda()
    torch.cuda.manual_seed(20261014)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prediction,_=model.forward_test(x,sampler='ddim',ddim_steps=args.ddim_steps)
        decoded=vae.decode(prediction['vertex_heatmap_latent_recon']).float()
    if not torch.isfinite(decoded).all(): raise FloatingPointError('Nonfinite prediction')
    mask=prediction['seg_mask'][0].cpu().numpy().astype(np.uint8)
    heat=((decoded[0].clamp(-1,1)+1.)*.5).mean(0).cpu().numpy(); vertices=extract_vertices(heat,.5,1000,3)
    quality=polygon_job((args.image.stem,None,mask,vertices,str(output)))
    atomic_json(output/'vertices.json',dict(vertices=vertices,coordinate_system='pixel'))
    palette=np.asarray(json.loads((Path(__file__).resolve().parents[1]/'configs/classes.json').read_text())['palette'],dtype=np.uint8)
    Image.fromarray(palette[mask]).save(output/'semantic_preview.png')
    Image.fromarray((heat.clip(0,1)*255).astype(np.uint8)).save(output/'vertex_heatmap.png')
    overlay=Image.fromarray(image); draw=ImageDraw.Draw(overlay)
    polygons=json.loads((output/'polygons'/f'{args.image.stem}.json').read_text())['features']
    for feature in polygons:
        for ring in [feature['exterior']]+feature['holes']:
            draw.line([tuple(p) for p in ring],fill=tuple(map(int,palette[feature['class_id']])),width=2)
    overlay.save(output/'overlay.png')
    atomic_json(output/'metadata.json',dict(checkpoint=str(args.checkpoint),checkpoint_step=checkpoint_step,
        class_names=CLASSES,ddim_steps=args.ddim_steps,coordinate_system='pixel_boundary',ground_truth_used=False,quality=quality))
    print(json.dumps(dict(output=str(output),quality=quality),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--image',required=True,type=Path)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'weights/model.pth')
    parser.add_argument('--output',type=Path); parser.add_argument('--ddim-steps',type=int,default=200)
    main(parser.parse_args())
