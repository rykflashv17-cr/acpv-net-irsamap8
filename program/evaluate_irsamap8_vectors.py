"""Stream DDIM/VAE/PSLG evaluation without full-dataset probability dumps."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from irsamap8_runtime import ROOT,SOURCE,DATA,CLASSES,IRSData,loader,atomic_json

def fixed_ids(split='val',count=64):
    rows=json.loads((DATA/split/'manifest.json').read_text()); chosen=[]
    for cls in range(8):
        for row in sorted(rows,key=lambda x:x['class_pixel_counts'][cls],reverse=True):
            if row['id'] not in chosen: chosen.append(row['id']); break
    if len(chosen)>=count: return chosen[:count]
    rng=np.random.default_rng(20261014)
    for index in rng.permutation(len(rows)):
        if rows[index]['id'] not in chosen: chosen.append(rows[index]['id'])
        if len(chosen)>=count: break
    return chosen

def worker_init():
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'

def polygon_job(item):
    import cv2
    import rasterio.features
    from affine import Affine
    from shapely.geometry import Polygon,mapping,LineString,box
    from shapely.ops import unary_union,polygonize
    from rasterio.enums import MergeAlg
    sys.path.insert(0,str(SOURCE/'tools'))
    from polygonize_pslg_one_image import process_one_image_categories
    from extract_vertices_from_heatmap import match_vertices
    ident,split,pred,vertices,output=item
    output=Path(output); temporary=output/'temporary'; temporary.mkdir(parents=True,exist_ok=True)
    seg_path=temporary/f'{ident}.npy'; vertex_path=temporary/f'{ident}.json'
    np.save(seg_path,pred.astype(np.uint8)); vertex_path.write_text(json.dumps(vertices))
    start=time.monotonic()
    try:
        results=process_one_image_categories(str(seg_path),str(vertex_path),list(range(8)),vis_dir=None,
            dist_thresh=5.,corner_eps=2.,vss=True,dp_fix=True,dp_epsilon_dp=1.,dp_epsilon_fix=2.5)
    finally:
        seg_path.unlink(missing_ok=True); vertex_path.unlink(missing_ok=True)
    features=[]; shapes=[]; invalid=0; total=0; raw_invalid=0; raw_count=0
    # Node all predicted shared edges together and polygonize once. This repairs
    # crossings/invalid rings while preserving a complete planar partition.
    lines=[box(0,0,512,512).boundary]
    for cls,instances in results.items():
        for instance in instances:
            raw_count+=1
            exterior=np.asarray(instance[0],dtype=float).reshape(-1,2)
            holes=[np.asarray(h,dtype=float).reshape(-1,2) for h in instance[1:]]
            p=Polygon(exterior,holes)
            if not p.is_valid: raw_invalid+=1
            for ring in [exterior]+holes:
                if len(ring)>=3: lines.append(LineString(np.vstack([ring,ring[:1]])))
    for p in polygonize(unary_union(lines)):
        if p.is_empty or p.area<1e-10: continue
        total+=1
        if not p.is_valid: invalid+=1; raise RuntimeError('Invalid noded PSLG face')
        x0,y0,x1,y1=p.bounds
        x0=max(0,int(np.floor(x0))); y0=max(0,int(np.floor(y0))); x1=min(512,int(np.ceil(x1))); y1=min(512,int(np.ceil(y1)))
        if x1>x0 and y1>y0:
            interior=rasterio.features.rasterize([(mapping(p),1)],out_shape=(y1-y0,x1-x0),fill=0,dtype='uint8',transform=Affine.translation(x0,y0))
            values=pred[y0:y1,x0:x1][interior.astype(bool)]
        else: values=np.empty(0,dtype=np.uint8)
        if len(values): cls=int(np.bincount(values,minlength=8).argmax())
        else:
            x,y=p.representative_point().coords[0]; cls=int(pred[min(511,max(0,int(y))),min(511,max(0,int(x)))])
        shapes.append((mapping(p),cls))
        features.append(dict(class_id=cls,class_name=CLASSES[cls],exterior=list(map(list,p.exterior.coords)),
            holes=[list(map(list,ring.coords)) for ring in p.interiors]))
    if shapes:
        raster=rasterio.features.rasterize(shapes,out_shape=(512,512),fill=8,dtype='uint8',transform=Affine.identity())
        coverage=rasterio.features.rasterize([(geom,1) for geom,_ in shapes],out_shape=(512,512),fill=0,dtype='uint16',
            merge_alg=MergeAlg.add,transform=Affine.identity())
    else: raster=np.full((512,512),8,dtype=np.uint8); coverage=np.zeros((512,512),dtype=np.uint16)
    atomic_json(output/'polygons'/f'{ident}.json',dict(id=ident,coordinate_system='pixel_boundary',class_names=CLASSES,features=features,
        reconstruction='shared noded PSLG with full image frame and majority predicted class per face',
        raw_invalid_polygons=raw_invalid,raw_polygon_count=raw_count))
    Image.fromarray(pred.astype(np.uint8)).save(output/'masks'/f'{ident}.png')
    if split is None:
        return dict(id=ident,gap_pixels=int((coverage==0).sum()),overlap_pixels=int((coverage>1).sum()),invalid_polygons=invalid,
                    polygons=total,ground_truth_used=False)
    with Image.open(DATA/split/'masks'/f'{ident}.png') as im: truth=np.asarray(im).astype(np.int64)
    conf=np.bincount((truth*9+raster).ravel(),minlength=72).reshape(8,9)
    def boundary(a):
        b=np.zeros_like(a,dtype=np.uint8); h=a[:,1:]!=a[:,:-1]; v=a[1:]!=a[:-1]
        b[:,1:]|=h; b[:,:-1]|=h; b[1:]|=v; b[:-1]|=v; return b
    pb=boundary(raster); tb=boundary(truth); kernel=np.ones((5,5),dtype=np.uint8)
    boundary_counts=[int((pb & cv2.dilate(tb,kernel)).sum()),int(pb.sum()),int((tb & cv2.dilate(pb,kernel)).sum()),int(tb.sum())]
    gt=json.loads((DATA/split/'vertices'/f'{ident}.json').read_text())
    if isinstance(gt,dict): gt=gt.get('vertices',gt.get('junctions',gt.get('points')))
    if gt is None: raise ValueError('Unrecognized ground truth vertices')
    counts=match_vertices(vertices,gt,5.)
    return dict(id=ident,confusion=conf.tolist(),gap_pixels=int((coverage==0).sum()),overlap_pixels=int((coverage>1).sum()),
        invalid_polygons=invalid,polygons=total,vertices_counts=list(map(int,counts)),boundary_counts=boundary_counts,
        raw_invalid_polygons=raw_invalid,raw_polygon_count=raw_count,
        polygon_seconds=time.monotonic()-start)

@torch.inference_mode()
def evaluate_vectors(model,split,output,guard,monitor,step,*,amp=True,batch=8,ddim_steps=50,ids=None):
    from frozen_vae import FrozenVAE
    sys.path.insert(0,str(SOURCE/'tools'))
    from extract_vertices_from_heatmap import extract_vertices
    output=Path(output); guard.check(64_000_000)
    for name in ['masks','polygons','temporary']: (output/name).mkdir(parents=True,exist_ok=True)
    ids=ids or fixed_ids(split)
    dl=loader(IRSData(split,ids=ids),batch=batch,workers=2)
    was_training=model.training; model.eval(); start=time.monotonic()
    vae=FrozenVAE().cuda(); vae.use_sdpa(); conf=np.zeros((8,9),dtype=np.int64)
    gap=overlap=invalid=poly_count=seen=raw_invalid=raw_poly_count=0; vertex=np.zeros(3,dtype=np.int64); boundary=np.zeros(4,dtype=np.int64)
    inference_seconds=0.; polygon_seconds=0.
    # These worker processes perform mechanical polygonization; they never use CUDA.
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
        for group in dl:
            guard.check(64_000_000); x=group['image'].cuda(non_blocking=True)
            rng=torch.cuda.get_rng_state(); torch.cuda.manual_seed(20261014+seen)
            begin=time.monotonic()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                prediction,_=model.forward_test(x,sampler='ddim',ddim_steps=ddim_steps)
                decoded=vae.decode(prediction['vertex_heatmap_latent_recon']).float()
            torch.cuda.set_rng_state(rng); torch.cuda.synchronize(); inference_seconds+=time.monotonic()-begin
            if not torch.isfinite(decoded).all(): raise FloatingPointError('Nonfinite decoded vertex heatmap')
            masks=prediction['seg_mask'].cpu().numpy().astype(np.uint8)
            heat=((decoded.clamp(-1,1)+1.)*.5).mean(1).cpu().numpy()
            tasks=[]
            for ident,mask,hm in zip(group['id'],masks,heat):
                pts=extract_vertices(hm,.5,1000,3)
                tasks.append((ident,split,mask,pts,str(output)))
            for result in pool.map(polygon_job,tasks):
                conf+=np.asarray(result['confusion']); gap+=result['gap_pixels']; overlap+=result['overlap_pixels']
                invalid+=result['invalid_polygons']; poly_count+=result['polygons']; vertex+=result['vertices_counts']; boundary+=result['boundary_counts']
                raw_invalid+=result['raw_invalid_polygons']; raw_poly_count+=result['raw_polygon_count']
                polygon_seconds+=result['polygon_seconds']; seen+=1
            del prediction,decoded,x
            atomic_json(output/'progress.json',dict(samples=seen,total=len(ids),ddim_steps=ddim_steps,seconds=time.monotonic()-start))
    true=conf.sum(1); predicted=conf[:,:8].sum(0); intersection=np.diag(conf[:,:8]).astype(float)
    union=true+predicted-intersection; iou=np.divide(intersection,union,out=np.full(8,np.nan),where=union>0)
    tp,fp,fn=vertex.tolist(); precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
    bp=boundary[0]/max(1,boundary[1]); br=boundary[2]/max(1,boundary[3])
    scores={'vector/mIoU':float(np.nanmean(iou)),'vector/gap_fraction':gap/max(1,seen*512*512),
        'vector/overlap_fraction':overlap/max(1,seen*512*512),'vector/invalid_fraction':invalid/max(1,poly_count),
        'vector/raw_invalid_fraction':raw_invalid/max(1,raw_poly_count),
        'vertex/precision':float(precision),'vertex/recall':float(recall),'vertex/F1':float(2*precision*recall/max(1e-12,precision+recall)),
        'vector/boundary_F1':float(2*bp*br/max(1e-12,bp+br)),'vector/seconds':time.monotonic()-start,
        'vector/inference_seconds':inference_seconds,'vector/polygon_worker_seconds':polygon_seconds,'vector/samples':seen,'vector/ddim_steps':ddim_steps}
    for i,name in enumerate(CLASSES):
        if np.isfinite(iou[i]): scores['vector/IoU/'+name]=float(iou[i])
    details=dict(scores=scores,samples=seen,ddim_steps=ddim_steps,confusion_matrix_with_gap=conf.tolist(),vertex_counts=vertex.tolist(),
        boundary_counts=boundary.tolist(),invalid_polygons=invalid,polygon_count=poly_count,class_names=CLASSES,
        fixed_seed=20261014,gt_reference='official SegLabel_rvwsb for raster and corrected shared-boundary vertices for vertex matching')
    atomic_json(output/'summary.json',details)
    if monitor: monitor.log({('test/' if split=='test' else 'val/')+k:v for k,v in scores.items()},step=step)
    del vae,dl
    if was_training: model.train()
    torch.cuda.empty_cache()
    return scores
