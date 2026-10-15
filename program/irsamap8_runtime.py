"""IRSAMap-specific data, model, resource budgets and resumable checkpoints."""
from __future__ import annotations
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
from PIL import Image
import psutil
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from project_paths import ROOT, SOURCE, DATA, RAW


CLASSES=['farmland','tree','grass','water','building','road_area','sport','bareland']
D4=['rot0','rot90','rot180','rot270','flip_h','flip_v','flip_diag','flip_anti_diag']
MEAN=np.array([109.730,103.832,98.681],dtype=np.float32).reshape(3,1,1)
STD=np.array([22.275,22.124,23.229],dtype=np.float32).reshape(3,1,1)

def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp'); tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); tmp.replace(path)

def configure_runtime():
    if hasattr(os,'sched_getaffinity'): os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:32]))
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.cuda.set_device(0)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.,64*1024**3/total),0)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)

def weights():
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/train.json').read_text())
    return np.asarray(config['class_weights'],dtype=np.float64)

def transform(a,name):
    if name.startswith('rot'): a=np.rot90(a,int(name[3:])//90)
    elif name=='flip_h': a=a[:,::-1]
    elif name=='flip_v': a=a[::-1]
    elif name=='flip_diag': a=np.swapaxes(a,0,1)
    elif name=='flip_anti_diag': a=np.swapaxes(a,0,1)[::-1,::-1]
    else: raise ValueError(name)
    return np.ascontiguousarray(a)

class IRSData(Dataset):
    def __init__(self,split,augment=False,latent=False,ids=None):
        self.split=split; self.root=DATA/split; self.augment=augment; self.latent=latent
        manifest=json.loads((self.root/'manifest.json').read_text())
        self.ids=ids or [row['id'] for row in manifest]
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        ident=self.ids[i]; name=random.choice(D4) if self.augment else 'rot0'
        with Image.open(self.root/'images'/f'{ident}.png') as im: rgb=np.asarray(im.convert('RGB'))
        with Image.open(self.root/'masks'/f'{ident}.png') as im: mask=np.asarray(im)
        if rgb.shape!=(512,512,3) or mask.shape!=(512,512) or mask.max()>7: raise ValueError('Invalid sample '+ident)
        rgb=transform(rgb,name); mask=transform(mask,name)
        image=torch.from_numpy((rgb.transpose(2,0,1).astype(np.float32)-MEAN)/STD)
        result={'image':image,'mask':torch.from_numpy(mask.copy()).long(),'id':ident}
        if self.latent:
            z=torch.load(self.root/'heatmap_augmented_latent_kl-4'/name/'z'/f'{ident}.pt',map_location='cpu',weights_only=True)
            if z.shape!=(3,128,128) or z.dtype!=torch.float32 or not torch.isfinite(z).all(): raise ValueError('Invalid latent '+ident)
            result['latent']=z
        return result

def worker_seed(i):
    seed=torch.initial_seed()%2**32; np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(1)

def loader(dataset,batch=8,shuffle=False,workers=8,generator=None):
    return DataLoader(dataset,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,
        drop_last=shuffle,worker_init_fn=worker_seed,generator=generator,
        persistent_workers=workers>0,**({'prefetch_factor':2} if workers else {}))

def make_config():
    from topomapper.config import cfg
    c=cfg.clone(); c.defrost(); c.merge_from_file(str(SOURCE/'config-files/deventer512_vmamba-s_m_vh-ldm_kl4_b8.yaml'))
    c.DATASETS.NUM_CLASSES=8; c.MODEL.DECODE_HEAD.NUM_CLASSES=8; c.MODEL.AUX_HEAD.NUM_CLASSES=8
    c.MODEL.BACKBONE.PRETRAINED=None
    c.DATASETS.TRAIN=('irsamap8_train_with_latent_vertex_heatmap',)
    c.DATASETS.VAL=('irsamap8_val_with_latent_vertex_heatmap',)
    c.DATASETS.TEST=('irsamap8_test_with_latent_vertex_heatmap',)
    c.DATASETS.AUGMENTATIONS=D4; c.DATALOADER.NUM_WORKERS=8
    c.OUTPUT_DIR=str(ROOT/'outputs/irsamap8/formal'); c.freeze(); return c

def make_model():
    from topomapper.upernet_detector_vh_m_ldm import BuildingUPerNetDetector
    class IRSModel(BuildingUPerNetDetector):
        def __init__(self,cfg):
            super().__init__(cfg)
            self.register_buffer('irs_class_weight',torch.as_tensor(weights(),dtype=torch.float32),persistent=False)
        def segmentation_loss(self,logits,target,loss_type='ce'):
            return F.cross_entropy(logits.float(),target.long(),weight=self.irs_class_weight.float())
        def semantic_logits(self,images,aux=False):
            inputs=self.backbone(images); features=self.uper_head(inputs)
            main=F.interpolate(self.mask_predictor(features),size=images.shape[-2:],mode='bilinear',align_corners=False)
            if aux:
                second=self.mask_aux_predictor(self.auxiliary_head(inputs))
                return main,second
            return main
    return IRSModel(make_config())

def initialize(model):
    path=ROOT/'sources/acpv_official/model_iter_0160000.pth'
    # This is the hash-verified official checkpoint transferred from the user's machine.
    doc=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    state=doc.get('model',doc.get('state_dict',doc))
    if all(k.startswith('module.') for k in state): state={k[7:]:v for k,v in state.items()}
    wanted=model.state_dict(); selected={}; mismatched={}; unexpected=[]
    for key,value in state.items():
        if key not in wanted: unexpected.append(key)
        elif wanted[key].shape!=value.shape: mismatched[key]={'old':list(value.shape),'new':list(wanted[key].shape)}
        else: selected[key]=value
    permitted={'mask_predictor.2.weight','mask_predictor.2.bias','mask_aux_predictor.3.weight','mask_aux_predictor.3.bias'}
    missing=set(wanted)-set(selected)
    if missing!=permitted or set(mismatched)!=permitted or unexpected:
        raise RuntimeError('Unexpected checkpoint incompatibilities '+str(dict(missing=sorted(missing),mismatched=mismatched,unexpected=unexpected)))
    result=model.load_state_dict(selected,strict=False)
    if 'model_ema' in doc:
        model.model_ema.load_state_dict(doc['model_ema'],strict=True)
    model.model_ema.num_updates.zero_()
    report=dict(checkpoint=str(path),loaded_keys=len(selected),loaded_state_tensor_elements=sum(v.numel() for v in selected.values()),
        skipped_classification_layers=mismatched,missing_keys=list(result.missing_keys),optimizer_restored=False,
        iteration_reset=0,class_names=CLASSES,class_weights=weights().tolist(),source_commit='cf28ca7adbabd653d39a7a3847eb160096fb200b')
    atomic_json(ROOT/'qa/irsamap8_weight_migration.json',report)
    return report

def set_phase(model,phase):
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(('mask_predictor.2.','mask_aux_predictor.3.')) if phase=='head'
            else not name.startswith(('mask_head.','jloc_head.')))
    if phase=='head':
        model.eval(); model.mask_predictor[-1].train(); model.mask_aux_predictor[-1].train()
    else: model.train()

def optimizer_for(model,phase):
    groups=[]
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        lr=1e-4 if phase=='head' else 3e-6 if name.startswith('backbone.') else 1e-5 if name.startswith('model.') else 3e-5
        wd=0. if 'norm' in name or 'pos_embed' in name else .01
        groups.append(dict(params=[p],lr=lr,weight_decay=wd))
    return torch.optim.AdamW(groups,betas=(.9,.999))

def scheduler_for(optim,phase,total):
    warm=min(100 if phase=='head' else 500,max(1,total//10))
    def factor(step):
        if step<warm: return max(.01,(step+1)/warm)
        progress=min(1.,(step-warm)/max(1,total-warm)); return .5*(1.+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optim,factor)

class ResourceGuard:
    def __init__(self,output):
        self.output=Path(output)
        spec=importlib.util.spec_from_file_location('guard',ROOT/'storage_guard.py')
        self.g=importlib.util.module_from_spec(spec); spec.loader.exec_module(self.g)
        self.watch=[ROOT/'outputs/irsamap8',ROOT/'wandb',ROOT/'cache/triton',ROOT/'cache/cuda',ROOT/'cache/matplotlib',
                    ROOT/'cache/torch_extensions',ROOT/'tmp',ROOT/'logs',ROOT/'qa',ROOT/'program',SOURCE]
        full=self.g.usage(ROOT)['effective_bytes']
        self.base=full-self.dynamic_bytes()
        self.setup_bytes=sum(self.g.usage(p)['effective_bytes'] for p in [ROOT/'train_env',ROOT/'cache/pip-training',
            ROOT/'sources/acpv_official',ROOT/'cache/torch_extensions'] if p.exists())
        if self.setup_bytes>25_000_000_000: raise RuntimeError('Training environment/weight setup exceeds 25 GB')
        self.last={}; self.check()
        atomic_json(ROOT/'qa/training_resource_baseline.json',dict(total_bytes=full,immutable_base_bytes=self.base,
            project_limit_bytes=130_000_000_000,quota_policy='no separate quota, confirmed by user'))
    def dynamic_bytes(self):
        return sum(self.g.usage(p)['effective_bytes'] for p in self.watch if p.exists())
    def check(self,pending=0,kind='other'):
        usage=self.base+self.dynamic_bytes(); fs=os.statvfs(ROOT); free=fs.f_bavail*fs.f_frsize
        output=ROOT/'outputs/irsamap8'; seen=set(); checkpoint_bytes=0
        for path in output.rglob('*'):
            if '.pth' not in path.name: continue
            try: info=path.stat()
            except FileNotFoundError: continue
            ident=(info.st_dev,info.st_ino)
            if ident in seen: continue
            seen.add(ident); checkpoint_bytes+=max(info.st_size,info.st_blocks*512)
        output_bytes=self.g.usage(output)['effective_bytes'] if output.exists() else 0
        extra_bytes=max(0,output_bytes-checkpoint_bytes)+sum(self.g.usage(p)['effective_bytes'] for p in [ROOT/'wandb',ROOT/'logs'] if p.exists())
        checkpoint_pending=pending if kind=='checkpoint' else 4_000_000_000 if kind=='reserve' else 0
        other_pending=pending if kind=='other' else 1_000_000_000 if kind=='reserve' else 0
        if checkpoint_bytes+checkpoint_pending>16_000_000_000: raise RuntimeError('16 GB checkpoint component budget reached')
        if extra_bytes+other_pending>8_000_000_000: raise RuntimeError('8 GB evaluation/log component budget reached')
        proc=psutil.Process(); processes=[proc]+proc.children(recursive=True)
        rss=0
        for p in processes:
            try: rss+=p.memory_info().rss
            except psutil.NoSuchProcess: pass
        self.last={'storage/project_bytes':usage,'storage/filesystem_available_bytes':free,
            'storage/checkpoint_bytes':checkpoint_bytes,'storage/evaluation_and_log_bytes':extra_bytes,
            'runtime/process_tree_rss_bytes':rss,'runtime/gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        if usage+pending>=130_000_000_000: raise RuntimeError('130 GB project pre-write budget reached')
        if free-pending<int(os.environ.get('ACPV_MIN_FREE_BYTES','5000000000')): raise RuntimeError('Filesystem free-space reserve reached')
        if rss>64*1024**3: raise RuntimeError('CPU process-tree memory budget reached')
        if torch.cuda.max_memory_allocated()>64*1024**3: raise RuntimeError('GPU memory budget reached')
        return self.last

def rng_state(generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),
                loader=generator.get_state())

def restore_rng(state,generator):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda']); generator.set_state(state['loader'])

def save_checkpoint(path,model,optim,scheduler,phase,phase_step,global_step,generator,guard,extra=None):
    guard.check(4_000_000_000,kind='checkpoint')
    payload=dict(model=model.state_dict(),optimizer=optim.state_dict(),scheduler=scheduler.state_dict(),
        phase=phase,phase_step=phase_step,global_step=global_step,rng=rng_state(generator),class_names=CLASSES,
        dataset_version='official-rvwsb-semantic-v3',extra=extra or {})
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+'.tmp')
    try:
        torch.save(payload,tmp)
        if tmp.stat().st_size>4_000_000_000: raise RuntimeError('Checkpoint exceeded 4 GB reservation')
        tmp.replace(path)
    finally:
        if tmp.exists(): tmp.unlink()
    return path.stat().st_size

def confusion_scores(conf):
    conf=np.asarray(conf,dtype=np.int64); tp=np.diag(conf).astype(float)
    support=conf.sum(1); pred=conf.sum(0); union=support+pred-tp
    iou=np.divide(tp,union,out=np.full(8,np.nan),where=union>0)
    f1=np.divide(2*tp,support+pred,out=np.full(8,np.nan),where=support+pred>0)
    scores={'mIoU':float(np.nanmean(iou)),'macro_F1':float(np.nanmean(f1)),'pixel_accuracy':float(tp.sum()/max(1,conf.sum()))}
    for i,name in enumerate(CLASSES):
        if np.isfinite(iou[i]): scores['IoU/'+name]=float(iou[i])
        if np.isfinite(f1[i]): scores['F1/'+name]=float(f1[i])
    details=dict(scores=scores,confusion_matrix=conf.tolist(),support_pixels=support.tolist(),predicted_pixels=pred.tolist())
    return scores,details

@torch.inference_mode()
def validate(model,valid_loader,amp,monitor,step,output,guard,limit=None,prefix='val'):
    was_training=model.training; model.eval(); start=time.monotonic()
    conf=torch.zeros(64,device='cuda',dtype=torch.int64); imgs=[]; truths=[]; predictions=[]; seen=0
    for batch in valid_loader:
        x=batch['image'].cuda(non_blocking=True); truth=batch['mask'].cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp): logits=model.semantic_logits(x)
        if logits.shape[1]!=8 or not torch.isfinite(logits).all(): raise RuntimeError('Invalid validation logits')
        pred=logits.argmax(1); conf+=torch.bincount((truth*8+pred).flatten(),minlength=64)
        if len(imgs)<8:
            for i in range(min(8-len(imgs),len(x))):
                image=((batch['image'][i].numpy()*STD)+MEAN).clip(0,255).astype(np.uint8).transpose(1,2,0)
                imgs.append(image); truths.append(truth[i].cpu().numpy()); predictions.append(pred[i].cpu().numpy())
        seen+=len(x)
        if limit and seen>=limit: break
    scores,details=confusion_scores(conf.reshape(8,8).cpu().numpy())
    scores['seconds']=time.monotonic()-start; scores['samples']=seen
    guard.check(5_000_000); atomic_json(Path(output)/f'{prefix}_latest.json',dict(**details,step=step,samples=seen,seconds=scores['seconds']))
    monitor.log({prefix+'/'+k:v for k,v in scores.items()},step=step)
    monitor.log_predictions(imgs,truths,predictions,CLASSES,step=step)
    if was_training: model.train()
    return scores
