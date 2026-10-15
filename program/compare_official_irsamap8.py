#!/usr/bin/env python3
"""Same-image common-five-class evaluation; no optimization or weight changes."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from project_paths import ROOT, SOURCE, DATA, RAW
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(ROOT/'program'))
from irsamap8_runtime import configure_runtime,make_config,IRSData,loader,ResourceGuard,atomic_json,confusion_scores
from topomapper.upernet_detector_vh_m_ldm import BuildingUPerNetDetector
from topomapper.utils.wandb_monitor import WandbMonitor

NAMES=['building','road','unvegetated','vegetation','water']
# 5 is an unmappable prediction. Only GT 5 is ignored; predicted 5 remains an error.
MAP=[5,3,3,4,0,1,5,2]
OUT=ROOT/'outputs/irsamap8/comparison_common5'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def scores(conf):
    conf=np.asarray(conf,dtype=np.int64)
    true=conf.sum(1); predicted=conf[:,:5].sum(0); tp=np.diag(conf[:,:5]).astype(float)
    union=true+predicted-tp
    iou=np.divide(tp,union,out=np.full(5,np.nan),where=union>0)
    f1=np.divide(2*tp,true+predicted,out=np.full(5,np.nan),where=(true+predicted)>0)
    return dict(mIoU=float(np.nanmean(iou)),macro_F1=float(np.nanmean(f1)),pixel_accuracy=float(tp.sum()/conf.sum()),
        class_IoU={name:float(iou[i]) for i,name in enumerate(NAMES)},class_F1={name:float(f1[i]) for i,name in enumerate(NAMES)},
        support_pixels=true.tolist(),unmappable_prediction_pixels=int(conf[:,5].sum()),confusion_matrix_5_by_6=conf.tolist())

@torch.inference_mode()
def main():
    configure_runtime(); OUT.mkdir(parents=True,exist_ok=True)
    atomic_json(ROOT/'qa/official_comparison_progress.json',dict(stage='storage_preflight',updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    guard=ResourceGuard(OUT)
    official_path=ROOT/'sources/acpv_official/model_iter_0160000.pth'
    expected='42f35412bc78a5b729650f8dd13fdf745b900534e8a4805200decc992a99f27a'
    digest=sha256(official_path)
    if digest!=expected: raise RuntimeError('Official checkpoint checksum mismatch')
    cfg=make_config().clone(); cfg.defrost()
    cfg.DATASETS.NUM_CLASSES=5; cfg.MODEL.DECODE_HEAD.NUM_CLASSES=5; cfg.MODEL.AUX_HEAD.NUM_CLASSES=5; cfg.freeze()
    model=BuildingUPerNetDetector(cfg)
    doc=torch.load(official_path,map_location='cpu',weights_only=False,mmap=True)
    state=doc['model']
    if all(k.startswith('module.') for k in state): state={k[7:]:v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    del state,doc
    model.cuda().eval().requires_grad_(False)
    options=json.loads((ROOT/'configs/wandb_monitor.json').read_text()); options.update(workspace=str(ROOT),console='wrap',log_prediction_images=False)
    contract=dict(purpose='same IRSAMap test images, original official model versus frozen fine-tuned predictions',
        split='test',samples=3648,common_classes=NAMES,irsamap_id_to_common_id=MAP,
        ignored_ground_truth_classes=['farmland','sport'],unmappable_prediction_id=5,
        unmappable_predictions_on_evaluated_GT='counted as classification errors',
        prediction_method='standard argmax semantic labels mapped to common classes; no forced renormalization',
        class_mapping_source='official topomapper/upernet_detector_vh_m_ldm.py class-ID dictionary',
        original_checkpoint=str(official_path),original_sha256=digest,
        finetuned_checkpoint=str(ROOT/'outputs/irsamap8/formal/best.pth'),finetuned_step=25000,
        finetuned_predictions=str(ROOT/'outputs/irsamap8/formal/test_results/masks'),
        optimizer_used=False,model_weights_changed=False,precision='bf16',microbatch=8,seed=42,
        dataset_version='official-rvwsb-semantic-v3')
    mapping=torch.tensor(MAP,device='cuda',dtype=torch.int64)
    original=torch.zeros(30,device='cuda',dtype=torch.int64); finetuned=torch.zeros_like(original)
    native=torch.zeros(64,device='cuda',dtype=torch.int64); seen=0; ignored=0; elapsed=time.monotonic()
    with WandbMonitor(options,config=contract,output_dir=OUT,run_name='official-vs-finetuned-common5-'+time.strftime('%Y%m%d-%H%M%S'),job_type='comparison') as monitor:
        for batch in loader(IRSData('test'),batch=8,workers=8):
            x=batch['image'].cuda(non_blocking=True); truth=batch['mask'].cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                feature=model.uper_head(model.backbone(x))
                logits=F.interpolate(model.mask_predictor(feature),size=(512,512),mode='bilinear',align_corners=False)
            if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite official-model logits')
            pred=logits.argmax(1)
            saved=[]
            for ident in batch['id']:
                with Image.open(ROOT/'outputs/irsamap8/formal/test_results/masks'/f'{ident}.png') as image:
                    a=np.asarray(image)
                    if a.shape!=(512,512) or a.max()>7: raise RuntimeError('Invalid cached fine-tuned prediction')
                    saved.append(torch.from_numpy(a.copy()).long())
            new=torch.stack(saved).cuda(non_blocking=True)
            native+=torch.bincount((truth*8+new).flatten(),minlength=64)
            common_truth=mapping[truth]; valid=common_truth<5
            original+=torch.bincount((common_truth[valid]*6+pred[valid]).flatten(),minlength=30)
            finetuned+=torch.bincount((common_truth[valid]*6+mapping[new][valid]).flatten(),minlength=30)
            ignored+=int((~valid).sum()); seen+=len(x)
            if seen%256==0 or seen==3648:
                guard.check(5_000_000)
                atomic_json(ROOT/'qa/official_comparison_progress.json',dict(stage='comparing',samples=seen,total=3648,
                    seconds=time.monotonic()-elapsed,dashboard=monitor.record.get('url')))
        old_result=scores(original.reshape(5,6).cpu().numpy()); new_result=scores(finetuned.reshape(5,6).cpu().numpy())
        native_scores,_=confusion_scores(native.reshape(8,8).cpu().numpy())
        published=json.loads((ROOT/'qa/irsamap8_final_test_report.json').read_text())['semantic_metrics']['mIoU']
        if abs(native_scores['mIoU']-published)>1e-5: raise RuntimeError('Cached predictions do not reproduce the published native-eight-class mIoU')
        if old_result['support_pixels']!=new_result['support_pixels']: raise RuntimeError('Evaluated GT pixels differ between models')
        result=dict(completed=True,contract=contract,official=old_result,finetuned=new_result,
            mIoU_gain_percentage_points=(new_result['mIoU']-old_result['mIoU'])*100,
            pixel_accuracy_gain_percentage_points=(new_result['pixel_accuracy']-old_result['pixel_accuracy'])*100,
            ignored_GT_pixels=ignored,evaluated_GT_pixels=sum(old_result['support_pixels']),
            evaluated_pixel_fraction=sum(old_result['support_pixels'])/(seen*512*512),
            finetuned_native8_mIoU_recomputed=native_scores['mIoU'],cached_prediction_provenance_verified=True,
            samples=seen,seconds=time.monotonic()-elapsed,dashboard=monitor.record.get('url'),resource_snapshot=guard.check())
        metrics={}
        for label,data in [('official',old_result),('finetuned',new_result)]:
            for key in ['mIoU','macro_F1','pixel_accuracy']: metrics['test/common5/'+label+'/'+key]=data[key]
            for name,value in data['class_IoU'].items(): metrics['test/common5/'+label+'/IoU/'+name]=value
        metrics['test/common5/mIoU_gain_pp']=result['mIoU_gain_percentage_points']
        monitor.log(metrics,step=0)
        guard.check(5_000_000)
        atomic_json(ROOT/'qa/official_vs_finetuned_common5.json',result)
        atomic_json(ROOT/'qa/official_comparison_progress.json',dict(stage='completed',samples=seen,total=3648,dashboard=monitor.record.get('url')))
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    try: main()
    except BaseException as exc:
        atomic_json(ROOT/'qa/official_comparison_progress.json',dict(stage='failed',error_type=type(exc).__name__,error=str(exc)))
        raise
