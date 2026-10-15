#!/usr/bin/env python3
"""Single-GPU pilot and staged fine-tuning with real validation and bounded writes."""
from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch

from project_paths import ROOT, SOURCE, DATA, RAW
sys.path.insert(0,str(SOURCE))
from irsamap8_runtime import (CLASSES,DATA,SOURCE,ResourceGuard,IRSData,atomic_json,configure_runtime,
    initialize,loader,make_config,make_model,optimizer_for,restore_rng,save_checkpoint,scheduler_for,seed_all,set_phase,validate)
from topomapper.utils.wandb_monitor import WandbMonitor

STOP=False
def stop_signal(signum,frame):
    global STOP
    STOP=True

def status(stage,**kwargs):
    atomic_json(ROOT/'qa/training_experiment_status.json',dict(stage=stage,pid=os.getpid(),
        updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs))

def choose_small_ids(count=32):
    rows=json.loads((DATA/'train/manifest.json').read_text())
    selected=[]
    for cls in range(8):
        ordered=sorted(rows,key=lambda x:x['class_pixel_counts'][cls],reverse=True)
        for row in ordered:
            if row['id'] not in selected: selected.append(row['id']); break
    rng=np.random.default_rng(42)
    for i in rng.permutation(len(rows)):
        if rows[i]['id'] not in selected: selected.append(rows[i]['id'])
        if len(selected)==count: break
    return selected

def gpu_batch(batch):
    x=batch['image'].cuda(non_blocking=True); mask=batch['mask'].cuda(non_blocking=True)
    if 'latent' in batch:
        z=batch['latent'].cuda(non_blocking=True)
        anns=[dict(mask=m,vertex_heatmap_latent=latent) for m,latent in zip(mask,z)]
    else: anns=None
    return x,mask,anns

def train_loss(model,x,mask,anns,phase):
    if phase=='head':
        main,aux=model.semantic_logits(x,aux=True)
        return dict(loss_mask=model.segmentation_loss(main,mask),loss_mask_aux=model.segmentation_loss(aux,mask))
    # iteration=None disables the official five-colour debug visualizer and uncontrolled dumps.
    losses,_=model.forward_train(x,anns,iteration=None,output_dir=None)
    return losses

def rotate_checkpoint(output,model,optim,scheduler,phase,phase_step,step,generator,guard,extra):
    latest=output/'latest.pth'; previous=output/'previous.pth'
    guard.check(4_000_000_000,kind='checkpoint')
    if latest.exists():
        if previous.exists(): previous.unlink()
        latest.replace(previous)
    size=save_checkpoint(latest,model,optim,scheduler,phase,phase_step,step,generator,guard,extra)
    return latest,size

def set_best(latest,output,guard):
    guard.check(4096)
    tmp=output/'best.pth.link'; best=output/'best.pth'
    if tmp.exists(): tmp.unlink()
    os.link(latest,tmp); tmp.replace(best)

def resume_self_check(path,model,optim,scheduler,generator):
    doc=torch.load(path,map_location='cpu',weights_only=False)
    p=next(model.parameters()); expected=p.detach().clone()
    lrs=[g['lr'] for g in optim.param_groups]; epoch=scheduler.last_epoch
    expected_random=torch.rand(8,device='cuda')
    with torch.no_grad(): p.add_(1.)
    for g in optim.param_groups: g['lr']=1.
    scheduler.last_epoch=-123
    model.load_state_dict(doc['model'],strict=True); optim.load_state_dict(doc['optimizer']); scheduler.load_state_dict(doc['scheduler'])
    restore_rng(doc['rng'],generator)
    actual_random=torch.rand(8,device='cuda')
    if not torch.equal(p,expected) or [g['lr'] for g in optim.param_groups]!=lrs or scheduler.last_epoch!=epoch or not torch.equal(expected_random,actual_random):
        raise RuntimeError('Checkpoint resume self-check failed')
    restore_rng(doc['rng'],generator)
    del doc,expected
    return dict(model_restore_verified=True,optimizer_lr_verified=True,scheduler_step_verified=True,rng_verified=True,
                scheduler_last_epoch=epoch)

def run(args):
    configure_runtime(); seed_all(42)
    output=ROOT/'outputs/irsamap8'/('pilot' if args.pilot else 'formal')
    output.mkdir(parents=True,exist_ok=True)
    logger=logging.getLogger('training'); logger.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in [logging.StreamHandler(sys.stdout),logging.FileHandler(output/'train.log')]:
        handler.setFormatter(formatter); logger.addHandler(handler)
    for sig in [signal.SIGTERM,signal.SIGINT]: signal.signal(sig,stop_signal)
    status('resource_baseline',output=str(output),pilot=args.pilot)
    guard=ResourceGuard(output)
    status('model_initialization',output=str(output),pilot=args.pilot)
    model=make_model(); migration=initialize(model); model.cuda()
    (ROOT/'configs/irsamap8_model.yaml').write_text('# Model and input normalization only. Use program/train_irsamap8.py for the staged schedule.\n'+make_config().dump())
    generator=torch.Generator().manual_seed(42)
    val_loader=loader(IRSData('val'),batch=args.batch,workers=args.workers)
    options=json.loads((ROOT/'configs/wandb_monitor.json').read_text())
    options.update(workspace=str(ROOT),console='wrap',log_prediction_images=True)
    settings=dict(dataset_version='official-rvwsb-semantic-v3',classes=CLASSES,seed=42,microbatch=args.batch,
        accumulation=8//args.batch,effective_batch=8,amp='bf16' if args.amp else 'fp32',pilot=args.pilot,
        source_commit='cf28ca7adbabd653d39a7a3847eb160096fb200b',class_weights=migration['class_weights'],
        training_script=str(Path(__file__)),resource_limits=dict(cpu_cores=32,gpu_count=1,gpu_GiB=64,host_GiB=64,project_GB=130))
    settings['schedule']=dict(head_steps=1000,head_lr=1e-4,head_warmup=100,joint_steps=30000,joint_warmup=500,
        backbone_lr=3e-6,semantic_lr=3e-5,diffusion_lr=1e-5,optimizer='AdamW',weight_decay=.01,
        decay='cosine',gradient_clip_norm=1.,loss_weights=dict(mask=1.,mask_aux=.4,vertex_noise_l1=10.),
        ema_decay=.9999,validation_period=2000,vector_validation_period=5000)
    atomic_json(output/'experiment_config.json',settings)
    phases=[('joint',300)] if args.pilot else [('head',1000),('joint',30000)]
    global_step=0; best_score=-1.; patience_best=-1.; bad_validations=0; resume_report=None; times=[]; small_losses=[]
    begin=time.monotonic(); final_state='completed'; reached_wall_budget=False
    resume_doc=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if resume_doc:
        model.load_state_dict(resume_doc['model'],strict=True); global_step=resume_doc['global_step']
        best_score=resume_doc['extra'].get('best_score',-1.); bad_validations=resume_doc['extra'].get('bad_validations',0)
        patience_best=resume_doc['extra'].get('patience_best',best_score)
        begin-=resume_doc['extra'].get('training_wall_elapsed_seconds',0.)
    with WandbMonitor(options,config=settings,output_dir=output,resume=bool(args.resume),
            run_name=('pilot' if args.pilot else 'finetune')+'-irsamap8-'+time.strftime('%Y%m%d-%H%M%S'),
            job_type='pilot' if args.pilot else 'training') as monitor:
        status('pilot_training' if args.pilot else 'head_training',output=str(output),dashboard=monitor.record.get('url'),step=global_step)
        for phase,total in phases:
            if resume_doc and phase=='head' and resume_doc['phase']=='joint': continue
            set_phase(model,phase); optim=optimizer_for(model,phase); scheduler=scheduler_for(optim,phase,total)
            start_step=0
            if resume_doc and resume_doc['phase']==phase:
                optim.load_state_dict(resume_doc['optimizer']); scheduler.load_state_dict(resume_doc['scheduler'])
                restore_rng(resume_doc['rng'],generator); start_step=resume_doc['phase_step']; resume_doc=None
            if not args.pilot and phase=='joint' and start_step==0:
                from evaluate_irsamap8_vectors import evaluate_vectors
                status('joint_baseline_vector_validation',step=global_step,dashboard=monitor.record.get('url'))
                evaluate_vectors(model,'val',output/'vector_validation',guard,monitor,global_step,
                    amp=args.amp,batch=args.batch,ddim_steps=50)
                set_phase(model,phase)
            dataset=IRSData('train',augment=True,latent=phase=='joint')
            full=loader(dataset,args.batch,True,args.workers,generator)
            small=loader(IRSData('train',latent=True,ids=choose_small_ids()),args.batch,True,0,generator) if args.pilot else None
            stream=iter(small if args.pilot and start_step<100 else full)
            status('pilot_training' if args.pilot else phase+'_training',phase=phase,step=global_step,dashboard=monitor.record.get('url'))
            for phase_step in range(start_step+1,total+1):
                if args.pilot and phase_step==101: stream=iter(full)
                if STOP or (not args.pilot and time.monotonic()-begin>=48*3600):
                    final_state='stopped'; reached_wall_budget=not STOP; break
                start=time.monotonic(); optim.zero_grad(set_to_none=True); logs={}; data_seconds=0.
                for micro in range(8//args.batch):
                    t=time.monotonic()
                    try: batch=next(stream)
                    except StopIteration:
                        stream=iter(small if args.pilot and phase_step<=100 else full); batch=next(stream)
                    data_seconds+=time.monotonic()-t
                    x,mask,anns=gpu_batch(batch)
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.amp):
                        losses=train_loss(model,x,mask,anns,phase)
                        coefficients={'loss_mask':1.,'loss_mask_aux':.4,'loss_vertex_heatmap_l1':10.}
                        loss=sum(coefficients[k]*v for k,v in losses.items())
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                    (loss/(8//args.batch)).backward()
                    for k,v in losses.items(): logs[k]=logs.get(k,0.)+float(v.detach())/(8//args.batch)
                    del loss,losses,x,mask,anns
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
                optim.step()
                if phase=='joint': model.on_train_batch_end()
                scheduler.step(); global_step+=1
                torch.cuda.synchronize(); seconds=time.monotonic()-start
                times.append(seconds)
                if args.pilot and phase_step<=100: small_losses.append(logs['loss_mask'])
                if global_step%20==0 or phase_step==total:
                    remaining=(total-phase_step)+(30000 if phase=='head' and not args.pilot else 0)
                    metrics={f'train/{k}':v for k,v in logs.items()}
                    metrics.update({f'train/{k}_weighted':coefficients[k]*v for k,v in logs.items()})
                    metrics.update({'train/loss_total':sum(coefficients[k]*v for k,v in logs.items()),'train/lr':optim.param_groups[0]['lr'],
                        'train/lr_min':min(g['lr'] for g in optim.param_groups),'train/lr_max':max(g['lr'] for g in optim.param_groups),
                        'train/grad_norm':float(norm),'runtime/batch_seconds':seconds,'runtime/data_seconds':data_seconds,
                        'runtime/samples_per_second':8/seconds,'runtime/eta_seconds':float(np.mean(times[-100:]))*remaining})
                    metrics.update(guard.check(5_000_000_000,kind='reserve')); monitor.log(metrics,step=global_step)
                    logger.info('phase=%s step=%s/%s loss=%.5f sec/update=%.3f peak_GiB=%.2f',phase,phase_step,total,
                        metrics['train/loss_total'],seconds,torch.cuda.max_memory_allocated()/1024**3)
                    status('pilot_training' if args.pilot else phase+'_training',phase=phase,step=global_step,
                        phase_step=phase_step,phase_total=total,dashboard=monitor.record.get('url'),metrics=metrics)
                if args.pilot and phase_step==150:
                    path,size=rotate_checkpoint(output,model,optim,scheduler,phase,phase_step,global_step,generator,guard,{})
                    monitor.log_checkpoint(path,step=global_step)
                    resume_report=resume_self_check(path,model,optim,scheduler,generator)
                    atomic_json(ROOT/'qa/irsamap8_resume_check.json',resume_report)
                should_validate=phase_step==total or (not args.pilot and phase=='joint' and phase_step%2000==0)
                if should_validate:
                    scores=validate(model,val_loader,args.amp,monitor,global_step,output,guard)
                    set_phase(model,phase)
                    improved=scores['mIoU']>best_score
                    if scores['mIoU']>patience_best+.002:
                        bad_validations=0; patience_best=scores['mIoU']
                    else: bad_validations+=1
                    best_score=max(best_score,scores['mIoU'])
                    path,size=rotate_checkpoint(output,model,optim,scheduler,phase,phase_step,global_step,generator,guard,
                        dict(best_score=best_score,patience_best=patience_best,bad_validations=bad_validations,
                             training_wall_elapsed_seconds=time.monotonic()-begin))
                    monitor.log_checkpoint(path,step=global_step)
                    if improved: set_best(path,output,guard)
                    if not args.pilot and phase=='joint' and phase_step>=10000 and bad_validations>=6:
                        logger.info('Early stopping after six validations without a 0.2 percentage point improvement'); break
                if not args.pilot and phase=='joint' and (phase_step%5000==0 or phase_step==total):
                    from evaluate_irsamap8_vectors import evaluate_vectors
                    status('periodic_vector_validation',step=global_step,dashboard=monitor.record.get('url'))
                    evaluate_vectors(model,'val',output/'vector_validation',guard,monitor,global_step,
                        amp=args.amp,batch=args.batch,ddim_steps=50)
                    set_phase(model,phase)
            if final_state=='stopped':
                path,size=rotate_checkpoint(output,model,optim,scheduler,phase,phase_step-1,global_step,generator,guard,
                    dict(best_score=best_score,patience_best=patience_best,bad_validations=bad_validations,
                         training_wall_elapsed_seconds=time.monotonic()-begin))
                monitor.log_checkpoint(path,step=global_step); break
            del full,stream,small
        guard.check()
        report=dict(status=final_state,steps=global_step,pilot=args.pilot,wall_budget_reached=reached_wall_budget,
            best_val_mIoU=best_score,seconds=time.monotonic()-begin,peak_gpu_GiB=torch.cuda.max_memory_allocated()/1024**3,
            stable_optimizer_seconds=float(np.median(times[-150:])),checkpoint_bytes=(output/'latest.pth').stat().st_size,
            resource_snapshot=guard.last,dashboard=monitor.record.get('url'),resume_check=resume_report,
            small_sample_mask_loss_first20=float(np.mean(small_losses[:20])) if small_losses else None,
            small_sample_mask_loss_last20=float(np.mean(small_losses[-20:])) if small_losses else None)
        if args.pilot:
            if global_step!=300 or not resume_report or small_losses[-20:]==[]: raise RuntimeError('Pilot did not complete all acceptance checks')
            if report['small_sample_mask_loss_last20']>=report['small_sample_mask_loss_first20']:
                raise RuntimeError('Fixed-sample segmentation loss did not improve')
            report['pilot_passed']=True
        atomic_json(ROOT/'qa'/('irsamap8_pilot_report.json' if args.pilot else 'irsamap8_training_report.json'),report)
        logger.info('Completed: %s',json.dumps(report))
        status('pilot_passed' if args.pilot else 'training_'+final_state,**report)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--pilot',action='store_true'); parser.add_argument('--batch',type=int,choices=[2,4,8],default=8)
    parser.add_argument('--workers',type=int,default=8); parser.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    try: run(args)
    except BaseException as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc),pilot=args.pilot)
        raise
