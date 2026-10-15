"""Evaluate downloaded inference weights or a local training checkpoint."""
import argparse,json
from pathlib import Path
import torch
from project_paths import ROOT
from irsamap8_runtime import make_model,configure_runtime,set_phase,IRSData,loader,validate,ResourceGuard,atomic_json
from evaluate_irsamap8_vectors import evaluate_vectors
from topomapper.utils.wandb_monitor import WandbMonitor

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--semantic-only',action='store_true')
    p.add_argument('--batch',type=int,default=8)
    args=p.parse_args()
    checkpoint=args.checkpoint or ROOT/'outputs/irsamap8/formal/best.pth'
    if not checkpoint.exists():checkpoint=ROOT/'weights/model.pth'
    configure_runtime();output=ROOT/'outputs/irsamap8/formal';output.mkdir(parents=True,exist_ok=True)
    doc=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=make_model();model.load_state_dict(doc['model'],strict=True)
    step=doc['global_step'];del doc
    model.cuda();set_phase(model,'joint');model.eval()
    guard=ResourceGuard(output)
    opts=json.loads((ROOT/'configs/wandb_monitor.json').read_text());opts['workspace']=str(ROOT)
    with WandbMonitor(opts,output_dir=output,job_type='evaluation') as monitor:
        semantics=validate(model,loader(IRSData(args.split),batch=args.batch),True,monitor,step,output,guard,prefix=args.split)
        result={'split':args.split,'checkpoint_step':step,'semantic_metrics':semantics}
        if not args.semantic_only:
            result['vector_metrics']=evaluate_vectors(model,args.split,output/f'{args.split}_results',guard,monitor,step,
                batch=args.batch,ddim_steps=200,ids=IRSData(args.split).ids)
        atomic_json(output/f'{args.split}_evaluation.json',result)
        print(json.dumps(result,indent=2))
if __name__=='__main__':main()
