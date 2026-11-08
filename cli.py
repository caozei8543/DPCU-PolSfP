"""CLI commands. No command fabricates experimental results or missing checkpoints."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import random
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import NPZDataset
from .model import DPCUPolSfP,ModelConfig
from .losses import total_loss
from .metrics import angular_summary,oracle_decomposition
from .physics import stokes
from .calibration import calibrate
from .demo import make_demo
from .geometry import integrate_normals,depth_rmse,depth_to_points,chamfer_distance,reference_diagonal


def set_seed(seed:int)->None:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def device_from_arg(s:str)->torch.device:
    if s.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    return torch.device(s)


def load_checkpoint(path:str,device:torch.device)->DPCUPolSfP:
    if not Path(path).is_file():raise FileNotFoundError('Checkpoint missing: '+path)
    # Safe weights-only loader: never deserialize arbitrary pickle objects.
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(ck,dict) or 'state_dict' not in ck or 'config' not in ck:
        raise ValueError('Invalid checkpoint: require state_dict and config')
    model=DPCUPolSfP(ModelConfig(**ck['config']))
    model.load_state_dict(ck['state_dict'],strict=True)
    model.to(device).eval()
    return model


def write_json(path:str|Path,obj:object):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def command_train(args):
    set_seed(args.seed)
    device=device_from_arg(args.device)
    dataset=NPZDataset(args.train,augment=args.augment,crop_size=args.crop)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0)
    val=DataLoader(NPZDataset(args.val),batch_size=1,num_workers=0) if args.val else None
    cfg=ModelConfig(width=args.width,gate=args.gate)
    model=DPCUPolSfP(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    best=float('inf');history=[]
    for ep in range(args.epochs):
        model.train();running=[]
        tau=1.0*pow(0.1,min(ep,79)/79) if args.epochs>1 else 1.0
        for batch in loader:
            images=batch['images'].to(device);normals=batch['normals'].to(device);mask=batch['mask'].to(device)
            opt.zero_grad(set_to_none=True)
            o=model(images,tau=tau,stochastic=True)
            parts=total_loss(o,normals,mask)
            if not torch.isfinite(parts['total']):raise FloatingPointError('nonfinite training loss')
            parts['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            opt.step();running.append(float(parts['total'].detach().cpu()))
        scheduler.step();row=dict(epoch=ep+1,train_loss=float(np.mean(running)),tau=tau)
        score=row['train_loss']
        if val:
            model.eval();scores=[]
            with torch.no_grad():
                for b in val:
                    o=model(b['images'].to(device),stochastic=False)
                    scores.append(angular_summary(o['normals'],b['normals'].to(device),b['mask'].to(device))['mae_deg'])
            score=float(np.mean(scores));row['val_mae_deg']=score
        history.append(row)
        if score<best:
            best=score
            p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'state_dict':model.state_dict(),'config':asdict(cfg),
                        'epoch':ep+1,'seed':args.seed,'source':'paper-derived reference, NOT original weights'},p)
        print(json.dumps(row))
    write_json(str(args.out)+'.history.json',{'runs':history,'best_score':best,
               'warning':'Toy/demo results must not be reported as benchmark accuracy'})
    print('Checkpoint:',args.out)


def _scene_metrics(model,scene,device,geometry:bool=False,fx:float|None=None,fy:float|None=None,
                   cx:float|None=None,cy:float|None=None,range_m:float|None=None,offset:bool=True):
    inp=scene['images'][None].to(device)
    with torch.no_grad():out=model(inp,stochastic=False)
    mask=scene['mask'][None].to(device)
    result={'invalid_inversion_pct':float(100*out['invalid_inversion'][mask>0].float().mean().cpu())}
    if 'normals' in scene:
        gt=scene['normals'][None].to(device)
        result.update(angular_summary(out['normals'],gt,mask))
        result.update(oracle_decomposition(out['candidates'],gt,out['onehot'].argmax(dim=1),mask))
    if geometry:
        if 'depth' not in scene: raise ValueError('Geometry evaluation needs reference depth')
        if any(x is None for x in (fx,fy,cx,cy,range_m)):
            raise ValueError('Geometry evaluation requires calibrated fx,fy,cx,cy,range_m')
        if range_m<=0:raise ValueError('range_m must be positive')
        pred=out['normals'][0].detach().cpu().numpy().transpose(1,2,0)
        m=scene['mask'][0].numpy()>0
        z=integrate_normals(pred,m,dx=range_m/fx,dy=range_m/fy)
        ref=scene['depth'][0].numpy()
        # Align a single global depth offset for both depth and point cloud.
        valid=m & np.isfinite(ref) & np.isfinite(z)
        offset_z=(ref[valid]-z[valid]).mean() if offset and valid.any() else 0.
        z=z+offset_z
        cloud=depth_to_points(z,m,fx,fy,cx,cy)
        reference=depth_to_points(ref,m,fx,fy,cx,cy)
        diag=reference_diagonal(reference)
        result['depth_rmse_pct_D']=depth_rmse(z,ref,m,diag,align_offset=False)
        result['chamfer_pct_D']=chamfer_distance(cloud,reference,diag)
        result['reference_diagonal_m']=diag
    return result


def command_evaluate(args):
    device=device_from_arg(args.device)
    model=load_checkpoint(args.checkpoint,device)
    ds=NPZDataset(args.manifest,need_normals=not args.no_gt)
    rows=[]
    for i in range(len(ds)):
        scene=ds[i]
        m=_scene_metrics(model,scene,device,args.geometry,args.fx,args.fy,args.cx,args.cy,args.range_m)
        m['id']=ds.rows[i]['id'];rows.append(m)
        print(json.dumps(m,ensure_ascii=False))
    names=[x for x in rows[0] if x!='id']
    avg={key:float(np.mean([x[key] for x in rows])) for key in names}
    write_json(args.out,{'per_scene':rows,'macro_average':avg,
              'checkpoint':args.checkpoint,'manifest':args.manifest,'n_scenes':len(rows),
              'warning':'No comparison with paper values is implied unless exact original data and weights were used.'})
    print('Saved:',args.out)


def command_calibrate(args):
    ds=NPZDataset(args.train)
    z=[];p=[]
    set_seed(args.seed)
    rg=np.random.default_rng(args.seed)
    for i in range(len(ds)):
        s=ds[i];n=s['normals'].numpy();m=s['mask'][0].numpy()>0
        theta=np.arccos(np.clip(n[2][m],-1,1))
        dolp=stokes(s['images'][None])['dolp'][0,0].numpy()[m]
        if len(theta)>args.max_pixels_per_scene:
            idx=rg.choice(len(theta),size=args.max_pixels_per_scene,replace=False)
            theta,dolp=theta[idx],dolp[idx]
        z.append(theta);p.append(dolp)
    result=calibrate(np.concatenate(z),np.concatenate(p),step=args.step,maxiter=args.maxiter)
    write_json(args.out,result);print(json.dumps(result,indent=2))


def command_sensitivity(args):
    device=device_from_arg(args.device)
    model=load_checkpoint(args.checkpoint,device)
    ds=NPZDataset(args.manifest)
    from .physics import ZenithLookup
    scores=[];orig_eta=model.cfg.eta;orig_rer=model.cfg.rer
    for sf_eta in (0.8,0.9,1.0,1.1,1.2):
        for sf_rer in (0.8,0.9,1.0,1.1,1.2):
            try:
                model.cfg.eta=orig_eta*sf_eta;model.cfg.rer=orig_rer*sf_rer
                model.lookup=ZenithLookup(model.cfg.eta,model.cfg.rer,model.cfg.max_zenith_degrees)
                rows=[_scene_metrics(model,ds[i],device) for i in range(len(ds))]
                scores.append(dict(eta_scale=sf_eta,rer_scale=sf_rer,
                                   mae_deg=float(np.mean([r['mae_deg'] for r in rows])),
                                   oracle_mae_deg=float(np.mean([r['oracle_mae_deg'] for r in rows])),
                                   selection_error_deg=float(np.mean([r['selection_error_deg'] for r in rows])),
                                   invalid_inversion_pct=float(np.mean([r['invalid_inversion_pct'] for r in rows]))))
            except ValueError as e:
                scores.append(dict(eta_scale=sf_eta,rer_scale=sf_rer,status='nonmonotonic',reason=str(e)))
    write_json(args.out,{'values':scores,'note':'fixed weights, altered physical inversion; no retraining'})
    print('Saved:',args.out)


def command_profile(args):
    device=device_from_arg(args.device)
    model=load_checkpoint(args.checkpoint,device)
    scene=NPZDataset(args.manifest,need_normals=False)[0]
    images=scene['images'][None].to(device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    with torch.no_grad():
        for _ in range(args.warmup):model(images,stochastic=False)
        sync();start=perf_counter()
        for _ in range(args.runs):model(images,stochastic=False)
        sync();lat=(perf_counter()-start)*1000/args.runs
    result=dict(device=str(device),image_hw=list(images.shape[-2:]),batch_size=1,
                model_forward_ms=lat,warmup=args.warmup,runs=args.runs,
                note='forward only, not comparable to full 180 ms manuscript latency; integration excluded')
    write_json(args.out,result);print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(prog='dpcu-polsfp',description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('demo-data',help='Make tiny synthetic smoke-test scenes (not real results)')
    d.add_argument('--out',default='data/demo');d.add_argument('--size',type=int,default=64);d.set_defaults(func=lambda a:print(make_demo(a.out,a.size)))
    t=sub.add_parser('train',help='Train reference CNN on a real author-provided NPZ manifest')
    t.add_argument('--train',required=True);t.add_argument('--val');t.add_argument('--out',default='weights/reference.pt')
    t.add_argument('--device',default='cpu');t.add_argument('--epochs',type=int,default=100)
    t.add_argument('--width',type=int,default=32);t.add_argument('--batch-size',type=int,default=1)
    t.add_argument('--lr',type=float,default=1e-4);t.add_argument('--seed',type=int,default=42)
    t.add_argument('--gate',choices=['entropy','fixed','dolp','softmax_confidence','learned','local','context'],default='entropy')
    t.add_argument('--augment',action='store_true');t.add_argument('--crop',type=int,default=None);t.set_defaults(func=command_train)
    e=sub.add_parser('evaluate',help='Evaluate a REAL checkpoint, optionally geometry')
    e.add_argument('--manifest',required=True);e.add_argument('--checkpoint',required=True)
    e.add_argument('--out',default='results/eval.json');e.add_argument('--device',default='cpu')
    e.add_argument('--no-gt',action='store_true');e.add_argument('--geometry',action='store_true')
    for x in ('fx','fy','cx','cy','range-m'):e.add_argument('--'+x,type=float)
    e.set_defaults(func=command_evaluate)
    c=sub.add_parser('calibrate',help='Fit eta/RER only from a TRAIN manifest with normals')
    c.add_argument('--train',required=True);c.add_argument('--out',default='results/calibration.json')
    c.add_argument('--seed',type=int,default=42);c.add_argument('--step',type=float,default=0.05)
    c.add_argument('--maxiter',type=int,default=500);c.add_argument('--max-pixels-per-scene',type=int,default=3000)
    c.set_defaults(func=command_calibrate)
    s=sub.add_parser('sensitivity',help='Run fixed-weight 5x5 physical parameter perturbation grid')
    s.add_argument('--checkpoint',required=True);s.add_argument('--manifest',required=True)
    s.add_argument('--device',default='cpu');s.add_argument('--out',default='results/sensitivity.json')
    s.set_defaults(func=command_sensitivity)
    pr=sub.add_parser('profile',help='Forward-only model runtime (NOT full geometry latency)')
    pr.add_argument('--checkpoint',required=True);pr.add_argument('--manifest',required=True)
    pr.add_argument('--device',default='cpu');pr.add_argument('--out',default='results/forward_profile.json')
    pr.add_argument('--runs',type=int,default=20);pr.add_argument('--warmup',type=int,default=5)
    pr.set_defaults(func=command_profile)
    args=p.parse_args();args.func(args)

if __name__=='__main__':main()
