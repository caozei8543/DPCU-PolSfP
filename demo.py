"""Tiny synthetic sanity-check, NOT ThermoPol and NOT validation of the paper metrics."""
from pathlib import Path
import csv
import numpy as np
import torch
from .physics import physical_dolp


def make_demo(out_dir:str|Path,size:int=64,seed:int=42)->dict:
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    # Synthetic front-facing sphere; polarizer-mode convention offset=pi/2.
    yy,xx=np.mgrid[:size,:size]
    x=(xx-(size-1)/2)/((size-1)/2)*0.85
    y=(yy-(size-1)/2)/((size-1)/2)*0.85
    rr=x*x+y*y
    mask=rr<0.72
    nz=np.sqrt(np.maximum(1-rr,0.01))
    normal=np.stack((-x,-y,nz),axis=-1)
    normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    theta=np.arccos(np.clip(normal[...,2],-1,1))
    phi=np.arctan2(normal[...,1],normal[...,0])
    aop=phi-np.pi/2
    rho=physical_dolp(torch.from_numpy(theta),1.82,1.65).numpy()
    # Foreground radiance is intentionally synthetic and linear. Noise-free.
    s0=np.full((size,size),1.0)
    s1=s0*rho*np.cos(2*aop);s2=s0*rho*np.sin(2*aop)
    channels=[(s0+s1)/2,(s0+s2)/2,(s0-s1)/2,(s0-s2)/2]
    chans=[np.where(mask,a,0.3+0*a).astype('float32') for a in channels]
    z=1-nz
    path=out/'synthetic_sphere.npz'
    np.savez_compressed(path,i0=chans[0],i45=chans[1],i90=chans[2],i135=chans[3],
                        normals=normal.astype('float32'),mask=mask.astype('uint8'),depth=z.astype('float32'))
    for split in ('train','val','test'):
        # Same scene is used only for a pipeline smoke test. No fair performance claim.
        with (out/f'{split}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['id','path']);w.writerow(['toy_sphere','synthetic_sphere.npz'])
    return {'path':str(path),'warning':'single synthetic case reused across splits: smoke test only'}
