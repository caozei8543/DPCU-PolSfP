#!/usr/bin/env python3
"""Create evidence-only visual diagnostic panels from author-supplied NPZ + checkpoint.

Do not use toy panels as published experimental evidence. Images saved at 350 DPI.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from dpcu_polsfp.data import load_scene
from dpcu_polsfp.physics import stokes
from dpcu_polsfp.cli import load_checkpoint,device_from_arg
from dpcu_polsfp.metrics import angular_error_degrees

p=argparse.ArgumentParser()
p.add_argument('--scene',required=True);p.add_argument('--checkpoint',required=True)
p.add_argument('--out',default='results/diagnostic.png');p.add_argument('--device',default='cpu')
a=p.parse_args();device=device_from_arg(a.device)
s=load_scene(a.scene,need_normals=False)
model=load_checkpoint(a.checkpoint,device)
with torch.no_grad():o=model(s['images'][None].to(device),stochastic=False)
d=stokes(s['images'][None]);pred=o['normals'][0].cpu().numpy().transpose(1,2,0)
mask=s['mask'][0].numpy()>0
imgs=[(d['dolp'][0,0].numpy(),'Measured DoLP','viridis',0,0.12),
      (d['aop'][0,0].numpy(),'Measured AoP (rad)','twilight',-np.pi/2,np.pi/2),
      (o['entropy'][0,0].cpu().numpy(),'Predictive entropy','magma',0,1),
      (np.clip((pred+1)/2,0,1),'Predicted normal RGB',None,None,None)]
if 'normals' in s:
    gt=s['normals'][None].to(device)
    err=angular_error_degrees(o['normals'],gt)[0].cpu().numpy()
    imgs.append((err,'Normal angular error (°)','inferno',0,40))
fig,axes=plt.subplots(1,len(imgs),figsize=(4.3*len(imgs),4.3),layout='constrained')
for ax,(arr,title,cmap,vmin,vmax) in zip(np.atleast_1d(axes),imgs):
    if arr.ndim==2: arr=np.ma.masked_where(~mask,arr)
    else: arr=arr.copy();arr[~mask]=np.nan
    im=ax.imshow(arr,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_title(title);ax.set_axis_off()
    if arr.ndim==2:fig.colorbar(im,ax=ax,shrink=.68)
Path(a.out).parent.mkdir(exist_ok=True,parents=True)
fig.savefig(a.out,dpi=350,facecolor='white')
plt.close(fig)
print('Saved:',a.out)
