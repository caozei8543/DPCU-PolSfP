#!/usr/bin/env python3
"""Evaluate third-party predicted normal maps under identical masks and normal metrics.

Manifest columns: id,path,pred_path (prediction NPZ with normals as HxWx3 or 3xHxW).
Model execution and checkpoint licensing remain third-party responsibilities.
"""
import argparse,csv
from pathlib import Path
import numpy as np,torch
from dpcu_polsfp.data import load_scene
from dpcu_polsfp.metrics import angular_summary
from dpcu_polsfp.cli import write_json

p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
p.add_argument('--out',default='results/external_baseline.json');a=p.parse_args()
rows=[];base=Path(a.manifest).parent
with open(a.manifest,newline='',encoding='utf-8') as f:
    for row in csv.DictReader(f):
        im=Path(row['path']); pred=Path(row['pred_path'])
        if not im.is_absolute():im=base/im
        if not pred.is_absolute():pred=base/pred
        scene=load_scene(im,need_normals=True)
        with np.load(pred,allow_pickle=False) as d:n=np.asarray(d['normals'],dtype=np.float32)
        if n.shape==(*scene['mask'].shape[1:],3):n=n.transpose(2,0,1)
        if n.shape!=scene['normals'].shape:raise ValueError('prediction dimensions mismatch: '+str(pred))
        stats=angular_summary(torch.from_numpy(n)[None],scene['normals'][None],scene['mask'][None]);rows.append(dict(id=row['id'],**stats))
if not rows:raise SystemExit('Empty manifest')
keys=[k for k in rows[0] if k!='id'];avg={k:float(np.mean([r[k] for r in rows])) for k in keys}
write_json(a.out,{'per_scene':rows,'macro_average':avg,
      'warning':'Set camera-frame axis convention and mask EXACTLY; no model runtime is measured here'})
print('Saved:',a.out)
