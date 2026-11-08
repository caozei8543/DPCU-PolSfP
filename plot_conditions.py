#!/usr/bin/env python3
"""Plot per-condition experiment values from user-provided CSV; no hardcoded numbers."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--csv',required=True)
p.add_argument('--out',default='results/conditions.png');a=p.parse_args()
with open(a.csv,newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
required={'condition','baseline_mae','ours_mae'}
if not rows or not required.issubset(rows[0]):raise ValueError('expected condition,baseline_mae,ours_mae columns')
idx=np.arange(len(rows));names=[x['condition'] for x in rows]
fig,ax=plt.subplots(figsize=(max(8,len(rows)),4),layout='constrained')
ax.bar(idx-.18,[float(x['baseline_mae']) for x in rows],.36,label='Baseline')
ax.bar(idx+.18,[float(x['ours_mae']) for x in rows],.36,label='DPCU')
ax.set_xticks(idx,names,rotation=35,ha='right');ax.set_ylabel('Mean normal angular error (°)')
ax.legend();ax.spines[['top','right']].set_visible(False)
Path(a.out).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=350,facecolor='white')
plt.close(fig);print('Saved:',a.out)
