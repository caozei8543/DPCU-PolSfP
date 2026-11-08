#!/usr/bin/env python3
"""Render Figure-style 5x5 parameter sensitivity heatmaps from ACTUAL CLI JSON."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--input',required=True)
p.add_argument('--out',default='results/sensitivity.png');a=p.parse_args()
rows=json.loads(Path(a.input).read_text())['values']
axes_keys=['mae_deg','oracle_mae_deg','invalid_inversion_pct']
axes_titles=['Final normal MAE (°)','Candidate-oracle MAE (°)','Out-of-range inversion (%)']
scales=[0.8,0.9,1.0,1.1,1.2]
fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,key,title in zip(axs,axes_keys,axes_titles):
    m=np.full((5,5),np.nan)
    for d in rows:
        if key in d:m[scales.index(d['rer_scale']),scales.index(d['eta_scale'])]=d[key]
    im=ax.imshow(m,cmap='viridis')
    ax.set_xticks(range(5),scales);ax.set_yticks(range(5),scales)
    ax.set_xlabel('eta / eta*');ax.set_ylabel('RER / RER*');ax.set_title(title)
    for i in range(5):
        for j in range(5):
            if np.isfinite(m[i,j]):ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=8,
                                             color='white' if m[i,j]<(np.nanmin(m)+np.nanmax(m))/2 else 'black')
    fig.colorbar(im,ax=ax,fraction=.04)
Path(a.out).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=350,facecolor='white')
plt.close(fig);print('Saved:',a.out)
