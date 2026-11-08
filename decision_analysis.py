#!/usr/bin/env python3
"""Generate oracle/selection decomposition and decision reversal report per scene."""
import argparse
import json
from pathlib import Path
import torch
from dpcu_polsfp.cli import load_checkpoint,device_from_arg,write_json
from dpcu_polsfp.data import NPZDataset
from dpcu_polsfp.analysis import decision_reversal,ambiguity_bins

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',default='results/decision_analysis.json')
    p.add_argument('--device',default='cpu');args=p.parse_args()
    device=device_from_arg(args.device)
    model=load_checkpoint(args.checkpoint,device)
    ds=NPZDataset(args.manifest)
    rows=[]
    for idx in range(len(ds)):
        b=ds[idx]
        with torch.no_grad():out=model(b['images'][None].to(device),stochastic=False)
        ref=b['normals'][None].to(device);mask=b['mask'][None].to(device)
        rows.append({'id':ds.rows[idx]['id'],'reversals':decision_reversal(out,ref,mask),
                     'entropy_bins':ambiguity_bins(out,ref,mask)})
    write_json(args.out,{'per_scene':rows,'note':'within-checkpoint analysis; no claim of paper-reproduced percentages'})
    print('Saved:',args.out)
if __name__=='__main__':main()
