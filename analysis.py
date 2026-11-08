"""Within-checkpoint decision reversals and uncertainty stratification."""
from __future__ import annotations
import numpy as np
import torch
from .metrics import angular_error_degrees


def decision_reversal(output:dict,reference:torch.Tensor,mask:torch.Tensor,threshold:float=0.8)->dict:
    """Pair local/fused decisions on SAME candidate set and SAME trained checkpoint."""
    candidates=output['candidates']
    err0=angular_error_degrees(candidates[:,0],reference)
    err1=angular_error_degrees(candidates[:,1],reference)
    oracle=torch.where(err0<=err1,0,1)
    local=output['local_logits'].argmax(dim=1)
    fused=output['logits'].argmax(dim=1)
    valid=mask[:,0].bool()
    high=valid & (output['entropy'][:,0]>=threshold)
    def rates(m):
        count=int(m.sum())
        if count==0:return dict(count=0,beneficial_pct=float('nan'),harmful_pct=float('nan'),net_gain_pp=float('nan'))
        beneficial=((local!=oracle)&(fused==oracle)&m).sum().item()/count*100
        harmful=((local==oracle)&(fused!=oracle)&m).sum().item()/count*100
        return dict(count=count,beneficial_pct=beneficial,harmful_pct=harmful,net_gain_pp=beneficial-harmful)
    return {'all':rates(valid),'high_ambiguity':rates(high),'high_fraction_pct':float(100*high.sum()/valid.sum().clamp_min(1))}


def ambiguity_bins(output:dict,reference:torch.Tensor,mask:torch.Tensor,edges=(0.,0.2,0.4,0.6,0.8,1.000001))->list[dict]:
    e0=angular_error_degrees(output['candidates'][:,0],reference)
    e1=angular_error_degrees(output['candidates'][:,1],reference)
    oracle=torch.minimum(e0,e1)
    actual=torch.where(output['onehot'].argmax(dim=1)==0,e0,e1)
    en=output['entropy'][:,0]
    valid=mask[:,0].bool()
    rows=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        region=valid & (en>=lo) & (en<hi)
        n=int(region.sum())
        rows.append(dict(entropy_min=lo,entropy_max=hi,count=n,
                         oracle_mae_deg=float(oracle[region].detach().mean()) if n else None,
                         selection_error_deg=float((actual-oracle)[region].detach().mean()) if n else None,
                         final_mae_deg=float(actual[region].detach().mean()) if n else None))
    return rows
