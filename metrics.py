"""Exact per-pixel candidate-oracle decomposition and angular summaries."""
from __future__ import annotations
import numpy as np
import torch


def angular_error_degrees(pred:torch.Tensor,ref:torch.Tensor)->torch.Tensor:
    # atan2 cross/dot is more stable than acos near perfect agreement
    a=torch.nn.functional.normalize(pred,dim=1)
    b=torch.nn.functional.normalize(ref,dim=1)
    return torch.rad2deg(torch.atan2(torch.linalg.vector_norm(torch.cross(a,b,dim=1),dim=1),
                                      (a*b).sum(dim=1).clamp(-1,1)))


def angular_summary(pred,reference,mask)->dict[str,float]:
    e=angular_error_degrees(pred,reference)
    m=mask[:,0]>0
    v=e[m].detach().cpu().numpy()
    if not len(v):raise ValueError("No valid pixels")
    return {'mae_deg':float(v.mean()),'median_deg':float(np.median(v)),
            'rmse_deg':float(np.sqrt((v*v).mean())),
            'lt_11_25_pct':float(100*(v<11.25).mean()),
            'lt_22_5_pct':float(100*(v<22.5).mean()),
            'lt_30_pct':float(100*(v<30).mean())}


def oracle_decomposition(candidates,ref,selected,mask)->dict[str,float]:
    e0=angular_error_degrees(candidates[:,0],ref)
    e1=angular_error_degrees(candidates[:,1],ref)
    o=torch.minimum(e0,e1)
    actual=torch.where(selected==0,e0,e1)
    valid=mask[:,0]>0
    preferred=torch.where(e0<=e1,0,1)
    return {'oracle_mae_deg':float(o[valid].detach().mean()),
            'selection_error_deg':float((actual-o)[valid].detach().mean()),
            'selection_accuracy_pct':float(100*(selected[valid]==preferred[valid]).float().mean()),
            'actual_mae_deg':float(actual[valid].detach().mean())}
