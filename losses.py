"""Eqs. (7), (8), (10) implemented with foreground masks and stable denominators."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


def masked_mean(values:Tensor,mask:Tensor)->Tensor:
    w=mask.to(values.dtype)
    if w.shape!=values.shape:
        w=torch.broadcast_to(w,values.shape)
    return (values*w).sum()/w.sum().clamp_min(1)


def supervised_candidate_loss(output:dict,normal_gt:Tensor,mask:Tensor,lambda_n:float=0.5)->Tensor:
    cs=output['candidates']
    dots=(cs*normal_gt.unsqueeze(1)).sum(dim=2).clamp(-1,1)
    targets=dots.argmax(dim=1)
    ce=F.cross_entropy(output['logits'],targets,reduction='none').unsqueeze(1)
    expected=(output['probs']*(1-dots)).sum(dim=1,keepdim=True)
    return masked_mean(ce+lambda_n*expected,mask)


def polarimetric_loss(output:dict,mask:Tensor,delta_phy:float=1e-3)->Tensor:
    err=(output['expected_dolp']-output['dolp']).abs()+delta_phy
    s=output['log_scale']
    return masked_mean(torch.exp(-s)*err+s,mask)


def integrability_loss(normals:Tensor,mask:Tensor,eps:float=1e-4)->Tensor:
    """Orthographic curl: |d_y p - d_x q| on 2x2 valid neighborhoods."""
    if min(normals.shape[-2:])<2:
        return normals.sum()*0
    z=normals[:,2:3].clamp_min(eps)
    p=-normals[:,0:1]/z
    q=-normals[:,1:2]/z
    dpdy=p[:,:,1:,:-1]-p[:,:,:-1,:-1]
    dqdx=q[:,:,:-1,1:]-q[:,:,:-1,:-1]
    valid=mask[:,:,1:,:-1]*mask[:,:,:-1,:-1]*mask[:,:,:-1,1:]
    return masked_mean((dpdy-dqdx).abs(),valid)


def total_loss(output:dict,normal_gt:Tensor,mask:Tensor,lambda_n:float=0.5,
               lambda_phy:float=0.05,lambda_geo:float=0.10,delta_phy:float=1e-3)->dict[str,Tensor]:
    sup=supervised_candidate_loss(output,normal_gt,mask,lambda_n)
    phy=polarimetric_loss(output,mask,delta_phy)
    geo=integrability_loss(output['normals'],mask)
    total=sup+lambda_phy*phy+lambda_geo*geo
    return dict(total=total,supervised=sup,physics=phy,geometry=geo)
