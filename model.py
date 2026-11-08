"""Paper-aligned *reference* neural architecture.

The manuscript specifies branches and losses but not every convolution/width.
Encoder, context block, and heads below are documented implementation choices.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .physics import ZenithLookup, stokes, normal_candidates, physical_dolp


def convblock(ci:int,co:int,dilation:int=1)->nn.Sequential:
    return nn.Sequential(nn.Conv2d(ci,co,3,padding=dilation,dilation=dilation),nn.GroupNorm(8,co),nn.SiLU())

@dataclass
class ModelConfig:
    eta:float=1.82
    rer:float=1.65
    offset_degrees:float=90.0
    max_zenith_degrees:float=75.0
    delta_max_degrees:float=10.0
    width:int=32
    log_scale_min:float=-7.0
    log_scale_max:float=3.0
    gate:str="entropy" # entropy,fixed,dolp,softmax_confidence,learned,local,context
    gumbel_tau:float=1.0

class DPCUPolSfP(nn.Module):
    def __init__(self, cfg:ModelConfig|None=None):
        super().__init__()
        self.cfg=cfg or ModelConfig()
        c=self.cfg.width
        if c%8: raise ValueError("width must be divisible by 8")
        self.lookup=ZenithLookup(self.cfg.eta,self.cfg.rer,self.cfg.max_zenith_degrees)
        self.encoder=nn.Sequential(convblock(8,c),convblock(c,c),convblock(c,c))
        self.local_head=nn.Conv2d(c,2,1)
        self.correction_head=nn.Conv2d(c,1,1)
        self.obs_scale_head=nn.Conv2d(c,1,1)
        self.context_branch=nn.Sequential(convblock(c+6,c),convblock(c,c,dilation=2),convblock(c,c,dilation=4),nn.Conv2d(c,2,1))
        self.learned_gate=nn.Sequential(convblock(c+6,c),nn.Conv2d(c,1,1))

    def forward(self,images:Tensor,tau:float|None=None,stochastic:bool|None=None)->dict[str,Tensor]:
        s=stokes(images)
        v=torch.cat([images,s['s0'],s['dolp'],torch.cos(2*s['aop']),torch.sin(2*s['aop'])],dim=1)
        features=self.encoder(v)
        theta0,invalid=self.lookup.invert(s['dolp'].detach())
        bounded=math.radians(self.cfg.delta_max_degrees)*torch.tanh(self.correction_head(features))
        theta=(theta0+bounded).clamp(0,math.radians(self.cfg.max_zenith_degrees))
        candidates=normal_candidates(theta,s['aop'],self.cfg.offset_degrees)
        local=self.local_head(features)
        context_input=torch.cat((features,candidates.flatten(start_dim=1,end_dim=2)),dim=1)
        context=self.context_branch(context_input)
        q=F.softmax(local,dim=1)
        entropy=-(q*q.clamp_min(1e-8).log()).sum(dim=1,keepdim=True)/math.log(2)
        mode=self.cfg.gate
        if mode=="entropy": r=entropy
        elif mode=="fixed": r=torch.full_like(entropy,0.5)
        elif mode=="dolp": r=(1-s['dolp']/0.1).clamp(0,1)
        elif mode=="softmax_confidence": r=1-q.max(dim=1,keepdim=True).values
        elif mode=="learned": r=torch.sigmoid(self.learned_gate(context_input))
        elif mode=="local": r=torch.zeros_like(entropy)
        elif mode=="context": r=torch.ones_like(entropy)
        else: raise ValueError(f"Unknown gate: {mode}")
        logits=(1-r)*local+r*context
        probs=F.softmax(logits,dim=1)
        if stochastic is None: stochastic=self.training
        if stochastic:
            # forward one-hot, surrogate gradient; categorical sampling for training
            onehot=F.gumbel_softmax(logits,tau=tau or self.cfg.gumbel_tau,hard=True,dim=1)
        else:
            onehot=F.one_hot(logits.argmax(dim=1),num_classes=2).permute(0,3,1,2).to(logits.dtype)
        selected=(onehot.unsqueeze(2)*candidates).sum(dim=1)
        return dict(normals=selected,candidates=candidates,logits=logits,probs=probs,onehot=onehot,
                    theta=theta,theta0=theta0,invalid_inversion=invalid,entropy=entropy,gate=r,
                    local_logits=local,context_logits=context,
                    log_scale=self.obs_scale_head(features).clamp(self.cfg.log_scale_min,self.cfg.log_scale_max),
                    dolp=s['dolp'],expected_dolp=physical_dolp(theta,self.cfg.eta,self.cfg.rer))
