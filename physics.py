"""Stokes descriptors and an *effective, real-index* emission/reflection model.

Paper Eq. (1)--(4). The single-real-index Fresnel implementation is an explicit
reference-model assumption; it is not a measured complex MWIR material model.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor


def stokes(images: Tensor, eps: float = 1e-8) -> dict[str, Tensor]:
    """images: Bx4xHxW, ordered I0, I45, I90, I135, linear-calibrated intensities."""
    if images.ndim != 4 or images.shape[1] != 4:
        raise ValueError("Expected [batch, 4, height, width], angles (0,45,90,135)")
    if not torch.isfinite(images).all():
        raise ValueError("Nonfinite radiometric measurements")
    i0,i45,i90,i135 = images[:,0:1],images[:,1:2],images[:,2:3],images[:,3:4]
    s0 = i0+i90
    s1 = i0-i90
    s2 = i45-i135
    rho = (torch.sqrt(s1.square()+s2.square())/(s0.clamp_min(eps))).clamp_min(0)
    psi = 0.5*torch.atan2(s2,s1)
    return dict(s0=s0,s1=s1,s2=s2,dolp=rho,aop=psi)


def fresnel_reflectances(theta: Tensor, eta: Tensor | float) -> tuple[Tensor,Tensor]:
    """Unpolarized incidence from n=1 to lossless isotropic dielectric n=eta > 1."""
    n=torch.as_tensor(eta,device=theta.device,dtype=theta.dtype)
    if bool(torch.any(n<=1)):
        raise ValueError("This lossless Fresnel reference requires eta > 1")
    ci=torch.cos(theta)
    ct=torch.sqrt((1-(torch.sin(theta)/n)**2).clamp_min(0))
    rs=((ci-n*ct)/(ci+n*ct).clamp_min(1e-8)).square()
    rp=((n*ci-ct)/(n*ci+ct).clamp_min(1e-8)).square()
    return rp,rs


def signed_polarization(theta:Tensor,eta:float=1.82,rer:float=1.65)->Tensor:
    """Signed contrast m from manuscript Eq. (2), with emissivity=1-reflectance."""
    if rer<=0: raise ValueError("RER must be positive")
    rp,rs=fresnel_reflectances(theta,eta)
    ep,es=1-rp,1-rs
    return ((rp-rs)+rer*(ep-es))/((rp+rs)+rer*(ep+es)).clamp_min(1e-8)


def physical_dolp(theta:Tensor,eta:float=1.82,rer:float=1.65)->Tensor:
    return signed_polarization(theta,eta,rer).abs()


class ZenithLookup:
    """Monotone 0--75 degree LUT with a validity mask before boundary clipping.

    For alternative parameters, the DoLP response may not be monotone. Such
    inputs fail loudly instead of silently selecting a wrong inversion branch.
    """
    def __init__(self,eta:float=1.82,rer:float=1.65,max_degrees:float=75.0,step:float=0.1):
        self.eta,self.rer,self.max_degrees,self.step=eta,rer,max_degrees,step
        angles=torch.linspace(0, math.radians(max_degrees), round(max_degrees/step)+1, dtype=torch.float64)
        response=physical_dolp(angles,eta,rer)
        if not bool(torch.all(response[1:] > response[:-1])):
            raise ValueError("DoLP is not strictly increasing on chosen inversion branch")
        self.theta=angles
        self.response=response

    def invert(self, measured:Tensor)->tuple[Tensor,Tensor]:
        """Return theta0 and out-of-calibration-range mask, both same shape as DoLP."""
        v=self.response.to(measured.device,measured.dtype)
        t=self.theta.to(measured.device,measured.dtype)
        invalid=(measured < v[0]) | (measured > v[-1]) | ~torch.isfinite(measured)
        x=torch.nan_to_num(measured,nan=0.0,posinf=float(v[-1]),neginf=0.0).clamp(float(v[0]),float(v[-1]))
        j=torch.searchsorted(v.contiguous(),x.contiguous()).clamp(1,len(v)-1)
        lo,hi=v[j-1],v[j]
        alpha=(x-lo)/(hi-lo).clamp_min(1e-12)
        return t[j-1]+alpha*(t[j]-t[j-1]),invalid


def normal_candidates(theta:Tensor,psi:Tensor,offset_degrees:float=90.)->Tensor:
    """B x 2 x 3 x H x W, normal candidates separated by pi azimuth."""
    if theta.shape!=psi.shape or theta.ndim!=4 or theta.shape[1]!=1:
        raise ValueError("theta and psi must be [B,1,H,W]")
    phi=psi+math.radians(offset_degrees)
    nx=torch.sin(theta)*torch.cos(phi)
    ny=torch.sin(theta)*torch.sin(phi)
    nz=torch.cos(theta)
    n0=torch.cat((nx,ny,nz),dim=1)
    n1=torch.cat((-nx,-ny,nz),dim=1)
    return torch.stack((n0,n1),dim=1)


def stokes_to_intensity(s0:Tensor,s1:Tensor,s2:Tensor)->Tensor:
    """Synthesize ideal linear-analyzer measurements; angle order 0,45,90,135."""
    return torch.cat(((s0+s1)/2,(s0+s2)/2,(s0-s1)/2,(s0-s2)/2),dim=1)
