import math
import torch
import pytest
from dpcu_polsfp.physics import physical_dolp,stokes,stokes_to_intensity,ZenithLookup,normal_candidates


def test_paper_fresnel_tabulated_dolp():
    theta=torch.deg2rad(torch.tensor([0.,15.,30.,45.,60.,70.,75.],dtype=torch.float64))
    expected=torch.tensor([0,.00265,.01132,.02828,.05711,.08287,.09500],dtype=torch.float64)
    assert torch.allclose(physical_dolp(theta),expected,atol=6e-6)


def test_stokes_roundtrip():
    s0=torch.ones(2,1,8,7)
    s1=torch.full_like(s0,.04)
    s2=torch.full_like(s0,-.07)
    desc=stokes(stokes_to_intensity(s0,s1,s2))
    for key,ex in [('s0',s0),('s1',s1),('s2',s2)]:assert torch.allclose(desc[key],ex,atol=2e-7)


def test_zenith_lut_and_outside_boundary():
    tab=ZenithLookup()
    angle=torch.tensor([0.,math.radians(30),math.radians(75)],dtype=torch.float64)
    theta,invalid=tab.invert(physical_dolp(angle))
    assert torch.allclose(angle,theta,atol=1e-6)
    assert not invalid.any()
    theta2,bad=tab.invert(torch.tensor([.2],dtype=torch.float64))
    assert bool(bad.item()) and theta2.item()==pytest.approx(math.radians(75))


def test_azimuth_candidate_opposites():
    th=torch.full((1,1,4,4),.5);psi=torch.zeros_like(th)
    cs=normal_candidates(th,psi)
    assert torch.allclose(cs[:,0,:2],-cs[:,1,:2])
    assert torch.allclose(cs[:,0,2],cs[:,1,2])
    assert torch.allclose(torch.linalg.vector_norm(cs,dim=2),torch.ones_like(cs[:,:,0]),atol=1e-6)
