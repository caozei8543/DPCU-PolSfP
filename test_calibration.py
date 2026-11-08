import numpy as np
import torch
from dpcu_polsfp.physics import physical_dolp
from dpcu_polsfp.calibration import calibrate


def test_calibration_returns_bounded_parameters():
    theta=np.linspace(0,np.deg2rad(75),90)
    rho=physical_dolp(torch.from_numpy(theta)).numpy()
    out=calibrate(theta,rho,step=.25,maxiter=30)
    assert 1.4<=out['eta']<=2.3 and 1.2<=out['rer']<=2.4
    assert out['loss']<1e-5
