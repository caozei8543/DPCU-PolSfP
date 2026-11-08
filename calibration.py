"""Fitting effective eta/RER to training-set DoLP and ground truth zenith."""
from __future__ import annotations
import numpy as np
import torch
from scipy.optimize import minimize
from .physics import physical_dolp


def calibrate(theta:np.ndarray,dolp:np.ndarray,bounds=((1.40,2.30),(1.20,2.40)),
              step:float=0.05,huber_delta:float=0.02,maxiter:int=500)->dict:
    theta=np.asarray(theta,np.float64).ravel()
    dolp=np.asarray(dolp,np.float64).ravel()
    valid=np.isfinite(theta)&np.isfinite(dolp)&(theta>=0)&(theta<=np.deg2rad(75))
    theta,dolp=theta[valid],dolp[valid]
    if not len(theta):raise ValueError("No valid training calibration pixels")
    t=torch.from_numpy(theta)
    def objective(x):
        pred=physical_dolp(t,float(x[0]),float(x[1])).numpy()
        err=np.abs(pred-dolp)
        d=huber_delta
        return float(np.mean(np.where(err<=d,0.5*err*err,d*(err-0.5*d))))
    xs=np.arange(bounds[0][0],bounds[0][1]+step/2,step)
    ys=np.arange(bounds[1][0],bounds[1][1]+step/2,step)
    initial=min(((float(x),float(y)) for x in xs for y in ys),key=objective)
    res=minimize(objective,initial,method='L-BFGS-B',bounds=bounds,
                 options={'maxiter':maxiter,'ftol':1e-13})
    return {'eta':float(res.x[0]),'rer':float(res.x[1]),'loss':float(res.fun),
            'optimizer_success':bool(res.success),'optimizer_message':str(res.message),
            'initial_eta':initial[0],'initial_rer':initial[1], 'num_pixels':len(theta)}
