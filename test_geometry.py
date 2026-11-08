import numpy as np
from dpcu_polsfp.geometry import integrate_normals,depth_rmse,chamfer_distance,depth_to_points,reference_diagonal


def test_poisson_integrates_planar_surface():
    h,w=15,16
    ax,by=.1,-.2
    normal=np.array([-ax,-by,1.]);normal/=np.linalg.norm(normal)
    field=np.broadcast_to(normal,(h,w,3)).copy()
    mask=np.ones((h,w),bool)
    z=integrate_normals(field,mask,dx=.02,dy=.03)
    yy,xx=np.mgrid[:h,:w]
    gt=ax*xx*.02+by*yy*.03
    assert depth_rmse(z,gt,mask,1.0)<1e-3


def test_disconnected_components_and_chamfer():
    mask=np.zeros((9,9),bool);mask[:3,:3]=1;mask[6:,6:]=1
    n=np.zeros((9,9,3));n[...,2]=1
    z=integrate_normals(n,mask)
    assert np.isfinite(z[mask]).all() and np.isnan(z[~mask]).all()
    pts=depth_to_points(np.ones((9,9)),mask,100,100,4,4)
    assert chamfer_distance(pts,pts,reference_diagonal(pts))==0.
