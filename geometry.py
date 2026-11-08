"""Foreground-masked orthographic normal integration and geometry metrics."""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from scipy.ndimage import label as connected_components
from scipy.spatial import cKDTree


def integrate_normals(normals:np.ndarray,mask:np.ndarray,dx:float=1.,dy:float=1.,tol:float=1e-6,maxiter:int=2000)->np.ndarray:
    """Solve least-squares neighbor depth differences with one gauge fix / mask component.

    Input normals HxWx3 in right/down/towards-camera convention with nz>0,
    dx,dy metric spacing in x,y on the local orthographic surface.
    Outside mask returns NaN. Absolute depth is undetermined without an anchor.
    """
    normals=np.asarray(normals,dtype=np.float64)
    mask=np.asarray(mask,dtype=bool)
    if normals.ndim!=3 or normals.shape[-1]!=3 or normals.shape[:2]!=mask.shape:
        raise ValueError("normals must be HxWx3 and mask HxW")
    if dx<=0 or dy<=0:raise ValueError("dx,dy must be positive")
    h,w=mask.shape
    if not mask.any():return np.full(mask.shape,np.nan)
    if not np.all(np.isfinite(normals[mask])): raise ValueError("nonfinite normals on valid pixels")
    if np.any(normals[:,:,2][mask] <= 1e-4):raise ValueError("nz must be positive and away from zero")
    idx=np.full((h,w),-1,dtype=int)
    idx[mask]=np.arange(mask.sum())
    p=-normals[:,:,0]/np.where(mask,normals[:,:,2],1)
    q=-normals[:,:,1]/np.where(mask,normals[:,:,2],1)
    rows=[];cols=[];vals=[];rhs=[];r=0
    for y in range(h):
        for x in range(w):
            if not mask[y,x]: continue
            if x+1<w and mask[y,x+1]:
                rows.extend([r,r]);cols.extend([idx[y,x],idx[y,x+1]]);vals.extend([-1.,1.]);rhs.append(dx*(p[y,x]+p[y,x+1])/2);r+=1
            if y+1<h and mask[y+1,x]:
                rows.extend([r,r]);cols.extend([idx[y,x],idx[y+1,x]]);vals.extend([-1.,1.]);rhs.append(dy*(q[y,x]+q[y+1,x])/2);r+=1
    # Each disconnected mask component has an arbitrary gauge; fix one pixel.
    cc,count=connected_components(mask,np.array([[0,1,0],[1,1,1],[0,1,0]]))
    for k in range(1,count+1):
        y,x=np.argwhere(cc==k)[0]
        rows.append(r);cols.append(idx[y,x]);vals.append(1.);rhs.append(0.);r+=1
    matrix=coo_matrix((vals,(rows,cols)),shape=(r,int(mask.sum()))).tocsr()
    solution=lsmr(matrix,np.asarray(rhs),atol=tol,btol=tol,maxiter=maxiter)[0]
    z=np.full((h,w),np.nan);z[mask]=solution
    return z


def depth_rmse(pred:np.ndarray,gt:np.ndarray,mask:np.ndarray,diagonal:float,align_offset:bool=True)->float:
    if diagonal<=0:raise ValueError("positive diagonal needed")
    valid=np.asarray(mask,bool)&np.isfinite(pred)&np.isfinite(gt)
    if not valid.any():return float('nan')
    err=np.asarray(pred)[valid]-np.asarray(gt)[valid]
    if align_offset:err=err-err.mean()
    return float(np.sqrt((err*err).mean())/diagonal*100.)


def depth_to_points(z:np.ndarray,mask:np.ndarray,fx:float,fy:float,cx:float,cy:float)->np.ndarray:
    """Pinhole backprojection with metric z (not orthographic depth differentials)."""
    if fx<=0 or fy<=0:raise ValueError("fx and fy must be positive")
    y,x=np.where(np.asarray(mask,bool)&np.isfinite(z))
    zz=np.asarray(z)[y,x]
    return np.stack(((x-cx)*zz/fx,(y-cy)*zz/fy,zz),axis=-1)


def chamfer_distance(p:np.ndarray,q:np.ndarray,diagonal:float)->float:
    """Mean bidirectional Euclidean (unsquared) nearest-neighbor distance, in %D."""
    if diagonal<=0:raise ValueError("positive diagonal needed")
    a,b=np.asarray(p,dtype=np.float64),np.asarray(q,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=3 or b.shape[1]!=3 or not len(a) or not len(b):
        raise ValueError("nonempty Nx3 point sets required")
    return float(50*(cKDTree(b).query(a)[0].mean()+cKDTree(a).query(b)[0].mean())/diagonal)


def reference_diagonal(points:np.ndarray)->float:
    p=np.asarray(points)
    if not len(p):raise ValueError("no points")
    d=np.linalg.norm(p.max(axis=0)-p.min(axis=0))
    if d<=0:raise ValueError("degenerate reference bounding box")
    return float(d)
