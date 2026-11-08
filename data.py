"""Dataset contracts and physically consistent image/normal augmentation.

No public benchmark images are distributed. Each NPZ is a radiometrically
calibrated four-angle measurement with independent GT normals (when training).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .physics import stokes_to_intensity

REQUIRED=("i0","i45","i90","i135")


def load_scene(path:str|Path,need_normals:bool=False)->dict:
    with np.load(path,allow_pickle=False) as data:
        missing=[x for x in REQUIRED if x not in data.files]
        if missing: raise ValueError(f"{path}: missing {missing}")
        imgs=[np.asarray(data[x],dtype=np.float32) for x in REQUIRED]
        shape=imgs[0].shape
        if len(shape)!=2 or any(im.shape!=shape for im in imgs): raise ValueError(f"{path}: channels must be same HxW")
        if not all(np.isfinite(im).all() for im in imgs): raise ValueError(f"{path}: nonfinite intensities")
        images=torch.from_numpy(np.stack(imgs))
        if 'mask' in data.files:
            m=np.asarray(data['mask']);
            if m.shape!=shape: raise ValueError(f"{path}: mask mismatch")
            mask=torch.from_numpy((m>0).astype(np.float32))[None]
        else: mask=torch.ones((1,*shape),dtype=torch.float32)
        result=dict(images=images,mask=mask)
        if 'normals' in data.files:
            normals=np.asarray(data['normals'],dtype=np.float32)
            if normals.shape==(*shape,3): normals=np.moveaxis(normals,-1,0)
            if normals.shape!=(3,*shape): raise ValueError(f"{path}: normals shape")
            if not np.isfinite(normals).all(): raise ValueError(f"{path}: nonfinite normals")
            nr=torch.from_numpy(normals)
            result['normals']=torch.nn.functional.normalize(nr,dim=0,eps=1e-8)
        elif need_normals: raise ValueError(f"{path}: normals required for training/evaluation")
        if 'depth' in data.files:
            z=np.asarray(data['depth'],dtype=np.float32)
            if z.shape!=shape: raise ValueError(f"{path}: depth shape")
            result['depth']=torch.from_numpy(z)[None]
    return result


def augment_scene(scene:dict,flip_x:bool=False,flip_y:bool=False,rot90:int=0)->dict:
    """Stokes and normal vector camera-frame transformations under image reflection/rotation.

    Rotation is visually CCW while image x points right and y points down;
    therefore (nx,ny) maps to (ny,-nx). Transform Stokes as spin-2. All four analyzer frames are regenerated.
    """
    image=scene['images'].unsqueeze(0)
    i0,i45,i90,i135=image[:,0:1],image[:,1:2],image[:,2:3],image[:,3:4]
    s0,s1,s2=i0+i90,i0-i90,i45-i135
    nm=scene.get('normals')
    x,y,z=(nm[0:1].unsqueeze(0),nm[1:2].unsqueeze(0),nm[2:3].unsqueeze(0)) if nm is not None else (None,None,None)
    mask=scene['mask'].unsqueeze(0)
    depth=scene.get('depth')
    if depth is not None: depth=depth.unsqueeze(0)
    def tr(t,op):return op(t) if t is not None else None
    if flip_x:
        flip=lambda t:torch.flip(t,(-1,))
        s0,s1,s2=flip(s0),flip(s1),-flip(s2)
        if nm is not None:x,y,z=-flip(x),flip(y),flip(z)
        mask=flip(mask);depth=tr(depth,flip)
    if flip_y:
        flip=lambda t:torch.flip(t,(-2,))
        s0,s1,s2=flip(s0),flip(s1),-flip(s2)
        if nm is not None:x,y,z=flip(x),-flip(y),flip(z)
        mask=flip(mask);depth=tr(depth,flip)
    for _ in range(rot90%4):
        rot=lambda t:torch.rot90(t,1,(-2,-1))
        s0,s1,s2=rot(s0),-rot(s1),-rot(s2)
        if nm is not None:
            ox,oy=rot(x),rot(y);x,y=oy,-ox;z=rot(z)
        mask=rot(mask);depth=tr(depth,rot)
    out=dict(images=stokes_to_intensity(s0,s1,s2)[0],mask=mask[0])
    if nm is not None:out['normals']=torch.cat((x,y,z),dim=1)[0]
    if depth is not None:out['depth']=depth[0]
    return out


def random_crop_scene(scene: dict, size: int) -> dict:
    """Apply the same square crop to all measurements, masks, normals and depth."""
    if size <= 0:
        raise ValueError("size must be a positive integer")
    height, width = scene['images'].shape[-2:]
    if height < size or width < size:
        raise ValueError(f"Crop {size}x{size} exceeds image {height}x{width}")
    y = int(torch.randint(0, height - size + 1, (1,)))
    x = int(torch.randint(0, width - size + 1, (1,)))
    return {k: v[..., y:y+size, x:x+size].clone() for k, v in scene.items()}


class NPZDataset(Dataset):
    def __init__(self,manifest:str|Path,augment:bool=False,need_normals:bool=True,
                 crop_size:int|None=None,intensity_gain:float=0.05,noise_sigma:float=0.005):
        self.manifest=Path(manifest)
        if not self.manifest.exists(): raise FileNotFoundError(f"Manifest missing: {manifest}")
        with self.manifest.open(newline='',encoding='utf-8') as f:
            self.rows=list(csv.DictReader(f))
        if not self.rows or not {'id','path'}.issubset(self.rows[0]):
            raise ValueError("Manifest requires id,path columns and at least one row")
        self.augment,self.need_normals=augment,need_normals
        self.crop_size=crop_size
        self.intensity_gain=intensity_gain
        self.noise_sigma=noise_sigma
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];path=Path(row['path'])
        if not path.is_absolute(): path=self.manifest.parent/path
        scene=load_scene(path,need_normals=self.need_normals)
        if self.crop_size is not None:
            scene=random_crop_scene(scene,self.crop_size)
        if self.augment:
            scene=augment_scene(scene, bool(torch.randint(0,2,(1,))), bool(torch.randint(0,2,(1,))),int(torch.randint(0,4,(1,))))
            # Radiance gain is common to all analyzer frames; sensor noise is
            # independent per image. No augmentation is used for evaluation.
            gain=1+(2*torch.rand(1).item()-1)*self.intensity_gain
            scene['images']=(scene['images']*gain+
                torch.randn_like(scene['images'])*self.noise_sigma).clamp_min(0)
        return scene
