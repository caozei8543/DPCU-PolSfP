import torch
from dpcu_polsfp.demo import make_demo
from dpcu_polsfp.data import NPZDataset,augment_scene
from dpcu_polsfp.physics import stokes


def test_scene_loader_and_rotated_stokes(tmp_path):
    make_demo(tmp_path,size=32)
    original=NPZDataset(tmp_path/'train.csv')[0]
    assert original['images'].shape==(4,32,32)
    old=stokes(original['images'][None]); flipped=augment_scene(original,flip_x=True)
    new=stokes(flipped['images'][None])
    assert torch.allclose(old['s1'].flip(-1),new['s1'],atol=2e-6)
    assert torch.allclose(-old['s2'].flip(-1),new['s2'],atol=2e-6)
    rotated=augment_scene(original,rot90=1)
    r=stokes(rotated['images'][None])
    assert torch.allclose(r['s1'],-torch.rot90(old['s1'],1,(-2,-1)),atol=2e-6)
    assert torch.allclose(r['s2'],-torch.rot90(old['s2'],1,(-2,-1)),atol=2e-6)
    assert torch.allclose(rotated['normals'][0],torch.rot90(original['normals'][1],1,(-2,-1)))
    assert torch.allclose(rotated['normals'][1],-torch.rot90(original['normals'][0],1,(-2,-1)))
