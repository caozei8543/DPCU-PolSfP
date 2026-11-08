import torch
from dpcu_polsfp.demo import make_demo
from dpcu_polsfp.data import NPZDataset,random_crop_scene


def test_crop_shapes_and_geometric_consistency(tmp_path):
    make_demo(tmp_path,size=30)
    ds=NPZDataset(tmp_path/'train.csv',crop_size=24)
    b=ds[0]
    assert b['images'].shape==(4,24,24)
    assert b['mask'].shape==(1,24,24)
    assert b['normals'].shape==(3,24,24)
    aug=NPZDataset(tmp_path/'train.csv',augment=True,crop_size=24)[0]
    assert aug['images'].shape==(4,24,24)
    assert torch.isfinite(aug['images']).all()
