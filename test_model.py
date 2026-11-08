import torch
from dpcu_polsfp.demo import make_demo
from dpcu_polsfp.data import NPZDataset
from dpcu_polsfp.model import DPCUPolSfP,ModelConfig
from dpcu_polsfp.losses import total_loss
from dpcu_polsfp.metrics import angular_summary,oracle_decomposition
from dpcu_polsfp.analysis import decision_reversal,ambiguity_bins


def test_shapes_training_gradients_and_inference(tmp_path):
    make_demo(tmp_path,size=24)
    scene=NPZDataset(tmp_path/'train.csv')[0]
    model=DPCUPolSfP(ModelConfig(width=8))
    im=scene['images'][None];n=scene['normals'][None];m=scene['mask'][None]
    model.train()
    o=model(im,stochastic=True)
    assert o['candidates'].shape==(1,2,3,24,24)
    assert o['normals'].shape==(1,3,24,24)
    losses=total_loss(o,n,m)
    assert torch.isfinite(losses['total'])
    losses['total'].backward()
    assert model.correction_head.weight.grad is not None
    model.eval();o=model(im)
    assert torch.all((o['onehot']==0)|(o['onehot']==1))
    a=angular_summary(o['normals'],n,m)
    d=oracle_decomposition(o['candidates'],n,o['onehot'].argmax(dim=1),m)
    assert abs(a['mae_deg']-d['actual_mae_deg'])<1e-4
    assert abs(d['oracle_mae_deg']+d['selection_error_deg']-a['mae_deg'])<1e-4
    assert decision_reversal(o,n,m)['all']['count']>0
    assert len(ambiguity_bins(o,n,m))==5


def test_gate_variants():
    for gate in ('entropy','fixed','dolp','softmax_confidence','learned','local','context'):
        model=DPCUPolSfP(ModelConfig(width=8,gate=gate)).eval()
        x=torch.rand(1,4,12,12)
        with torch.no_grad(): out=model(x)
        assert out['normals'].shape==(1,3,12,12)
        assert bool(((out['gate']>=0)&(out['gate']<=1)).all())
