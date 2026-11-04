"""使用CPU核验09各臂的主路径配对、辅助梯度来源和推理行为。"""
from pathlib import Path
import hashlib
import json
import sys

import torch
from torch import nn
from torch.nn import functional as F

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'training'))
from prd_r01.auxiliary_supervision import trainable_identity, training_loss
from prd_r01.model import ResponseModel


def model(mode):
    torch.manual_seed(3407)
    encoder = nn.Sequential(nn.Linear(3, 4, bias=False), nn.Dropout(0.1))
    return ResponseModel(encoder, 4, {}, ['vqgan'], hidden=5, dropout=0.2,
                         cached=True, slots=2, residual='signed_diff', auxiliary_input=mode).double()


torch.set_num_threads(2)
modes = ['none', 'original_aux', 'residual_aux']
models, identities, after_init = {}, {}, {}
for mode in modes:
    models[mode] = model(mode)
    after_init[mode] = torch.get_rng_state().clone()
    identities[mode] = trainable_identity(models[mode])

plain = models['none']
legacy_hash = hashlib.sha256()
for name, parameter in plain.named_parameters():
    if parameter.requires_grad:
        legacy_hash.update(name.encode()); legacy_hash.update(parameter.detach().numpy().tobytes())
assert identities['none']['trainable_initial_sha256'] == legacy_hash.hexdigest()
for mode in modes[1:]:
    assert torch.equal(after_init[mode],after_init['none'])
    assert identities[mode]['main_trainable_initial_sha256']==legacy_hash.hexdigest()
    assert identities[mode]['auxiliary_parameters']==10  # 玩具模型4→2；真实头1024→2应为2050。
    assert identities[mode]['trainable_parameters']==identities['none']['trainable_parameters']+10
    for name,tensor in plain.state_dict().items():
        assert torch.equal(tensor,models[mode].state_dict()[name]),name
assert identities['original_aux']==identities['residual_aux']

x=torch.tensor([[1.,-2.,4.],[-3.,2.,1.],[2.,3.,0.],[-1.,4.,2.]],dtype=torch.float64)
y=torch.tensor([[2.,1.,-2.],[1.,-1.,3.],[-3.,2.,1.],[2.,0.,-3.]],dtype=torch.float64)
labels=torch.tensor([1,0,0,1])
forward_rng,logits,auxiliary_grad={},{},{}
for mode,m in models.items():
    embeddings=[]
    hook=m.encoder.register_forward_hook(lambda mod,args,value:embeddings.append(value))
    torch.manual_seed(811)
    m.train();out=m(x,{'vqgan':y})
    forward_rng[mode]=torch.get_rng_state().clone();logits[mode]=out['logits'].detach()
    total,main,aux=training_loss(out,labels,mode,1.0)
    assert torch.isfinite(total)
    z,zr=embeddings;torch.testing.assert_close(out['features'],torch.cat([z,z-zr],dim=1),rtol=0,atol=0)
    main_grad=torch.autograd.grad(main,m.encoder[0].weight,retain_graph=True)[0]
    assert torch.isfinite(main_grad).all() and main_grad.norm()>0
    if mode=='none':
        assert aux is None and 'auxiliary_logits' not in out
        torch.testing.assert_close(total,F.cross_entropy(out['logits'],labels),rtol=0,atol=0)
    else:
        expected_input=z if mode=='original_aux' else z-zr
        independent=F.cross_entropy(F.linear(expected_input,m.auxiliary_head.weight,m.auxiliary_head.bias),labels)
        torch.testing.assert_close(aux,independent,rtol=0,atol=0)
        torch.testing.assert_close(total,main+independent,rtol=0,atol=0)
        gz,gzr,gw=torch.autograd.grad(aux,(z,zr,m.encoder[0].weight),allow_unused=True,retain_graph=True)
        assert gz is not None and gz.norm()>0 and torch.isfinite(gw).all() and gw.norm()>0
        if mode=='original_aux':
            assert gzr is None  # 原图辅助CE没有重构视图梯度。
        else:
            torch.testing.assert_close(gzr,-gz,rtol=0,atol=0)  # 有符号差将相反梯度送回两视图。
        auxiliary_grad[mode]=dict(z_norm=float(gz.norm()),zr_norm=None if gzr is None else float(gzr.norm()),encoder_norm=float(gw.norm()))
    total.backward();hook.remove()
for mode in modes[1:]:
    torch.testing.assert_close(logits[mode],logits['none'],rtol=0,atol=0)
    assert torch.equal(forward_rng[mode],forward_rng['none'])
    m=models[mode].eval()
    def reject_aux_call(*args):raise AssertionError('评估调用了辅助头')
    hook=m.auxiliary_head.register_forward_pre_hook(reject_aux_call)
    with torch.no_grad():
        evaluation=m(x,{'vqgan':y});main_logits=evaluation['logits'].clone()
        assert set(evaluation)=={'logits','features'}
        m.auxiliary_head.weight.fill_(999);m.auxiliary_head.bias.fill_(-100)
        torch.testing.assert_close(m(x,{'vqgan':y})['logits'],main_logits,rtol=0,atol=0)
    hook.remove()

print(json.dumps(dict(passed=True,cpu_only=True,main_initial_and_buffers_equal=True,
                      auxiliary_initial_equal=True,construction_and_forward_rng_equal=True,
                      main_logits_equal_before_update=True,main_auxiliary_losses_checked=True,
                      auxiliary_gradients=auxiliary_grad,inference_skips_auxiliary_head=True)))
