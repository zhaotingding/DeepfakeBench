"""在CPU区分05和06的前向、共享编码器反向及零残差行为。"""
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn


root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('unit_model_test', root/'training/prd_r01/model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(1)
epsilon = 1e-6


def model(mode=None):
    torch.manual_seed(3407)
    options = {} if mode is None else {'residual': mode}
    return module.ResponseModel(nn.Linear(3, 3, bias=False).double(), 3, {}, ['vqgan'],
                                hidden=4, dropout=0, cached=True, slots=2, **options).double().eval()


signed = model()
separate = model('normalized_feature_diff')
unit = model('unit_residual')
for other in (separate, unit):
    assert all(torch.equal(v, other.state_dict()[k]) for k, v in signed.state_dict().items())

# 第二行同方向但幅度不同，第三行为完全相同特征，第四行差值与ε同量级。
x = torch.tensor([[1.,-2.,4.],[2.,0.,0.],[1.,-3.,2.],[1.,2.,1.]], dtype=torch.float64)
y = torch.tensor([[2.,1.,-2.],[1.,0.,0.],[1.,-3.,2.],[1.,2.,1.-1e-6]], dtype=torch.float64)
z, zr = unit.encoder(x), unit.encoder(y)
difference = z - zr
length = difference.norm(dim=1, keepdim=True)
expected = difference / (length + epsilon)
expected_separate = z/(z.norm(dim=1,keepdim=True)+epsilon) - zr/(zr.norm(dim=1,keepdim=True)+epsilon)

features = unit(x, {'vqgan': y})['features']
signed_features = signed(x, {'vqgan': y})['features']
separate_features = separate(x, {'vqgan': y})['features']
torch.testing.assert_close(features[:,:3], z, rtol=0, atol=0)  # 原图槽保留幅度。
torch.testing.assert_close(features[:,3:], expected, rtol=0, atol=0)
torch.testing.assert_close(signed_features[:,3:], difference, rtol=0, atol=0)
torch.testing.assert_close(separate_features[:,3:], expected_separate, rtol=0, atol=0)
assert features[1,3:].norm() > .99 and separate_features[1,3:].norm() < 1e-4
assert torch.equal(features[2,3:], torch.zeros_like(features[2,3:]))
assert not torch.allclose(features[3,3:], difference[3]/length[3].clamp_min(epsilon))

cotangent = torch.tensor([[1.,2.,-3.],[-2.,1.,.5],[1.,3.,2.],[2.,-.5,1.]], dtype=torch.float64)
actual = torch.autograd.grad((features[:,3:]*cotangent).sum(), unit.encoder.weight)[0]
v = difference.detach()
n = length.detach()
safe_n = torch.where(n>0, n, torch.ones_like(n))  # 零残差时第二项为零。
vjp = cotangent/(n+epsilon) - v*(v*cotangent).sum(dim=1,keepdim=True)/(safe_n*(n+epsilon)**2)
manual = vjp.T @ (x-y)  # 解析Jacobian同时包含原图和重构视图对同一权重的梯度。
torch.testing.assert_close(actual, manual, rtol=1e-9, atol=1e-8)

zero = model('unit_residual')
with torch.no_grad():
    zero.encoder.weight.zero_()
zero_features = zero(x, {'vqgan': y})['features']
zero_gradient = torch.autograd.grad((zero_features[:,3:]*cotangent).sum(), zero.encoder.weight)[0]
assert torch.equal(zero_features, torch.zeros_like(zero_features)) and torch.isfinite(zero_gradient).all()
torch.testing.assert_close(zero_gradient, (cotangent/epsilon).T@(x-y), rtol=1e-9, atol=1e-8)

print(json.dumps(dict(passed=True, formula_max_abs=float((features[:,3:]-expected).abs().max()),
                      encoder_gradient_max_abs=float((actual-manual).abs().max()),
                      collinear_unit_norm=float(features[1,3:].norm()), collinear_separate_norm=float(separate_features[1,3:].norm()),
                      zero_residual_and_gradient_checked=True, epsilon_addition_checked=True,
                      signed_and_05_preserved=True, initialization_identical=True, cpu_only=True)))
