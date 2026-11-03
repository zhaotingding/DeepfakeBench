"""只使用CPU核对公式、共享编码器梯度、零范数和初始化。"""
from pathlib import Path
import importlib.util,json
import torch
from torch import nn
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('normalized_model',root/'training/prd_r01/model.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(2)
def build(mode=None):
    torch.manual_seed(3407)  # 模式切换不应改变任何可训练初值。
    options={} if mode is None else dict(residual=mode)
    return module.ResponseModel(nn.Linear(3,3,bias=False).double(),3,{},['vqgan'],hidden=4,dropout=0,cached=True,slots=2,**options).double().eval()
signed=build();model=build('normalized_feature_diff')
assert all(torch.equal(v,model.state_dict()[k]) for k,v in signed.state_dict().items())
x=torch.tensor([[1.,-2.,4.],[-3.,2.,1.]],dtype=torch.float64)
y=torch.tensor([[2.,1.,-2.],[1.,-1.,3.]],dtype=torch.float64)
z=model.encoder(x);zr=model.encoder(y)
f=model(x,{'vqgan':y})['features'];expected=z/(z.norm(dim=1,keepdim=True)+1e-6)-zr/(zr.norm(dim=1,keepdim=True)+1e-6)
torch.testing.assert_close(f[:,:3],z,rtol=0,atol=0)  # 原图直通不能一起被归一化。
torch.testing.assert_close(f[:,3:],expected,rtol=0,atol=0)
sf=signed(x,{'vqgan':y})['features'];torch.testing.assert_close(sf[:,3:],z-zr,rtol=0,atol=0)  # 原默认路径数值不变。
unit=(z-zr)/((z-zr).norm(dim=1,keepdim=True)+1e-6)
assert not torch.allclose(expected,unit)  # 避免把两端归一化误写成差分单位化。
actual=torch.autograd.grad(f[:,3:].sum(),model.encoder.weight)[0]
def vjp(values):
    a=values.detach();n=a.norm(dim=1,keepdim=True)
    return torch.ones_like(a)/(n+1e-6)-a*a.sum(dim=1,keepdim=True)/(n*(n+1e-6)**2)  # 手写归一化Jacobian的全一向量乘积。
manual=vjp(z).T@x-vjp(zr).T@y  # 两分支均须对共享编码器产生梯度。
torch.testing.assert_close(actual,manual,rtol=1e-12,atol=1e-12)
zero=build('normalized_feature_diff')
with torch.no_grad():zero.encoder.weight.zero_()
zf=zero(x,{'vqgan':y})['features'];zg=torch.autograd.grad(zf[:,3:].sum(),zero.encoder.weight)[0]
assert torch.equal(zf,torch.zeros_like(zf)) and torch.isfinite(zg).all()  # 零范数路径输出零且梯度有限。
try:build('unknown')
except ValueError:pass
else:raise AssertionError('未知残差模式未拒绝')
print(json.dumps(dict(passed=True,formula_max_abs=float((f[:,3:]-expected).abs().max()),encoder_gradient_max_abs=float((actual-manual).abs().max()),zero_norm_finite=True,default_signed_preserved=True,initialization_identical=True,cpu_only=True)))
