import importlib.util, json, sys
from pathlib import Path
import torch
from torch import nn
root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("response_test",root/"training/prd_r01/model.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(2)
def model(mode):
    torch.manual_seed(3407)  # 两模式必须从完全相同的权重出发。
    return module.ResponseModel(nn.Linear(3,3,bias=False).double(),3,{},["vqgan"],hidden=4,dropout=0,cached=True,slots=2,residual=mode).double()
signed=model("signed_diff");absolute=model("absolute_diff")
assert all(torch.equal(v,absolute.state_dict()[k]) for k,v in signed.state_dict().items())  # 映射切换不改变初值。
x=torch.tensor([[1.,-2.,4.],[-3.,2.,1.]],dtype=torch.float64)
y=torch.tensor([[2.,1.,-2.],[1.,-1.,3.]],dtype=torch.float64)
signed.eval();absolute.eval()
sf=signed(x,{"vqgan":y})["features"];af=absolute(x,{"vqgan":y})["features"]
expected=signed.encoder(x)-signed.encoder(y)
assert (expected>0).any() and (expected<0).any()  # 确认测例真正包含正负差。
torch.testing.assert_close(sf[:,3:],expected,rtol=0,atol=0)
torch.testing.assert_close(af[:,:3],sf[:,:3],rtol=0,atol=0)
torch.testing.assert_close(af[:,3:],expected.abs(),rtol=0,atol=0)
actual_grad=torch.autograd.grad(af[:,3:].sum(),absolute.encoder.weight)[0]
w=absolute.encoder.weight.detach().clone().requires_grad_(True)
manual_grad=torch.autograd.grad((x@w.T-y@w.T).abs().sum(),w)[0]
torch.testing.assert_close(actual_grad,manual_grad,rtol=0,atol=0)  # 独立公式验证两支编码器梯度。
assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum()>0
try: model("unknown")
except ValueError: pass
else: raise AssertionError("未知模式未拒绝")
print(json.dumps({"passed":True,"signed_default_equivalent":True,"absolute_formula":True,"encoder_gradient_error":float((actual_grad-manual_grad).abs().max()),"identical_initialization":True,"input_features":6,"cpu_only":True}))
