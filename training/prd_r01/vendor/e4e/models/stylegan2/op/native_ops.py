"""冻结StyleGAN推理所用的PyTorch算子；无CUDA编译依赖。"""
import torch
from torch import nn
from torch.nn import functional as F

def fused_leaky_relu(input,bias,negative_slope=.2,scale=2**.5):
    shape=[1,-1]+[1]*(input.ndim-2)  # 与官方通道偏置广播一致。
    return F.leaky_relu(input+bias.view(*shape),negative_slope=negative_slope)*scale

class FusedLeakyReLU(nn.Module):
    def __init__(self,channel,negative_slope=.2,scale=2**.5):
        super().__init__();self.bias=nn.Parameter(torch.zeros(channel));self.negative_slope=negative_slope;self.scale=scale
    def forward(self,input):return fused_leaky_relu(input,self.bias,self.negative_slope,self.scale)

def upfirdn2d(input,kernel,up=1,down=1,pad=(0,0)):
    n,c,h,w=input.shape  # 先插零，再滤波，再按固定步长取样。
    x=input.reshape(n*c,h,1,w,1)
    x=F.pad(x,(0,up-1,0,0,0,up-1)).reshape(n*c,1,h*up,w*up)
    p0,p1=pad;x=F.pad(x,(max(p0,0),max(p1,0),max(p0,0),max(p1,0)))
    x=x[:,:,max(-p0,0):x.shape[2]-max(-p1,0),max(-p0,0):x.shape[3]-max(-p1,0)]
    x=F.conv2d(x,kernel.flip((0,1))[None,None]);x=x[:,:,::down,::down]
    return x.reshape(n,c,x.shape[-2],x.shape[-1])
