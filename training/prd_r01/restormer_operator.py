"""冻结的官方Restormer实拍去噪算子，保持224像素与FP32。"""
from pathlib import Path
import torch
from torch import nn
from .cache import file_sha
from .vendor.restormer.restormer_arch import Restormer

class RestormerProbe(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.register_buffer('mean',torch.tensor(config['mean']).view(1,3,1,1))  # CLIP反归一化。
        self.register_buffer('std',torch.tensor(config['std']).view(1,3,1,1))
        with torch.random.fork_rng(devices=[]):  # 构造冻结算子不消耗训练随机序列。
            self.network=Restormer(LayerNorm_type='BiasFree')  # 官方实拍去噪配置。
        weight=Path(config['restormer_weights'])
        if file_sha(weight)!=config['restormer_weights_sha256']:raise ValueError('Restormer权重身份变化')
        state=torch.load(weight,map_location='cpu',weights_only=True)
        self.network.load_state_dict(state['params'],strict=True)  # 禁止漏层或随机初始化。
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self,images):
        self.network.eval()
        pixels=images*self.std+self.mean  # 原图RGB，不额外加噪。
        output=[]
        for frame in pixels.split(1):  # 固定单帧内部推理，缓存与端点使用相同数值路径。
            output.append(self.network(frame).clamp(0,1))  # 沿用官方输出范围。
        restored=torch.cat(output)
        if restored.shape!=images.shape or not torch.isfinite(restored).all():raise ValueError('Restormer输出无效')
        return (restored-self.mean)/self.std

def restormer_spec(config):
    return dict(operator='restormer',recipe='official-real-denoising-fp32-224-v1',resolution=config['resolution'],
                mean=config['mean'],std=config['std'],storage='normalized-float32-nchw',weights_sha256=config['restormer_weights_sha256'],
                upstream_commit='68dc6ac472db26f16361150cb7a96a1bc87da93f',layer_norm='BiasFree',output_clip=[0,1],internal_batch=1)
