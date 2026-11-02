"""官方ImageNet f16/1024码本的冻结推理接入。"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml
from .cache import file_sha
from .vendor.vqgan.model import Encoder, Decoder
from .vendor.vqgan.quantize import VectorQuantizer2

class VQGANProbe(nn.Module):
    def __init__(self, config):
        super().__init__()
        parameters=yaml.safe_load((Path(__file__).parent/'vendor/vqgan/model.yaml').read_text())['model']['params']
        assert parameters['n_embed']==1024 and parameters['embed_dim']==256  # 拒绝混用另一种码本。
        self.register_buffer('mean',torch.tensor(config['mean']).view(1,3,1,1))
        self.register_buffer('std',torch.tensor(config['std']).view(1,3,1,1))
        with torch.random.fork_rng(devices=[]):  # 恢复器构造不消耗分类器初始化随机序列。
            network=nn.Module()
            network.encoder=Encoder(**parameters['ddconfig'])
            network.decoder=Decoder(**parameters['ddconfig'])
            network.quantize=VectorQuantizer2(1024,256,beta=0.25,remap=None,legacy=True)
            network.quant_conv=nn.Conv2d(parameters['ddconfig']['z_channels'],256,1)
            network.post_quant_conv=nn.Conv2d(256,parameters['ddconfig']['z_channels'],1)
        weight=Path(config['vqgan_weights'])
        if file_sha(weight)!=config['vqgan_weights_sha256']:raise ValueError('VQGAN权重身份改变')
        raw=torch.load(weight,map_location='cpu',weights_only=True)
        state=raw['state_dict'];expected=set(network.state_dict())
        extras=set(state)-expected
        if any(not key.startswith('loss.') for key in extras):raise ValueError('出现未登记的checkpoint键')
        network.load_state_dict({key:state[key] for key in expected},strict=True)  # 推理所需所有层严格加载。
        self.load_audit=dict(reconstruction_tensors=len(expected),excluded_training_tensors=len(extras),excluded_prefix='loss.')
        self.network=network
        self.requires_grad_(False).eval()  # 不训练码本、编码器、解码器或判别器。

    def encode(self, pixels):
        z=self.network.quant_conv(self.network.encoder(pixels*2-1))  # 官方[-1,1]输入。
        quantized,_,info=self.network.quantize(z)  # 实际最近码字量化，不能跳过。
        return z,quantized,info[2]

    @torch.no_grad()
    def forward(self, images):
        self.network.eval()
        pixels=images*self.std+self.mean
        outputs=[]
        for frame in pixels.split(1):  # 缓存与评估固定同一内部batch，不改变数值路径。
            _,quantized,_=self.encode(frame)
            decoded=self.network.decoder(self.network.post_quant_conv(quantized))
            outputs.append((decoded+1)/2)  # 保留官方decoder输出，不另行裁剪。
        restored=torch.cat(outputs)
        if restored.shape!=images.shape or not torch.isfinite(restored).all():raise ValueError('VQGAN输出形状或数值异常')
        return (restored-self.mean)/self.std

    @torch.no_grad()
    def audit_quantization(self, image):
        z,q,indices=self.encode(image*self.std+self.mean)
        if tuple(z.shape)!=(1,256,14,14):raise ValueError('实际潜变量网格不符')
        ids=indices.reshape(-1);vectors=z.permute(0,2,3,1).reshape(-1,256)
        sample=np.linspace(0,len(ids)-1,16,dtype=np.int64)  # 预先固定16个空间位置，不看标签。
        inputs=vectors[sample].cpu().double().numpy();book=self.network.quantize.embedding.weight.cpu().double().numpy()
        distances=((inputs[:,None,:]-book[None,:,:])**2).sum(axis=-1)
        expected=distances.argmin(axis=1)
        if not np.array_equal(expected,ids[sample].cpu().numpy()):raise ValueError('独立CPU最近邻与官方码字不一致')
        nearest=self.network.quantize.embedding(ids).reshape(1,14,14,256).permute(0,3,1,2)
        error=float((nearest-q).abs().max())
        if error>1e-6 or int(ids.min())<0 or int(ids.max())>=1024:raise ValueError('量化输出不对应码本')
        return dict(grid=[14,14],codebook_size=1024,embedding_dim=256,checked_positions=16,
                    nearest_index_match=True,quantized_codeword_max_abs=error,unique_codes=int(ids.unique().numel()))

def vqgan_spec(config):
    root=Path(__file__).parent/'vendor/vqgan'
    return dict(operator='vqgan',recipe='imagenet-f16-1024-native224-fp32-raw-v1',resolution=config['resolution'],
                mean=config['mean'],std=config['std'],storage='normalized-float32-nchw',weights_sha256=config['vqgan_weights_sha256'],
                config_sha256=file_sha(root/'model.yaml'),upstream_commit='3ba01b241669f5ade541ce990f7650a3b8f65318',
                native_input=224,training_configuration_resolution=256,codebook_size=1024,embedding_dim=256,
                internal_batch=1,output_clipping=False,quantizer_legacy=True)
