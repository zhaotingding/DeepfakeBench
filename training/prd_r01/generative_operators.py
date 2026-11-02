"""第四批：降采样超分与人脸码本预测；仅提供冻结逐图重构。"""
import torch  # 算子与训练器分别控制精度。
from pathlib import Path
from torch import nn
from torch.nn import functional as F

LABELS={'superresolution':'降采样超分重构','codeformer':'人脸码本预测重构'}
ASSETS=Path.home()/'.cache/prd-models/重构第四批-20261102'
WEIGHTS={'superresolution':('RealESRGAN_x4plus.pth','4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1'),'codeformer':('codeformer.pth','1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7')}

def spec(config,name):
    from .cache import file_sha  # 固定下载资产身份，不接受无声换权重。
    filename,expected=WEIGHTS[name];actual=file_sha(ASSETS/filename)
    if actual!=expected:raise ValueError('重构权重身份不符：'+name)
    base=dict(operator=name,resolution=224,mean=config['mean'],std=config['std'],storage='normalized-float32-nchw',internal_batch='one_image',output_dtype='float32',weights_sha256=actual,extra_face_detection=False,extra_alignment=False)
    if name=='superresolution':
        base.update(recipe='bicubic-antialias56-realesrgan-x4plus-prepad10-v1',architecture='RRDBNet64-23-32-x4',low_resolution=56,input_resize='FP32 bicubic antialias align_cornersFalse then clamp01',pre_pad=10,padding='reflect right-bottom then crop scale4',precision='frozen network FP16; output FP32',output_clamp=[0,1],basicsr_revision='8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a',realesrgan_revision='a4abfb2979a7bbff3f69f58f58ae324608821e27')
    else:
        base.update(recipe='codeformer-face-code-prediction-w0-adain-v1',revision='b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b',internal_resolution=512,codebook_size=1024,latent_tokens=256,fidelity_weight=0.0,adain=True,precision='frozen FP32 weights with CUDA autocast FP16',input_resize='FP32 bilinear512 align_cornersFalse then scale[-1,1]',output_resize='clamp[-1,1] then FP32 antialiased bilinear224',input='existing224crop; no new warp',output_quantization='none')
    return base

class GenerativeProbe(nn.Module):
    def __init__(self,config,name):
        super().__init__();self.name=name
        self.register_buffer('mean',torch.tensor(config['mean']).view(1,3,1,1));self.register_buffer('std',torch.tensor(config['std']).view(1,3,1,1))
        spec(config,name)  # 加载前核对固定官方资产。
        if name=='superresolution':
            from .vendor.generative.rrdbnet_arch import RRDBNet
            self.net=RRDBNet(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=4)
        elif name=='codeformer':
            from .vendor.generative.codeformer_arch import CodeFormer
            self.net=CodeFormer(dim_embd=512,codebook_size=1024,n_head=8,n_layers=9,connect_list=['32','64','128','256'])
        else:raise ValueError(name)
        checkpoint=torch.load(ASSETS/WEIGHTS[name][0],map_location='cpu',weights_only=True)
        self.net.load_state_dict(checkpoint['params_ema'],strict=True)  # 不忽略缺失或多余权重。
        if name=='superresolution':self.net.half()  # RRDB官方支持冻结半精度推理。
        self.requires_grad_(False).eval()

    def _one(self,rgb):
        if self.name=='superresolution':
            small=F.interpolate(rgb,size=(56,56),mode='bicubic',align_corners=False,antialias=True).clamp(0,1)
            padded=F.pad(small,(0,10,0,10),mode='reflect')  # 与官方预补边/裁切一致。
            return self.net(padded.half()).float()[:,:,:224,:224].clamp(0,1)
        inp=F.interpolate(rgb,size=(512,512),mode='bilinear',align_corners=False)*2-1
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            restored,_,_=self.net(inp,w=0.0,adain=True)  # 固定码本预测，不使用可调特征旁路。
        pixels=(restored.float().clamp(-1,1)+1)/2
        return F.interpolate(pixels,size=(224,224),mode='bilinear',align_corners=False,antialias=True)

    @torch.inference_mode()
    def forward(self,images):
        if images.device.type!='cuda':raise ValueError('本批冻结精度仅验收CUDA路径')
        self.net.eval();pixels=(images*self.std+self.mean).float();outputs=[]
        for one in pixels.split(1):outputs.append(self._one(one))  # 缓存、BS16与BS64均逐张调用。
        result=torch.cat(outputs).float()
        if result.shape!=images.shape or not torch.isfinite(result).all():raise ValueError('重构输出形状或有限性失败')
        return (result-self.mean)/self.std

    @torch.inference_mode()
    def audit(self,images):
        cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();out=self(images[:1])
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
        pixels=images[:1]*self.std+self.mean;restored=out*self.std+self.mean
        report=dict(operator=self.name,global_rng_unchanged=True,mean_absolute_rgb_change=float((restored-pixels).abs().mean()),range=[float(restored.min()),float(restored.max())],frozen=not any(p.requires_grad for p in self.parameters()))
        if self.name=='superresolution':
            assert self.net.scale==4 and len(self.net.body)==23
            report.update(low_resolution=56,prepad=10,network_scale=4,extra_face_enhancement=False)
        else:
            inp=F.interpolate(pixels,size=(512,512),mode='bilinear',align_corners=False)*2-1
            with torch.autocast(device_type='cuda',dtype=torch.float16):_,logits,feature=self.net(inp,w=0.0,adain=True)
            assert logits.shape==(1,256,1024) and feature.shape==(1,256,16,16)
            report.update(codebook_size=1024,latent_tokens=256,fidelity_weight=0.0,adain=True,extra_alignment=False)
        return report
