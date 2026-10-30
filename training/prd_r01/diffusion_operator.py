"""固定SD1.5潜变量加噪与DDIM去噪；沿用本批原VAE和256尺寸往返。"""
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import AutoencoderKL,UNet2DConditionModel,DDIMScheduler

ASSETS=Path.home()/'.cache/prd-models/sd15-diffusion-20261030'
REVISION='451f4fe16113bff5a5d2269ed5ad43b0592e9a14'
TIMESTEPS=[181,161,141,121,101,81,61,41,21,1]

def diffusion_spec(config):
    from .cache import file_sha
    manifest=json.loads((ASSETS/'download_manifest.json').read_text())
    assert manifest['revision']==REVISION
    vae=Path(config['vae_path']);weight=next(vae/n for n in ['diffusion_pytorch_model.safetensors','diffusion_pytorch_model.bin'] if (vae/n).exists())
    return dict(operator='diffusion',recipe='sd15-original-vae-mode-ddim50-strength02-v1',resolution=config['resolution'],mean=config['mean'],std=config['std'],storage='normalized-float32-nchw',weights={n:x['sha256'] for n,x in manifest['files'].items()},revision=REVISION,vae_weights_sha256=file_sha(weight),vae_config_sha256=file_sha(vae/'config.json'),internal_resolution=256,internal_batch=1,dtype='float16',latent_scaling=.18215,posterior='mode',scheduler='DDIM',grid_steps=50,strength=.2,timesteps=TIMESTEPS,eta=0.,prompt='',guidance_scale=1.,noise_identity='sha256-round-rgb-uint8/cpu-float32-normal-to-fp16',output_clip=False)

class DiffusionProbe(nn.Module):
    def __init__(self,config,device):
        super().__init__()
        from transformers import CLIPTextModel,CLIPTokenizer
        self.register_buffer('mean',torch.tensor(config['mean'],device=device).view(1,3,1,1))
        self.register_buffer('std',torch.tensor(config['std'],device=device).view(1,3,1,1))
        self.vae=AutoencoderKL.from_pretrained(config['vae_path'],torch_dtype=torch.float16,local_files_only=True).to(device)
        self.unet=UNet2DConditionModel.from_pretrained(ASSETS/'unet',variant='fp16',torch_dtype=torch.float16,local_files_only=True).to(device)
        encoder=CLIPTextModel.from_pretrained(ASSETS/'text_encoder',variant='fp16',torch_dtype=torch.float16,local_files_only=True).to(device).eval()
        tokenizer=CLIPTokenizer.from_pretrained(ASSETS/'tokenizer',local_files_only=True)
        tokens=tokenizer('',padding='max_length',max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt').input_ids.to(device)
        with torch.inference_mode():embedding=encoder(tokens)[0]
        self.register_buffer('empty_embedding',embedding);del encoder  # 空条件只计算一次，未使用标签或图像描述。
        self.scheduler=DDIMScheduler(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',clip_sample=False,set_alpha_to_one=False,steps_offset=1,prediction_type='epsilon',timestep_spacing='leading')
        self.scheduler.set_timesteps(50,device=device);self.timesteps=self.scheduler.timesteps[40:]
        assert self.timesteps.tolist()==TIMESTEPS and self.unet.config.in_channels==4 and self.vae.config.latent_channels==4
        assert self.unet.config.cross_attention_dim==self.empty_embedding.shape[-1]
        self.requires_grad_(False).eval();self.forward_calls=0

    def encode(self,pixels):
        inputs=F.interpolate(pixels*2-1,size=(256,256),mode='bilinear',align_corners=False)
        return self.vae.encode(inputs.to(torch.float16)).latent_dist.mode()  # 同本批原VAE的确定性编码。

    def decode(self,latent):
        decoded=self.vae.decode(latent).sample.float()
        decoded=F.interpolate(decoded,size=(224,224),mode='bilinear',align_corners=False)
        return ((decoded+1)/2-self.mean)/self.std  # 不提前量化或裁剪，沿用原VAE端点入口。

    def noise(self,pixels,shape):
        identity=pixels.mul(255).round().clamp(0,255).to(torch.uint8).cpu().numpy().tobytes()
        seed=int.from_bytes(hashlib.sha256(identity).digest()[:8],'little')%(2**63-1)
        generator=torch.Generator(device='cpu').manual_seed(seed)
        return torch.randn(shape,generator=generator,dtype=torch.float32).to(pixels.device,torch.float16)

    def denoise(self,latent):
        for timestep in self.timesteps:
            noise=self.unet(self.scheduler.scale_model_input(latent,timestep),timestep,encoder_hidden_states=self.empty_embedding).sample
            latent=self.scheduler.step(noise,timestep,latent,eta=0.,return_dict=False)[0]
            self.forward_calls+=1  # 真实去噪网络调用计数，防止退化为纯VAE。
        return latent

    @torch.inference_mode()
    def forward(self,images):
        outputs=[]
        for image in images.split(1):
            pixels=image*self.std+self.mean;latent=self.encode(pixels)*.18215
            noise=self.noise(pixels,latent.shape)
            latent=self.scheduler.add_noise(latent,noise,self.timesteps[:1])
            outputs.append(self.decode(self.denoise(latent)/.18215))
        result=torch.cat(outputs)
        if result.shape!=images.shape or not torch.isfinite(result).all():raise ValueError('扩散输出形状或有限性异常')
        return result

    @torch.inference_mode()
    def audit(self,images,config):
        from .model import VAEProbe
        x=images[:1];pixels=x*self.std+self.mean;latent=self.encode(pixels)
        raw_vae=self.decode(latent);reference=VAEProbe(config['vae_path'],config['mean'],config['std'],torch.float16).to(x.device)
        expected=reference(x);vae_error=float((raw_vae-expected).abs().max());assert vae_error<=1e-6;del reference
        latent=latent*.18215;noise=self.noise(pixels,latent.shape);first=self.timesteps[0]
        initial=self.scheduler.add_noise(latent,noise,self.timesteps[:1])
        alpha=self.scheduler.alphas_cumprod.to(x.device,latent.dtype)[first];formula=alpha.sqrt()*latent+(1-alpha).sqrt()*noise
        add_error=float((initial-formula).abs().max());assert add_error<=1e-6
        predicted=self.unet(initial,first,encoder_hidden_states=self.empty_embedding).sample
        step=self.scheduler.step(predicted,first,initial,eta=0.,return_dict=False)[0]
        at=self.scheduler.alphas_cumprod[first];ap=self.scheduler.alphas_cumprod[first-20]
        x0=(initial-(1-at)**.5*predicted)/at**.5;manual=ap**.5*x0+(1-ap)**.5*predicted
        ddim_error=float((step-manual).abs().max());assert ddim_error<=1e-6
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();calls=self.forward_calls
        output=self(x);assert self.forward_calls-calls==10
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        deviation=float((output-raw_vae).abs().mean());assert deviation>1e-5
        return dict(vae_reference_max_abs=vae_error,add_noise_formula_max_abs=add_error,ddim_step_formula_max_abs=ddim_error,unet_calls=10,timesteps=TIMESTEPS,global_rng_unchanged=True,mean_abs_vs_pure_vae=deviation)
