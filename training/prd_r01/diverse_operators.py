"""第三批二维重构：上色、自相似、稀疏字典、人脸生成反演。"""
import hashlib,json,sys,types  # 冻结资产身份与隔离官方模块入口。
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LABELS={'colorization':'灰度上色重构','nlm':'非局部自相似重构','sparse':'稀疏字典重构','e4e':'人脸生成反演重构'}
ASSETS=Path.home()/'.cache/prd-models/重构第三批-20261031'

def sparse_code(patches,dictionary,count=8):
    """批量OMP；对选中原子做最小二乘，小对角项只稳定病态Gram。"""
    n=patches.shape[0];atoms=dictionary.shape[1]  # 输入每行为一个去均值图块。
    residual=patches.clone();chosen=torch.empty(n,0,dtype=torch.long,device=patches.device)
    available=torch.ones(n,atoms,dtype=torch.bool,device=patches.device)
    for step in range(count):
        score=(residual@dictionary).abs().masked_fill(~available,-1)
        index=score.argmax(1);available.scatter_(1,index[:,None],False)
        chosen=torch.cat([chosen,index[:,None]],1)
        basis=dictionary.T[chosen].transpose(1,2)  # 每块只解已选原子的系统。
        gram=basis.transpose(1,2)@basis
        rhs=basis.transpose(1,2)@patches[:,:,None]
        coef=torch.linalg.solve(gram+1e-6*torch.eye(step+1,device=patches.device,dtype=patches.dtype),rhs)
        residual=patches-(basis@coef).squeeze(-1)
    codes=torch.zeros(n,atoms,device=patches.device,dtype=patches.dtype)
    codes.scatter_(1,chosen,coef.squeeze(-1))
    return codes

def spec(config,name):
    from .cache import file_sha
    base=dict(operator=name,resolution=224,mean=config['mean'],std=config['std'],storage='normalized-float32-nchw',internal_batch='one_image',output_dtype='float32')
    if name=='colorization':
        base.update(recipe='eccv16-lab-original-L-predicted-ab-v1',weights_sha256=file_sha(ASSETS/'colorization_release_v2-9b330a0b.pth'),revision='4f6009ed1495b1300231ebeb41cc4015557ddef7',internal_resolution=256,resize='official PIL bicubic uint8',precision='float32',output='official Lab2RGB gamut clipping')
    elif name=='nlm':
        base.update(recipe='skimage-0.22.0-fast-nlm-rgb-v1',patch_size=5,patch_distance=6,h=.08,sigma=0.,fast_mode=True,channel_axis=-1,preserve_range=True,injected_noise=False)
    elif name=='sparse':
        base.update(recipe='source-only-ksvd-L8x8-256atoms-omp8-v1',dictionary_sha256=file_sha(ASSETS/'source_ksvd_dictionary.npy'),fit_metadata_sha256=file_sha(ASSETS/'source_ksvd_fit.json'),patch_size=8,stride=4,sparsity=8,gram_stabilizer=1e-6,lightness='Lab L divided by100',chroma='preserved from input',overlap='uniform average',precision='float32')
    elif name=='e4e':
        base.update(recipe='e4e-ffhq-fixed-noise-original-crop-v1',weights_sha256=file_sha(ASSETS/'e4e_ffhq_encode.pt'),revision='99ea50578695d2e8a1cf7259d8ee89b23eea942b',weight_distribution='AIRI-Institute/StyleFeatureEditor ee3aac06a197410d63327b16917739a50210f068',encoder_resolution=256,generator_resolution=1024,output_resize='official adaptive-average256 then bilinear224',alignment='existing224crop; no extra warp or detection',noise='checkpoint fixed buffers',precision='float32',fused_ops='equivalent PyTorch reference; no CUDA extension')
    else:raise ValueError(name)
    return base

class DiverseProbe(nn.Module):
    def __init__(self,config,name):
        super().__init__();self.name=name
        self.register_buffer('mean',torch.tensor(config['mean']).view(1,3,1,1));self.register_buffer('std',torch.tensor(config['std']).view(1,3,1,1))
        if name=='colorization':
            from .vendor.colorization.colorizers.eccv16 import ECCVGenerator
            self.net=ECCVGenerator();self.net.load_state_dict(torch.load(ASSETS/'colorization_release_v2-9b330a0b.pth',map_location='cpu',weights_only=True),strict=True)
        elif name=='nlm':
            import skimage
            if skimage.__version__!='0.22.0':raise ValueError('NLM implementation changed')
        elif name=='sparse':
            matrix=np.load(ASSETS/'source_ksvd_dictionary.npy',allow_pickle=False)
            if matrix.shape!=(64,256):raise ValueError('dictionary shape')
            self.register_buffer('dictionary',torch.from_numpy(matrix).float())
        elif name=='e4e':
            vendor=Path(__file__).parent/'vendor/e4e';sys.path.insert(0,str(vendor))
            from models.encoders.psp_encoders import Encoder4Editing
            from models.stylegan2.model import Generator
            checkpoint=torch.load(ASSETS/'e4e_ffhq_encode.pt',map_location='cpu',weights_only=True)
            if checkpoint['opts']['encoder_type']!='Encoder4Editing':raise ValueError('not e4e encoder')
            self.encoder=Encoder4Editing(50,'ir_se',types.SimpleNamespace(stylegan_size=1024));self.decoder=Generator(1024,512,8,channel_multiplier=2)
            for key,module in [('encoder',self.encoder),('decoder',self.decoder)]:module.load_state_dict({k[len(key)+1:]:v for k,v in checkpoint['state_dict'].items() if k.startswith(key+'.')},strict=True)
            self.register_buffer('latent_avg',checkpoint['latent_avg']);self.net=None
            if not checkpoint['opts'].get('start_from_latent_avg'):raise ValueError('unexpected latent centering')
        else:raise ValueError(name)
        self.requires_grad_(False).eval()

    def _color_one(self,one):
        from .vendor.colorization.colorizers.util import preprocess_img,postprocess_tens
        rgb=(one[0].permute(1,2,0)*255).round().clamp(0,255).to(torch.uint8).cpu().numpy()
        lum,small=preprocess_img(rgb);ab=self.net(small.to(one.device))
        restored=postprocess_tens(lum,ab.cpu())  # 原图亮度与网络预测色度组合。
        return torch.from_numpy(restored.copy()).permute(2,0,1)[None].to(one.device).float()

    def _sparse_one(self,one):
        from skimage.color import rgb2lab,lab2rgb
        lab=rgb2lab(one[0].permute(1,2,0).cpu().numpy());light=torch.from_numpy(lab[:,:,0].copy()).to(one.device)[None,None]/100
        patches=F.unfold(light,kernel_size=8,stride=4)[0].T;means=patches.mean(1,keepdim=True)
        codes=sparse_code(patches-means,self.dictionary);rebuilt=codes@self.dictionary.T+means
        folded=F.fold(rebuilt.T[None],output_size=(224,224),kernel_size=8,stride=4)
        weights=F.fold(torch.ones_like(rebuilt.T)[None],output_size=(224,224),kernel_size=8,stride=4)
        if not bool((weights>0).all()):raise ValueError('uncovered sparse pixels')
        lab[:,:,0]=(folded/weights)[0,0].cpu().numpy()*100
        return torch.from_numpy(lab2rgb(lab).copy()).permute(2,0,1)[None].to(one.device).float()

    @torch.inference_mode()
    def forward(self,images):
        pixels=(images*self.std+self.mean).float();outputs=[]
        for one in pixels.split(1):
            if self.name=='colorization':out=self._color_one(one)
            elif self.name=='nlm':
                from skimage.restoration import denoise_nl_means
                a=one[0].permute(1,2,0).cpu().numpy()
                y=denoise_nl_means(a,patch_size=5,patch_distance=6,h=.08,sigma=0.,fast_mode=True,channel_axis=-1,preserve_range=True)
                out=torch.from_numpy(y.copy()).permute(2,0,1)[None].to(images.device)
            elif self.name=='sparse':out=self._sparse_one(one)
            else:
                inp=F.interpolate(one*2-1,size=(256,256),mode='bilinear',align_corners=False)
                latent=self.encoder(inp)+self.latent_avg.reshape(1,-1,512)
                generated,_=self.decoder([latent],input_is_latent=True,randomize_noise=False)
                out=(F.interpolate(F.adaptive_avg_pool2d(generated,(256,256)),size=(224,224),mode='bilinear',align_corners=False)+1)/2
            outputs.append(out)
        result=torch.cat(outputs).float()
        if result.shape!=images.shape or not torch.isfinite(result).all():raise ValueError('invalid reconstruction')
        return (result-self.mean)/self.std

    @torch.inference_mode()
    def audit(self,images):
        rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone();out=self(images[:1])
        assert torch.equal(rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
        pixels=images[:1]*self.std+self.mean;restored=out*self.std+self.mean
        report=dict(global_rng_unchanged=True,mean_absolute_rgb_change=float((restored-pixels).abs().mean()),operator=self.name)
        if self.name=='colorization':
            from .vendor.colorization.colorizers.util import preprocess_img,postprocess_tens
            rgb=(pixels[0].permute(1,2,0)*255).round().clamp(0,255).to(torch.uint8).cpu().numpy()
            lum,small=preprocess_img(rgb);ab=self.net(small.to(images.device));expected=postprocess_tens(lum,ab.cpu())
            err=float(np.max(np.abs(restored[0].permute(1,2,0).cpu().numpy()-expected)));assert err<1e-6
            report.update(original_luminance_preserved_before_gamut_conversion=True,learned_ab_max=float(ab.abs().max()),official_pipeline_max_error=err)
        elif self.name in ['nlm','sparse']:
            constant=(torch.full_like(images[:1],.5)-self.mean)/self.std
            err=float((self(constant)-constant).abs().max());assert err<2e-5
            report['constant_max_error']=err
            if self.name=='sparse':
                test=torch.arange(64,device=images.device,dtype=torch.float32)[None]/64;test-=test.mean()
                codes=sparse_code(test,self.dictionary);assert int((codes.abs()>1e-8).sum())<=8
                chosen=(codes[0].abs()>1e-8).nonzero().flatten();d=self.dictionary[:,chosen].double().cpu().numpy();y=test[0].double().cpu().numpy()
                reference=np.linalg.solve(d.T@d+1e-6*np.eye(len(chosen)),d.T@y)
                err=float(np.max(np.abs(d@reference-(codes@self.dictionary.T)[0].cpu().numpy())));assert err<2e-4
                report.update(sparsity=int(len(chosen)),cpu_least_squares_max_error=err)
        else:
            assert self.encoder.style_count==18 and self.decoder.n_latent==18
            report.update(encoder_styles=18,generator_resolution=1024,fixed_noise=True,extra_alignment=False)
        return report
