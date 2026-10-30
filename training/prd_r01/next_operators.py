"""本批固定MAE、学习压缩与TV工作点；不包含检测器训练逻辑。"""
import hashlib
from pathlib import Path
import types
import numpy as np
import torch
from torch import nn

LABELS={'mae':'MAE遮挡重构','compression':'学习压缩重构','tv':'TV变分重构'}  # 中文运行名称。
MODEL_DIR=Path.home()/'.cache/prd-models/重构第二批-20261030'

def spec(config,name):
    from .cache import file_sha
    common=dict(operator=name,resolution=224,mean=config['mean'],std=config['std'],storage='normalized-float32-nchw',internal_batch='one_image',output_dtype='float32')
    if name=='mae':
        common.update(recipe='mae-vit-large-pixel-four-complementary-masked-only-v1',weights_sha256=file_sha(MODEL_DIR/'mae_visualize_vit_large.pth'),source_revision=(MODEL_DIR/'mae_revision.txt').read_text().strip(),visible_patches=49,patches=196,each_patch_predictions=3,mask_identity='sha256-round-rgb-uint8',normalization='imagenet',norm_pix_loss=False,clip_output=False)
    elif name=='compression':
        common.update(recipe='compressai-bmshj2018-factorized-mse-quality3-v1',weights_sha256=file_sha(MODEL_DIR/'bmshj2018-factorized-prior-3-5c6f152b.pth.tar'),library='compressai1.2.8',quality=3,metric='mse',quantization='round-around-learned-medians',clip_output=True,entropy_bitstream=False)
    elif name=='tv':
        common.update(recipe='skimage-0.22.0-chambolle-rgb-v1',weight=.1,eps=2e-4,max_num_iter=100,channel_axis=-1,clip_output=False)
    else: raise ValueError(name)
    return common

class NextProbe(nn.Module):
    def __init__(self,config,name):
        super().__init__();self.name=name
        self.register_buffer('mean',torch.tensor(config['mean']).view(1,3,1,1))
        self.register_buffer('std',torch.tensor(config['std']).view(1,3,1,1))
        self.register_buffer('imagenet_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('imagenet_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        if name=='mae':
            from .mae_vendor.models_mae import mae_vit_large_patch16
            self.net=mae_vit_large_patch16(norm_pix_loss=False)
            weights=torch.load(MODEL_DIR/'mae_visualize_vit_large.pth',map_location='cpu',weights_only=True)
            self.net.load_state_dict(weights['model'],strict=True)  # 完整编码器与像素解码器，不加载分类微调权重。
            self.net.random_masking=types.MethodType(self._fixed_mask,self.net)
        elif name=='compression':
            from compressai.zoo import bmshj2018_factorized
            from compressai.zoo.pretrained import load_pretrained
            self.net=bmshj2018_factorized(quality=3,metric='mse',pretrained=False)
            weights=torch.load(MODEL_DIR/'bmshj2018-factorized-prior-3-5c6f152b.pth.tar',map_location='cpu',weights_only=True)
            self.net.load_state_dict(load_pretrained(weights),strict=True)
        elif name=='tv':
            import skimage
            if skimage.__version__!='0.22.0':raise ValueError('TV实现版本变化')
        else: raise ValueError(name)
        self.requires_grad_(False).eval()

    @staticmethod
    def _fixed_mask(net,x,mask_ratio):
        ids=net.fixed_ids.to(x.device);keep=ids[:,:49]  # 每行保留互补集合的49个patch。
        restore=torch.argsort(ids,dim=1)
        visible=torch.gather(x,1,keep.unsqueeze(-1).expand(-1,-1,x.shape[-1]))
        mask=torch.ones((4,196),device=x.device);mask[:,:49]=0
        return visible,torch.gather(mask,1,restore),restore

    def _mae_one(self,pixels):
        identity=pixels.mul(255).round().clamp(0,255).to(torch.uint8).cpu().numpy().tobytes()
        seed=int.from_bytes(hashlib.sha256(identity).digest()[:8],'little')%(2**63-1)
        generator=torch.Generator(device='cpu').manual_seed(seed)
        permutation=torch.randperm(196,generator=generator)
        ids=torch.stack([torch.cat([permutation[g*49:(g+1)*49],permutation[:g*49],permutation[(g+1)*49:]]) for g in range(4)])
        self.net.fixed_ids=ids  # 不消耗检测器全局RNG；同图跨端点固定掩码。
        inputs=((pixels-self.imagenet_mean)/self.imagenet_std).expand(4,-1,-1,-1)
        latent,mask,restore=self.net.forward_encoder(inputs,.75)
        predictions=self.net.forward_decoder(latent,restore)
        if not torch.equal(mask.sum(0),torch.full((196,),3.,device=mask.device)):raise ValueError('掩码覆盖错误')
        prediction=(predictions*mask.unsqueeze(-1)).sum(0,keepdim=True)/3  # 只用被遮挡时的预测，不粘贴可见原图。
        return self.net.unpatchify(prediction)*self.imagenet_std+self.imagenet_mean

    @torch.inference_mode()
    def forward(self,images):
        pixels=(images*self.std+self.mean).float();outputs=[]
        for one in pixels.split(1):  # 调用批次改变时仍保持相同内部计算形状。
            if self.name=='mae':out=self._mae_one(one)
            elif self.name=='compression':out=self.net(one)['x_hat'].clamp(0,1)
            else:
                from skimage.restoration import denoise_tv_chambolle
                array=one[0].permute(1,2,0).cpu().numpy()
                restored=denoise_tv_chambolle(array,weight=.1,eps=2e-4,max_num_iter=100,channel_axis=-1)
                out=torch.from_numpy(restored.copy()).permute(2,0,1).unsqueeze(0).to(images.device)
            outputs.append(out)
        result=torch.cat(outputs)
        if result.shape!=images.shape or not torch.isfinite(result).all():raise ValueError('重构输出异常')
        return ((result-self.mean)/self.std).float()

    @torch.inference_mode()
    def audit(self,images):
        pixels=images[:1]*self.std+self.mean
        output=self(images[:1])*self.std+self.mean
        if self.name=='mae':
            ids=self.net.fixed_ids;visible=torch.zeros(196,dtype=torch.int64)
            for row in ids:visible[row[:49]]+=1
            assert torch.equal(visible,torch.ones_like(visible))
            assert self.net.norm_pix_loss is False and self.net.decoder_pred.out_features==768
            rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
            again=self(images[:1]);assert torch.equal(rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
            # 官方encoder/decoder在同一固定mask下的输出，独立按patch累加核对合成。
            inp=((pixels-self.imagenet_mean)/self.imagenet_std).expand(4,-1,-1,-1)
            _,pred,mask=self.net(inp,mask_ratio=.75)
            manual=torch.stack([pred[:,j][mask[:,j].bool()].mean(0) for j in range(196)]).unsqueeze(0)
            expected=self.net.unpatchify(manual)*self.imagenet_std+self.imagenet_mean
            error=float((output-expected).abs().max());assert error<1e-6
            return dict(mask_coverage=3,no_visible_pixel_paste=True,pixel_decoder=True,official_forward_error=error,global_rng_unchanged=True)
        if self.name=='compression':
            latent=self.net.g_a(pixels);median=self.net.entropy_bottleneck._get_medians()
            quant=torch.round(latent-median)+median
            expected=self.net.g_s(quant).clamp(0,1)
            error=float((output-expected).abs().max());assert error<1e-6
            return dict(independent_quantization_error=error,latent_shape=list(latent.shape),actual_rounding=True,bitstream_claim=False)
        def objective(y,x):
            dy=torch.zeros_like(y);dx=torch.zeros_like(y)
            dy[:,:,:-1,:]=y[:,:,1:,:]-y[:,:,:-1,:];dx[:,:,:,:-1]=y[:,:,:,1:]-y[:,:,:,:-1]
            return float(.5*(y-x).square().sum()+.1*(dy.square()+dx.square()).sqrt().sum())
        before=objective(pixels,pixels);after=objective(output,pixels);assert after<before
        constant=(torch.full_like(images[:1],.5)-self.mean)/self.std
        error=float((self(constant)-constant).abs().max());assert error<1e-6
        return dict(rof_objective_before=before,rof_objective_after=after,constant_max_error=error)
