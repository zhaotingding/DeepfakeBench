"""三槽同头模型：原图、单响应、双响应共用实现。"""
import torch
from torch import nn
from torch.nn import functional as F

class ResponseModel(nn.Module):
    def __init__(self, encoder, feature_dim, probes, responses, hidden=1024, dropout=0.5, cached=False, slots=3, residual="signed_diff", auxiliary_input="none"):
        super().__init__()
        if slots not in (2, 3) or len(responses) > slots - 1 or (not cached and any(name not in probes for name in responses)):
            raise ValueError("At most two registered response slots are supported")
        if residual not in {"signed_diff", "absolute_diff"}:
            raise ValueError(f"Unsupported residual: {residual}")  # 拒绝把未知模式当作有符号差。
        self.residual = residual  # 保存响应定义，不增加模型参数或消耗随机数。
        self.slots = slots  # 单算子用两槽，双响应保留三槽。
        self.encoder = encoder
        self.probes = nn.ModuleDict(probes)
        self.cached = cached
        self.responses = tuple(responses)
        self.head = nn.Sequential(
            nn.Linear(feature_dim * slots, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 2),
        )
        self.auxiliary_input = auxiliary_input
        self.auxiliary_head = None
        if auxiliary_input not in {"none", "original_aux", "residual_aux"}:
            raise ValueError(f"Unknown auxiliary input: {auxiliary_input}")
        if auxiliary_input != "none":
            if self.responses != ("vqgan",) or residual != "signed_diff" or slots != 2:
                raise ValueError("VQGAN辅助监督须使用有符号差和两槽主头")
            with torch.random.fork_rng(devices=[]):  # 在CPU构造辅助头，退出后恢复共同主路径的CPU随机状态。
                self.auxiliary_head = nn.Linear(feature_dim, 2)
        for probe in self.probes.values():
            probe.requires_grad_(False)  # 探测算子冻结，编码器仍可训练。

    def forward(self, images, reconstructions=None):
        if self.cached and set(reconstructions or {}) != set(self.responses):
            raise ValueError("缓存响应不完整或含多余算子")
        z = self.encoder(images)
        responses = {}
        devices = [images.device.index] if images.is_cuda else []
        for name in dict.fromkeys(self.responses):
            # 附加视图不消耗原图/头/下一batch的全局Torch随机序列。
            with torch.random.fork_rng(devices=devices):
                if self.cached:
                    reconstructed = reconstructions[name]  # 跳过算子，但不冻结下游编码器。
                else:
                    probe = self.probes[name].eval()
                    with torch.no_grad():
                        reconstructed = probe(images.detach())
                if reconstructed.shape != images.shape:
                    raise ValueError(f"{name} changed the image shape")
                difference = z - self.encoder(reconstructed.detach())  # 两次编码共享可训练权重，冻结重构图输入。
                responses[name] = difference.abs() if self.residual == "absolute_diff" else difference  # 绝对值仅去除符号，默认有符号路径不变。
        slots = [responses[name] for name in self.responses]
        slots += [torch.zeros_like(z)] * (self.slots - 1 - len(slots))
        features = torch.cat([z, *slots], dim=1)  # 重复VAE槽复用同一响应，不重复计算或加噪。
        output = {"logits": self.head(features), "features": features}
        if self.training and self.auxiliary_head is not None:
            auxiliary_features = z if self.auxiliary_input == "original_aux" else slots[0]
            output["auxiliary_logits"] = self.auxiliary_head(auxiliary_features)  # 保留辅助CE到共享编码器的梯度。
        return output

class CLIPEncoder(nn.Module):
    def __init__(self, path, lora):
        super().__init__()
        from transformers import CLIPModel
        from peft import LoraConfig, get_peft_model
        model = CLIPModel.from_pretrained(path, local_files_only=True)
        vision = model.vision_model
        self.feature_dim = vision.config.hidden_size
        self.backbone = get_peft_model(vision, LoraConfig(
            r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"], bias="none",
        ))

    def forward(self, images):
        return self.backbone(images)["pooler_output"]

class VAEProbe(nn.Module):
    def __init__(self, path, mean, std, dtype):
        super().__init__()
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(path, torch_dtype=dtype, local_files_only=True)
        self.vae.requires_grad_(False).eval()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images):
        self.vae.eval()
        pixels = images * self.std + self.mean
        inputs = F.interpolate(pixels * 2 - 1, size=(256, 256), mode="bilinear", align_corners=False)
        latent = self.vae.encode(inputs.to(self.vae.dtype)).latent_dist.mode()  # 确定性均值，不采样posterior。
        decoded = self.vae.decode(latent).sample.to(images.dtype)
        decoded = F.interpolate(decoded, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return ((decoded + 1) / 2 - self.mean) / self.std

def build_model(config, device):
    from .config import readiness
    missing = readiness(config)
    if missing:
        raise ValueError("; ".join(missing))  # 不把未实现的扩散/DINO偷偷替换成VAE/CLIP。
    # 固定构造次序，使三种已支持配置的编码器与分类头初值相同。
    encoder = CLIPEncoder(config["clip_path"], config["lora"])
    rng = torch.get_rng_state()
    probes = {}
    cached = config.get("reconstruction_mode") == "cached_paired"
    if "vae" in config["responses"] and not cached:
        probes["vae"] = VAEProbe(
            config["vae_path"], config["mean"], config["std"],
            torch.float16 if device.type == "cuda" else torch.float32,
        )
    if not cached:
        from .operators import build_operator
        for name in config["responses"]:
            if name in {"wavelet", "lowrank"}:
                probes[name] = build_operator(config, name, device)
    if "vqgan" in config["responses"] and not cached:
        from .vqgan_operator import VQGANProbe  # 复用原已核验的逐图 FP32 码本恢复路径。
        probes["vqgan"] = VQGANProbe(config)  # 冻结算子；恢复后的图像仍由可训练共享 CLIP 编码。
    if "superresolution" in config["responses"] and not cached:
        from .generative_operators import GenerativeProbe  # 复用已冻结的 Real-ESRGAN 降采样恢复。
        probes["superresolution"] = GenerativeProbe(config, "superresolution")
    torch.set_rng_state(rng)  # 算子构造的随机初始化不得改变共同主头初值。
    model = ResponseModel(encoder, encoder.feature_dim, probes, config["responses"],
                          config["head"]["hidden"], config["head"]["dropout"], cached=cached, slots=config["head"]["slots"],
                          residual=config.get("residual", "signed_diff"),
                          auxiliary_input=config.get("auxiliary_input", "none"))  # 旧配置仅使用原主头。
    return model.to(device)
