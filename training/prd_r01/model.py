"""三槽同头模型：原图、单响应、双响应共用实现。"""
import torch
from torch import nn
from torch.nn import functional as F

class ResponseModel(nn.Module):
    def __init__(self, encoder, feature_dim, probes, responses, hidden=1024, dropout=0.5, cached=False, slots=3, residual="signed_diff"):
        super().__init__()
        if slots not in (2, 3) or len(responses) > slots - 1 or (not cached and any(name not in probes for name in responses)):
            raise ValueError("At most two registered response slots are supported")
        if residual not in {"signed_diff", "normalized_feature_diff", "unit_residual"}:
            raise ValueError(f"Unsupported residual: {residual}")  # 不把新模式隐式当作原差分。
        self.residual = residual  # 映射切换不增加参数或消耗初始化随机数。
        self.slots = slots  # 单算子用两槽，双响应保留三槽。
        self.encoder = encoder
        self.probes = nn.ModuleDict(probes)
        self.cached = cached
        self.responses = tuple(responses)
        self.head = nn.Sequential(
            nn.Linear(feature_dim * slots, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 2),
        )
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
                z_reconstructed = self.encoder(reconstructed.detach())  # 两次编码共享可训练权重。
                if self.residual == "normalized_feature_diff":
                    z_unit = z / (torch.linalg.vector_norm(z, dim=1, keepdim=True) + 1e-6)  # 分别归一化两端，ε固定且零向量输出为零。
                    zr_unit = z_reconstructed / (torch.linalg.vector_norm(z_reconstructed, dim=1, keepdim=True) + 1e-6)
                    responses[name] = z_unit - zr_unit  # 不是先求差再把残差单位化。
                elif self.residual == "unit_residual":
                    difference = z - z_reconstructed  # 先计算两个共享编码器输出的有符号差。
                    magnitude = torch.linalg.vector_norm(difference, dim=1, keepdim=True)  # 每个样本独立计算残差长度。
                    responses[name] = difference / (magnitude + 1e-6)  # 零残差保持零，原图槽仍使用未归一化的z。
                else:
                    responses[name] = z - z_reconstructed  # 原有符号差路径保持。
        slots = [responses[name] for name in self.responses]
        slots += [torch.zeros_like(z)] * (self.slots - 1 - len(slots))
        features = torch.cat([z, *slots], dim=1)  # 重复VAE槽复用同一响应，不重复计算或加噪。
        return {"logits": self.head(features), "features": features}

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
    torch.set_rng_state(rng)
    model = ResponseModel(encoder, encoder.feature_dim, probes, config["responses"],
                          config["head"]["hidden"], config["head"]["dropout"], cached=cached, slots=config["head"]["slots"],
                          residual=config.get("residual", "signed_diff"))  # 新模式必须显式来自科学配置。
    return model.to(device)
