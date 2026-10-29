"""固定工作点的传统重构算子，输入输出均为CLIP归一化RGB。"""
import torch
from torch import nn
from torch.nn import functional as F

class ClassicalProbe(nn.Module):
    def __init__(self, config, name):
        super().__init__()
        self.name = name  # 两个已批准的确定性算子。
        self.options = config["operators"][name]
        self.register_buffer("mean", torch.tensor(config["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(config["std"]).view(1, 3, 1, 1))

    def forward(self, images):
        pixels = (images * self.std + self.mean).float()  # 在真实RGB尺度操作。
        if self.name == "wavelet":
            if pixels.shape[-1] % 2 or pixels.shape[-2] % 2:
                raise ValueError("一层Haar要求偶数高宽")
            coarse = F.avg_pool2d(pixels, 2).repeat_interleave(2, -2).repeat_interleave(2, -1)
            result = coarse + self.options["detail_gain"] * (pixels - coarse)  # 等价于正交Haar三个细节子带同倍衰减。
        elif self.name == "lowrank":
            rank = self.options["rank"]
            if not 1 <= rank <= min(pixels.shape[-2:]):
                raise ValueError("低秩阶数超出图像大小")
            kwargs = {"driver": "gesvd"} if pixels.is_cuda else {}  # 固定精确SVD，不使用随机近似。
            u, s, vh = torch.linalg.svd(pixels, full_matrices=False, **kwargs)
            result = (u[..., :rank] * s[..., :rank].unsqueeze(-2)) @ vh[..., :rank, :]
        else:
            raise ValueError("未注册传统算子：" + self.name)
        return ((result - self.mean) / self.std).to(images.dtype)  # 不提前量化或裁剪SVD重建。

def classical_spec(config, name):
    return dict(operator=name, recipe="rgb-" + name + "-v1", parameters=config["operators"][name],
                resolution=config["resolution"], mean=config["mean"], std=config["std"],
                storage="normalized-float32-nchw")  # 参数身份阻止不同工作点混用缓存。

def build_operator(config, name, device):
    if name in {"wavelet", "lowrank"}:
        return ClassicalProbe(config, name).to(device).requires_grad_(False).eval()
    if name == "vae":
        from .model import VAEProbe
        return VAEProbe(config["vae_path"], config["mean"], config["std"],
                        torch.float16 if device.type == "cuda" else torch.float32).to(device)
    raise ValueError("未实现算子：" + name)

