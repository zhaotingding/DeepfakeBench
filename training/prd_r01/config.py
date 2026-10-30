"""共享科学配置与机器路径分离。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = "b5fb097740a0815f468faac7d6feac402b29e518"
VARIANTS = {
    "original": ("clip", []),
    "vae": ("clip", ["vae"]),
    "wavelet": ("clip", ["wavelet"]),
    "lowrank": ("clip", ["lowrank"]),
    "vqgan": ("clip", ["vqgan"]),
    "mae": ("clip", ["mae"]),
    "compression": ("clip", ["compression"]),
    "tv": ("clip", ["tv"]),
    "repeat_vae": ("clip", ["vae", "vae"]),
    "diffusion": ("clip", ["diffusion"]),
    "restoration": ("clip", ["restoration"]),
    "vae_diffusion": ("clip", ["vae", "diffusion"]),
    "vae_restoration": ("clip", ["vae", "restoration"]),
    "dinov2_original": ("dinov2", []),
    "dinov2_vae": ("dinov2", ["vae"]),
    "dinov2_diffusion": ("dinov2", ["diffusion"]),
    "dinov2_vae_diffusion": ("dinov2", ["vae", "diffusion"]),
    "siglip_original": ("siglip", []),
    "siglip_vae": ("siglip", ["vae"]),
    "siglip_diffusion": ("siglip", ["diffusion"]),
    "siglip_vae_diffusion": ("siglip", ["vae", "diffusion"]),
}
PATH_KEYS = {"data_root", "clip_path", "vae_path", "output_root"}

def read_config(machine_file, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    config = json.loads((ROOT / "training/config/r01/common.json").read_text())
    machine = json.loads(Path(machine_file).read_text())
    if set(machine) != PATH_KEYS:
        raise ValueError(f"Machine file must contain only {sorted(PATH_KEYS)}")
    for key, value in machine.items():
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{key} must be an explicit absolute path")
        config[key] = str(path.resolve())  # 不按主机名猜数据路径或回落到NAS。
    config["variant"] = variant
    config["backbone"], config["responses"] = VARIANTS[variant]
    return config

def readiness(config):
    missing = []
    if config["backbone"] != "clip":
        missing.append(f"{config['backbone']} adapter and local weights are not prepared")
    for name in dict.fromkeys(config["responses"]):
        if name not in {"vae", "wavelet", "lowrank", "vqgan", "mae", "compression", "tv"}:
            missing.append(f"{name} operator/checkpoint/preprocessing/workpoint is not prepared")
    required = [
        Path(config["data_root"]) / "DeepfakeBench/lmdb/FaceForensics++_lmdb/data.mdb",
        Path(config["data_root"]) / "DeepfakeBench/config/dataset_json/FaceForensics++.json",
        Path(config["clip_path"]) / "config.json",
    ]
    if "vae" in config["responses"]:
        required.append(Path(config["vae_path"]) / "config.json")
    for path in required:
        if not path.is_file():
            missing.append(f"Missing input: {path}")
    if not any((Path(config["clip_path"]) / x).is_file() for x in ["model.safetensors", "pytorch_model.bin"]):
        missing.append("Missing local CLIP weights")
    if "vae" in config["responses"] and not any(
        (Path(config["vae_path"]) / x).is_file()
        for x in ["diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"]
    ):
        missing.append("Missing local VAE weights")
    return missing

def read_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("status") != "approved":
        raise ValueError("Formal training requires a separately approved protocol; current draft does not authorize it")
    for key in ["id", "epochs", "seed", "train_manifest", "val_manifest", "max_wall_seconds"]:
        if protocol.get(key) is None:
            raise ValueError(f"Protocol field is not fixed: {key}")
    if not isinstance(protocol["epochs"], int) or protocol["epochs"] < 1 or protocol["max_wall_seconds"] <= 0:
        raise ValueError("Invalid finite training budget")
    if not isinstance(protocol["seed"], int):
        raise ValueError("Seed must be fixed before execution")
    for key in ["train_manifest", "val_manifest"]:
        if not Path(protocol[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path to a frozen manifest")
    if protocol.get("checkpoint_rule") != "source_val_video_balanced_ce":
        raise ValueError("Only the declared source-val rule is supported")
    if protocol.get("reconstruction_mode") != "cached_paired" or protocol.get("augmentation_protocol") != "paired_float32_v1":
        raise ValueError("正式协议须明确预计算重构与配对增强，不能隐式沿用旧增强顺序")
    return protocol
