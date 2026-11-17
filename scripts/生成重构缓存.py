#!/usr/bin/env python3
"""按显式清单生成冻结VAE重构；不会启动分类器训练。"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

def main():
    from prd_r01.config import read_config
    from prd_r01.data import load_manifest
    from prd_r01.cache import operator_spec, row_key, build_cache
    parser = argparse.ArgumentParser(description="生成一次重构缓存，供多个epoch和实验复用")
    parser.add_argument("--machine", default=str(ROOT / "machine.local.json"))
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args()
    config = read_config(args.machine, "vae")
    rows, seen = [], {}
    for path in args.manifest:
        for row in load_manifest(path)["rows"]:
            key = row_key(row)
            if key in seen and seen[key] != row:
                raise ValueError("多个清单的同一帧元数据冲突")
            if key not in seen:
                seen[key] = row
                rows.append(row)
    spec = operator_spec(config, "vae")
    import torch
    from prd_r01.model import VAEProbe
    from prd_r01.engine import seed_all
    seed_all(3407)  # 确定性算子不采样；仅固定实现随机状态。
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("无CUDA设备，不自动回落到CPU")
    probe = VAEProbe(config["vae_path"], config["mean"], config["std"],
                     torch.float16 if device.type == "cuda" else torch.float32).to(device)
    meta = build_cache(rows, config, probe, args.output, spec, device, args.batch_size, args.max_seconds)
    print(json.dumps(dict(status=meta["status"], frames=len(rows), file_bytes=meta["file_bytes"],
                          generation_seconds=meta["generation_seconds"], optimizer_updates=0), ensure_ascii=False))

if __name__ == "__main__":
    main()
