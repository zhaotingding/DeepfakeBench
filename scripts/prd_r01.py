#!/usr/bin/env python3
"""新阶段唯一入口：默认只检查；smoke和正式训练必须显式选择。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

def main():
    from prd_r01.config import read_config, readiness, read_protocol, VARIANTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", default=str(ROOT / "r01.machine.json"))
    parser.add_argument("--variant", choices=VARIANTS, default="original")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--cache-smoke", action="store_true")
    parser.add_argument("--protocol")
    parser.add_argument("--run-id")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = read_config(args.machine, args.variant)
    missing = readiness(config)
    if not args.smoke and not args.train and not args.cache_smoke:
        print(json.dumps(dict(variant=args.variant, implementation_ready=not missing, missing=missing,
                              formal_training_approved=False, config=config), ensure_ascii=False, indent=2))
        return 2 if missing else 0
    if missing:
        raise ValueError("; ".join(missing))
    protocol = None
    if args.train:
        if not args.protocol or not args.run_id:
            raise ValueError("--train requires an approved --protocol and unique --run-id")
        protocol = read_protocol(args.protocol)  # 在加载模型或写输出前拒绝草案。
    import torch
    from prd_r01.engine import seed_all, run_training
    from prd_r01.model import build_model
    from prd_r01.data import LMDBImages, smoke_rows
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA device; no automatic CPU training fallback")
    if args.cache_smoke:
        if args.variant != "vae":
            raise ValueError("--cache-smoke目前只验收VAE公共接口")
        from prd_r01.cache_smoke import run_cache_smoke
        print(json.dumps(run_cache_smoke(config, device), ensure_ascii=False, indent=2))
        return 0
    if args.train:
        print(json.dumps(run_training(config, protocol, args.run_id, device), indent=2))
        return 0
    config = dict(config, reconstruction_mode="online_after_augment")  # 旧smoke显式验收在线接口。
    torch.set_num_threads(2)
    seed_all(3407)  # 仅接口验收固定随机状态，不预选正式实验seed。
    rows = smoke_rows(config)
    dataset = LMDBImages(rows, config, augment=False)
    images = torch.stack([dataset[i][0] for i in range(2)]).to(device)
    labels = torch.tensor([r["label"] for r in rows], device=device)
    dataset.close()
    model = build_model(config, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.monotonic()
    model.train()
    logits = model(images)["logits"]
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()  # 验证反向，不执行optimizer.step、不保存权重。
    grads = [p.grad for p in model.encoder.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):
        raise ValueError("Encoder gradients missing, nonfinite or all zero")
    if any(p.grad is not None for p in model.probes.parameters()):
        raise ValueError("Frozen operator received gradients")
    model.eval()
    with torch.no_grad():
        first = model(images)["logits"]
        second = model(images)["logits"]
    if not torch.isfinite(first).all() or not torch.equal(first, second):
        raise ValueError("Repeated eval forward differs")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(json.dumps(dict(status="smoke_pass",variant=args.variant,source_frames=2,
                          optimizer_updates=0,new_checkpoints=0,loss=float(loss.detach()),
                          elapsed_seconds=time.monotonic()-start,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                          eval_repeat_max_abs=float((first-second).abs().max())), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
