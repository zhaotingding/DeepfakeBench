"""两张真实源图的缓存接口验收；没有optimizer更新或正式结果。"""
import json
from pathlib import Path
import tempfile
import time
import torch
from .cache import operator_spec, build_cache, read_cache
from .data import LMDBImages, smoke_rows
from .model import build_model, ResponseModel
from .engine import seed_all

def run_cache_smoke(config, device):
    torch.set_num_threads(2)
    seed_all(3407)
    online_config = dict(config, reconstruction_mode="online_after_augment", responses=["vae"])
    rows = smoke_rows(config)
    online = build_model(online_config, device).eval()
    cached = ResponseModel(online.encoder, online.encoder.feature_dim, {}, ["vae"],
                           config["head"]["hidden"], config["head"]["dropout"], cached=True).to(device)
    cached.head = online.head  # 相同参数比较，不另训一套模型。
    cached.eval()
    source = LMDBImages(rows, online_config)
    images = torch.stack([source[i][0] for i in range(2)]).to(device)
    source.close()
    counts = {"online": 0}
    hook = online.probes["vae"].register_forward_hook(lambda *args: counts.__setitem__("online", counts["online"] + 1))
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def measure(fn):
        sync()
        start = time.perf_counter()
        result = fn()
        sync()
        return result, time.perf_counter() - start
    try:
        with tempfile.TemporaryDirectory(prefix="prd-cache-smoke-") as temp:
            spec = operator_spec(config, "vae")
            meta = build_cache(rows, online_config, online.probes["vae"], str(Path(temp) / "cache"),
                               spec, device, batch_size=2, max_seconds=180)
            values, _, _ = read_cache(Path(temp) / "cache", rows, config, spec, verify_file=True)
            config = dict(config, reconstruction_caches={"vae": str(Path(temp) / "cache")})
            dataset = LMDBImages(rows, config, augment=False)
            with torch.inference_mode():
                expected, online_seconds = measure(lambda: online(images)["logits"])
                before = counts["online"]
                def cached_forward():
                    batch = [dataset[i] for i in range(2)]
                    x = torch.stack([b[0] for b in batch]).to(device)
                    views = {"vae": torch.stack([b[3]["vae"] for b in batch]).to(device)}
                    return cached(x, views)["logits"]
                actual, cached_seconds = measure(cached_forward)
                repeat = cached_forward()
            error = float((expected - actual).abs().max())
            if error > 1e-6 or not torch.equal(actual, repeat) or counts["online"] != before:
                raise ValueError("缓存读回/重复前向/零算子调用验收失败")
            dataset.close()
            train_data = LMDBImages(rows, config, augment=True)
            batch = [train_data[i] for i in range(2)]
            x = torch.stack([b[0] for b in batch]).to(device)
            views = {"vae": torch.stack([b[3]["vae"] for b in batch]).to(device)}
            cached.train()
            loss = torch.nn.functional.cross_entropy(cached(x, views)["logits"], torch.tensor([0, 1], device=device))
            loss.backward()  # 只验收梯度，绝不调用optimizer.step。
            grads = [p.grad for p in cached.encoder.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().sum() > 0 for g in grads):
                raise ValueError("缓存训练路径缺少有限非零编码器梯度")
            train_data.close()
            return dict(status="cache_smoke_pass", device=str(device), source_frames=2,
                        cache_bytes=meta["file_bytes"], cache_generation_seconds=meta["generation_seconds"],
                        online_eval_seconds=online_seconds, cached_eval_including_read_seconds=cached_seconds,
                        max_logit_error=error, cached_operator_calls=counts["online"]-before,
                        optimizer_updates=0, new_checkpoints=0,
                        timing_scope="one CPU/GPU two-frame interface check; not full-run speedup or accuracy")
    finally:
        hook.remove()
