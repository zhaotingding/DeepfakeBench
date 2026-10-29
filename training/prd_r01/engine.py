"""共同训练/源验证入口；不沿用旧目标集每轮两次评测和早停。"""
import json
import hashlib
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import load_manifest, validate_source_pair, make_loader
from .model import build_model

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

def video_metrics(logits, rows):
    from sklearn.metrics import roc_auc_score
    groups = {}
    for z, row in zip(logits, rows):
        key = (row["domain"], row["video_id"])
        entry = groups.setdefault(key, {"label": row["label"], "values": []})
        if entry["label"] != row["label"]:
            raise ValueError("Conflicting video labels")
        entry["values"].append(z)
    per_domain = {}
    for domain in sorted({key[0] for key in groups}):
        selected = [v for k, v in groups.items() if k[0] == domain]
        z = np.asarray([np.mean(v["values"], axis=0) for v in selected], dtype=np.float64)
        y = np.asarray([v["label"] for v in selected], dtype=np.int64)
        if set(y.tolist()) != {0, 1}:
            raise ValueError(f"{domain} needs both video labels")
        normalizer = np.logaddexp(z[:, 0], z[:, 1])
        ce = normalizer - z[np.arange(len(y)), y]
        prob = np.exp(z[:, 1] - normalizer)
        per_domain[domain] = dict(auc=float(roc_auc_score(y, prob)), ce=float(ce.mean()),
                                 balanced_ce=float((ce[y == 0].mean() + ce[y == 1].mean()) / 2),
                                 videos=int(len(y)))
    return {"domains": per_domain, "macro": {
        metric: float(np.mean([v[metric] for v in per_domain.values()]))
        for metric in ["auc", "ce", "balanced_ce"]
    }}

def move_batch(batch, device):
    images, labels, indices, views = batch
    return (images.to(device, non_blocking=True), labels.to(device, non_blocking=True), indices,
            {name: value.to(device, non_blocking=True) for name, value in views.items()})

def evaluate(model, loader, rows, device, deadline=None):
    model.eval()
    logits = np.empty((len(rows), 2), dtype=np.float32)
    with torch.inference_mode():
        for batch in loader:
            images, labels, indices, views = move_batch(batch, device)
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Approved wall-time budget exhausted during evaluation")
            result = model(images, views)["logits"]
            if not torch.isfinite(result).all():
                raise ValueError("Nonfinite evaluation logits")
            logits[indices.numpy()] = result.cpu().numpy()
    return video_metrics(logits, rows), logits

def run_training(config, protocol, run_id, device):
    from .config import ROOT
    import subprocess
    if not run_id or any(not (c.isalnum() or c in "-_") for c in run_id):
        raise ValueError("使用唯一的中文或字母数字运行名称，不含路径分隔符")
    train = load_manifest(protocol["train_manifest"], "source_train")
    val = load_manifest(protocol["val_manifest"], "source_val")
    validate_source_pair(train, val)
    targets = []
    for path in protocol.get("evaluation_manifests", []):
        manifest = load_manifest(path)
        if manifest.get("role") not in ["target_dev", "source_test"]:
            raise ValueError("Endpoint manifests must be source_test or target_dev")
        targets.append(manifest)
    from .cache import bind_caches
    config, cache_identities = bind_caches(config, protocol, [train, val] + targets)
    if len(train["rows"]) % config["train_batch_size"] == 1:
        raise ValueError("BatchNorm head cannot train on a final singleton batch; fix the protocol explicitly")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"], text=True).strip():
        raise ValueError("Commit the runnable code before formal training")
    out = Path(config["output_root"]) / "r01" / run_id
    out.mkdir(parents=True, exist_ok=False)  # 不覆盖旧run，也不自动接续失败队列。
    started = time.monotonic()
    deadline = started + protocol["max_wall_seconds"]
    code = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    meta = dict(code_commit=code, config=config, protocol=protocol, run_id=run_id,
                cache_identities=cache_identities, torch_version=torch.__version__, device=str(device), started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    for name, manifest in [("train", train), ("val", val)]:
        (out / (name + "_manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False))
    seed_all(protocol["seed"])
    model = build_model(config, device)
    opt = config["optimizer"]
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=opt["lr"], betas=tuple(opt["betas"]), eps=opt["eps"], weight_decay=opt["weight_decay"])
    train_loader = make_loader(train["rows"], config, True, protocol["seed"], device)
    val_loader = make_loader(val["rows"], config, False, protocol["seed"], device)
    curve, best, step = [], float("inf"), 0
    best_epoch = None
    trainable = [p for p in model.parameters() if p.requires_grad]
    initial = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            initial.update(name.encode()); initial.update(parameter.detach().cpu().numpy().tobytes())
    meta["trainable_initial_sha256"] = initial.hexdigest()
    meta["trainable_parameters"] = sum(p.numel() for p in trainable)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    trace = (out / "batch_trace.jsonl").open("w", buffering=1)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(protocol["epochs"]):
            model.train()
            iterator = iter(train_loader)
            waiting = 0.0
            while True:
                start = time.monotonic()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                waiting += time.monotonic() - start  # 主线程取batch等待，非GPU分段时间。
                if time.monotonic() > deadline:
                    raise TimeoutError("Approved wall-time budget exhausted")
                images, labels, indices, views = move_batch(batch, device)
                image_hash = hashlib.sha256(images.detach().cpu().numpy().tobytes()).hexdigest()
                rng_hash = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
                if device.type == "cuda":
                    rng_hash += ":" + hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(images, views)["logits"], labels)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss before optimizer update")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, float("inf"), error_if_nonfinite=True)  # 仅核验，不裁剪有限梯度。
                optimizer.step()
                step += 1
                trace.write(json.dumps(dict(step=step, epoch=epoch+1, indices=indices.tolist(), image_sha256=image_hash, rng_sha256=rng_hash)) + "\n")
                if step == 1 or step % config["log_every"] == 0:
                    value = float(loss.detach())
                    if not np.isfinite(value):
                        raise ValueError("Nonfinite training loss")
                    print(json.dumps(dict(epoch=epoch + 1, step=step, loss=value,
                                          elapsed_seconds=time.monotonic() - started, host_data_wait_seconds=waiting, gradient_norm=float(grad_norm))), flush=True)
            metrics, logits = evaluate(model, val_loader, val["rows"], device, deadline)
            np.save(out / f"source_val_epoch{epoch + 1}.npy", logits)  # 保存逐样本源验证证据。
            score = metrics["macro"]["balanced_ce"]
            curve.append(dict(epoch=epoch + 1, source_val=metrics))
            (out / "learning_curve.json").write_text(json.dumps(curve, indent=2))
            if score < best:
                best = score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), out / "source_best.pth")
            print(json.dumps(curve[-1]), flush=True)
        torch.save(model.state_dict(), out / "terminal.pth")
        endpoint_results = {}
        for endpoint in ["source_best", "terminal"]:
            model.load_state_dict(torch.load(out / (endpoint + ".pth"), map_location=device, weights_only=True), strict=True)
            endpoint_results[endpoint] = []
            for n, manifest in enumerate(targets):
                loader = make_loader(manifest["rows"], config, False, protocol["seed"], device)
                metrics, logits = evaluate(model, loader, manifest["rows"], device, deadline)
                np.save(out / f"{endpoint}_eval{n}.npy", logits)
                (out / f"eval{n}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False))
                endpoint_results[endpoint].append(dict(role=manifest["role"], metrics=metrics))
                loader.dataset.close()
        result = dict(status="completed", optimizer_updates=step, endpoints=endpoint_results, source_best_epoch=best_epoch,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                      elapsed_seconds=time.monotonic() - started, independent_final_test=False)
        (out / "result.json").write_text(json.dumps(result, indent=2))
        return result
    except Exception as exc:
        (out / "failure.json").write_text(json.dumps(dict(status="failed",error=repr(exc),optimizer_updates=step)))
        raise
    finally:
        trace.close()
        train_loader.dataset.close()
        val_loader.dataset.close()
