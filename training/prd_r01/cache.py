"""冻结重构图缓存：保存FP32归一化张量，不缓存可训练编码器特征。"""
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

def file_sha(path):
    digest = hashlib.sha256()  # 身份校验替代再次执行昂贵的重构。
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def row_key(row):
    return (row["domain"], row["key"])  # 原LMDB键不做斜杠规范化。

def operator_spec(config, name):
    if name in {"codeformer", "superresolution"}:
        from .generative_operators import spec
        return spec(config, name)  # 新算子的冻结资产、精度与输入工作点。
    if name in {"colorization", "nlm", "sparse", "e4e"}:
        from .diverse_operators import spec
        return spec(config, name)  # 第三批各算子完整身份。
    if name in {"mae", "compression", "tv"}:
        from .next_operators import spec
        return spec(config, name)  # 完整工作点身份，拒绝混用其他缓存。
    if name == "vqgan":
        from .vqgan_operator import vqgan_spec
        return vqgan_spec(config)
    if name in {"wavelet", "lowrank"}:
        from .operators import classical_spec
        return classical_spec(config, name)
    if name != "vae":
        raise ValueError("尚未实现该重构算子的缓存生成：" + name)
    root = Path(config["vae_path"])
    weight = next((root / n for n in ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"]
                   if (root / n).is_file()), None)
    if weight is None:
        raise FileNotFoundError("缺少本地VAE权重")
    return dict(operator=name, recipe="vae-mode-256-bilinear-v1", resolution=config["resolution"],
                mean=config["mean"], std=config["std"], storage="normalized-float32-nchw",
                weights_sha256=file_sha(weight), config_sha256=file_sha(root / "config.json"),
                **({"execution":config["vae_inference"]} if "vae_inference" in config else {}))  # 新对照固定内部批次和推理精度；旧身份不变。

def read_cache(path, rows, config, expected_spec=None, verify_file=False):
    root = Path(path).resolve()
    meta = json.loads((root / "缓存清单.json").read_text())
    if meta.get("status") != "complete" or meta.get("schema") != 1:
        raise ValueError("缓存未完整生成")
    spec = meta["spec"]
    for key in ["resolution", "mean", "std"]:
        if spec[key] != config[key]:
            raise ValueError("缓存预处理不匹配：" + key)
    if spec.get("storage") != "normalized-float32-nchw":
        raise ValueError("不支持的缓存精度或布局")
    if expected_spec is not None and spec != expected_spec:
        raise ValueError("缓存算子权重或工作点不匹配")
    file = root / "重构张量.npy"
    if file.stat().st_size != meta["file_bytes"]:
        raise ValueError("缓存文件大小不符")
    if verify_file and file_sha(file) != meta["tensor_sha256"]:
        raise ValueError("缓存内容校验失败")
    values = np.load(file, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(meta["entries"]), 3, config["resolution"], config["resolution"])
    if values.dtype != np.float32 or values.shape != expected_shape:
        raise ValueError("缓存形状或精度不符")
    lookup = {row_key(e["row"]): (i, e) for i, e in enumerate(meta["entries"])}
    if len(lookup) != len(meta["entries"]):
        raise ValueError("缓存包含重复原始键")
    slots = []
    for row in rows:
        if row_key(row) not in lookup:
            raise ValueError("缓存缺少清单中的帧")
        slot, entry = lookup[row_key(row)]
        if entry["row"] != row:
            raise ValueError("缓存标签/视频来源或清单元数据不同")
        slots.append(slot)
    return values, meta, slots

def build_cache(rows, config, probe, destination, spec, device, batch_size=16, max_seconds=3600):
    from .data import LMDBImages
    if not rows or len({row_key(r) for r in rows}) != len(rows):
        raise ValueError("缓存清单必须非空且无重复帧")
    if batch_size < 1 or max_seconds <= 0:
        raise ValueError("缓存需要有限批大小与时限")
    root = Path(destination).expanduser()
    if not root.is_absolute():
        raise ValueError("缓存路径必须为绝对路径")
    if root.exists():
        raise FileExistsError("拒绝覆盖已有缓存目录：" + str(root))
    root.parent.mkdir(parents=True, exist_ok=True)
    needed = len(rows) * 3 * config["resolution"] ** 2 * 4 + 1024 * 1024
    if shutil.disk_usage(root.parent).free < needed * 1.1:
        raise OSError("缓存空间不足，未启动重构")
    root.mkdir()  # 不复用中断目录，避免把部分缓存当作完整产物。
    started = time.monotonic()
    (root / "生成状态.json").write_text(json.dumps(dict(status="building", frames=len(rows)), ensure_ascii=False))
    dataset = LMDBImages(rows, dict(config, reconstruction_mode="online_after_augment"), augment=False)
    array = np.lib.format.open_memmap(root / "重构张量.npy", mode="w+", dtype=np.float32,
                                     shape=(len(rows), 3, config["resolution"], config["resolution"]))
    entries = []
    probe.requires_grad_(False).eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("缓存生成达到批准时限")
                batch, batch_entries = [], []
                for i in range(start, min(start + batch_size, len(rows))):
                    pixels, source_sha = dataset.read_pixels(i)
                    image = torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float().div_(255)
                    batch.append((image - dataset.mean) / dataset.std)
                    batch_entries.append(dict(row=rows[i], source_sha256=source_sha))
                images = torch.stack(batch).to(device)
                reconstructed = probe(images)
                if reconstructed.shape != images.shape or not torch.isfinite(reconstructed).all():
                    raise ValueError("算子返回形状错误或非有限重构")
                array[start:start + len(batch)] = reconstructed.float().cpu().numpy()
                entries.extend(batch_entries)
                if start == 0 or (start // batch_size) % 100 == 0:
                    print(json.dumps(dict(stage="cache", frames=len(entries), total=len(rows), elapsed_seconds=time.monotonic()-started)), flush=True)
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("缓存生成达到批准时限")
        array.flush()
        del array
        tensor_file = root / "重构张量.npy"
        meta = dict(schema=1, status="complete", spec=spec, entries=entries,
                    file_bytes=tensor_file.stat().st_size, tensor_sha256=file_sha(tensor_file),
                    device=str(device), torch_version=torch.__version__, generation_seconds=time.monotonic()-started,
                    generation_dtype=str(next(probe.parameters(), torch.empty(0)).dtype),
                    optimizer_updates=0, new_checkpoints=0)
        (root / "缓存清单.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        (root / "生成状态.json").write_text(json.dumps(dict(status="complete"), ensure_ascii=False))
        return meta
    except Exception as exc:
        (root / "生成状态.json").write_text(json.dumps(dict(status="failed", error=repr(exc)), ensure_ascii=False))
        raise
    finally:
        dataset.close()

def bind_caches(config, protocol, manifests):
    if config.get("reconstruction_mode") != "cached_paired" or not config.get("responses"):
        return dict(config), {}
    supplied = protocol.get("reconstruction_caches", {})
    names = set(config["responses"])
    if set(supplied) != names:
        raise ValueError("协议必须为每种实际重构算子指定缓存目录")
    merged, seen = [], {}
    for manifest in manifests:
        for row in manifest["rows"]:
            key = row_key(row)
            if key in seen and seen[key] != row:
                raise ValueError("跨清单相同帧元数据冲突")
            if key not in seen:
                merged.append(row)
                seen[key] = row
    identities = {}
    for name in sorted(names):
        if not Path(supplied[name]).is_absolute():
            raise ValueError("协议缓存路径必须是绝对路径")
        values, meta, _ = read_cache(supplied[name], merged, config, operator_spec(config, name), verify_file=True)
        del values
        identities[name] = dict(path=supplied[name], spec=meta["spec"], tensor_sha256=meta["tensor_sha256"],
                                generation_seconds=meta["generation_seconds"], device=meta["device"])
    return dict(config, reconstruction_caches=supplied), identities
