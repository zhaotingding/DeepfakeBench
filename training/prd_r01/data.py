"""清单驱动的LMDB数据入口，保留原始键并拒绝坏样本替换。"""
import json
from pathlib import Path
import random
import os

_ENVIRONMENTS = {}  # 同一进程复用同一路径的LMDB句柄，避免train/val重复打开。
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_manifest(path, expected_role=None):
    manifest = json.loads(Path(path).read_text())
    if expected_role is not None and manifest.get("role") != expected_role:
        raise ValueError(f"Expected manifest role {expected_role}")
    rows = manifest.get("rows", [])
    if not rows:
        raise ValueError("Manifest is empty")
    seen = set()
    for row in rows:
        for key in ["domain", "key", "label", "video_id"]:
            if key not in row:
                raise ValueError(f"Missing row field {key}")
        if row["label"] not in (0, 1) or not isinstance(row["key"], str):
            raise ValueError("Invalid label or LMDB key")
        if "/" in row["domain"] or "\\" in row["domain"] or row["domain"] in ("", ".", ".."):
            raise ValueError("Domain must be a dataset name")
        identity = (row["domain"], row["key"])
        if identity in seen:
            raise ValueError("Duplicate frame in manifest")
        seen.add(identity)
    return manifest

def validate_source_pair(train, val):
    if train["role"] != "source_train" or val["role"] != "source_val":
        raise ValueError("Training and checkpoint selection must use source manifests")
    if any(r["domain"] != "FaceForensics++" for r in train["rows"] + val["rows"]):
        raise ValueError("R01 source data must be FaceForensics++")
    identities = []
    for manifest in [train, val]:
        ids = set()
        for row in manifest["rows"]:
            if not row.get("source_ids"):
                raise ValueError("Source manifest needs source_ids for split-overlap verification")
            ids.update(row["source_ids"])
        identities.append(ids)
    if identities[0] & identities[1]:
        raise ValueError("Source train/val identity components overlap")
    if {r["label"] for r in val["rows"]} != {0, 1}:
        raise ValueError("Validation must contain both classes")

def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)  # 各配置使用相同worker初始化策略。

class LMDBImages(Dataset):
    def __init__(self, rows, config, augment=False):
        self.rows, self.config, self.augment = rows, config, augment
        self.envs = {}
        self.env_keys = {}
        self.transform = None
        self.caches = {}  # mmap按worker延迟打开。
        self.mean = torch.tensor(config["mean"]).view(3, 1, 1)
        self.std = torch.tensor(config["std"]).view(3, 1, 1)

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["envs"] = {}
        state["env_keys"] = {}  # spawn worker不携带LMDB句柄。
        state["transform"] = None
        state["caches"] = {}
        return state

    def read_pixels(self, index):
        import hashlib
        import cv2
        import lmdb
        row = self.rows[index]
        domain = row["domain"]
        if domain not in self.envs:
            path = Path(self.config["data_root"]) / "DeepfakeBench/lmdb" / (domain + "_lmdb")
            token = (os.getpid(), str(path.resolve()))
            if token not in _ENVIRONMENTS:
                _ENVIRONMENTS[token] = [lmdb.open(str(path), readonly=True, lock=False, create=False), 0]
            _ENVIRONMENTS[token][1] += 1
            self.envs[domain] = _ENVIRONMENTS[token][0]
            self.env_keys[domain] = token
        key = row["key"]
        if key.startswith("./datasets\\"):
            key = key[len("./datasets\\"):]  # 只沿用旧实现的已知前缀，不改斜杠或替换样本。
        with self.envs[domain].begin(write=False) as txn:
            raw = txn.get(key.encode())
        if raw is None:
            raise KeyError(f"Missing LMDB key: {domain}/{key}")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode: {key}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config["resolution"],) * 2, interpolation=cv2.INTER_CUBIC)
        return image, hashlib.sha256(raw).hexdigest()  # 原图变化则拒绝旧缓存。

    def __getitem__(self, index):
        from .cache import read_cache
        image, source_sha = self.read_pixels(index)
        row = self.rows[index]
        paired = self.config.get("augmentation_protocol") == "paired_float32_v1"
        views = {}
        if self.config.get("reconstruction_mode") == "cached_paired":
            for name in dict.fromkeys(self.config.get("responses", [])):
                if name not in self.caches:
                    path = self.config.get("reconstruction_caches", {}).get(name)
                    if path is None:
                        raise ValueError("缺少缓存目录，禁止回落在线重构：" + name)
                    self.caches[name] = read_cache(path, self.rows, self.config)
                values, meta, slots = self.caches[name]
                slot = slots[index]
                if meta["entries"][slot]["source_sha256"] != source_sha:
                    raise ValueError("原图内容变化，缓存已失效")
                cached = torch.from_numpy(np.array(values[slot], copy=True))
                if not torch.isfinite(cached).all():
                    raise ValueError("缓存帧包含非有限数值")
                views[name] = cached
        if self.augment and paired:
            targets = {}
            for name, value in views.items():
                pixels = (value * self.std + self.mean).permute(1, 2, 0).numpy()
                targets["reconstruction_" + name] = pixels.clip(0, 1)  # 增强时统一合法像素范围。
            if self.transform is None:
                self.transform = paired_transform(list(targets))
            original = self.normalize_pixels(image.astype(np.float32) / 255)
            original = (original * self.std + self.mean).permute(1, 2, 0).numpy().clip(0, 1)
            augmented = self.transform(image=original, **targets)  # 两路经历相同浮点往返，防止增强放大舍入差。
            image = augmented["image"]
            views = {name: self.normalize_pixels(augmented["reconstruction_" + name]) for name in views}
            tensor = self.normalize_pixels(image)
        else:
            if self.augment:
                if self.transform is None:
                    import albumentations as A
                    self.transform = A.Compose([
                        A.HorizontalFlip(p=0.5), A.Rotate(limit=(-10, 10), p=0.5),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                        A.OneOf([A.RandomBrightnessContrast(brightness_limit=.1, contrast_limit=.1),
                                 A.FancyPCA(), A.HueSaturationValue()], p=.5),
                        A.ImageCompression(quality_lower=40, quality_upper=100, p=.5),
                    ])  # 历史增强仅供显式旧协议。
                image = self.transform(image=image)["image"]
            tensor = self.normalize_pixels(image.astype(np.float32) / 255)
        return tensor, row["label"], index, views

    def normalize_pixels(self, image):
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        return (tensor - self.mean) / self.std

    def close(self):
        for token in self.env_keys.values():
            _ENVIRONMENTS[token][1] -= 1
            if _ENVIRONMENTS[token][1] == 0:
                _ENVIRONMENTS.pop(token)[0].close()
        self.envs.clear()
        self.env_keys.clear()
        self.caches.clear()

def paired_transform(names):
    import albumentations as A
    return A.Compose([
        A.HorizontalFlip(p=.5), A.Rotate(limit=(-10, 10), p=.5),
        A.GaussianBlur(blur_limit=(3, 7), p=.5),
        A.OneOf([A.RandomBrightnessContrast(brightness_limit=.1, contrast_limit=.1),
                 A.HueSaturationValue()], p=.5),
        A.ImageCompression(quality_lower=40, quality_upper=100, p=.5),
    ], additional_targets={name: "image" for name in names})  # 同次参数；浮点协议不用仅uint8的FancyPCA。

def make_loader(rows, config, training, seed, device):
    dataset = LMDBImages(rows, config, augment=training)
    kwargs = dict(
        batch_size=config["train_batch_size"] if training else config["test_batch_size"],
        shuffle=training, num_workers=config["workers"], pin_memory=config["pin_memory"] and device.type == "cuda",
        worker_init_fn=worker_seed, generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )
    if config["workers"] > 0:
        kwargs.update(prefetch_factor=config["prefetch_factor"], persistent_workers=config["persistent_workers"])
    return DataLoader(dataset, **kwargs)

def smoke_rows(config):
    path = Path(config["data_root"]) / "DeepfakeBench/config/dataset_json/FaceForensics++.json"
    data = json.loads(path.read_text())["FaceForensics++"]
    selected = {}
    for label, splits in data.items():
        for video, item in splits["train"]["c23"].items():
            y = 0 if item["label"] == "FF-real" else 1
            if y not in selected and item["frames"]:
                selected[y] = dict(domain="FaceForensics++", key=item["frames"][0], label=y, video_id=video)
            if len(selected) == 2:
                return [selected[0], selected[1]]  # 仅源数据接口验证，不评估目标AUC。
    raise ValueError("Source JSON does not contain two classes")
