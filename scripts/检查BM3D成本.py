#!/usr/bin/env python3
"""八张固定源图的CPU可用性与成本检查，不训练分类器。"""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json
import time
import importlib.metadata
from pathlib import Path
import sys
import numpy as np
import torch
import bm3d
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from prd_r01.config import read_config
from prd_r01.data import LMDBImages, load_manifest

def main():
    torch.set_num_threads(1)
    config = read_config(ROOT / "r01.machine.json", "original")
    manifest = Path.home() / "prd-inputs/重构算子首批-20261029/训练清单.json"
    rows = load_manifest(manifest)["rows"]
    indices = [0, 1, 200, 201, 5000, 5001, 20000, 20001]
    dataset = LMDBImages([rows[i] for i in indices], config, False)
    timings, errors, outputs = [], [], []
    for i in range(len(indices)):
        pixels, _ = dataset.read_pixels(i)
        pixels = pixels.astype(np.float32) / 255
        start = time.monotonic()
        reconstructed = bm3d.bm3d_rgb(pixels, 25/255, profile="np", colorspace="opp")
        elapsed = time.monotonic() - start
        if reconstructed.shape != pixels.shape or not np.isfinite(reconstructed).all():
            raise ValueError("BM3D形状或数值异常")
        timings.append(elapsed)
        outputs.append(reconstructed.astype(np.float32))
        print(json.dumps(dict(index=indices[i], seconds=elapsed)), flush=True)
    for i in [0, 1]:
        pixels, _ = dataset.read_pixels(i)
        repeat = bm3d.bm3d_rgb(pixels.astype(np.float32)/255, 25/255, profile="np", colorspace="opp").astype(np.float32)
        errors.append(float(np.max(np.abs(repeat-outputs[i]))))
    dataset.close()
    seconds = float(np.median(timings[1:]))
    result = dict(stage="cpu_cost_preflight", frames=8, formal_updates=0, new_checkpoints=0,
        package_versions={p:importlib.metadata.version(p) for p in ["bm3d", "PyWavelets", "numpy", "scipy", "torch"]},
        sigma=25/255, profile="np", colorspace="opp", threads=1,
        seconds_per_frame_median=seconds, repeated_max_abs=max(errors),
        sequential_train_val_hours=29468*seconds/3600,
        four_process_train_val_hours_ideal=29468*seconds/4/3600,
        four_process_one_full_eval_hours_ideal=171358*seconds/4/3600,
        note="四进程投影只是假设，不代替真实并发吞吐验收。")
    state=ROOT/".state/bm3d-20261029";state.mkdir(parents=True,exist_ok=True)
    (state/"cpu_cost.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__ == "__main__":main()
