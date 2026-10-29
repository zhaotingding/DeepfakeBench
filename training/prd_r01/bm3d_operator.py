"""BM3D-LC固定算子与四进程CPU批处理；不改变可训练编码器。"""
import importlib.metadata
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch import nn

def restore_one(item):
    import bm3d
    image, mean, std = item
    pixels = image * np.asarray(std,dtype=np.float32)[:,None,None] + np.asarray(mean,dtype=np.float32)[:,None,None]
    restored = bm3d.bm3d_rgb(np.ascontiguousarray(pixels.transpose(1,2,0)),25/255,profile=bm3d.BM3DProfileLC(),colorspace="opp")
    if restored.shape != pixels.transpose(1,2,0).shape or not np.isfinite(restored).all():
        raise ValueError("BM3D-LC返回异常重构")
    value = restored.astype(np.float32).transpose(2,0,1)
    return np.ascontiguousarray((value-np.asarray(mean,dtype=np.float32)[:,None,None])/np.asarray(std,dtype=np.float32)[:,None,None])

class BM3DProbe(nn.Module):
    def __init__(self, config, parallel=False):
        super().__init__()
        self.mean,self.std=config["mean"],config["std"]
        self.pool=ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn")) if parallel else None

    def forward(self, images):
        items=[(x,self.mean,self.std) for x in images.detach().cpu().numpy()]
        values=list(self.pool.map(restore_one,items)) if self.pool else [restore_one(x) for x in items]
        return torch.from_numpy(np.stack(values)).to(device=images.device,dtype=images.dtype)

    def close(self):
        if self.pool:self.pool.shutdown();self.pool=None

def bm3d_spec(config):
    return dict(operator="bm3d",recipe="cbm3d-lc-opp-sigma25-v1",resolution=config["resolution"],mean=config["mean"],std=config["std"],
                storage="normalized-float32-nchw",profile="lc",sigma=25/255,colorspace="opp",
                versions={p:importlib.metadata.version(p) for p in ["bm3d","PyWavelets","numpy","scipy"]})
