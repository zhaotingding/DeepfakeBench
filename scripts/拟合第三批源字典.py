"""只用源训练清单拟合一次固定K-SVD字典，不读取目标或验证标签。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
from pathlib import Path
import sys,time,json,hashlib
import numpy as np
import torch
from skimage.color import rgb2lab
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'training'))
from prd_r01.config import read_config
from prd_r01.data import LMDBImages,load_manifest
from prd_r01.diverse_operators import sparse_code,ASSETS

def main():
    torch.set_num_threads(2);started=time.monotonic();rng=np.random.default_rng(3407)
    target=ASSETS/'source_ksvd_dictionary.npy'
    if target.exists():raise FileExistsError('已有冻结字典，不覆盖或重新挑选')
    manifest=Path.home()/'prd-inputs/重构算子首批-20261029/训练清单.json'
    rows=load_manifest(manifest)['rows'];ids=rng.choice(len(rows),2048,replace=False)
    c=read_config(ROOT/'r01.machine.json','original');data=LMDBImages(rows,dict(c,reconstruction_mode='online_after_augment'),False)
    patches=[];samples=[]
    for idx in ids:
        rgb,sha=data.read_pixels(int(idx));light=rgb2lab(rgb)[:,:,0].astype(np.float32)/100
        points=rng.integers(0,217,size=(8,2))  # 每张源图固定抽8块，不按类别挑选。
        for y,x in points:
            p=light[y:y+8,x:x+8].reshape(-1);patches.append(p-p.mean())
        samples.append(dict(index=int(idx),key=rows[int(idx)]['key'],points=points.tolist(),source_sha256=sha))
    data.close();y=np.asarray(patches,dtype=np.float32).T;norm=np.linalg.norm(y,axis=0)
    candidates=np.flatnonzero(norm>1e-4);chosen=rng.choice(candidates,256,replace=False)
    dictionary=y[:,chosen]/norm[chosen];curve=[]
    for iteration in range(10):
        if time.monotonic()-started>1800:raise TimeoutError('字典准备超过30分钟')
        codes=sparse_code(torch.from_numpy(y.T.copy()),torch.from_numpy(dictionary.copy())).numpy().T
        residual=y-dictionary@codes;before=float(np.mean(residual**2))
        for atom in range(256):
            active=np.flatnonzero(np.abs(codes[atom])>1e-10)
            if not len(active):
                idx=int(np.argmax(np.sum(residual**2,axis=0)));v=residual[:,idx];dictionary[:,atom]=v/max(np.linalg.norm(v),1e-8);continue
            error=residual[:,active]+dictionary[:,atom,None]*codes[atom,active][None]
            u,s,vt=np.linalg.svd(error,full_matrices=False)
            dictionary[:,atom]=u[:,0];codes[atom,active]=s[0]*vt[0]
            residual[:,active]=error-dictionary[:,atom,None]*codes[atom,active][None]
        after=float(np.mean(residual**2))
        if not np.isfinite(dictionary).all() or after>before+1e-7:raise ValueError('K-SVD更新未降低重构目标')
        entry=dict(iteration=iteration+1,before_mse=before,after_mse=after,seconds=time.monotonic()-started);curve.append(entry);print(json.dumps(entry),flush=True)
    assert np.allclose(np.linalg.norm(dictionary,axis=0),1,atol=1e-5)
    np.save(target,dictionary.astype(np.float32))
    metadata=dict(method='K-SVD with stabilized OMP8',source_role='source_train',source_manifest=str(manifest),source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),images=2048,patches=16384,patch_size=8,atoms=256,iterations=10,seed=3407,label_conditioning=False,target_data_used=False,curve=curve,seconds=time.monotonic()-started,dictionary_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),samples=samples)
    (ASSETS/'source_ksvd_fit.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in metadata.items() if k!='samples'}),flush=True)
if __name__=='__main__':main()
