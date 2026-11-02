#!/usr/bin/env python3
"""人脸码本预测与降采样超分；沿用当前14组的共同训练评估协议。"""
import os
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",HF_HUB_OFFLINE="1",TOKENIZERS_PARALLELISM="false")
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"training"))
from prd_r01.config import read_config
from prd_r01.data import load_manifest,LMDBImages,make_loader,worker_seed
from prd_r01.cache import build_cache,file_sha
from prd_r01.generative_operators import GenerativeProbe as NextProbe,spec,LABELS
from prd_r01.engine import seed_all,run_training,video_metrics
from prd_r01.model import build_model
VARIANT=None
STATE=None
INPUTS=Path.home()/"prd-inputs/重构算子首批-20261029"
RUN=None

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def git(*args):return subprocess.check_output(["git","-C",str(ROOT),*args],text=True).strip()
def settings():
    c=read_config(ROOT/"r01.machine.json",VARIANT)
    paths=[INPUTS/(n+".json") for n in ["训练清单","源验证清单","源测试清单","目标开发清单"]]
    return c,paths,[load_manifest(p) for p in paths]

def endpoint_loader(rows,c,device):
    data=LMDBImages(rows,dict(c,reconstruction_mode='online_after_augment'),False)
    return torch.utils.data.DataLoader(data,batch_size=64,num_workers=4,shuffle=False,
        pin_memory=device.type=='cuda',worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(3407),
        multiprocessing_context='spawn',prefetch_factor=2)

def smoke(device):
    c,paths,items=settings();area=STATE/"smoke";area.mkdir(parents=True,exist_ok=False)
    if torch.cuda.mem_get_info(device)[0]<26*1024**3:raise RuntimeError("预检空闲不足26GiB")
    rows=items[0]["rows"][:64];raw=LMDBImages(rows,dict(c,reconstruction_mode="online_after_augment"),False)
    images=torch.stack([raw[i][0] for i in range(64)]);raw.close()
    probe=NextProbe(c,VARIANT).to(device)
    quantization_audit=probe.audit(images[:1].to(device))
    probe(images[:1].to(device))  # 预热恢复网络。
    torch.cuda.synchronize(device);started=time.monotonic();restored=probe(images.to(device));torch.cuda.synchronize(device)
    elapsed=time.monotonic()-started
    error=float((probe(images[:2].to(device))-restored[:2]).abs().max())
    if error>1e-6:raise ValueError('重构重复不一致')
    cache=area/'缓存';build_cache(rows,c,probe,str(cache),spec(c,VARIANT),device,16,600)
    c["reconstruction_caches"]={VARIANT:str(cache)}
    seed_all(3407);model=build_model(c,device);trainable=[p for p in model.parameters() if p.requires_grad]
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if p.requires_grad:h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    if h.hexdigest()!='9aa889456c6661e4c09eb03229689f17e2f9766a4e8d09d4c009badf83f133e5':raise ValueError("训练初值与本批不同")
    opt=c['optimizer'];optimizer=torch.optim.Adam(trainable,lr=opt['lr'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'])
    loader=make_loader(rows,c,True,3407,device);times=[];trace=[];torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for epoch in range(2):
        for x,y,index,views in loader:
            x=x.to(device);y=y.to(device);views={k:v.to(device) for k,v in views.items()}
            trace.append(dict(indices=index.tolist(),image_sha=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()))
            torch.cuda.synchronize(device);start=time.monotonic();optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x,views)['logits'],y);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(trainable,float('inf'),error_if_nonfinite=True)
            if not torch.isfinite(loss) or float(norm)<=0:raise ValueError('loss/梯度异常')
            if not any(p.grad is not None and float(p.grad.abs().max())>0 for n,p in model.named_parameters() if 'lora_B' in n):raise ValueError('LoRA没有真实梯度')
            optimizer.step();torch.cuda.synchronize(device);times.append(time.monotonic()-start)
    loader.dataset.close();optimizer.zero_grad(set_to_none=True);del optimizer
    old=ROOT.parent/'重构算子探索/.state/重构算子首批-20261029/wavelet-smoke/验收.json'
    if trace!=json.loads(old.read_text())['trace']:raise ValueError('与旧两臂原图增强不匹配')
    checkpoint=area/'临时重载权重.pth';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
    second=build_model(c,device);second.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval();second.eval();x=images.to(device);views={VARIANT:restored.to(device)}
    with torch.inference_mode():
        torch.cuda.synchronize(device);start=time.monotonic();a=model(x,views)['logits'];b=second(x,views)['logits'];torch.cuda.synchronize(device)
        dual_seconds=time.monotonic()-start;reload_error=float((a-b).abs().max())
    if reload_error>1e-6:raise ValueError('双端点重载不一致')
    eval_loader=endpoint_loader(rows,c,device)
    with torch.inference_mode():
        for x,y,index,views in eval_loader:
            repeated=probe(x.to(device))
            torch.testing.assert_close(repeated,restored[index],atol=1e-6,rtol=0)
            cache_loader=make_loader(rows,c,False,3407,device)
            cached=next(iter(cache_loader))[3][VARIANT].to(device)
            cache_loader.dataset.close()
            torch.testing.assert_close(cached,restored,atol=1e-6,rtol=0)
    eval_loader.dataset.close()
    seconds_per_frame=elapsed/64;train_step=float(np.median(times[2:]));peak=torch.cuda.max_memory_reserved(device)
    formal_projection=(21576*train_step+12*700/64*dual_seconds+(5600+165758)*seconds_per_frame+(5600+165758)/64*dual_seconds)*1.2+600
    cache_projection=29468*seconds_per_frame*1.2+300
    summary=dict(status='passed',time=now(),operator_identity=spec(c,VARIANT),operator_audit=quantization_audit,code_commit=git('rev-parse','HEAD'),updates=8,formal_updates=0,
        seconds_per_frame_gpu=seconds_per_frame,train_step_median=train_step,dual_eval64_seconds=dual_seconds,
        formal_projection_seconds=formal_projection,cache_projection_seconds=cache_projection,
        budget_pass=formal_projection<=43200 and cache_projection<=43200,required_free_bytes=int(peak+max(peak*.2,2*1024**3)),
        peak_reserved_bytes=peak,reload_max_abs=reload_error,parallel_max_abs=error,trace=trace,initial_sha256=h.hexdigest())
    save(area/'验收.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='trace'},ensure_ascii=False),flush=True)
    if not summary['budget_pass']:raise RuntimeError('真实预检超过预算，停止正式启动')

def evaluate_two(c,out,items,deadline,device):
    probe=NextProbe(c,VARIANT).to(device)  # 两个端点共用一份冻结恢复器。
    models={}
    for endpoint in ['source_best','terminal']:
        model=build_model(c,device);model.load_state_dict(torch.load(out/(endpoint+'.pth'),map_location='cpu',weights_only=True),strict=True);model.eval();models[endpoint]=model
    results={k:[] for k in models}
    with torch.inference_mode():
        for i,manifest in enumerate(items):
            arrays={k:np.empty((len(manifest['rows']),2),np.float32) for k in models}
            loader=endpoint_loader(manifest['rows'],c,device);started=time.monotonic()
            for batch,(x,y,index,views) in enumerate(loader):
                if time.monotonic()>deadline:raise TimeoutError('正式训练及评估达到12小时')
                x=x.to(device,non_blocking=True);views={VARIANT:probe(x)}
                for name,model in models.items():
                    logits=model(x,views)['logits']
                    if not torch.isfinite(logits).all():raise ValueError('评估非有限')
                    arrays[name][index.numpy()]=logits.cpu().numpy()
                if batch%20==0:print(json.dumps(dict(stage='evaluation',manifest=i,frames=min((batch+1)*64,len(manifest['rows'])),total=len(manifest['rows']),seconds=time.monotonic()-started)),flush=True)
            loader.dataset.close();save(out/f'eval{i}_manifest.json',manifest)
            for name,logits in arrays.items():
                np.save(out/f'{name}_eval{i}.npy',logits);results[name].append(dict(role=manifest['role'],metrics=video_metrics(logits,manifest['rows'])))
    return results

def run(device,claim):
    c,paths,items=settings();check=json.loads((STATE/'smoke/验收.json').read_text())
    if check['code_commit']!=git('rev-parse','HEAD') or not check['budget_pass']:raise ValueError('预检版本或预算不符')
    if git('status','--porcelain'):raise ValueError('运行代码未提交')
    if torch.cuda.mem_get_info(device)[0]<check['required_free_bytes']:raise RuntimeError('空闲显存低于实测余量')
    cache_root=Path('/tmp')/('prd-'+str(os.getuid()))/'重构第四批-20261102'
    cache_root.mkdir(parents=True,exist_ok=True)  # 本批可重建缓存独立放置，不删除历史资产。
    required=2*29468*3*224**2*4
    if shutil.disk_usage(cache_root).free<required+80*1024**3:raise OSError('本机临时缓存盘余量不足')
    if shutil.disk_usage(ROOT).free<2*3*1024**3+20*1024**3:raise OSError('持久结果目录余量不足')
    status=dict(stage='cache',pid=os.getpid(),cwd=str(ROOT),time=now(),claim_commit=claim,code_commit=git('rev-parse','HEAD'),run_id=RUN)
    state=STATE/'status.json';save(state,status)
    try:
        cache=cache_root/(LABELS[VARIANT]+'-训练验证')
        probe=NextProbe(c,VARIANT).to(device)
        cache_meta=build_cache(items[0]['rows']+items[1]['rows'],c,probe,str(cache),spec(c,VARIANT),device,16,43200)
        del probe
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info(device)[0]<check['required_free_bytes']:raise RuntimeError('缓存后GPU资源不足，保留缓存并停止')
        protocol=dict(status='approved',id=LABELS[VARIANT]+'-20261102',epochs=12,seed=3407,max_wall_seconds=43200,
            train_manifest=str(paths[0]),val_manifest=str(paths[1]),evaluation_manifests=[],checkpoint_rule='source_val_video_balanced_ce',
            reconstruction_mode='cached_paired',augmentation_protocol='paired_float32_v1',reconstruction_caches={VARIANT:str(cache)},
            endpoint_manifests=[str(p) for p in paths[2:]],endpoint_mode='shared_gpu_restoration_two_gpu_models',
            authorization='用户2026-11-02授权继续在4090双卡探索重构算子；本批仅人脸码本预测与降采样超分，各12轮，从共同初值开始，0自动重试，今日22:00停止未完成计算。')
        save(STATE/'protocol.json',protocol);status.update(stage='training',time=now(),formal_started=now(),cache_seconds=cache_meta['generation_seconds']);save(state,status)
        start=time.monotonic();deadline=start+43200
        result=run_training(c,protocol,RUN,device)
        torch.cuda.empty_cache();out=Path(c['output_root'])/'r01'/RUN
        status.update(stage='evaluation',time=now());save(state,status)
        result['endpoints']=evaluate_two(c,out,items[2:],deadline,device)
        result.update(elapsed_seconds=time.monotonic()-start,cache_seconds=cache_meta['generation_seconds'],complete_evaluation=True)
        save(out/'complete_result.json',result)
        status.update(stage='compute_finished',time=now(),result=result);save(state,status)
    except BaseException as exc:
        status.update(stage='failed',time=now(),error=repr(exc));save(state,status);raise

def main():
    global VARIANT,STATE,RUN
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['codeformer','superresolution']);p.add_argument('mode',choices=['smoke','run']);p.add_argument('--claim-commit');a=p.parse_args()
    VARIANT=a.variant;STATE=ROOT/'.state/重构第四批-20261102'/VARIANT;RUN='4090-'+LABELS[VARIANT]+'-20261102-种子3407'
    gpu='GPU-83850660-4196-4569-27bc-a3f9e72ada01' if VARIANT=='codeformer' else 'GPU-c3093e18-2d2f-8eb8-be80-f5802f43bdcc'
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('GPU绑定不符')
    import fcntl  # 与既有本项目GPU槽共用排他锁。
    lock_path=Path.home()/'.cache/prd-locks'/('重构算子并行-gpu0.lock' if VARIANT=='codeformer' else '重构算子探索-gpu1.lock')
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path.open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)  # 不删除或释放旧批次持有的锁。
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    torch.set_num_threads(1)
    device=torch.device('cuda:0');seed_all(3407)
    if a.mode=='smoke':smoke(device)
    else:
        if not a.claim_commit or len(a.claim_commit)!=40:raise ValueError('缺认领提交')
        run(device,a.claim_commit)

if __name__=='__main__':main()
