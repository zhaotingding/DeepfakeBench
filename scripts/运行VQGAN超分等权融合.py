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
from prd_r01.cache import build_cache,file_sha,read_cache,operator_spec
from prd_r01.generative_operators import GenerativeProbe
from prd_r01.vqgan_operator import VQGANProbe
LABELS={"vqgan_sr_mean":"VQGAN超分等权响应融合"}
CUTOFF=datetime.datetime.fromisoformat("2026-11-03T22:00:00+08:00")
FULL_CACHES={"vqgan":Path.home()/"prd-runs/重构缓存-20261029/VQGAN码本-训练验证", "superresolution":Path("/tmp/prd-1007/重构第四批-20261102/降采样超分重构-训练验证")}
def make_probes(c,device):
    return {"vqgan":VQGANProbe(c).to(device),"superresolution":GenerativeProbe(c,"superresolution").to(device)}
from prd_r01.engine import seed_all,run_training,video_metrics
from prd_r01.model import build_model
VARIANT=None
STATE=None
INPUTS=Path.home()/"prd-inputs/重构算子首批-20261029"
RUN=None

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+".tmp");t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8");t.replace(p)
def git(*args):return subprocess.check_output(["git","-C",str(ROOT),*args],text=True).strip()
def settings():
    c=read_config(ROOT/"r01.machine.json",VARIANT)
    c["vqgan_weights"]=str(Path.home()/".cache/prd-models/vqgan-imagenet-f16-1024/last.ckpt")
    c["vqgan_weights_sha256"]="e38a3aed8c86e8d47436f611325b3a23888cab70a0cc56c13cd84e8d4aab6e71"
    paths=[INPUTS/(n+".json") for n in ["训练清单","源验证清单","源测试清单","目标开发清单"]]
    return c,paths,[load_manifest(p) for p in paths]

def endpoint_loader(rows,c,device):
    data=LMDBImages(rows,dict(c,reconstruction_mode='online_after_augment'),False)
    return torch.utils.data.DataLoader(data,batch_size=64,num_workers=4,shuffle=False,
        pin_memory=device.type=='cuda',worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(3407),
        multiprocessing_context='spawn',prefetch_factor=2)

def smoke(device):
    c,paths,items=settings();area=STATE/'smoke';area.mkdir(parents=True,exist_ok=False)
    if torch.cuda.mem_get_info(device)[0]<28*1024**3:raise RuntimeError('三视图预检空闲显存不足28GiB')
    rows=items[0]['rows'][:64];raw=LMDBImages(rows,dict(c,reconstruction_mode='online_after_augment'),False)
    images=torch.stack([raw[i][0] for i in range(64)]);raw.close()
    frozen=make_probes(c,device);restored={};operator_audits={};seconds={}
    for name,probe in frozen.items():
        operator_audits[name]=probe.audit_quantization(images[:1].to(device)) if name=='vqgan' else probe.audit(images[:1].to(device))
        probe(images[:1].to(device));torch.cuda.synchronize(device);start=time.monotonic()
        restored[name]=probe(images.to(device));torch.cuda.synchronize(device);seconds[name]=(time.monotonic()-start)/64
        torch.testing.assert_close(probe(images[:2].to(device)),restored[name][:2],atol=1e-6,rtol=0)
        cache=area/('缓存-'+name);build_cache(rows,c,probe,str(cache),operator_spec(c,name),device,16,600)
    c['reconstruction_caches']={name:str(area/('缓存-'+name)) for name in frozen}
    cache_loader=make_loader(rows,c,False,3407,device);cached=next(iter(cache_loader))[3];cache_loader.dataset.close()
    for name in frozen:torch.testing.assert_close(cached[name].to(device),restored[name],atol=1e-6,rtol=0)
    full_cache_audit={}
    for name,path in FULL_CACHES.items():
        values,meta,slots=read_cache(path,items[0]['rows']+items[1]['rows'],c,operator_spec(c,name),verify_file=True)
        assert len(slots)==29468
        full_cache_audit[name]=dict(path=str(path),frames=len(slots),tensor_sha256=meta['tensor_sha256'],generation_seconds=meta['generation_seconds'],spec=meta['spec']);del values
    seed_all(3407);model=build_model(c,device);trainable=[p for p in model.parameters() if p.requires_grad]
    initial=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad:initial.update(name.encode());initial.update(p.detach().cpu().numpy().tobytes())
    assert initial.hexdigest()=='9aa889456c6661e4c09eb03229689f17e2f9766a4e8d09d4c009badf83f133e5'
    model.eval();observed=[];hook=model.encoder.register_forward_hook(lambda module,args,result:observed.append(result.detach()))
    with torch.inference_mode():output=model(images[:2].to(device),{n:v[:2] for n,v in restored.items()})
    hook.remove();assert len(observed)==3 and output['features'].shape==(2,2048)
    expected=(observed[0]-observed[1]+observed[0]-observed[2])/2
    torch.testing.assert_close(output['features'][:,1024:],expected,atol=1e-6,rtol=1e-6)
    formula_error=float((output['features'][:,1024:]-expected).abs().max());del observed,output,expected
    seed_all(3407);model=build_model(c,device);trainable=[p for p in model.parameters() if p.requires_grad]
    opt=c['optimizer'];optimizer=torch.optim.Adam(trainable,lr=opt['lr'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'])
    loader=make_loader(rows,c,True,3407,device);times=[];trace=[];torch.cuda.reset_peak_memory_stats(device);model.train()
    for epoch in range(2):
        for x,y,index,views in loader:
            x=x.to(device);y=y.to(device);views={n:v.to(device) for n,v in views.items()}
            trace.append(dict(indices=index.tolist(),image_sha=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()))
            torch.cuda.synchronize(device);start=time.monotonic();optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x,views)['logits'],y);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(trainable,float('inf'),error_if_nonfinite=True)
            assert torch.isfinite(loss) and float(norm)>0 and any(p.grad is not None and float(p.grad.abs().max())>0 for n,p in model.named_parameters() if 'lora_B' in n)
            optimizer.step();torch.cuda.synchronize(device);times.append(time.monotonic()-start)
    loader.dataset.close();optimizer.zero_grad(set_to_none=True);del optimizer
    old=ROOT.parent/'重构算子探索/.state/重构算子首批-20261029/wavelet-smoke/验收.json'
    assert trace==json.loads(old.read_text(encoding='utf-8'))['trace']
    checkpoint=area/'临时重载权重.pth';torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
    second=build_model(c,device);second.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval();second.eval()
    with torch.inference_mode():
        torch.cuda.synchronize(device);start=time.monotonic();a=model(images.to(device),restored)['logits'];b=second(images.to(device),restored)['logits'];torch.cuda.synchronize(device)
        dual_seconds=time.monotonic()-start;reload_error=float((a-b).abs().max())
    assert reload_error<=1e-6
    train_step=float(np.median(times[2:]));peak=torch.cuda.max_memory_reserved(device)
    formal_projection=(21576*train_step+12*700/64*dual_seconds+(5600+165758)*sum(seconds.values())+(5600+165758)/64*dual_seconds)*1.2+600
    summary=dict(status='passed',time=now(),code_commit=git('rev-parse','HEAD'),updates=8,formal_updates=0,operator_audits=operator_audits,full_cache_audit=full_cache_audit,
        train_step_median=train_step,dual_eval64_seconds=dual_seconds,operator_seconds_per_frame=seconds,formal_projection_seconds=formal_projection,cache_projection_seconds=60,
        budget_pass=formal_projection<=43200 and formal_projection+660<(CUTOFF-datetime.datetime.now(datetime.timezone.utc)).total_seconds(),required_free_bytes=int(peak+max(peak*.2,2*1024**3)),
        peak_reserved_bytes=peak,reload_max_abs=reload_error,formula_max_abs=formula_error,trace=trace,initial_sha256=initial.hexdigest(),response_fusion='equal_mean',power_cutoff=CUTOFF.isoformat())
    save(area/'验收.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in {'trace','full_cache_audit'}},ensure_ascii=False),flush=True)
    if not summary['budget_pass']:raise RuntimeError('当前实测预算不通过，未启动正式训练')


def evaluate_two(c,out,items,deadline,device):
    frozen=make_probes(c,device)  # 两个端点共享两种冻结重构，各按原内部逐图路径。
    models={}
    for endpoint in ['source_best','terminal']:
        model=build_model(c,device);model.load_state_dict(torch.load(out/(endpoint+'.pth'),map_location='cpu',weights_only=True),strict=True);model.eval();models[endpoint]=model
    results={k:[] for k in models}
    with torch.inference_mode():
        for i,manifest in enumerate(items):
            arrays={k:np.empty((len(manifest['rows']),2),np.float32) for k in models}
            loader=endpoint_loader(manifest['rows'],c,device);started=time.monotonic()
            for batch,(x,y,index,views) in enumerate(loader):
                if time.monotonic()>deadline or datetime.datetime.now(datetime.timezone.utc)>=CUTOFF:raise TimeoutError('达到正式预算或供电截止')
                x=x.to(device,non_blocking=True);views={name:probe(x) for name,probe in frozen.items()}
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
    c,paths,items=settings();check=json.loads((STATE/'smoke/验收.json').read_text(encoding='utf-8'))
    if check['code_commit']!=git('rev-parse','HEAD') or not check['budget_pass'] or git('status','--porcelain'):raise ValueError('预检/代码/预算身份不符')
    if torch.cuda.mem_get_info(device)[0]<check['required_free_bytes']:raise RuntimeError('空闲显存低于实测峰值及余量')
    if check['formal_projection_seconds']+660>(CUTOFF-datetime.datetime.now(datetime.timezone.utc)).total_seconds():raise RuntimeError('剩余供电窗口不足')
    if shutil.disk_usage(ROOT).free<12*1024**3:raise OSError('结果盘空间不足')
    status=dict(stage='cache_verification',time=now(),pid=os.getpid(),cwd=str(ROOT),claim_commit=claim,code_commit=git('rev-parse','HEAD'),run_id=RUN)
    state=STATE/'status.json';save(state,status)
    try:
        start=time.monotonic();costs={}
        for name,path in FULL_CACHES.items():
            values,meta,slots=read_cache(path,items[0]['rows']+items[1]['rows'],c,operator_spec(c,name),verify_file=True)
            assert meta['tensor_sha256']==check['full_cache_audit'][name]['tensor_sha256'];costs[name]=meta['generation_seconds'];del values
        reuse_seconds=time.monotonic()-start
        protocol=dict(status='approved',id='VQGAN超分等权融合-20261103',epochs=12,seed=3407,max_wall_seconds=min(43200,(CUTOFF-datetime.datetime.now(datetime.timezone.utc)).total_seconds()),
            train_manifest=str(paths[0]),val_manifest=str(paths[1]),evaluation_manifests=[],checkpoint_rule='source_val_video_balanced_ce',
            reconstruction_mode='cached_paired',augmentation_protocol='paired_float32_v1',reconstruction_caches={name:str(path) for name,path in FULL_CACHES.items()},
            endpoint_manifests=[str(p) for p in paths[2:]],endpoint_mode='two_frozen_operators_shared_by_two_endpoints',response_fusion='equal_mean',weights=[0.5,0.5],
            authorization='用户2026-11-02明确授权GPU1运行队列VQGAN加超分等权响应融合，持续供电至11-03 22:00；仅本项，0自动重试。')
        save(STATE/'protocol.json',protocol);status.update(stage='training',time=now(),formal_started=now(),cache_reused=True,cache_verification_seconds=reuse_seconds,historical_cache_seconds=costs);save(state,status)
        start=time.monotonic();deadline=start+protocol['max_wall_seconds'];result=run_training(c,protocol,RUN,device)
        torch.cuda.empty_cache();out=Path(c['output_root'])/'r01'/RUN;status.update(stage='evaluation',time=now());save(state,status)
        result['endpoints']=evaluate_two(c,out,items[2:],deadline,device)
        result.update(elapsed_seconds=time.monotonic()-start,cache_seconds=sum(costs.values()),new_cache_generation_seconds=0,cache_reused=True,historical_cache_seconds=costs,cache_reuse_verification_seconds=reuse_seconds,cache_cost_attribution='两份既有缓存复用；增量生成成本0，独立方法总成本各归属一次',complete_evaluation=True,response_fusion='equal_mean')
        save(out/'complete_result.json',result);status.update(stage='compute_finished',time=now(),result=result);save(state,status)
    except BaseException as exc:
        status.update(stage='failed',time=now(),error=repr(exc));save(state,status);raise


def main():
    global VARIANT,STATE,RUN
    p=argparse.ArgumentParser();p.add_argument('variant',choices=['vqgan_sr_mean']);p.add_argument('mode',choices=['smoke','run']);p.add_argument('--claim-commit');a=p.parse_args()
    VARIANT=a.variant;STATE=ROOT/'.state/VQGAN超分等权融合-20261103'/VARIANT;RUN='4090-'+LABELS[VARIANT]+'-20261103-种子3407'
    gpu='GPU-c3093e18-2d2f-8eb8-be80-f5802f43bdcc'
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('GPU绑定不符')
    import fcntl  # 与既有本项目GPU槽共用排他锁。
    lock_path=Path.home()/'.cache/prd-locks'/'重构算子探索-gpu1.lock'
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
