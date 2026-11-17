#!/usr/bin/env python3
"""只评估既有绝对残差两份权重；不创建优化器或修改训练产物。"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,subprocess,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from prd_r01.data import LMDBImages,load_manifest,worker_seed
from prd_r01.model import build_model
from prd_r01.engine import seed_all,video_metrics
from prd_r01.vqgan_operator import VQGANProbe,vqgan_spec
STATE=ROOT/'.state/VQGAN绝对残差评估-20261104'
TRAIN_STATE=ROOT/'.state/VQGAN绝对残差训练-20261103'
INPUTS=Path.home()/'prd-inputs/重构算子首批-20261029'
RUN='4090-VQGAN绝对残差-20261103-种子3407'
TRAIN_CODE='8ff4ce93c10dc21221dfbd1d0b510c48c0d2f366'
GPU='GPU-c3093e18-2d2f-8eb8-be80-f5802f43bdcc'
WEIGHTS={'source_best':'042d3930c09df40eb7089210bc65a4adc2aa39a27205ea152fa48f017c7daf53','terminal':'d2e7a003ad40f994982730192a7f74fed33cde1ad547014400c5dd56567e10f9'}
MAX_SECONDS=28800

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()

def inputs():
    machine=read(ROOT/'machine.local.json');out=Path(machine['output_root'])/'r01'/RUN
    meta=read(out/'metadata.json');result=read(out/'result.json');training=read(TRAIN_STATE/'training_verification.json')
    if meta['code_commit']!=TRAIN_CODE or meta['config']['residual']!='absolute_diff' or result['optimizer_updates']!=21576:raise ValueError('原训练身份不符')
    if training['status']!='training_verified_pending_evaluation' or training['source_best_epoch']!=8 or not training['pairing_verified']:raise ValueError('原训练未验收')
    if result['endpoints']!={'source_best':[],'terminal':[]}:raise ValueError('原训练端点内容被改写')
    if meta['config']['test_batch_size']!=64 or meta['config']['head']['slots']!=2:raise ValueError('原评估模型形状或batch改变')
    for ep,digest in WEIGHTS.items():
        if sha(out/(ep+'.pth'))!=digest:raise ValueError('checkpoint身份改变：'+ep)
    for name in ['metadata.json','result.json','val_manifest.json','source_val_epoch8.npy','source_val_epoch12.npy']:
        if sha(out/name)!=training['artifacts_sha256'][name]:raise ValueError('源重放依据与训练验收hash不符：'+name)
    val=load_manifest(out/'val_manifest.json','source_val');source=load_manifest(INPUTS/'源测试清单.json','source_test');target=load_manifest(INPUTS/'目标开发清单.json','target_dev')
    if [len(x['rows']) for x in [val,source,target]]!=[700,5600,165758]:raise ValueError('完整数据清单不符')
    if len({(r['domain'],r['video_id']) for r in target['rows']})!=5876:raise ValueError('目标视频数量不符')
    return meta['config'],out,meta,result,training,val,[source,target]

def loader(rows,c,device):
    data=LMDBImages(rows,dict(c,reconstruction_mode='online_after_augment'),False)  # 仅冻结VQ在线重构，不增强评估图片。
    return torch.utils.data.DataLoader(data,batch_size=64,shuffle=False,num_workers=4,pin_memory=device.type=='cuda',worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(3407),multiprocessing_context='spawn',prefetch_factor=2)

def models(c,out,device):
    probe=VQGANProbe(c).to(device).eval();pair={}
    for ep in WEIGHTS:
        m=build_model(c,device);m.load_state_dict(torch.load(out/(ep+'.pth'),map_location='cpu',weights_only=True),strict=True);m.eval();m.requires_grad_(False);pair[ep]=m
    return probe,pair

def smoke(device):
    area=STATE/'preflight';area.mkdir(parents=True,exist_ok=False)
    c,out,meta,result,training,val,full=inputs()
    if (out/'complete_result.json').exists():raise FileExistsError('已有完整结果，不重复评估')
    seed_all(3407);probe,pair=models(c,out,device);torch.cuda.reset_peak_memory_stats(device)
    batches=loader(val['rows'][:128],c,device);times=[];errors={ep:0.0 for ep in pair};cached_errors={ep:0.0 for ep in pair}
    expected={ep:np.load(out/f'source_val_epoch{8 if ep=="source_best" else 12}.npy',allow_pickle=False)[:128] for ep in pair}
    # 同一源验证BS64重放检查原保存预测；不使用目标成绩调整实现。
    with torch.inference_mode():
        for x,y,index,views in batches:
            x=x.to(device);torch.cuda.synchronize(device);start=time.monotonic();restored=probe(x)
            for ep,m in pair.items():
                z=m(x,{'vqgan':restored})['logits'];errors[ep]=max(errors[ep],float(np.max(np.abs(z.cpu().numpy()-expected[ep][index.numpy()]))))
                if not torch.isfinite(z).all():raise ValueError('源验证重放非有限')
            torch.cuda.synchronize(device);times.append(time.monotonic()-start)
        # 另用完整128帧中的原缓存核对在线恢复、模型与旧评估输入是否一致。
        cached=LMDBImages(val['rows'][:128],c,False)
        for start in [0,64]:
            examples=[cached[i] for i in range(start,start+64)];x=torch.stack([t[0] for t in examples]).to(device);views={'vqgan':torch.stack([t[3]['vqgan'] for t in examples]).to(device)}
            for ep,m in pair.items():cached_errors[ep]=max(cached_errors[ep],float(np.max(np.abs(m(x,views)['logits'].cpu().numpy()-expected[ep][start:start+64]))))
        cached.close()
    batches.dataset.close()
    # 未通过重放时也先留证，不放宽既定1e-6容差。
    numerical=max([*errors.values(),*cached_errors.values()])<=1e-6
    seconds_per_batch=max(times);projection=(5600+165758)/64*seconds_per_batch*1.2+600
    peak=torch.cuda.max_memory_reserved(device);check=dict(status='passed' if numerical and projection<=MAX_SECONDS else 'preflight_rejected',time=now(),code_commit=git('rev-parse','HEAD'),training_code_commit=TRAIN_CODE,training_updates=0,checkpoint_sha256=WEIGHTS,source_replay_frames=128,source_replay_max_abs=errors,cached_source_replay_max_abs=cached_errors,tolerance=1e-6,bs64_dual_with_restoration_seconds=times,projected_seconds=projection,max_seconds=MAX_SECONDS,budget_pass=projection<=MAX_SECONDS,numerical_pass=numerical,peak_reserved_bytes=peak,required_free_bytes=int(peak+max(peak*.2,1024**3)),operator_spec=vqgan_spec(c),source_frames=5600,target_frames=165758,target_videos=5876)
    save(area/'验收.json',check);print(json.dumps(check,ensure_ascii=False),flush=True)
    if check['status']!='passed':raise RuntimeError('绝对残差补评估预检未通过；保留差异或预算证据')

def run(device,claim):
    check=read(STATE/'preflight/验收.json')
    if check['status']!='passed' or check['code_commit']!=git('rev-parse','HEAD'):raise ValueError('预检状态或版本不符')
    if git('status','--porcelain'):raise ValueError('运行代码未提交')
    if torch.cuda.mem_get_info(device)[0]<check['required_free_bytes']:raise RuntimeError('GPU1显存不足')
    if (STATE/'status.json').exists():raise FileExistsError('已有正式评估状态，禁止自动重试')
    c,out,meta,original,training,val,manifests=inputs()
    forbidden=[out/'complete_result.json',out/'evaluation_metadata.json']+[out/f'eval{i}_manifest.json' for i in range(2)]+[out/f'{ep}_eval{i}.npy' for ep in WEIGHTS for i in range(2)]
    if any(p.exists() for p in forbidden):raise FileExistsError('已有完整或部分正式评估产物，不覆盖')
    started=time.monotonic();deadline=started+MAX_SECONDS
    state=dict(stage='loading',time=now(),pid=os.getpid(),cwd=str(ROOT),run_id=RUN,claim_commit=claim,code_commit=git('rev-parse','HEAD'),training_code_commit=TRAIN_CODE,training_updates=0,max_seconds=MAX_SECONDS,completed_frames=0)
    save(STATE/'status.json',state)
    try:
        seed_all(3407);probe,pair=models(c,out,device)
        save(out/'evaluation_metadata.json',dict(task='VQGAN绝对残差评估-20261104',code_commit=state['code_commit'],training_code_commit=TRAIN_CODE,training_result_commit='48dd05410da84593a6d95732240eed21b0682900',claim_commit=claim,checkpoint_sha256=WEIGHTS,residual='absolute_diff',training_updates=0,operator_spec=vqgan_spec(c),bs=64,max_seconds=MAX_SECONDS))
        results={ep:[] for ep in pair}
        with torch.inference_mode():
            for i,manifest in enumerate(manifests):
                n=len(manifest['rows']);partial={ep:STATE/f'{ep}_eval{i}.partial.npy' for ep in pair}
                if any(p.exists() for p in partial.values()):raise FileExistsError('已有partial，不自动续跑')
                arrays={ep:np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(n,2)) for ep,p in partial.items()}
                dl=loader(manifest['rows'],c,device);completed=0;state.update(stage='evaluation',manifest=i,role=manifest['role'],completed_frames=0,total_frames=n,time=now());save(STATE/'status.json',state)
                for batch,(x,y,index,views) in enumerate(dl):
                    if time.monotonic()>deadline:raise TimeoutError('补评估8小时限时')
                    if index.tolist()!=list(range(completed,completed+len(index))):raise ValueError('评估索引缺失或重复')
                    x=x.to(device,non_blocking=True);views={'vqgan':probe(x)}
                    for ep,m in pair.items():
                        logits=m(x,views)['logits']
                        if not torch.isfinite(logits).all():raise ValueError('评估非有限')
                        arrays[ep][index.numpy()]=logits.cpu().numpy();arrays[ep].flush()  # 每batch持久化，但只有游标内是有效前缀。
                    completed+=len(index);state.update(completed_frames=completed,elapsed_seconds=time.monotonic()-started,time=now());save(STATE/'status.json',state)
                    if batch%20==0:print(json.dumps(state,ensure_ascii=False),flush=True)
                dl.dataset.close()
                if completed!=n:raise ValueError('清单评估未完整结束')
                for ep,a in arrays.items():
                    if not np.isfinite(a).all():raise ValueError('完整预测非有限')
                    results[ep].append(dict(role=manifest['role'],metrics=video_metrics(a,manifest['rows'])));a.flush()
                arrays.clear();del a
                save(out/f'eval{i}_manifest.json',manifest)
                for ep,p in partial.items():p.replace(out/f'{ep}_eval{i}.npy')  # 只发布完整文件，前置拒绝任何覆盖。
        for ep,digest in WEIGHTS.items():
            if sha(out/(ep+'.pth'))!=digest:raise ValueError('评估期间权重改变')
        previous=read(out/'training_stage_complete.json');elapsed=time.monotonic()-started
        result=dict(original,endpoints=results,complete_evaluation=True,residual='absolute_diff',training_code_commit=TRAIN_CODE,evaluation_code_commit=state['code_commit'],evaluation_claim_commit=claim,evaluation_seconds=elapsed,evaluation_training_updates=0,checkpoint_sha256=WEIGHTS,cache_seconds=previous['historical_cache_seconds'],cache_reused=True,new_cache_generation_seconds=0,cache_reuse_verification_seconds=previous['cache_reuse_verification_seconds'],training_seconds=original['elapsed_seconds'],elapsed_seconds=original['elapsed_seconds']+elapsed,cost_note='训练与本次完整评估分列；原VQ缓存生成和原训练缓存核验各计一次；预检不计正式成本')
        save(out/'complete_result.json',result);state.update(stage='compute_finished',time=now(),result=result);save(STATE/'status.json',state)
    except BaseException as exc:
        state.update(stage='failed',time=now(),error=repr(exc));save(STATE/'status.json',state);raise

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('须绑定物理GPU1')
    lock=Path.home()/'.cache/prd-locks/重构算子探索-gpu1.lock';lock.parent.mkdir(parents=True,exist_ok=True);handle=lock.open('a+')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handle.seek(0);handle.truncate();handle.write(str(os.getpid()));handle.flush()
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','run']);p.add_argument('--claim-commit');a=p.parse_args();torch.set_num_threads(1);device=torch.device('cuda:0')
    if a.mode=='smoke':smoke(device)
    else:
        if not a.claim_commit or len(a.claim_commit)!=40:raise ValueError('缺少本地认领SHA')
        run(device,a.claim_commit)
if __name__=='__main__':main()
