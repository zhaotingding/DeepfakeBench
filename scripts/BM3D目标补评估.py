"""只补BM3D完整目标评估；固定权重，分块落盘，0训练更新。"""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse,datetime,fcntl,gc,hashlib,json,shutil,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from prd_r01.config import read_config
from prd_r01.data import LMDBImages,load_manifest,worker_seed
from prd_r01.bm3d_operator import restore_one
from prd_r01.engine import seed_all,video_metrics
from prd_r01.model import build_model
STATE=ROOT/'.state/BM3D目标补评估-20261103'
RUN='4090-BM3D低复杂度重构-20261103-种子3407'
INPUTS=Path.home()/'prd-inputs/重构算子首批-20261029'
CUTOFF=datetime.datetime.fromisoformat('2026-11-03T22:00:00+08:00')
WEIGHT_SHA='7aff2ed2fd4388a1a13cbdf491855c973e300f67b0576cc564b7c8eb24f19ae2'
SOURCE_PRED_SHA='1b42da158c85e6dd679ffcfe9ec506f951ef9751dbb34d219941bb3e86615a69'
CHUNK=4096  # 64的倍数，保持原评估batch边界并定期回收worker。

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    temp.replace(path)
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def memory():
    keys=('MemAvailable:','SwapFree:');report={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(keys)}
    group=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split('::',1)[1].lstrip('/')
    report['cgroup']={name:(group/name).read_text().strip() for name in ['memory.current','memory.max','memory.peak','memory.events'] if (group/name).is_file()}
    return report

class EndpointImages(torch.utils.data.Dataset):
    def __init__(self,rows,c):
        self.base=LMDBImages(rows,dict(c,reconstruction_mode='online_after_augment'),False)
        self.mean,self.std=c['mean'],c['std']
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        x,y,index,_=self.base[i]
        restored=torch.from_numpy(restore_one((x.numpy(),self.mean,self.std)))
        return x,y,index,{'bm3d':restored}
    def close(self):self.base.close()

def loader(rows,c,device):
    return torch.utils.data.DataLoader(EndpointImages(rows,c),batch_size=64,num_workers=4,shuffle=False,pin_memory=True,
        worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(3407),multiprocessing_context='spawn',prefetch_factor=2)

def inputs():
    c=read_config(ROOT/'r01.machine.json','bm3d');out=Path(c['output_root'])/'r01'/RUN
    meta=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    assert meta['code_commit']=='3a8bfd5ebccb5851c8450b5bedfe2aabf08e573b'
    for name in ['resolution','mean','std','lora','head','residual','test_batch_size']:
        assert c[name]==meta['config'][name],name
    for name in ['source_best','terminal']:
        assert sha(out/(name+'.pth'))==WEIGHT_SHA
        assert sha(out/(name+'_eval0.npy'))==SOURCE_PRED_SHA
    source=load_manifest(INPUTS/'源测试清单.json');target=load_manifest(INPUTS/'目标开发清单.json')
    assert source==json.loads((out/'eval0_manifest.json').read_text(encoding='utf-8'))
    assert len(source['rows'])==5600 and len(target['rows'])==165758
    result=json.loads((out/'result.json').read_text(encoding='utf-8'));assert result['optimizer_updates']==21576 and result['source_best_epoch']==12
    return c,out,source,target,result

def model_for(c,out,device):
    seed_all(3407);m=build_model(c,device)
    m.load_state_dict(torch.load(out/'source_best.pth',map_location='cpu',weights_only=True),strict=True)
    m.requires_grad_(False).eval();return m  # 两端点权重完整SHA相同，只需计算一次唯一模型。

def preflight(device):
    if (STATE/'preflight.json').exists():raise FileExistsError('预检证据已存在')
    c,out,source,target,result=inputs();m=model_for(c,out,device)
    expected=np.load(out/'source_best_eval0.npy',allow_pickle=False);max_error=0.;offset=0;start=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for chunk in [source['rows'][:64],source['rows'][64:128]]:  # 验证重建worker后仍保持源测试预测。
            batches=loader(chunk,c,device)
            for x,y,index,views in batches:
                logits=m(x.to(device),{k:v.to(device) for k,v in views.items()})['logits'].cpu().numpy()
                error=float(np.max(np.abs(logits-expected[offset+index.numpy()])));max_error=max(max_error,error)
                if error>1e-6:raise ValueError('已有源预测重放超出1e-6：'+str(error))
            offset+=len(chunk);batches.dataset.close();del batches;gc.collect()
    probe=STATE/'partial-write-check.npy';arr=np.lib.format.open_memmap(probe,mode='w+',dtype=np.float32,shape=(128,2));arr[:64]=expected[:64];arr.flush()
    save(STATE/'partial-write-check.json',dict(completed_frames=64,total_frames=128,status='partial_not_complete'))
    check=np.load(probe,mmap_mode='r',allow_pickle=False);assert np.array_equal(check[:64],expected[:64]);del check,arr
    assert CHUNK%64==0
    required=torch.cuda.max_memory_reserved(device)+1024**3
    projection=165758/1280*183.57*1.15+300  # 使用原目标阶段实测保守吞吐，含worker回收和余量。
    report=dict(status='passed',time=now(),code_commit=git('rev-parse','HEAD'),training_updates=0,gpu_probe_frames=128,
        source_replay_max_abs=max_error,weight_sha256=WEIGHT_SHA,unique_checkpoints=1,equivalent_endpoints=['source_best','terminal'],
        workers=4,chunk_frames=CHUNK,partial_write_reopen=True,required_free_bytes=required,elapsed_seconds=time.monotonic()-start,
        projected_seconds=projection,budget_pass=projection<28800 and projection+600<(CUTOFF-datetime.datetime.now(datetime.timezone.utc)).total_seconds(),memory=memory())
    save(STATE/'preflight.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)
    if not report['budget_pass']:raise RuntimeError('补评估预算不通过')

def run(device,claim):
    check=json.loads((STATE/'preflight.json').read_text(encoding='utf-8'))
    assert check['code_commit']==git('rev-parse','HEAD') and check['budget_pass'] and not git('status','--porcelain')
    assert len(claim)==40 and not (STATE/'status.json').exists()
    if torch.cuda.mem_get_info(device)[0]<check['required_free_bytes']:raise RuntimeError('GPU0空闲低于补评估实测余量')
    if check['projected_seconds']+600>(CUTOFF-datetime.datetime.now(datetime.timezone.utc)).total_seconds():raise RuntimeError('剩余供电窗口不足')
    c,out,source,target,result=inputs()
    for name in ['source_best_eval1.npy','terminal_eval1.npy','complete_result.json']:
        if (out/name).exists():raise FileExistsError('拒绝覆盖已有目标产物：'+name)
    if shutil.disk_usage(STATE).free<4*1024**3:raise OSError('落盘空间不足')
    start=time.monotonic();deadline=min(start+28800,start+(CUTOFF-datetime.datetime.now(datetime.timezone.utc)).total_seconds())
    status=dict(stage='evaluation',time=now(),started_at=now(),pid=os.getpid(),cwd=str(ROOT),run_id=RUN,claim_commit=claim,code_commit=git('rev-parse','HEAD'),training_updates=0,completed_frames=0,total_frames=len(target['rows']),max_seconds=28800,power_cutoff=CUTOFF.isoformat())
    save(STATE/'status.json',status);partial=STATE/'目标预测.partial.npy'
    if partial.exists():raise FileExistsError('已有补评估输出，不自动重试')
    values=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(len(target['rows']),2));m=model_for(c,out,device)
    def interrupted(sig,frame):raise InterruptedError('received_signal_'+str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    completed=0
    try:
        with torch.inference_mode():
            for offset in range(0,len(target['rows']),CHUNK):
                batches=loader(target['rows'][offset:offset+CHUNK],c,device)
                try:
                    for x,y,index,views in batches:
                        if time.monotonic()>deadline or datetime.datetime.now(datetime.timezone.utc)>=CUTOFF:raise TimeoutError('补评估8小时或供电截止')
                        indices=offset+index.numpy();assert np.array_equal(indices,np.arange(completed,completed+len(indices)))
                        logits=m(x.to(device),{k:v.to(device) for k,v in views.items()})['logits'].cpu().numpy()
                        if logits.dtype!=np.float32 or not np.isfinite(logits).all():raise ValueError('目标预测异常')
                        values[indices]=logits;values.flush();completed+=len(indices)
                        status.update(completed_frames=completed,time=now(),elapsed_seconds=time.monotonic()-start)
                        save(STATE/'status.json',status)  # 每批完成游标与已刷盘预测对应，部分结果不冒充完整。
                        if completed==64 or (completed//64)%20==0:print(json.dumps(dict(stage='evaluation',frames=completed,total=len(values),seconds=time.monotonic()-start,memory=memory())),flush=True)
                finally:batches.dataset.close();del batches;gc.collect()
        assert completed==len(target['rows']) and np.isfinite(values).all();values.flush();del values
        for name in ['source_best.pth','terminal.pth']:assert sha(out/name)==WEIGHT_SHA
        for name in ['source_best_eval1.npy','terminal_eval1.npy']:
            destination=out/name;tmp=out/(name+'.recovery.tmp');assert not tmp.exists() and not destination.exists();shutil.copyfile(partial,tmp);tmp.replace(destination)
        save(out/'eval1_manifest.json',target)
        source_values=np.load(out/'source_best_eval0.npy',allow_pickle=False);target_values=np.load(partial,allow_pickle=False)
        entries=[dict(role='source_test',metrics=video_metrics(source_values,source['rows'])),dict(role='target_dev',metrics=video_metrics(target_values,target['rows']))]
        result['endpoints']={name:entries for name in ['source_best','terminal']}
        old=json.loads((ROOT/'.state/BM3D恢复-20261103/supervisor.json').read_text(encoding='utf-8'))
        original=json.loads((ROOT/'.state/BM3D恢复-20261103/bm3d/status.json').read_text(encoding='utf-8'))
        original_observed=(datetime.datetime.fromisoformat(old['time'])-datetime.datetime.fromisoformat(original['formal_started'])).total_seconds()
        recovery_seconds=time.monotonic()-start
        result.update(status='completed',complete_evaluation=True,cache_seconds=original['cache_seconds'],elapsed_seconds=original_observed+recovery_seconds,
            original_attempt_formal_seconds_upper_bound=original_observed,evaluation_recovery_seconds=recovery_seconds,evaluation_recovery_commit=git('rev-parse','HEAD'),
            recovery_training_updates=0,recovery_claim_commit=claim,unique_evaluated_checkpoints=1,checkpoint_sha256=WEIGHT_SHA,
            prior_failure_commit='ef2cf93221df59ba49f2405a12a9811804664201',cost_note='原尝试正式耗时按监督发现退出时上界计；含已耗中断评估，再加本次补评估；缓存只计一次')
        save(out/'complete_result.json',result);status.update(stage='compute_finished',time=now(),result=result);save(STATE/'status.json',status)
    except BaseException as exc:
        try:values.flush()
        except (NameError,AttributeError):pass
        status.update(stage='interrupted' if isinstance(exc,InterruptedError) else 'failed',time=now(),error=repr(exc),completed_frames=completed);save(STATE/'status.json',status);raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','run']);parser.add_argument('--claim-commit');args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-83850660-4196-4569-27bc-a3f9e72ada01':raise ValueError('GPU0绑定不符')
    STATE.mkdir(parents=True,exist_ok=True);lock=(Path.home()/'.cache/prd-locks/重构算子并行-gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush();torch.set_num_threads(1);device=torch.device('cuda:0')
    if args.mode=='preflight':preflight(device)
    else:run(device,args.claim_commit)
if __name__=='__main__':main()
