#!/usr/bin/env python3
"""有限双卡监督：占本项目锁，超时只停止自己的子进程组，结束归档。"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/".state/重构算子首批-20261029"
INPUTS=Path.home()/"prd-inputs/重构算子首批-20261029"
CONFIG=json.loads((ROOT/"training/config/r01/小波低秩-20261029.json").read_text())
PYTHON=sys.executable

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def independent_metrics(logits,rows):
    import numpy as np
    from scipy.stats import rankdata
    groups={}
    for prediction,row in zip(logits,rows):
        key=(row['domain'],row['video_id']);item=groups.setdefault(key,{'label':row['label'],'values':[]})
        if item['label']!=row['label']:raise ValueError('视频标签冲突')
        item['values'].append(prediction)
    domains={}
    for domain in sorted({k[0] for k in groups}):
        items=[v for k,v in groups.items() if k[0]==domain]
        z=np.array([np.mean(v['values'],axis=0) for v in items],dtype=np.float64)
        y=np.array([v['label'] for v in items]);pos=y==1;neg=y==0
        normalizer=np.logaddexp(z[:,0],z[:,1]);scores=np.exp(z[:,1]-normalizer)
        ranks=rankdata(scores);n=pos.sum();m=neg.sum()
        auc=(ranks[pos].sum()-n*(n+1)/2)/(n*m)
        loss=normalizer-z[np.arange(len(y)),y]
        domains[domain]={'auc':float(auc),'ce':float(loss.mean()),'balanced_ce':float((loss[pos].mean()+loss[neg].mean())/2),'videos':len(y)}
    return {'domains':domains,'macro':{k:float(np.mean([v[k] for v in domains.values()])) for k in ['auc','ce','balanced_ce']}}

def verify_run(variant):
    import numpy as np
    name=CONFIG['display_names'][variant]
    machine=json.loads((ROOT/'machine.local.json').read_text())
    out=Path(machine['output_root'])/'r01'/('4090-'+name+'-20261029-种子3407')
    result=json.loads((out/'result.json').read_text());meta=json.loads((out/'metadata.json').read_text())
    expected=CONFIG['epochs']*((CONFIG['source_train_frames']+15)//16)
    if result['optimizer_updates']!=expected:raise ValueError('更新数不完整')
    curve=json.loads((out/'learning_curve.json').read_text())
    if len(curve)!=CONFIG['epochs']:raise ValueError('源验证曲线不完整')
    val=json.loads((out/'val_manifest.json').read_text())['rows'];source_scores=[]
    max_error=0.0
    for epoch in range(1,CONFIG['epochs']+1):
        m=independent_metrics(np.load(out/f'source_val_epoch{epoch}.npy'),val)
        score=m['macro']['balanced_ce'];source_scores.append(score)
        if abs(score-curve[epoch-1]['source_val']['macro']['balanced_ce'])>1e-10:raise ValueError('源选点复算不符')
    selected=int(np.argmin(source_scores))+1
    if selected!=result['source_best_epoch']:raise ValueError('源选epoch不符')
    for endpoint,entries in result['endpoints'].items():
        for i,entry in enumerate(entries):
            rows=json.loads((out/f'eval{i}_manifest.json').read_text())['rows']
            logits=np.load(out/f'{endpoint}_eval{i}.npy')
            if logits.shape!=(len(rows),2) or not np.isfinite(logits).all():raise ValueError('预测不完整')
            measured=independent_metrics(logits,rows)
            for domain,values in measured['domains'].items():
                for metric in ['auc','ce','balanced_ce']:
                    error=abs(values[metric]-entry['metrics']['domains'][domain][metric]);max_error=max(max_error,error)
                    if error>1e-10:raise ValueError('独立指标复算失败')
    with (out/'batch_trace.jsonl').open() as f:trace=[json.loads(l) for l in f]
    if len(trace)!=expected or [r['step'] for r in trace]!=list(range(1,expected+1)):raise ValueError('trace步数不完整')
    for epoch in range(1,CONFIG['epochs']+1):
        indices=[i for r in trace if r['epoch']==epoch for i in r['indices']]
        if sorted(indices)!=list(range(CONFIG['source_train_frames'])):raise ValueError('训练帧缺失或重复')
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in {'.pth','.npy','.json','.jsonl'}}
    report=dict(status='verified',time=now(),variant=variant,output=str(out),code_commit=meta['code_commit'],
        updates=expected,source_best_epoch=selected,independent_metrics_max_error=max_error,
        trainable_initial_sha256=meta['trainable_initial_sha256'],trace_sha256=sha(out/'batch_trace.jsonl'),
        artifacts_sha256=hashes,result=result,scope='开发评估；原图/VAE匹配控制尚待完成，不宣称相对提点')
    save(STATE/variant/'verification.json',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','run'],required=True);p.add_argument('--claim-commit',required=True)
    args=p.parse_args()
    if len(args.claim_commit)!=40:raise ValueError('缺少本地认领SHA')
    STATE.mkdir(parents=True,exist_ok=True)
    if (STATE/(args.mode+'-supervisor.json')).exists():raise FileExistsError('本阶段已启动或已有失败，不自动重试')
    locks=[];children={};logs={};exits={}
    for name in ['batch','gpu0','gpu1']:
        path=Path.home()/'.cache/prd-locks'/('重构首批-'+name+'.lock');path.parent.mkdir(parents=True,exist_ok=True)
        f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    summary=dict(status='running',mode=args.mode,time=now(),supervisor_pid=os.getpid(),cwd=str(ROOT),claim_commit=args.claim_commit,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),pids={})
    state=STATE/(args.mode+'-supervisor.json');save(state,summary)
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).strip().splitlines()
    gpu_map={int(row.split(',')[0]):row.split(',')[1].strip() for row in gpu_rows}
    summary['physical_gpus']={v:gpu_map[i] for i,v in enumerate(CONFIG['variants'])}
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',CUDA_VISIBLE_DEVICES=','.join(gpu_map[i] for i in [0,1]))
    deadline=time.monotonic()+(CONFIG['smoke_max_seconds'] if args.mode=='smoke' else CONFIG['cache_max_seconds']+CONFIG['formal_max_seconds']+1800)
    def stop_own(signum=None,frame=None):
        for child in children.values():
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop_own);signal.signal(signal.SIGINT,stop_own)
    try:
        for i,variant in enumerate(CONFIG['variants']):
            log=(STATE/(args.mode+'-'+variant+'.log')).open('x');logs[variant]=log
            command=[PYTHON,str(ROOT/'scripts/运行重构首批.py'),args.mode,'--variant',variant,'--device',f'cuda:{i}','--claim-commit',args.claim_commit]
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children[variant]=child;summary['pids'][variant]=child.pid
        save(state,summary)
        while any(child.poll() is None for child in children.values()):
            if time.monotonic()>deadline:
                stop_own();time.sleep(5)
                for child in children.values():
                    if child.poll() is None:os.killpg(child.pid,signal.SIGKILL)
                summary['timeout']=True;break
            time.sleep(10)
        exits={k:p.wait() for k,p in children.items()};summary.update(exit_codes=exits,time=now())
        if any(exits.values()):summary['status']='failed'
        elif args.mode=='smoke':
            checks={v:json.loads((STATE/(v+'-smoke')/'验收.json').read_text()) for v in CONFIG['variants']}
            a,b=checks.values()
            if a['initial_sha256']!=b['initial_sha256'] or a['trace']!=b['trace']:raise ValueError('预检初始化或原图增强序列不配对')
            summary['status']='passed';summary['pairing_verified']=True
        else:
            reports={v:verify_run(v) for v in CONFIG['variants']};a,b=reports.values()
            if a['trainable_initial_sha256']!=b['trainable_initial_sha256'] or a['trace_sha256']!=b['trace_sha256']:raise ValueError('正式初始化或完整trace不配对')
            summary.update(status='verified',pairing_verified=True,reports=reports)
        save(state,summary)
        if args.mode=='run':
            archive=ROOT/'scripts/experiment_summaries/重构算子首批-20261029';archive.mkdir(parents=True,exist_ok=False)
            save(archive/'完成或失败.json',summary)
            save(archive/'清单验收.json',json.loads((INPUTS/'清单验收.json').read_text()))
            subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
            subprocess.run(['git','commit','-m','归档小波与低秩重构首批结果','--only','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
            sha_result=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            (STATE/'result_commit.txt').write_text(sha_result+'\n')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        if summary['status']=='failed':raise SystemExit(1)
    except BaseException as exc:
        stop_own();summary.update(status='failed',error=repr(exc),time=now(),exit_codes=exits);save(state,summary);raise
    finally:
        for log in logs.values():log.close()
        for f in locks:f.close()

if __name__=='__main__':main()
