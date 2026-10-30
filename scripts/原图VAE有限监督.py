#!/usr/bin/env python3
"""原图与VAE两卡有限监督；一臂失败不停止另一臂，不接续MAE。"""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.state/原图VAE对照-20261030'
ARMS={'original':('GPU-83850660-4196-4569-27bc-a3f9e72ada01','仅原图对照',45000),
      'vae':('GPU-c3093e18-2d2f-8eb8-be80-f5802f43bdcc','原VAE对照',88200)}

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)
def stop(child):
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)  # 只停止本监督创建的超时进程组。
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def verify(arm,label):
    spec=importlib.util.spec_from_file_location('control_verifier',ROOT/'scripts/VQGAN有限监督.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.STATE=STATE/arm;module.RUN=f'4090-{label}-20261030-种子3407'
    return module.verify()  # 复用已验收的完整清单、独立CPU指标、源选与trace核验。

def main():
    claim=sys.argv[1]
    if len(claim)!=40:raise ValueError('缺少本地正式认领')
    if (STATE/'supervisor.json').exists():raise FileExistsError('已有监督记录，禁止自动重试')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('代码未提交')
    summary=dict(status='running',time=now(),pid=os.getpid(),cwd=str(ROOT),claim_commit=claim,arms={})
    children={};logs={};started={}
    for arm,(gpu,label,budget) in ARMS.items():
        area=STATE/arm
        if (area/'status.json').exists():raise FileExistsError('已有正式任务，不重复启动')
        log=(area/'run.log').open('x');logs[arm]=log
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/运行原图VAE对照.py'),arm,'run','--claim-commit',claim],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[arm]=child;started[arm]=time.monotonic()
        info=dict(status='running',time=now(),supervisor_pid=os.getpid(),training_pid=child.pid,cwd=str(ROOT),claim_commit=claim,gpu_uuid=gpu)
        summary['arms'][arm]=info;save(area/'supervisor.json',info)
    save(STATE/'supervisor.json',summary)
    def interrupted(signum,frame):
        for child in children.values():stop(child)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    pending=set(children)
    while pending:
        for arm in list(pending):
            child=children[arm];gpu,label,budget=ARMS[arm];code=child.poll()
            if code is None and time.monotonic()-started[arm]>budget:
                stop(child);code=124  # 原图12小时＋余量，VAE缓存12＋正式12小时＋余量。
            if code is None:continue
            pending.remove(arm);logs[arm].close();info=summary['arms'][arm]
            info.update(exit_code=code,time=now(),status='failed' if code else 'cpu_verification')
            save(STATE/arm/'supervisor.json',info)
            if code==0:
                try:info.update(verification=verify(arm,label),status='verified',time=now())
                except Exception as exc:info.update(status='archive_failed',error=repr(exc),time=now())
            save(STATE/arm/'supervisor.json',info)
            archive=ROOT/'scripts/experiment_summaries/原图VAE对照-20261030'/arm
            archive.mkdir(parents=True,exist_ok=False);save(archive/'完成或失败.json',info)
            summary['time']=now();save(STATE/'supervisor.json',summary)
        if pending:time.sleep(15)
    archive=Path('scripts/experiment_summaries/原图VAE对照-20261030')
    summary.update(status='verified' if all(x['status']=='verified' for x in summary['arms'].values()) else 'finished_with_errors',time=now())
    save(ROOT/archive/'批次.json',summary)
    try:
        subprocess.run(['git','add','--',str(archive)],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','归档原图与VAE同协议对照结果','--',str(archive)],cwd=ROOT,check=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        for arm,info in summary['arms'].items():
            info['result_commit']=commit;(STATE/arm/'result_commit.txt').write_text(commit+'\n');save(STATE/arm/'supervisor.json',info)
        summary['result_commit']=commit;(STATE/'result_commit.txt').write_text(commit+'\n')
    except Exception as exc:summary.update(status='archive_failed',error=repr(exc))
    save(STATE/'supervisor.json',summary)
    return 0 if summary['status']=='verified' else 1

if __name__=='__main__':raise SystemExit(main())
