#!/usr/bin/env python3
"""单任务监督、独立CPU归档；不接续其他实验。"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.state/vqgan-20261029'
RUN='4090-VQGAN码本重构-20261029-种子3407'

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def verify():
    import numpy as np
    spec=importlib.util.spec_from_file_location('paired_verifier',ROOT/'scripts/首批双卡监督.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    metrics=module.independent_metrics  # 复用已登记的独立秩和AUC实现，不调用训练指标函数。
    machine=json.loads((ROOT/'r01.machine.json').read_text());out=Path(machine['output_root'])/'r01'/RUN
    result=json.loads((out/'complete_result.json').read_text());meta=json.loads((out/'metadata.json').read_text())
    if not result.get('complete_evaluation') or result['optimizer_updates']!=21576:raise ValueError('训练或完整评估未完成')
    if set(result['endpoints'])!={'source_best','terminal'}:raise ValueError('缺少权重端点')
    curve=json.loads((out/'learning_curve.json').read_text());val=json.loads((out/'val_manifest.json').read_text())['rows']
    if len(curve)!=12 or len(val)!=700:raise ValueError('源验证不完整')
    scores=[];max_error=0.0
    for epoch in range(1,13):
        values=np.load(out/f'source_val_epoch{epoch}.npy',allow_pickle=False)
        if values.shape!=(700,2) or not np.isfinite(values).all():raise ValueError('源验证预测错误')
        score=metrics(values,val)['macro']['balanced_ce'];scores.append(score)
        if abs(score-curve[epoch-1]['source_val']['macro']['balanced_ce'])>1e-10:raise ValueError('源验证复算不符')
    if int(np.argmin(scores))+1!=result['source_best_epoch']:raise ValueError('源选点错误')
    for endpoint,entries in result['endpoints'].items():
        if [e['role'] for e in entries]!=['source_test','target_dev']:raise ValueError('缺少完整源测试或开发域')
        for i,entry in enumerate(entries):
            manifest=json.loads((out/f'eval{i}_manifest.json').read_text());rows=manifest['rows']
            frozen=json.loads((Path.home()/'prd-inputs/重构算子首批-20261029'/(['源测试清单.json','目标开发清单.json'][i])).read_text())
            if manifest!=frozen or len(rows)!=[5600,165758][i]:raise ValueError('评估清单改变或缩减')
            logits=np.load(out/f'{endpoint}_eval{i}.npy',allow_pickle=False)
            if logits.shape!=(len(rows),2) or not np.isfinite(logits).all():raise ValueError('预测不完整')
            measured=metrics(logits,rows)
            for group in ['domains','macro']:
                expected=entry['metrics'][group];actual=measured[group]
                pairs=[(actual[k][m],expected[k][m]) for k in actual for m in ['auc','ce','balanced_ce']] if group=='domains' else [(actual[k],expected[k]) for k in actual]
                for actual_value,expected_value in pairs:
                    error=abs(actual_value-expected_value);max_error=max(max_error,error)
                    if error>1e-10:raise ValueError('独立指标复算失败')
    trace=[json.loads(line) for line in (out/'batch_trace.jsonl').read_text().splitlines()]
    if [x['step'] for x in trace]!=list(range(1,21577)):raise ValueError('trace步数错误')
    for epoch in range(1,13):
        indices=[i for x in trace if x['epoch']==epoch for i in x['indices']]
        if sorted(indices)!=list(range(28768)):raise ValueError('训练帧缺失或重复')
    reference=out.parent/'4090-小波重构-20261029-种子3407'
    previous=json.loads((reference/'metadata.json').read_text())
    if meta['trainable_initial_sha256']!=previous['trainable_initial_sha256'] or sha(out/'batch_trace.jsonl')!=sha(reference/'batch_trace.jsonl'):raise ValueError('与小波完整训练初值或增强RNG轨迹不配对')
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in {'.pth','.npy','.json','.jsonl'}}
    report=dict(status='verified',time=now(),output=str(out),code_commit=meta['code_commit'],updates=21576,
                independent_metrics_max_error=max_error,pairing_verified=True,artifacts_sha256=hashes,result=result,
                scope='开发评估；缺少本批原图/VAE匹配控制，不宣称相对提点')
    save(STATE/'verification.json',report)
    return report

def main():
    claim=sys.argv[1]
    if len(claim)!=40:raise ValueError('缺本地正式认领提交')
    state=STATE/'supervisor.json'
    if state.exists():raise FileExistsError('已有监督证据，不自动重试')
    info=dict(status='running',time=now(),supervisor_pid=os.getpid(),cwd=str(ROOT),claim_commit=claim)
    log=(STATE/'run.log').open('x')
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/运行VQGAN单项.py'),'run','--claim-commit',claim],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info['training_pid']=child.pid;save(state,info)
    def stop(signum=None,frame=None):
        if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        try:code=child.wait(timeout=88200)  # 缓存12小时、正式12小时及收尾余量；不续跑。
        except subprocess.TimeoutExpired:
            stop()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
        info.update(exit_code=code,time=now())
        if code!=0:
            info.update(status='failed');save(state,info)
        else:
            info.update(status='cpu_verification');save(state,info)
            info['verification']=verify();info.update(status='verified',time=now());save(state,info)
        archive=ROOT/'scripts/experiment_summaries/重构VQGAN-20261029';archive.mkdir(parents=True,exist_ok=False)
        save(archive/'完成或失败.json',info)
        subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','归档VQGAN重构正式结果','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        (STATE/'result_commit.txt').write_text(commit+'\n');info.update(result_commit=commit);save(state,info)
        if code!=0:raise SystemExit(code)
    except BaseException as exc:
        stop();info.update(error=repr(exc),time=now())
        if info.get('exit_code')==0:info['status']='archive_failed'
        save(state,info);raise
    finally:log.close()

if __name__=='__main__':main()
