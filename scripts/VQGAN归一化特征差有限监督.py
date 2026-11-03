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
STATE=ROOT/'.state/VQGAN归一化特征差-20261104'
RUN='4090-VQGAN归一化特征差-20261104-种子3407'
MAX_FORMAL_SECONDS=43200  # 独立监督从启动起最多12小时，0自动重试。

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8");temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def verify():
    import numpy as np
    spec=importlib.util.spec_from_file_location('paired_verifier',ROOT/'scripts/归一化差独立指标.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    metrics=module.independent_metrics  # 复用已登记的独立秩和AUC实现，不调用训练指标函数。
    machine=json.loads((ROOT/'r01.machine.json').read_text(encoding="utf-8"));out=Path(machine['output_root'])/'r01'/RUN
    result=json.loads((out/'complete_result.json').read_text(encoding="utf-8"));meta=json.loads((out/'metadata.json').read_text(encoding="utf-8"))
    if meta.get('config',{}).get('residual')!='normalized_feature_diff' or meta.get('protocol',{}).get('residual')!='normalized_feature_diff' or result.get('residual')!='normalized_feature_diff':raise ValueError('模型配置、冻结协议或完整结果不是归一化特征差')
    if any(value.get('normalization_epsilon')!=1e-6 for value in [meta['config'],meta['protocol'],result]):raise ValueError('归一化ε与固定协议不符')  # 不允许结果口径漏记固定ε。
    if not result.get('complete_evaluation') or result['optimizer_updates']!=21576:raise ValueError('训练或完整评估未完成')
    if set(result['endpoints'])!={'source_best','terminal'}:raise ValueError('缺少权重端点')
    curve=json.loads((out/'learning_curve.json').read_text(encoding="utf-8"));val=json.loads((out/'val_manifest.json').read_text(encoding="utf-8"))['rows']
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
            manifest=json.loads((out/f'eval{i}_manifest.json').read_text(encoding="utf-8"));rows=manifest['rows']
            frozen=json.loads((Path.home()/'prd-inputs/重构算子首批-20261029'/(['源测试清单.json','目标开发清单.json'][i])).read_text(encoding="utf-8"))
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
    trace=[json.loads(line) for line in (out/'batch_trace.jsonl').read_text(encoding="utf-8").splitlines()]
    if [x['step'] for x in trace]!=list(range(1,21577)):raise ValueError('trace步数错误')
    for epoch in range(1,13):
        indices=[i for x in trace if x['epoch']==epoch for i in x['indices']]
        if sorted(indices)!=list(range(28768)):raise ValueError('训练帧缺失或重复')
    reference=out.parent/'4090-小波重构-20261029-种子3407'
    previous=json.loads((reference/'metadata.json').read_text(encoding="utf-8"))
    if meta['trainable_initial_sha256']!=previous['trainable_initial_sha256'] or sha(out/'batch_trace.jsonl')!=sha(reference/'batch_trace.jsonl'):raise ValueError('与小波完整训练初值或增强RNG轨迹不配对')
    hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in {'.pth','.npy','.json','.jsonl'}}
    report=dict(status='verified',time=now(),output=str(out),code_commit=meta['code_commit'],updates=21576,
                independent_metrics_max_error=max_error,pairing_verified=True,artifacts_sha256=hashes,result=result,
                scope='同协议VQGAN有符号残差为匹配对照；源选主端点及末轮完整报告，仍属开发评估')
    save(STATE/'verification.json',report)
    return report

def main():
    claim=sys.argv[1]
    if len(claim)!=40:raise ValueError('缺本地正式认领提交')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('运行代码未提交')
    check=json.loads((STATE/'smoke/验收.json').read_text(encoding='utf-8'))
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if check['code_commit']!=code_commit or not check['budget_pass']:raise ValueError('预检版本/预算不符')
    if check['formal_projection_seconds']+60>MAX_FORMAL_SECONDS:raise ValueError('预估超12小时预算')
    state=STATE/'supervisor.json'
    if state.exists():raise FileExistsError('已有监督证据，不自动重试')
    info=dict(status='running',time=now(),supervisor_pid=os.getpid(),cwd=str(ROOT),claim_commit=claim,code_commit=code_commit,max_seconds=MAX_FORMAL_SECONDS,power_cutoff=None)
    log=(STATE/'run.log').open('x',encoding='utf-8')
    child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/运行VQGAN归一化特征差.py'),'run','--claim-commit',claim],
                           cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-83850660-4196-4569-27bc-a3f9e72ada01',PYTHONUTF8='1',PYTHONIOENCODING='utf-8'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info['training_pid']=child.pid;save(state,info)
    def stop(signum=None,frame=None):
        if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        timed_out=False
        try:code=child.wait(timeout=MAX_FORMAL_SECONDS)  # 相对12小时监督独立于本机，不放宽训练评估预算。
        except subprocess.TimeoutExpired:
            stop()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124;timed_out=True
        info.update(exit_code=code,time=now())
        if code!=0:
            info.update(status='time_budget_exhausted' if timed_out else 'failed');save(state,info)
        else:
            info.update(status='cpu_verification');save(state,info)
            info['verification']=verify();info.update(status='verified',time=now());save(state,info)
        archive=ROOT/'scripts/experiment_summaries/VQGAN归一化特征差-20261104';archive.mkdir(parents=True,exist_ok=False)
        save(archive/'完成或失败.json',info)
        subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','归档VQGAN归一化特征差结果','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        (STATE/'result_commit.txt').write_text(commit+'\n',encoding='utf-8');info.update(result_commit=commit);save(state,info)
        if code!=0:return code  # 终态证据归档后保留原取消/失败分类。
        return 0
    except BaseException as exc:
        stop();info.update(error=repr(exc),time=now())
        if info.get('exit_code')==0:info['status']='archive_failed'
        save(state,info);raise
    finally:log.close()

if __name__=='__main__':raise SystemExit(main())
