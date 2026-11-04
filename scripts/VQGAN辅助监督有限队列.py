"""GPU1按原图辅助、残差辅助顺序运行，两臂验收后统一提交结果。"""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.state/VQGAN辅助监督-20261104'
INPUTS = Path.home() / 'prd-inputs/重构算子首批-20261029'
ARMS = {'original_aux': '原图辅助监督', 'residual_aux': '残差辅助监督'}
GPU = 'GPU-c3093e18-2d2f-8eb8-be80-f5802f43bdcc'
LIMIT = 12 * 3600
TRACE_HASH = '6041ac84270ca91720d7e45f152f053466bed22d40091ad9f388488fc63583ea'
MAIN_INITIAL = '9aa889456c6661e4c09eb03229689f17e2f9766a4e8d09d4c009badf83f133e5'


def now():return datetime.datetime.now().astimezone().isoformat()
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def load(path):return Path(path).read_text(encoding='utf-8')
def report(path):return json.loads(load(path))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def compare_metrics(actual,expected):
    keys=('auc','ce','balanced_ce')
    if set(actual['domains'])!=set(expected['domains']):raise ValueError('指标域不符')
    errors=[abs(actual['macro'][k]-expected['macro'][k]) for k in keys]
    errors.extend(abs(actual['domains'][d][k]-expected['domains'][d][k]) for d in actual['domains'] for k in keys)
    if not all(0<=error<=1e-10 for error in errors):raise ValueError('独立视频指标复算不符')
    return max(errors)


def verify_arm(arm,code,claim,smoke):
    import numpy as np
    from 绝对残差独立指标 import independent_metrics  # 此纯CPU模块无需已结束实验的配置。
    s=STATE/arm;name=f'4090-VQGAN{ARMS[arm]}-20261104-种子3407';out=Path(report(ROOT/'r01.machine.json')['output_root'])/'r01'/name
    meta=report(out/'metadata.json');result=report(out/'complete_result.json');received=report(s/'status.json')
    if received['stage']!='compute_finished' or received['result']!=result or meta['code_commit']!=code or received['claim_commit']!=claim or meta['run_id']!=name or received['run_id']!=name:
        raise ValueError('计算退出/代码/认领不符')
    for obj in (meta['config'],meta['protocol'],result):
        if obj.get('auxiliary_input')!=arm or obj.get('auxiliary_lambda')!=1.0 or obj.get('residual')!='signed_diff':raise ValueError('训练臂/损失系数/残差定义错误')
    if not result['complete_evaluation'] or result['optimizer_updates']!=21576 or meta['protocol']['epochs']!=12 or result['elapsed_seconds']>LIMIT:raise ValueError('训练/完整评估未满足预算')
    ident=smoke['arms'][arm]['identity']
    for key,value in ident.items():
        if meta.get(key)!=value:raise ValueError('metadata初值或参数量不同：'+key)
    if meta['main_trainable_initial_sha256']!=MAIN_INITIAL or meta['auxiliary_parameters']!=2050 or meta['trainable_parameters']!=5250052:
        raise ValueError('主路径或辅助容量不符')
    manifests=[]
    for filename,canonical,expected_count in [('train_manifest.json','训练清单.json',28768),('val_manifest.json','源验证清单.json',700),('eval0_manifest.json','源测试清单.json',5600),('eval1_manifest.json','目标开发清单.json',165758)]:
        m=report(out/filename)
        if len(m['rows'])!=expected_count or m!=report(INPUTS/canonical):raise ValueError('清单不完整：'+filename)
        manifests.append(m)
    if len({(r['domain'],r['video_id']) for r in manifests[-1]['rows']})!=5876:raise ValueError('目标视频数不符')
    curve=report(out/'learning_curve.json');scores=[];errors=[]
    if [entry['epoch'] for entry in curve]!=list(range(1,13)):raise ValueError('源验证轮次不完整')
    for epoch,entry in enumerate(curve,start=1):
        logits=np.load(out/f'source_val_epoch{epoch}.npy',allow_pickle=False)
        if logits.shape!=(700,2) or logits.dtype!=np.float32 or not np.isfinite(logits).all():raise ValueError('源验证预测错误')
        values=independent_metrics(logits,manifests[1]['rows']);errors.append(compare_metrics(values,entry['source_val']));scores.append(values['macro']['balanced_ce'])
    if int(np.argmin(scores))+1!=result['source_best_epoch']:raise ValueError('源验证选择错误')
    for ep in ('source_best','terminal'):
        if [e['role'] for e in result['endpoints'][ep]]!=['source_test','target_dev']:raise ValueError('端点评估集错误')
        for idx,m in enumerate(manifests[2:]):
            values=np.load(out/f'{ep}_eval{idx}.npy',allow_pickle=False)
            if values.shape!=(len(m['rows']),2) or values.dtype!=np.float32 or not np.isfinite(values).all():raise ValueError('完整端点预测缺失')
            errors.append(compare_metrics(independent_metrics(values,m['rows']),result['endpoints'][ep][idx]['metrics']))
        if (out/(ep+'.pth')).stat().st_size<=0:raise ValueError('端点权重为空')
    trace=out/'batch_trace.jsonl'
    if sha(trace)!=TRACE_HASH or sum(1 for _ in trace.open())!=21576:raise ValueError('与已验收VQGAN的增强/初始RNG轨迹不匹配')
    artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in {'.pth','.npy','.json','.jsonl'}}
    checked=dict(status='verified',time=now(),output=str(out),code_commit=code,claim_commit=claim,arm=arm,updates=21576,
                 independent_metrics_max_error=max(errors),pairing_verified=True,identity=ident,artifacts_sha256=artifacts,result=result,
                 scope='同协议主头部署；与无辅助VQGAN和另一辅助臂作配对开发评估。')
    save(s/'verification.json',checked);return checked


def stop_own(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    if len(sys.argv)!=2 or len(sys.argv[1])!=40:raise ValueError('缺正式认领SHA')
    claim=sys.argv[1];code=git('rev-parse','HEAD');smoke=report(STATE/'smoke/验收.json')
    if git('status','--porcelain') or smoke['status']!='passed' or smoke['code_commit']!=code:raise ValueError('代码/预检版本不符')
    if (STATE/'supervisor.json').exists():raise FileExistsError('已有有限队列，不自动重试')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    summary=dict(status='starting',time=now(),pid=os.getpid(),cwd=str(ROOT),gpu=GPU,code_commit=code,claim_commit=claim,arms={arm:{'status':'queued'} for arm in ARMS})
    interrupted=None
    def mark(sig,frame):
        nonlocal interrupted
        interrupted='supervisor_signal_'+str(sig)
    signal.signal(signal.SIGTERM,mark);signal.signal(signal.SIGINT,mark)
    save(STATE/'supervisor.json',summary);child=None
    try:
        for arm in ARMS:
            if interrupted:break
            if git('rev-parse','HEAD')!=code or git('status','--porcelain'):raise RuntimeError('臂间冻结代码改变')
            s=STATE/arm;s.mkdir(parents=True,exist_ok=True)
            with (s/'run.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/运行VQGAN辅助监督.py'),'run','--arm',arm,'--claim-commit',claim],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                started=time.monotonic();a=summary['arms'][arm];a.update(status='running',time=now(),training_pid=child.pid);summary.update(status='running',active_arm=arm,time=now());save(STATE/'supervisor.json',summary)
                reason=None
                while child.poll() is None:
                    reason=interrupted or ('arm_time_budget_exhausted' if time.monotonic()-started>=LIMIT else None)
                    if reason:stop_own(child);break
                    try:child.wait(timeout=min(5,max(.05,LIMIT-(time.monotonic()-started))))
                    except subprocess.TimeoutExpired:pass
                rc=child.wait();a.update(exit_code=rc,stop_reason=reason,time=now());child=None
            if rc or reason:
                a['status']='failed';summary.update(status='failed',time=now());break
            a['status']='cpu_verification';save(STATE/'supervisor.json',summary)
            try:
                verified=verify_arm(arm,code,claim,smoke);a.update(status='verified',time=now(),verification=verified)
            except Exception as exc:
                a.update(status='archive_failed',error=repr(exc),time=now());summary.update(status='archive_failed',time=now());break
            save(STATE/'supervisor.json',summary)
        for a in summary['arms'].values():
            if a['status']=='queued':a['status']='not_started_after_stop'
        accepted=all(a['status']=='verified' for a in summary['arms'].values())
        summary.update(status='verified' if accepted else (summary['status'] if summary['status'] in ('archive_failed','failed') else 'cancelled'),active_arm=None,time=now())
        save(STATE/'supervisor.json',summary)
        archive=ROOT/'scripts/experiment_summaries/VQGAN辅助监督-20261104';archive.mkdir(parents=True,exist_ok=False)
        for arm,a in summary['arms'].items():save(archive/(arm+'.json'),a)
        save(archive/'批次.json',summary);relative=str(archive.relative_to(ROOT))
        subprocess.run(['git','add','--',relative],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','归档VQGAN辅助监督有限组','--',relative],cwd=ROOT,check=True)
        commit=git('rev-parse','HEAD');summary['result_commit']=commit;(STATE/'result_commit.txt').write_text(commit+'\n',encoding='utf-8');save(STATE/'supervisor.json',summary);os.sync()
        return 0 if accepted else 1
    except BaseException as exc:
        stop_own(child);summary.update(status='archive_failed' if any(a.get('exit_code')==0 for a in summary['arms'].values()) else 'failed',error=repr(exc),time=now());save(STATE/'supervisor.json',summary);raise


if __name__=='__main__':
    with (STATE/'queue.lock').open('a+') as queue_lock:
        fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)  # 同一组只接受一个监督者。
        raise SystemExit(main())
