#!/usr/bin/env python3
"""绝对残差仅评估有限监督：8小时、0重试、独立CPU验收和结果归档。"""
from pathlib import Path
import datetime,hashlib,importlib.util,json,os,signal,subprocess,sys,time
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.state/VQGAN绝对残差评估-20261104'
RUN='4090-VQGAN绝对残差-20261103-种子3407'
GPU='GPU-c3093e18-2d2f-8eb8-be80-f5802f43bdcc'
MAX_SECONDS=28800
WEIGHTS={'source_best':'042d3930c09df40eb7089210bc65a4adc2aa39a27205ea152fa48f017c7daf53','terminal':'d2e7a003ad40f994982730192a7f74fed33cde1ad547014400c5dd56567e10f9'}
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
def git(*a):return subprocess.check_output(['git','-C',str(ROOT),*a],text=True).strip()

def verify():
    # 原完整验收模块只调用独立CPU指标，不执行旧入口中的历史截止逻辑。
    spec=importlib.util.spec_from_file_location('absolute_complete_verifier',ROOT/'scripts/VQGAN绝对残差有限监督.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.STATE=STATE
    report=module.verify();out=Path(report['output']);result=read(out/'complete_result.json');old=read(ROOT/'.state/VQGAN绝对残差训练-20261103/training_verification.json')
    if report['code_commit']!='8ff4ce93c10dc21221dfbd1d0b510c48c0d2f366' or result['evaluation_code_commit']!=git('rev-parse','HEAD'):raise ValueError('训练/评估代码身份不符')
    if result['evaluation_training_updates']!=0 or result['source_best_epoch']!=8:raise ValueError('新训练或选点发生改变')
    if report['updates']!=21576 or not report['pairing_verified']:raise ValueError('原训练配对未通过')
    for ep,digest in WEIGHTS.items():
        if sha(out/(ep+'.pth'))!=digest:raise ValueError('权重hash不符：'+ep)
    for name,digest in old['artifacts_sha256'].items():
        if sha(out/name)!=digest:raise ValueError('原训练产物被改写：'+name)
    manifest=read(out/'eval1_manifest.json');videos={(r['domain'],r['video_id']) for r in manifest['rows']}
    if len(videos)!=5876:raise ValueError('目标视频不完整')
    report.update(evaluation_code_commit=result['evaluation_code_commit'],evaluation_training_updates=0,training_result_commit='48dd05410da84593a6d95732240eed21b0682900',original_training_artifacts_unchanged=True,scope='既有绝对残差第8轮/第12轮两权重完整评估；0新训练，目标仍为开发域。')
    save(STATE/'verification.json',report);return report

def stop(child):
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def main():
    claim=sys.argv[1]
    if len(claim)!=40:raise ValueError('缺少本地正式认领SHA')
    if git('status','--porcelain'):raise ValueError('运行代码未提交')
    code=git('rev-parse','HEAD');check=read(STATE/'preflight/验收.json')
    if check['status']!='passed' or check['code_commit']!=code or not check['budget_pass']:raise ValueError('预检状态/版本/预算不符')
    if (STATE/'supervisor.json').exists() or (STATE/'status.json').exists():raise FileExistsError('已有正式记录，不自动重试')
    log=(STATE/'run.log').open('x',encoding='utf-8');child=None;reason=None
    info=dict(status='starting',time=now(),pid=os.getpid(),cwd=str(ROOT),code_commit=code,claim_commit=claim,run_id=RUN,new_training_updates=0,max_seconds=MAX_SECONDS)
    def interrupted(sig,frame):
        nonlocal reason
        reason='supervisor_signal_'+str(sig)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/VQGAN绝对残差完整评估.py'),'run','--claim-commit',claim],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        info.update(status='running',computation_pid=child.pid);save(STATE/'supervisor.json',info);start=time.monotonic()
        while child.poll() is None:
            if time.monotonic()-start>=MAX_SECONDS:reason='evaluation_time_budget_exhausted'
            if reason:stop(child);break
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
        code_exit=child.wait();info.update(exit_code=code_exit,stop_reason=reason,time=now(),status='failed' if code_exit or reason else 'cpu_verification');save(STATE/'supervisor.json',info)
        if code_exit==0 and reason is None:
            try:info.update(verification=verify(),status='verified',time=now())
            except Exception as exc:info.update(status='archive_failed',error=repr(exc))
        save(STATE/'supervisor.json',info)
        archive=ROOT/'scripts/experiment_summaries/VQGAN绝对残差评估-20261104';archive.mkdir(parents=True,exist_ok=False);save(archive/'完整结果或中断.json',info)
        relative=str(archive.relative_to(ROOT));subprocess.run(['git','add','--',relative],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','归档VQGAN绝对残差完整评估','--',relative],cwd=ROOT,check=True)
        result_commit=git('rev-parse','HEAD');info['result_commit']=result_commit;(STATE/'result_commit.txt').write_text(result_commit+'\n',encoding='utf-8');save(STATE/'supervisor.json',info);os.sync()
        return 0 if info['status']=='verified' else 1
    except BaseException as exc:
        stop(child);info.update(status='archive_failed' if info.get('exit_code')==0 else 'failed',time=now(),error=repr(exc));save(STATE/'supervisor.json',info);raise
    finally:log.close()
if __name__=='__main__':raise SystemExit(main())
