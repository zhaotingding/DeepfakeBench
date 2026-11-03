"""一次获批的BM3D目标补评估：8小时／22:00截止，保留原中断。"""
from pathlib import Path
import datetime,importlib.util,json,os,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.state/BM3D目标补评估-20261103'
RUN='4090-BM3D低复杂度重构-20261103-种子3407'
CUTOFF=datetime.datetime.fromisoformat('2026-11-03T22:00:00+08:00')
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def stop(child):
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def verify():
    spec=importlib.util.spec_from_file_location('bm3d_recovery_verifier',ROOT/'scripts/CodeFormer独立验收模块.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.RUN=RUN;module.STATE=STATE;report=module.verify()
    report.update(evaluation_recovery=True,evaluation_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),new_training_updates=0,scope='既有12轮模型补齐目标评估；源选和末轮相同权重经hash和源预测复现验证，复用唯一模型预测；目标仍为开发域。')
    save(STATE/'verification.json',report);return report
def main():
    claim=sys.argv[1];assert len(claim)==40
    assert not (STATE/'supervisor.json').exists() and not (STATE/'status.json').exists()
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    check=json.loads((STATE/'preflight.json').read_text(encoding='utf-8'));assert check['code_commit']==code and check['budget_pass']
    if check['projected_seconds']+600>(CUTOFF-datetime.datetime.now(datetime.timezone.utc)).total_seconds():raise RuntimeError('剩余供电窗口不足')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-83850660-4196-4569-27bc-a3f9e72ada01',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LD_LIBRARY_PATH=str(ROOT/'.state/bm3d-20261029/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    log=(STATE/'run.log').open('x',encoding='utf-8');child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/BM3D目标补评估.py'),'run','--claim-commit',claim],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(status='running',time=now(),pid=os.getpid(),computation_pid=child.pid,cwd=str(ROOT),claim_commit=claim,code_commit=code,run_id=RUN,new_training_updates=0,max_seconds=28800,power_cutoff=CUTOFF.isoformat())
    save(STATE/'supervisor.json',info);started=time.monotonic();reason=None;low_samples=0
    def interrupted(sig,frame):
        nonlocal reason
        reason='supervisor_signal_'+str(sig)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    while child.poll() is None:
        if datetime.datetime.now(datetime.timezone.utc)>=CUTOFF:reason='cancelled_before_power_outage'
        elif time.monotonic()-started>=28800:reason='evaluation_time_budget_exhausted'
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        low_samples=low_samples+1 if available<2*1024**3 else 0
        if low_samples>=2:reason='host_memory_guard'  # 连续两次低于2GiB，只停止自己的计算。
        if reason:stop(child);break
        time.sleep(10)
    exit_code=child.wait();log.close();info.update(exit_code=exit_code,time=now(),stop_reason=reason,status='failed' if exit_code or reason else 'cpu_verification');save(STATE/'supervisor.json',info)
    if exit_code==0 and reason is None:
        try:info.update(verification=verify(),status='verified',time=now())
        except Exception as exc:info.update(status='archive_failed',error=repr(exc))
    save(STATE/'supervisor.json',info)
    archive=ROOT/'scripts/experiment_summaries/BM3D目标补评估-20261103';archive.mkdir(parents=True,exist_ok=False);save(archive/'完成或中断.json',info)
    try:
        rel=str(archive.relative_to(ROOT));subprocess.run(['git','add','--',rel],cwd=ROOT,check=True);subprocess.run(['git','commit','--only','-m','归档BM3D目标补评估终态','--',rel],cwd=ROOT,check=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();info['result_commit']=commit;(STATE/'result_commit.txt').write_text(commit+'\n',encoding='utf-8')
    except Exception as exc:info.update(status='archive_failed',error=repr(exc))
    save(STATE/'supervisor.json',info);os.sync();return 0 if info['status']=='verified' else 1
if __name__=='__main__':raise SystemExit(main())
