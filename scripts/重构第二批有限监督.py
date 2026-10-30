"""第二批固定双卡有限队列：GPU0 MAE；GPU1 TV后学习压缩。"""
import datetime,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.state/重构第二批-20261030'
LABELS={'mae':'MAE遮挡重构','tv':'TV变分重构','compression':'学习压缩重构'}
LANES={'GPU-83850660-4196-4569-27bc-a3f9e72ada01':['mae'],'GPU-c3093e18-2d2f-8eb8-be80-f5802f43bdcc':['tv','compression']}
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2));tmp.replace(p)
def stop(child):
    if child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)  # 仅停止本监督创建的超时进程组。
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
def verify(arm):
    spec=importlib.util.spec_from_file_location('verified_archive',ROOT/'scripts/VQGAN有限监督.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.STATE=STATE/arm;module.RUN='4090-'+LABELS[arm]+'-20261030-种子3407'
    report=module.verify()
    report['scope']='本批原图/VAE匹配控制已齐；目标域反复开发；比较源选主端点和固定末轮，不选目标赢家。'
    save(STATE/arm/'verification.json',report)
    return report
def main():
    claim=sys.argv[1];assert len(claim)==40
    if (STATE/'supervisor.json').exists():raise FileExistsError('已有正式监督，不自动重试')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('代码未提交')
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    summary=dict(status='running',time=now(),pid=os.getpid(),cwd=str(ROOT),code_commit=code_commit,claim_commit=claim,arms={},queues=LANES)
    for arm in LABELS:
        smoke=json.loads((STATE/arm/'smoke/验收.json').read_text())
        if smoke['code_commit']!=code_commit or not smoke['budget_pass']:raise ValueError('预检版本/预算不符：'+arm)
        if (STATE/arm/'status.json').exists():raise FileExistsError('已有正式状态：'+arm)
        summary['arms'][arm]=dict(status='queued',time=now())
    running={};queues={gpu:list(items) for gpu,items in LANES.items()}
    def interrupted(sig,frame):
        for entry in running.values():stop(entry['child'])
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def launch(gpu,arm):
        log=(STATE/arm/'run.log').open('x')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/运行重构第二批.py'),arm,'run','--claim-commit',claim],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        running[gpu]=dict(arm=arm,child=child,log=log,started=time.monotonic())
        summary['arms'][arm].update(status='running',time=now(),supervisor_pid=os.getpid(),training_pid=child.pid,gpu_uuid=gpu,cwd=str(ROOT),claim_commit=claim)
        save(STATE/arm/'supervisor.json',summary['arms'][arm]);save(STATE/'supervisor.json',summary)
    while running or any(queues.values()):
        for gpu in queues:
            if gpu not in running and queues[gpu]:launch(gpu,queues[gpu].pop(0))
        for gpu,entry in list(running.items()):
            arm=entry['arm'];child=entry['child'];exit_code=child.poll();status_path=STATE/arm/'status.json'
            if exit_code is None:
                status=json.loads(status_path.read_text()) if status_path.exists() else {}
                start=status.get('formal_started')
                formal_seconds=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(start)).total_seconds() if start else 0
                elapsed=time.monotonic()-entry['started']
                if formal_seconds>43200 or elapsed>88200 or (not start and elapsed>43200):stop(child);exit_code=124
            if exit_code is None:continue
            entry['log'].close();running.pop(gpu);info=summary['arms'][arm]
            info.update(exit_code=exit_code,status='failed' if exit_code else 'cpu_verification',time=now());save(STATE/arm/'supervisor.json',info)
            if exit_code==0:
                try:info.update(verification=verify(arm),status='verified',time=now())
                except Exception as exc:info.update(status='archive_failed',error=repr(exc))
            save(STATE/arm/'supervisor.json',info)
            if info['status']!='verified':
                for waiting in queues[gpu]:summary['arms'][waiting].update(status='blocked_by_lane_failure',time=now(),reason=arm)
                queues[gpu].clear()  # 一条队列失败只停止本槽后续，不伤及健康的另一卡。
            save(STATE/'supervisor.json',summary)
        if running:time.sleep(15)
    archive=ROOT/'scripts/experiment_summaries/重构第二批-20261030';archive.mkdir(parents=True,exist_ok=False)
    summary.update(status='verified' if all(x['status']=='verified' for x in summary['arms'].values()) else 'finished_with_errors',time=now())
    for arm,info in summary['arms'].items():save(archive/(LABELS[arm]+'.json'),info)
    save(archive/'批次.json',summary)
    try:
        relative=str(archive.relative_to(ROOT));subprocess.run(['git','add','--',relative],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','归档MAE学习压缩与TV重构批次','--',relative],cwd=ROOT,check=True)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        summary['result_commit']=commit;(STATE/'result_commit.txt').write_text(commit+'\n')
        for arm,info in summary['arms'].items():
            info['result_commit']=commit;(STATE/arm/'result_commit.txt').write_text(commit+'\n');save(STATE/arm/'supervisor.json',info)
    except Exception as exc:summary.update(status='archive_failed',error=repr(exc))
    save(STATE/'supervisor.json',summary)
    return 0 if summary['status']=='verified' else 1
if __name__=='__main__':raise SystemExit(main())
