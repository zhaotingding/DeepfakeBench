"""只监督本次训练阶段；完整源测试和目标评估等待后续供电授权。"""
import datetime,hashlib,importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'.state/VQGAN绝对残差训练-20261103'
RUN='4090-VQGAN绝对残差-20261103-种子3407'
GPU='GPU-c3093e18-2d2f-8eb8-be80-f5802f43bdcc'
CUTOFF=datetime.datetime.fromisoformat('2026-11-03T22:00:00+08:00')
INPUTS=Path.home()/'prd-inputs/重构算子首批-20261029'

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()

def verify_training(code_commit):
    import numpy as np  # 独立验收仅使用CPU，不加载检测器或恢复器。
    spec=importlib.util.spec_from_file_location('absolute_independent_metrics',ROOT/'scripts/绝对残差独立指标.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    machine=load(ROOT/'machine.local.json');out=Path(machine['output_root'])/'r01'/RUN
    meta=load(out/'metadata.json');result=load(out/'result.json');curve=load(out/'learning_curve.json')
    if meta['code_commit']!=code_commit:raise ValueError('训练代码身份不匹配')
    if meta.get('config',{}).get('residual')!='absolute_diff' or meta.get('protocol',{}).get('residual')!='absolute_diff':raise ValueError('配置或冻结协议不是绝对残差')
    if result.get('status')!='completed' or result.get('optimizer_updates')!=21576:raise ValueError('训练未完整结束')
    if meta.get('protocol',{}).get('epochs')!=12 or meta['config']['train_batch_size']!=16:raise ValueError('训练预算或batch改变')
    if result.get('endpoints')!={'source_best':[],'terminal':[]}:raise ValueError('本阶段意外包含完整评估或端点身份错误')
    if (out/'complete_result.json').exists():raise ValueError('训练阶段不能冒充完整评估结果')
    for name,canonical,count in [('train','训练清单',28768),('val','源验证清单',700)]:
        manifest=load(out/(name+'_manifest.json'))
        if manifest!=load(INPUTS/(canonical+'.json')) or len(manifest['rows'])!=count:raise ValueError('源清单变化：'+name)
    rows=load(out/'val_manifest.json')['rows']
    if len(curve)!=12 or [entry['epoch'] for entry in curve]!=list(range(1,13)):raise ValueError('12轮源验证不齐')
    scores=[];max_error=0.0
    for epoch in range(1,13):
        logits=np.load(out/f'source_val_epoch{epoch}.npy',allow_pickle=False)
        if logits.shape!=(700,2) or logits.dtype!=np.float32 or not np.isfinite(logits).all():raise ValueError('源验证预测不完整')
        measured=module.independent_metrics(logits,rows);expected=curve[epoch-1]['source_val']
        if set(measured['domains'])!=set(expected['domains']):raise ValueError('源验证域身份不符')
        for key in ['auc','ce','balanced_ce']:
            errors=[abs(measured['macro'][key]-expected['macro'][key])]+[abs(measured['domains'][domain][key]-expected['domains'][domain][key]) for domain in measured['domains']]
            max_error=max(max_error,*errors)
            if max(errors)>1e-10:raise ValueError('独立源验证指标不一致')
        scores.append(measured['macro']['balanced_ce'])
    selected=int(np.argmin(scores))+1  # 最小源CE，平局保持最早轮次。
    if result['source_best_epoch']!=selected:raise ValueError('源选权重轮次错误')
    trace=[json.loads(line) for line in (out/'batch_trace.jsonl').read_text(encoding='utf-8').splitlines()]
    if [entry['step'] for entry in trace]!=list(range(1,21577)):raise ValueError('完整更新trace不齐')
    for epoch in range(1,13):
        indices=[index for entry in trace if entry['epoch']==epoch for index in entry['indices']]
        if sorted(indices)!=list(range(28768)):raise ValueError('训练帧缺失或重复')
    reference=out.parent/'4090-小波重构-20261029-种子3407';reference_meta=load(reference/'metadata.json')
    if meta['trainable_initial_sha256']!=reference_meta['trainable_initial_sha256'] or sha(out/'batch_trace.jsonl')!=sha(reference/'batch_trace.jsonl'):raise ValueError('共同初值或完整增强trace未配对')
    weights={}
    for endpoint in ['source_best','terminal']:
        path=out/(endpoint+'.pth')
        if not path.is_file() or path.stat().st_size==0:raise ValueError('缺少权重：'+endpoint)
        weights[endpoint]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
    artifacts={path.name:sha(path) for path in out.iterdir() if path.is_file() and path.suffix in {'.json','.jsonl','.npy'}}
    report=dict(status='training_verified_pending_evaluation',time=now(),run_id=RUN,output=str(out),code_commit=code_commit,
        epochs=12,optimizer_updates=21576,source_best_epoch=selected,weights=weights,artifacts_sha256=artifacts,
        trainable_initial_sha256=meta['trainable_initial_sha256'],trace_sha256=sha(out/'batch_trace.jsonl'),pairing_verified=True,
        source_validation_independent_max_error=max_error,complete_evaluation=False,source_test_pending=True,target_evaluation_pending=True,
        scope='仅完整训练和源验证验收；尚无源测试及三目标完整帧/视频指标，不能宣称实验全部完成。')
    save(STATE/'training_verification.json',report)
    return report

def main():
    claim=sys.argv[1]
    if len(claim)!=40:raise ValueError('缺少本地认领SHA')
    if datetime.datetime.now(datetime.timezone.utc)>=CUTOFF:raise RuntimeError('已过22:00，不启动')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('代码未提交')
    code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    state=STATE/'supervisor.json'
    if state.exists():raise FileExistsError('已有监督记录，不自动重试')
    STATE.mkdir(parents=True,exist_ok=True)
    summary=dict(status='starting',time=now(),pid=os.getpid(),cwd=str(ROOT),run_id=RUN,claim_commit=claim,
        code_commit=code_commit,gpu_uuid=GPU,power_cutoff=CUTOFF.isoformat(),max_wall_seconds=43200,complete_evaluation=False)
    save(state,summary)
    log=(STATE/'run.log').open('x',encoding='utf-8')
    child=None;stop_reason=None;started=time.monotonic()
    def requested_stop(signum,frame):
        nonlocal stop_reason
        stop_reason='cancelled_by_user'
    signal.signal(signal.SIGTERM,requested_stop);signal.signal(signal.SIGINT,requested_stop)
    def stop_own():
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    try:
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/运行VQGAN绝对残差训练.py'),claim],cwd=ROOT,
            env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        summary.update(status='running',time=now(),training_pid=child.pid);save(state,summary)
        while child.poll() is None:
            if datetime.datetime.now(datetime.timezone.utc)>=CUTOFF:stop_reason='cancelled_before_power_outage'
            elif time.monotonic()-started>=43200:stop_reason='training_budget_exhausted'
            if stop_reason:stop_own();break
            try:child.wait(timeout=min(10,max(0.05,(CUTOFF-datetime.datetime.now(datetime.timezone.utc)).total_seconds())))
            except subprocess.TimeoutExpired:pass
        exit_code=child.wait();summary.update(exit_code=exit_code,time=now())
        if exit_code==0 and stop_reason is None:
            summary.update(status='cpu_training_verification');save(state,summary)
            try:
                summary['verification']=verify_training(code_commit)
                summary.update(status='training_verified_pending_evaluation',time=now())
            except Exception as error:summary.update(status='archive_failed',error=repr(error),time=now())
        else:
            summary.update(status=stop_reason or 'failed',reason='供电取消不作为算法失败' if stop_reason=='cancelled_before_power_outage' else '仅训练阶段终止，保留现有产物')
        save(state,summary)
        archive=ROOT/'scripts/experiment_summaries/VQGAN绝对残差训练-20261103';archive.mkdir(parents=True,exist_ok=False)
        save(archive/'训练阶段结果或中断.json',summary)
        relative=str(archive.relative_to(ROOT))
        subprocess.run(['git','add','--',relative],cwd=ROOT,check=True)
        subprocess.run(['git','commit','--only','-m','归档VQGAN绝对残差训练阶段','--',relative],cwd=ROOT,check=True)
        result_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        summary['result_commit']=result_commit;(STATE/'result_commit.txt').write_text(result_commit+'\n',encoding='utf-8');save(state,summary)
        os.sync()
        return 0 if summary['status']=='training_verified_pending_evaluation' else 1
    except BaseException as error:
        stop_own();summary.update(status='archive_failed' if summary.get('exit_code')==0 else 'failed',error=repr(error),time=now());save(state,summary);raise
    finally:log.close()

if __name__=='__main__':raise SystemExit(main())
