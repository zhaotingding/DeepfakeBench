#!/usr/bin/env python3
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
from pathlib import Path
import datetime,fcntl,importlib.util,json,sys,time,subprocess
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('absolute_runner',ROOT/'scripts/运行VQGAN绝对残差.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)  # 复用已数值验收的模型与缓存接口。
STATE=ROOT/'.state/VQGAN绝对残差训练-20261103'
CUTOFF=datetime.datetime.fromisoformat('2026-11-03T22:00:00+08:00')
PRECHECK=ROOT/'.state/VQGAN绝对残差-20261103/smoke/验收.json'
def readiness():
    check=json.loads(PRECHECK.read_text(encoding='utf-8'))  # 原预检数值通过，完整评估窗口阻塞原样保留。
    if check['status']!='numeric_passed_budget_blocked' or check['updates']!=8 or check['formal_updates']!=0:raise ValueError('预检身份异常')
    if check['formula_max_abs']>1e-6 or check['reload_max_abs']>1e-6:raise ValueError('数值预检不通过')
    unchanged=['training','scripts/运行VQGAN绝对残差.py','scripts/绝对残差独立指标.py']
    subprocess.run(['git','diff','--quiet',check['code_commit'],'--',*unchanged],cwd=ROOT,check=True)  # 新阶段不改数值计算代码。
    needed=(21576*check['train_step_median']+12*700/64*check['dual_eval64_seconds'])*1.2+600  # 源验证按双模型耗时保守计入。
    available=(CUTOFF-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    return dict(status='passed' if needed+60<available else 'window_blocked',time=base.now(),code_commit=base.git('rev-parse','HEAD'),numeric_precheck_commit=check['code_commit'],training_projection_seconds=needed,cache_verification_allowance_seconds=60,remaining_seconds=available,power_cutoff=CUTOFF.isoformat(),required_free_bytes=check['required_free_bytes'],training_only=True,future_evaluation='仅下一供电窗口确认后执行；0额外训练')
def main():
    if len(sys.argv)!=2 or len(sys.argv[1])!=40:raise ValueError('缺少训练阶段正式认领SHA')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-c3093e18-2d2f-8eb8-be80-f5802f43bdcc':raise ValueError('仅绑定物理GPU1')
    if base.git('status','--porcelain'):raise ValueError('代码未提交')
    lock=Path.home()/'.cache/prd-locks/重构算子探索-gpu1.lock'
    handle=lock.open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)  # 保持直到本进程完成。
    handle.seek(0);handle.truncate();handle.write(str(os.getpid()));handle.flush()
    STATE.mkdir(exist_ok=True)
    if (STATE/'status.json').exists():raise FileExistsError('已有正式状态，禁止重复启动')
    check=readiness();base.save(STATE/'stage-readiness.json',check)
    if check['status']!='passed':raise RuntimeError('训练阶段也不能落入供电窗口')
    torch.set_num_threads(1);device=torch.device('cuda:0');base.seed_all(3407)
    if torch.cuda.mem_get_info(device)[0]<check['required_free_bytes']:raise RuntimeError('显存未达到实测余量')
    c,paths,items=base.settings();out=Path(c['output_root'])/'r01'/base.RUN
    if out.exists():raise FileExistsError('正式输出已存在，不覆盖或继续旧运行')
    status=dict(stage='cache_verification',time=base.now(),pid=os.getpid(),cwd=str(ROOT),claim_commit=sys.argv[1],code_commit=base.git('rev-parse','HEAD'),run_id=base.RUN,residual='absolute_diff',training_only=True)
    base.save(STATE/'status.json',status)
    try:
        t=time.monotonic();values,cache,slots=base.read_cache(base.FULL_CACHE,items[0]['rows']+items[1]['rows'],c,base.vqgan_spec(c),verify_file=True);del values
        expected=json.loads(PRECHECK.read_text(encoding='utf-8'))['full_cache_audit']
        if cache['tensor_sha256']!=expected['tensor_sha256'] or len(slots)!=29468:raise ValueError('缓存与预检不一致')
        reuse=time.monotonic()-t;check=readiness()
        if check['status']!='passed':raise RuntimeError('缓存复核后训练窗口不足')
        if torch.cuda.mem_get_info(device)[0]<check['required_free_bytes']:raise RuntimeError('启动前显存不足')
        protocol=dict(status='approved',id='VQGAN绝对残差训练-20261103',epochs=12,seed=3407,max_wall_seconds=min(43200,(CUTOFF-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-600),train_manifest=str(paths[0]),val_manifest=str(paths[1]),evaluation_manifests=[],checkpoint_rule='source_val_video_balanced_ce',reconstruction_mode='cached_paired',augmentation_protocol='paired_float32_v1',reconstruction_caches={'vqgan':str(base.FULL_CACHE)},residual='absolute_diff',power_cutoff=CUTOFF.isoformat(),execution_stage='training_only',pending_evaluation_manifests=[str(p) for p in paths[2:]],authorization='用户批准分阶段：今天完整12轮并保存源选末轮；下次确认供电后仅完整评估，0新增训练。')
        base.save(STATE/'protocol.json',protocol);status.update(stage='training',time=base.now(),formal_started=base.now(),cache_verification_seconds=reuse);base.save(STATE/'status.json',status)
        result=base.run_training(c,protocol,base.RUN,device)  # 保留逐轮源验证；本阶段不调用源测试或目标评估。
        if result['optimizer_updates']!=21576 or not all((out/n).exists() for n in ['source_best.pth','terminal.pth']):raise ValueError('训练更新或两个权重不完整')
        receipt=dict(status='training_complete_pending_evaluation',time=base.now(),result=result,code_commit=base.git('rev-parse','HEAD'),cache_reuse_verification_seconds=reuse,historical_cache_seconds=cache['generation_seconds'],new_cache_generation_seconds=0,evaluation_complete=False,new_target_predictions=0)
        base.save(out/'training_stage_complete.json',receipt);status.update(stage='training_complete_pending_evaluation',time=base.now(),result=result);base.save(STATE/'status.json',status)
    except BaseException as exc:
        status.update(stage='failed',time=base.now(),error=repr(exc));base.save(STATE/'status.json',status);raise
if __name__=='__main__':main()
