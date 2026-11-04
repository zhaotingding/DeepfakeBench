"""09的两个独立训练臂。共用原有符号VQGAN的数据、主头及完整评估入口。"""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import torch

from 运行VQGAN单项 import settings as vqgan_settings, evaluate_two
from prd_r01.cache import build_cache, read_cache
from prd_r01.data import LMDBImages, make_loader
from prd_r01.engine import run_training, seed_all, training_loss, trainable_identity
from prd_r01.model import build_model
from prd_r01.vqgan_operator import VQGANProbe, vqgan_spec


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.state/VQGAN辅助监督-20261104'
GPU = 'GPU-c3093e18-2d2f-8eb8-be80-f5802f43bdcc'
LOCK = Path.home() / '.cache/prd-locks/重构算子探索-gpu1.lock'
LIMIT = 12 * 3600
ARMS = {'original_aux': '原图辅助监督', 'residual_aux': '残差辅助监督'}
RUN = {arm: f'4090-VQGAN{label}-20261104-种子3407' for arm, label in ARMS.items()}
CACHE = Path.home() / 'prd-runs/重构缓存-20261029/VQGAN码本-训练验证'
BASE_RUN = '4090-VQGAN码本重构-20261029-种子3407'
BASE_STATE = ROOT / '.state/vqgan-20261029'
BASE_CODE = '8feaf3995c8ba8335cbca9f522bf85fad3a1a41b'
BASE_RESULT = 'd932ecb5a3d31af206fe0118d62a543cd735409b'
MAIN_INITIAL = '9aa889456c6661e4c09eb03229689f17e2f9766a4e8d09d4c009badf83f133e5'
TRACE_HASH = '6041ac84270ca91720d7e45f152f053466bed22d40091ad9f388488fc63583ea'


def now():
    return datetime.datetime.now().astimezone().isoformat()


def read(path):
    return Path(path).read_text(encoding='utf-8')


def load(path):
    return json.loads(read(path))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def settings(arm):
    config, paths, manifests = vqgan_settings()  # 原VQGAN权重、清单及接口。
    config.update(residual='signed_diff', auxiliary_input=arm,
                  auxiliary_lambda=1.0, reconstruction_caches={'vqgan': str(CACHE)})
    return config, paths, manifests


def cache_audit(config, manifests):
    rows = manifests[0]['rows'] + manifests[1]['rows']
    values, meta, slots = read_cache(CACHE, rows, config, vqgan_spec(config), verify_file=True)
    del values
    if len(slots) != 29468:
        raise ValueError('训练/源验证缓存清单不完整')
    return dict(path=str(CACHE), frames=len(slots), tensor_sha256=meta['tensor_sha256'],
                spec=meta['spec'], generation_seconds=meta['generation_seconds'])


def baseline_audit(config, manifests, cache, device):
    out = Path(config['output_root']) / 'r01' / BASE_RUN
    meta, complete = load(out / 'metadata.json'), load(out / 'complete_result.json')
    if meta['code_commit'] != BASE_CODE or read(BASE_STATE / 'result_commit.txt').strip() != BASE_RESULT:
        raise ValueError('历史无辅助VQGAN身份不匹配')
    archived = json.loads(git('show', BASE_RESULT + ':scripts/experiment_summaries/重构VQGAN-20261029/完成或失败.json'))
    original_verification = load(BASE_STATE / 'verification.json')
    if archived.get('verification') != original_verification or archived['exit_code'] != 0 or not original_verification['pairing_verified']:
        raise ValueError('无辅助基线未匹配已验收结果Git')
    checked = original_verification['artifacts_sha256']
    for filename in ('metadata.json','complete_result.json','batch_trace.jsonl',f"source_val_epoch{complete['source_best_epoch']}.npy"):
        if sha(out / filename) != checked[filename]:raise ValueError('历史基线产物已改变：'+filename)
    keys = ('resolution', 'mean', 'std', 'lora', 'head', 'residual', 'train_batch_size', 'test_batch_size',
            'workers', 'optimizer', 'amp', 'reconstruction_mode', 'augmentation_protocol', 'backbone', 'responses', 'clip_path', 'vqgan_weights_sha256')
    for key in keys:
        if config.get(key) != meta['config'].get(key):
            raise ValueError('无辅助参照的数值配置不符：' + key)
    if meta['trainable_initial_sha256'] != MAIN_INITIAL or sha(out / 'batch_trace.jsonl') != TRACE_HASH:
        raise ValueError('无辅助参照的初值/训练轨迹不符')
    if meta['protocol']['epochs'] != 12 or meta['protocol']['seed'] != 3407 or meta['protocol']['checkpoint_rule'] != 'source_val_video_balanced_ce':
        raise ValueError('无辅助参照的更新或选点协议不符')
    old_cache = meta['cache_identities']['vqgan']
    if any(cache[key] != old_cache[key] for key in ('path', 'spec', 'tensor_sha256')):
        raise ValueError('无辅助参照缓存不匹配')
    for filename, manifest in zip(('train_manifest.json', 'val_manifest.json', 'eval0_manifest.json', 'eval1_manifest.json'), manifests):
        if load(out / filename) != manifest:
            raise ValueError('与无辅助参照清单不同：' + filename)
    if complete['optimizer_updates'] != 21576 or not complete['complete_evaluation']:
        raise ValueError('无辅助参照未完整验收')
    plain = dict(config, auxiliary_input='none')
    seed_all(3407)
    model = build_model(plain, device)
    if trainable_identity(model)['main_trainable_initial_sha256'] != MAIN_INITIAL:
        raise ValueError('当前无辅助主路径初值不一致')
    checkpoint = out / 'source_best.pth'
    if sha(checkpoint) != checked['source_best.pth']:
        raise ValueError('参照source_best权重身份改变')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    rows = manifests[1]['rows'][:64]
    loader = make_loader(rows, plain, False, 3407, device)
    images, _, indices, views = next(iter(loader)); loader.dataset.close()
    with torch.inference_mode():
        logits = model(images.to(device), {'vqgan': views['vqgan'].to(device)})['logits'].cpu().numpy()
    expected = np.load(out / f"source_val_epoch{complete['source_best_epoch']}.npy", allow_pickle=False)[indices.numpy()]
    replay_error = float(np.max(np.abs(logits - expected)))
    if not np.isfinite(replay_error) or replay_error > 1e-6:
        raise ValueError(f'无辅助参照主路径重放超差：{replay_error}')
    return dict(code_commit=BASE_CODE, result_commit=BASE_RESULT, cache_hash=cache['tensor_sha256'],
                main_initial_sha256=MAIN_INITIAL, trace_sha256=TRACE_HASH, replay_frames=64,
                source_replay_max_abs=replay_error, matched_config_keys=keys,
                selected_epoch=complete['source_best_epoch'], manifests_matched=True)


def smoke_arm(arm, config, manifests, small_cache, images, restored, restoration_seconds, device):
    area = STATE / 'smoke' / arm
    area.mkdir(parents=True, exist_ok=False)
    c = dict(config, auxiliary_input=arm, reconstruction_caches={'vqgan': str(small_cache)})
    seed_all(3407)
    model = build_model(c, device)
    identity = trainable_identity(model)
    if identity['main_trainable_initial_sha256'] != MAIN_INITIAL or identity['auxiliary_parameters'] != 2050:
        raise ValueError('原主初值或辅助头参数量错误')
    parameters = [p for p in model.parameters() if p.requires_grad]
    lora = [p for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad]
    opt = c['optimizer']
    optimizer = torch.optim.Adam(parameters, lr=opt['lr'], betas=tuple(opt['betas']), eps=opt['eps'], weight_decay=opt['weight_decay'])
    rows = manifests[0]['rows'][:64]
    loader = make_loader(rows, c, True, 3407, device)
    expected_trace = load(BASE_STATE / 'smoke/验收.json')['trace']
    trace, times, component_gradients, first = [], [], {}, {}
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for _ in range(2):
        for x, y, index, views in loader:
            x, y = x.to(device), y.to(device)
            views = {key: value.to(device) for key, value in views.items()}
            trace.append(dict(indices=index.tolist(), image_sha=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()))
            torch.cuda.synchronize(device); start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            output = model(x, views)
            total, main_ce, aux_ce = training_loss(output, y, arm, 1.0)
            torch.testing.assert_close(total, main_ce + aux_ce, rtol=0, atol=0)
            if len(trace) == 1:
                for label, component in [('main', main_ce), ('auxiliary', aux_ce)]:
                    grads = torch.autograd.grad(component, lora, retain_graph=True, allow_unused=True)
                    magnitude = sum(float(g.detach().abs().sum()) for g in grads if g is not None)
                    if not np.isfinite(magnitude) or magnitude <= 0:
                        raise ValueError('真实首批LoRA分量梯度无效：' + label)
                    component_gradients[label] = magnitude
                features = output['features'][:, :1024] if arm == 'original_aux' else output['features'][:, 1024:]
                torch.testing.assert_close(output['auxiliary_logits'], model.auxiliary_head(features), atol=1e-6, rtol=0)
                first = dict(main_logits_sha256=hashlib.sha256(output['logits'].detach().cpu().numpy().tobytes()).hexdigest(),
                             main_ce=float(main_ce.detach()), aux_ce=float(aux_ce.detach()), total=float(total.detach()))
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
            if not torch.isfinite(total) or float(norm) <= 0 or model.auxiliary_head.weight.grad is None or not bool(model.auxiliary_head.weight.grad.abs().max() > 0):
                raise ValueError('训练损失或辅助头梯度无效')
            optimizer.step();torch.cuda.synchronize(device);times.append(time.monotonic() - start)
    loader.dataset.close()
    if trace != expected_trace or len(trace) != 8:
        raise ValueError('与原VQGAN八个原图增强batch未配对')
    optimizer.zero_grad(set_to_none=True);del optimizer
    with tempfile.TemporaryDirectory(dir=area) as tmp:
        checkpoint = Path(tmp) / '重载.pth'
        torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, checkpoint)
        second = build_model(c, device)
        second.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval();second.eval();x = images.to(device);views = {'vqgan': restored.to(device)}
    def reject_eval_aux(*args):
        raise ValueError('推理执行了辅助头')
    handle1 = model.auxiliary_head.register_forward_hook(reject_eval_aux)
    handle2 = second.auxiliary_head.register_forward_hook(reject_eval_aux)
    with torch.inference_mode():
        torch.cuda.synchronize(device);start=time.monotonic()
        a,b=model(x,views),second(x,views);torch.cuda.synchronize(device);dual=time.monotonic()-start
        if 'auxiliary_logits' in a or 'auxiliary_logits' in b:raise ValueError('推理产生辅助输出')
        err=float((a['logits']-b['logits']).abs().max());torch.testing.assert_close(a['logits'],b['logits'],rtol=0,atol=1e-6)
        z,zr=model.encoder(x),model.encoder(views['vqgan'])
        torch.testing.assert_close(a['features'],torch.cat([z,z-zr],dim=1),rtol=0,atol=1e-6)
    handle1.remove();handle2.remove()
    peak=torch.cuda.max_memory_reserved(device);step=float(np.median(times[2:]))
    forecast=(21576*step + 12*700/64*dual + (5600+165758)*restoration_seconds + (5600+165758)/64*dual)*1.2 + 600
    report=dict(arm=arm, time=now(), identity=identity, first=first, lora_gradient_l1=component_gradients, updates=8,
                trace=trace, test_batch=64, reload_max_abs=err, eval_aux_calls=0, train_step_median=step,
                dual_eval64_seconds=dual, formal_projection_seconds=forecast, budget_pass=forecast+60<=LIMIT,
                peak_reserved_bytes=peak, required_free_bytes=int(peak+max(.2*peak,2*1024**3)))
    save(area/'验收.json',report)
    return report


def smoke(device):
    area=STATE/'smoke';area.mkdir(parents=True,exist_ok=False)
    c,_,manifests=settings('original_aux')
    if torch.cuda.mem_get_info(device)[0]<26*1024**3 or shutil.disk_usage(ROOT).free<12*1024**3:
        raise RuntimeError('GPU1显存或结果盘不满足预检门槛')
    cache=cache_audit(c,manifests);baseline=baseline_audit(c,manifests,cache,device)
    rows=manifests[0]['rows'][:64]
    raw=LMDBImages(rows,dict(c,reconstruction_mode='online_after_augment'),False)
    images=torch.stack([raw[i][0] for i in range(64)]);raw.close()
    probe=VQGANProbe(c).to(device).eval()
    with torch.inference_mode():
        quant=probe.audit_quantization(images[:1].to(device));probe(images[:1].to(device))
        torch.cuda.synchronize(device);start=time.monotonic();restored=probe(images.to(device));torch.cuda.synchronize(device)
        restoration_seconds=(time.monotonic()-start)/len(images)
    small_cache=area/'小缓存';build_cache(rows,c,probe,str(small_cache),vqgan_spec(c),device,16,600)
    pairs=LMDBImages(rows,dict(c,reconstruction_caches={'vqgan':str(small_cache)}),False)
    cached=torch.stack([pairs[i][3]['vqgan'] for i in range(len(rows))]).to(device);pairs.close()
    torch.testing.assert_close(cached,restored,rtol=0,atol=1e-6);del cached,probe
    reports={}
    for arm in ARMS:
        torch.cuda.empty_cache()
        reports[arm]=smoke_arm(arm,c,manifests,small_cache,images,restored,restoration_seconds,device)
    b,c2=reports.values()
    if b['identity']!=c2['identity'] or b['first']['main_logits_sha256']!=c2['first']['main_logits_sha256'] or b['trace']!=c2['trace']:
        raise ValueError('两辅助臂初值、主logits或输入轨迹不配对')
    passed=all(x['budget_pass'] for x in reports.values())
    summary=dict(status='passed' if passed else 'numeric_passed_budget_blocked',time=now(),code_commit=git('rev-parse','HEAD'),baseline=baseline,
                 full_cache_audit=cache,quantization=quant,arms=reports,lambda_value=1.0,formal_updates=0,budget_pass=passed,seconds_per_frame_gpu=restoration_seconds)
    save(area/'验收.json',summary)
    if not passed:raise RuntimeError('某臂真实估算超过12小时，正式0更新')


def run(arm,device,claim):
    c,paths,manifests=settings(arm);s=STATE/arm
    check=load(STATE/'smoke/验收.json');head=git('rev-parse','HEAD')
    if git('status','--porcelain') or check['status']!='passed' or check['code_commit']!=head:
        raise ValueError('有未提交改动或预检版本不符')
    if (s/'status.json').exists():raise FileExistsError('已有正式记录，拒绝重入')
    gate=check['arms'][arm]
    if gate['formal_projection_seconds']+60>LIMIT or shutil.disk_usage(ROOT).free<12*1024**3:
        raise RuntimeError('时间或磁盘预算不足')
    status=dict(arm=arm,run_id=RUN[arm],stage='cache_verification',time=now(),pid=os.getpid(),cwd=str(ROOT),code_commit=head,claim_commit=claim)
    save(s/'status.json',status);start=time.monotonic();deadline=start+LIMIT
    try:
        cache=cache_audit(c,manifests);reuse=time.monotonic()-start
        if cache!=check['full_cache_audit'] or torch.cuda.mem_get_info(device)[0]<gate['required_free_bytes']:
            raise RuntimeError('缓存身份或启动显存不符')
        protocol=dict(status='approved',id=STATE.name+'-'+arm,epochs=12,seed=3407,max_wall_seconds=LIMIT,
            train_manifest=str(paths[0]),val_manifest=str(paths[1]),evaluation_manifests=[],checkpoint_rule='source_val_video_balanced_ce',
            reconstruction_mode='cached_paired',augmentation_protocol='paired_float32_v1',reconstruction_caches={'vqgan':str(CACHE)},
            endpoint_manifests=[str(p) for p in paths[2:]],endpoint_mode='shared_gpu_restoration_two_gpu_models',
            residual='signed_diff',auxiliary_input=arm,auxiliary_lambda=1.0,baseline_result_commit=BASE_RESULT,power_cutoff=None)
        save(s/'protocol.json',protocol);status.update(stage='training',time=now(),formal_started=now(),cache_reuse_verification_seconds=reuse);save(s/'status.json',status)
        train_start=time.monotonic();result=run_training(c,protocol,RUN[arm],device)
        torch.cuda.empty_cache();out=Path(c['output_root'])/'r01'/RUN[arm]
        status.update(stage='evaluation',time=now());save(s/'status.json',status)
        result['endpoints']=evaluate_two(c,out,manifests[2:],deadline,device)
        result.update(elapsed_seconds=time.monotonic()-train_start,cache_seconds=cache['generation_seconds'],cache_reused=True,
                      new_cache_generation_seconds=0,cache_reuse_verification_seconds=reuse,residual='signed_diff',
                      auxiliary_input=arm,auxiliary_lambda=1.0,complete_evaluation=True)
        save(out/'complete_result.json',result);status.update(stage='compute_finished',time=now(),result=result);save(s/'status.json',status)
    except BaseException as error:
        status.update(stage='failed',time=now(),error=repr(error));save(s/'status.json',status);raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','run']);parser.add_argument('--arm',choices=ARMS);parser.add_argument('--claim-commit');a=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('须绑定物理GPU1')
    torch.set_num_threads(1);seed_all(3407);device=torch.device('cuda:0')
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
        if a.mode=='smoke':smoke(device)
        else:
            if not a.arm or not a.claim_commit or len(a.claim_commit)!=40:raise ValueError('缺少明确臂或本地正式认领')
            run(a.arm,device,a.claim_commit)


if __name__=='__main__':main()
