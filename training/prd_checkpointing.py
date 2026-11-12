"""固定输入下检验CLIP层级重计算的数值等价性；不修改科学批量。"""
import copy
import gc
import hashlib
import json
import time
import numpy as np
import torch
import prd_fresh as fresh

IMPLEMENTATION='clip_transformer_nonreentrant_preserve_rng_v1'

def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)

def difference(left,right):
    if isinstance(left,torch.Tensor):
        assert left.dtype==right.dtype and left.shape==right.shape
        if not left.is_floating_point():
            assert torch.equal(left,right);return 0.
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        return float((left.double()-right.double()).abs().max()) if left.numel() else 0.
    if isinstance(left,np.ndarray):
        assert np.array_equal(left,right);return 0.
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        return max([difference(left[k],right[k]) for k in left]+[0.])
    if isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        return max([difference(a,b) for a,b in zip(left,right)]+[0.])
    assert left==right,(left,right)
    return 0.

def verify_checkpoint_parity(config,data,state,device,code):
    if config.get('activation_checkpointing')!=IMPLEMENTATION:
        raise ValueError('显式重计算身份缺失')
    state.mkdir(parents=True,exist_ok=True)
    destination=state/'checkpoint_parity.json'
    if destination.exists():raise FileExistsError('技术等价预检已存在，拒绝隐式覆盖')
    free,total=torch.cuda.mem_get_info(device)
    if free < 10*1024**3:raise RuntimeError('BS4等价探针须至少10GiB空闲显存')
    # 临时分配器上限只影响此技术预检进程，不改变模型批量或正式入口。
    torch.cuda.set_per_process_memory_fraction(min(.95,(free-2*1024**3)/total),device)
    rows=[r for pair in zip(*[[r for r in data['train']['rows'] if r['label']==y][:128] for y in (0,1)]) for r in pair]
    stream=fresh.loader(rows,config,1,True)
    source=iter(stream);batches=[]
    for _ in range(4):
        batch=next(source)
        batches.append((batch[0][:4].clone(),batch[1][:4].clone(),batch[2][:4].clone(),
                        {k:v[:4].clone() for k,v in batch[3].items()}))
    stream.dataset.close();del stream,source
    records=[];reference=[];started=time.monotonic()
    for enabled in (False,True):
        path=copy.deepcopy(config)
        if not enabled:path.pop('activation_checkpointing')
        fresh.seed_all(3407);model=fresh.build(path,device);opt=fresh.optimizer_for(model,path)
        initial=fresh.trainable_identity(model)
        assert initial['trainable_initial_sha256']=='aaa440a41a2f1e8cab40f8c0be19e3479c0b22880256d6e4ca96eee2c0b1fe7e'
        torch.cuda.reset_peak_memory_stats(device)
        errors={k:0. for k in ('logits','loss','gradients','parameters_and_buffers','optimizer')}
        for step,batch in enumerate(batches):
            if enabled:
                assert difference(cpu_tree(fresh.rng_state()),reference[step]['before_rng'])==0
                fresh.restore_rng(reference[step]['before_rng'])
            before=cpu_tree(fresh.rng_state());captured=[]
            hook=model.head.register_forward_hook(lambda module,args,output:captured.append(output.detach().cpu().clone()))
            loss,gradient,indices,input_hash=fresh.train_step(model,opt,batch,device)
            hook.remove()
            assert len(captured)==1, '分类头不应被激活重计算重复执行'
            payload={'before_rng':before,'after_rng':cpu_tree(fresh.rng_state()),'logits':captured[0],
                     'loss':torch.tensor(loss,dtype=torch.float64),
                     'gradients':{n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None},
                     'parameters_and_buffers':fresh.compact_state(model),'optimizer':cpu_tree(opt.state_dict()),
                     'input_sha256':input_hash,'indices':indices}
            assert any('lora_B' in k and float(v.abs().max())>0 for k,v in payload['gradients'].items())
            assert all(p.grad is None for p in model.probes.parameters())
            if enabled:
                expected=reference[step]
                assert payload['input_sha256']==expected['input_sha256'] and payload['indices']==expected['indices']
                assert difference(payload['before_rng'],expected['before_rng'])==0
                assert difference(payload['after_rng'],expected['after_rng'])==0
                # 可变buffer必须完全相同；BN只执行一次，不以1e-6容差掩盖状态漂移。
                assert all(torch.equal(v.detach().cpu(),expected['parameters_and_buffers'][n]) for n,v in model.named_buffers())
                for key in errors:
                    value=difference(payload[key],expected[key]);errors[key]=max(errors[key],value)
                    if value>1e-6:raise ValueError(f'激活重计算等价失败: step{step+1} {key}={value}')
            else:reference.append(payload)
        records.append({'checkpoint_enabled':enabled,'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),
                        'max_errors':errors,'rng_exact':True,'buffers_exact':True})
        opt.zero_grad(set_to_none=True)
        del model,opt,payload,captured,batch;gc.collect();torch.cuda.empty_cache()
    result={'status':'passed','time':fresh.now(),'code_commit':code,'implementation':IMPLEMENTATION,
            'technical_batch_size':4,'paired_optimizer_steps':4,'formal_updates':0,
            'formal_batch_unchanged':32,'records':records,'tolerance':1e-6,
            'input_sha256':[r['input_sha256'] for r in reference], 'seconds':time.monotonic()-started}
    fresh.save_json(destination,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
