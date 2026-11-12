"""原图辅助监督的真实梯度审计与激活重算等价门槛，不启动正式训练。"""
import copy
import gc
import hashlib
import numpy as np
import torch
from .auxiliary_supervision import training_loss, trainable_identity

MEMORY_MODE = "clip_even_layers_nonreentrant_preserve_rng_v1"


def auxiliary_audit(model, config, rows, device):
    import prd_fresh as fresh
    stream = fresh.loader(rows, config)
    batch = next(iter(stream)); stream.dataset.close()
    x = batch[0][:4].to(device); y = batch[1][:4].to(device)
    model.train()
    output = model(x, {})
    total, main, auxiliary = training_loss(output, y, "original_aux", 1.0)
    expected_auxiliary_logits = model.auxiliary_head(output["features"][:, :1024])
    assert torch.equal(output["auxiliary_logits"], expected_auxiliary_logits)
    assert torch.equal(total, main + auxiliary)
    named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(auxiliary, [p for _,p in named], allow_unused=True)
    shared = [float(g.abs().max()) for (name,_),g in zip(named,grads)
              if name.startswith("encoder.") and g is not None]
    head_absent = all(g is None for (name,_),g in zip(named,grads) if name.startswith("head."))
    aux = [float(g.abs().max()) for (name,_),g in zip(named,grads)
           if name.startswith("auxiliary_head.") and g is not None]
    assert shared and max(shared) > 0 and head_absent and aux and max(aux) > 0
    assert all(p.grad is None for p in model.probes.parameters())
    model.eval()
    with torch.inference_mode():
        evaluation = model(x,{})
        assert "auxiliary_logits" not in evaluation
        assert torch.equal(evaluation["logits"], model.head(evaluation["features"]))
        quantization = model.probes["vqgan"].audit_quantization(x[:1])
    return dict(status="passed", auxiliary_input="original_feature_z", auxiliary_parameters=2050,
                auxiliary_lambda=1.0, total_loss=float(total.detach()), main_ce=float(main.detach()),
                auxiliary_ce=float(auxiliary.detach()), exact_loss_sum=True,
                auxiliary_to_shared_lora_max_gradient=max(shared), auxiliary_head_max_gradient=max(aux),
                auxiliary_does_not_update_main_head=head_absent, inference_main_only=True,
                quantization=quantization)


def _cpu(item):
    if isinstance(item,torch.Tensor):return item.detach().cpu().clone()
    if isinstance(item,dict):return {k:_cpu(v) for k,v in item.items()}
    if isinstance(item,list):return [_cpu(v) for v in item]
    if isinstance(item,tuple):return tuple(_cpu(v) for v in item)
    return copy.deepcopy(item)


def _difference(left,right):
    if isinstance(left,torch.Tensor):
        assert isinstance(right,torch.Tensor) and left.shape==right.shape and left.dtype==right.dtype
        if torch.equal(left,right):return 0.0
        return float((left.to(torch.float64)-right.to(torch.float64)).abs().max())
    if isinstance(left,np.ndarray):
        assert isinstance(right,np.ndarray) and left.shape==right.shape and left.dtype==right.dtype
        if np.array_equal(left,right):return 0.0
        return float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64))))
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        return max((_difference(left[k],right[k]) for k in left),default=0.)
    if isinstance(left,(list,tuple)):
        assert type(left)==type(right) and len(left)==len(right)
        return max((_difference(a,b) for a,b in zip(left,right)),default=0.)
    if isinstance(left,float):
        assert np.isfinite(left) and np.isfinite(right)
        return abs(left-right)
    assert left==right
    return 0.


def checkpoint_parity(config, rows, device):
    import prd_fresh as fresh
    stream=fresh.loader(rows,config,1,True)
    batch=next(iter(stream));stream.dataset.close()
    inputs=[(batch[0][i:i+4].clone(),batch[1][i:i+4].clone()) for i in (0,4)]
    captures=[]
    for mode in (None,MEMORY_MODE):
        current=copy.deepcopy(config);current["activation_checkpointing"]=mode
        fresh.seed_all(3407);model=fresh.build(current,device);opt=fresh.optimizer_for(model,current)
        record={"initialization":trainable_identity(model),"steps":[]}
        for image,label in inputs:
            model.train();opt.zero_grad(set_to_none=True)
            output=model(image.to(device),{})
            total,main,auxiliary=training_loss(output,label.to(device),"original_aux",1.0)
            total.backward()
            named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
            norm=torch.nn.utils.clip_grad_norm_([p for _,p in named],float("inf"),error_if_nonfinite=True)
            gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in named}
            opt.step()
            record["steps"].append(dict(output=_cpu(output),losses=[float(total.detach()),float(main.detach()),float(auxiliary.detach())],
                gradient_norm=float(norm), gradients=gradients, model=fresh.compact_state(model),
                optimizer=_cpu(opt.state_dict()), rng=_cpu(fresh.rng_state())))
            del output,total,main,auxiliary,named,gradients
        captures.append(record)
        del model,opt,record
        gc.collect();torch.cuda.empty_cache()
    assert captures[0]["initialization"]==captures[1]["initialization"]
    errors={key:max(_difference(a[key],b[key]) for a,b in zip(captures[0]["steps"],captures[1]["steps"]))
            for key in ["output","losses","gradient_norm","gradients","model","optimizer","rng"]}
    # RNG及可变缓冲必须逐位一致；其他数值采用既定恢复预检1e-6绝对误差门槛。
    assert errors["rng"]==0
    buffers={}
    for a,b in zip(captures[0]["steps"],captures[1]["steps"]):
        for name in a["model"]:
            if name.endswith(("running_mean","running_var","num_batches_tracked","position_ids")):
                buffers[name]=max(buffers.get(name,0.),_difference(a["model"][name],b["model"][name]))
    assert max(buffers.values(),default=0.)==0
    assert max(errors.values())<=1e-6, errors
    return dict(status="passed",mode=MEMORY_MODE,checkpoint_layer_indices=list(range(0,24,2)),batch_size=4,steps=2,formal_updates=0,
                tolerance=1e-6,max_absolute_errors=errors,mutable_buffer_max_error=max(buffers.values(),default=0.),
                image_sha256=[hashlib.sha256(image.numpy().tobytes()).hexdigest() for image,_ in inputs],
                initialization=captures[0]["initialization"])

