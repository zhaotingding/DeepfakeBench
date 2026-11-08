#!/usr/bin/env python3
"""仅原图、经典 PRD 与冻结重构算子的验证早停训练入口。"""
import os
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"training"))
import numpy as np
import torch
from prd_r01.config import read_config
import prd_fresh as fresh


def git(*args):
    return subprocess.check_output(["git","-C",str(ROOT),*args],text=True).strip()


def settings(variant):
    config = read_config(ROOT/"r01.machine.json",variant)
    config.update(train_batch_size=32,test_batch_size=64,workers=0,persistent_workers=False,
                  augmentation_protocol="legacy_uint8_online_v1",reconstruction_mode="online_after_augment",
                  reconstruction_caches={},residual="signed_diff")
    config["lora"]={"rank":16,"alpha":16,"dropout":0.05,"target_modules":["q_proj","v_proj"]}
    config["head"]={"hidden":1024,"dropout":0.5,"slots":2}
    config["optimizer"]={"lr":2e-4,"betas":[0.9,0.999],"eps":1e-8,"weight_decay":0.0005}
    config["numerics"]={"deterministic_algorithms":True,"cuda_matmul_tf32":False,"cudnn_tf32":False,"cublas_workspace":":4096:8"}
    config["source_validation"]="official_fixed_val_frame_auc_every_epoch"
    config["target_evaluation"]="best_only_after_training_stops"
    config["note"]="本批源验证AUC五轮未创新高早停，最多30轮，训练前新初始化，增强后在线重构。"
    config.pop("operators",None)
    if variant == "vqgan":
        from prd_r01.vqgan_operator import vqgan_spec  # 记录原码本工作点，不复用增强前的旧缓存。
        config["vqgan_weights"] = str(Path.home()/".cache/prd-models/vqgan-imagenet-f16-1024/last.ckpt")
        config["vqgan_weights_sha256"] = "e38a3aed8c86e8d47436f611325b3a23888cab70a0cc56c13cd84e8d4aab6e71"
        config["operator_recipe"] = vqgan_spec(config)  # native224、FP32、内部逐图、不裁剪输出。
    if variant == "superresolution":
        from prd_r01.generative_operators import ASSETS, WEIGHTS, spec
        config["superresolution_weights"] = str(ASSETS / WEIGHTS[variant][0])  # 与旧超分使用同一官方资产。
        config["operator_recipe"] = spec(config, variant)  # FP32 抗混叠 56、反射补边、逐图 FP16 ×4。
    return config


def manifests(config):
    result={k:fresh.manifest(config,"FaceForensics++",split,"source_"+split) for k,split in
            [("train","train"),("val","val"),("source_test","test")]}
    fresh.validate_source_pair(result["train"],result["val"])
    for k in fresh.MAIN_DOMAINS+fresh.EXTRA_DOMAINS:
        result[k]=fresh.manifest(config,k,"test","target_test" if k in fresh.MAIN_DOMAINS else "supplementary_test")
    expected={"train":114884,"val":22354,"source_test":22388,
              "Celeb-DF-v2":16420,"DFDC":132116,"DeepFakeDetection":102541,"UADFV":3099,
              "Celeb-DF-v1":3136,"DFDCP":17222}
    if {k:len(v["rows"]) for k,v in result.items()} != expected:
        raise ValueError("实际清单与预检冻结规模不同")
    return result


def model_identity(config, data):
    return dict(base_weights=fresh.frozen_identity(config), config=config,
                manifests={k:hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                           for k,v in data.items()},
                validation_rule={"metric":"frame_auc","mode":"max","patience":5,"min_delta":0,
                                 "max_epochs":30,"tie":"earliest","fixed_views":True})


def trainable_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def selected(rows, amount):
    groups=[[r for r in rows if r["label"]==y][:amount//2] for y in (0,1)]
    if any(len(g)!=amount//2 for g in groups):raise ValueError("预检样本缺真假")
    return [r for pair in zip(*groups) for r in pair]


def cpu_tests():
    p={"best_auc":-1.0,"best_epoch":None,"bad_epochs":0}
    for epoch,score in enumerate([.7,.75,.74,.75,.73,.749,.75],1):
        fresh.update_best(p,score,epoch)
    assert p=={"best_auc":.75,"best_epoch":2,"bad_epochs":5}
    q={"best_auc":-1.0,"best_epoch":None,"bad_epochs":0}
    for epoch in range(1,31):fresh.update_best(q,.5+epoch*.001,epoch)
    assert q["bad_epochs"]==0 and q["best_epoch"]==30
    return {"patience5_tie_earliest":True,"max_epoch_is_separate_cap":True}


def preflight(config,data,state,device):
    state.mkdir(parents=True,exist_ok=False)
    identity=model_identity(config,data)
    fresh.save_json(state/"identity.json",identity)
    rows=selected(data["train"]["rows"],256);val=selected(data["val"]["rows"],64)
    fresh.seed_all(3407);model=fresh.build(config,device)
    initial=fresh.trainable_identity(model)
    assert initial["trainable_parameters"]==3675138
    assert initial["trainable_initial_sha256"]=="aaa440a41a2f1e8cab40f8c0be19e3479c0b22880256d6e4ca96eee2c0b1fe7e"  # 与首批共同 CLIP/LoRA/主头初值一致。
    probe_audit = None
    reconstruction_audit = None
    if config["variant"] == "vqgan":
        audit_loader = fresh.loader(val, config)
        audit_image = next(iter(audit_loader))[0][:1].to(device)
        probe_audit = model.probes["vqgan"].audit_quantization(audit_image)  # 独立最近邻验证实际码字路径。
        audit_loader.dataset.close()
        del audit_loader, audit_image
    if config["variant"] == "superresolution":
        audit_loader = fresh.loader(val, config)
        audit_image = next(iter(audit_loader))[0][:1].to(device)
        reconstruction_audit = model.probes["superresolution"].audit(audit_image)
        assert reconstruction_audit["frozen"] and reconstruction_audit["mean_absolute_rgb_change"] > 0
        audit_loader.dataset.close()
        del audit_loader, audit_image
    opt=fresh.optimizer_for(model,config)
    torch.cuda.reset_peak_memory_stats(device)
    steps=[];images=[]
    stream=fresh.loader(rows,config,1,True)
    iterator=iter(stream)
    for _ in range(4):
        torch.cuda.synchronize();t=time.monotonic();batch=next(iterator)
        loss,grad,ids,ih=fresh.train_step(model,opt,batch,device);torch.cuda.synchronize()
        steps.append(time.monotonic()-t);images.append(ih)
        assert any(p.grad is not None and float(p.grad.abs().max())>0 for n,p in model.named_parameters() if "lora_B" in n)
        assert all(p.grad is None for p in model.probes.parameters())
    metric_before,pred=fresh.evaluate(model,val,config,device,time.monotonic()+600)
    score_again,pred_again=fresh.evaluate(model,val,config,device,time.monotonic()+600)
    if not np.array_equal(pred,pred_again):raise ValueError("固定eval仍随机")
    progress=dict(next_epoch=1,offset=128,updates=4,best_auc=metric_before["frame_auc"],
                  best_epoch=0,bad_epochs=0,curve=[],training_complete=False,stop_reason=None,elapsed_seconds=0.)
    fresh.save_torch(state/"resume_test.pt",fresh.checkpoint(model,opt,progress,identity))
    expected_loss=[]
    for _ in range(4):
        torch.cuda.synchronize();t=time.monotonic();batch=next(iterator)
        loss,grad,ids,ih=fresh.train_step(model,opt,batch,device);torch.cuda.synchronize()
        steps.append(time.monotonic()-t);images.append(ih);expected_loss.append(loss)
    expected=fresh.compact_state(model);expected_hash=trainable_hash(expected)
    expected_optimizer=copy.deepcopy(opt.state_dict())
    expected_optimizer={k: v for k,v in expected_optimizer.items()}
    opt.zero_grad(set_to_none=True);stream.dataset.close()
    del iterator,stream,opt,model;gc.collect();torch.cuda.empty_cache()
    fresh.seed_all(3407);resumed=fresh.build(config,device);opt=fresh.optimizer_for(resumed,config)
    payload=torch.load(state/"resume_test.pt",map_location="cpu",weights_only=False)
    restored=fresh.load_resume(resumed,opt,payload,identity)
    assert restored==progress
    stream=fresh.loader(rows,config,1,True,restored["offset"])
    actual_loss=[]
    for batch in stream:
        loss,grad,ids,ih=fresh.train_step(resumed,opt,batch,device)
        actual_loss.append(loss)
        if ih!=images[4+len(actual_loss)-1]:raise ValueError("恢复后增强输入变化")
    actual=fresh.compact_state(resumed)
    param_error=max(float((actual[k].to(torch.float64)-expected[k].to(torch.float64)).abs().max()) for k in actual)
    optimizer_error=0.0
    for key,saved in expected_optimizer["state"].items():
        for name,value in saved.items():
            if isinstance(value,torch.Tensor):
                other=opt.state_dict()["state"][key][name]
                optimizer_error=max(optimizer_error,float((value.cpu()-other.cpu()).abs().max()))
    fresh.save_json(state/"resume_comparison.json",dict(parameter_error=param_error,optimizer_error=optimizer_error,
                    loss_error=max(abs(a-b) for a,b in zip(actual_loss,expected_loss))))
    if param_error>1e-6 or optimizer_error>1e-6 or max(abs(a-b) for a,b in zip(actual_loss,expected_loss))>1e-6:
        raise ValueError("中断恢复没有复现未中断的训练轨迹")
    stream.dataset.close();opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize();started=time.monotonic()
    final_metrics,final_pred=fresh.evaluate(resumed,val,config,device,time.monotonic()+600)
    torch.cuda.synchronize();eval_seconds=time.monotonic()-started
    best=dict(schema=1,model=fresh.compact_state(resumed),epoch=0,validation=final_metrics,identity=identity)
    fresh.save_torch(state/"best_test.pt",best)
    fresh.load_compact(resumed,torch.load(state/"best_test.pt",map_location="cpu",weights_only=True)["model"])
    _,reloaded=fresh.evaluate(resumed,val,config,device,time.monotonic()+600)
    assert np.array_equal(final_pred,reloaded)
    missing=copy.deepcopy(best["model"]);missing.pop(next(iter(missing)))
    try:fresh.load_compact(resumed,missing)
    except ValueError:pass
    else:raise ValueError("缺失权重没有被拒绝")
    comparison = fresh.loader(rows, settings("original"), 1, True)
    paired_hashes = [hashlib.sha256(batch[0].numpy().tobytes()).hexdigest() for batch in comparison]
    comparison.dataset.close()
    assert paired_hashes == images  # 重构加载与路径不能改变固定的原图增强输入。
    train_step=float(np.median(steps[1:]))
    epoch_est=math_ceil(len(data["train"]["rows"])/32)*train_step+len(data["val"]["rows"])/64*eval_seconds
    eval_frames=sum(len(data[k]["rows"]) for k in ["source_test",*fresh.MAIN_DOMAINS,*fresh.EXTRA_DOMAINS])
    summary=dict(status="passed",time=fresh.now(),code_commit=git("rev-parse","HEAD"),variant=config["variant"],
                 formal_updates=0,smoke_optimizer_updates=12,uninterrupted_updates=8,replayed_updates=4,quantization_audit=probe_audit,batch_size=32,test_batch_size=64,
                 initialization=initial,fixed_validation_max_error=float(np.max(np.abs(pred-pred_again))),
                 resume_parameter_max_error=param_error,resume_optimizer_max_error=optimizer_error,
                 resume_loss_max_error=max(abs(a-b) for a,b in zip(actual_loss,expected_loss)),
                 compact_state_sha256=expected_hash,initial_input_hashes=images,
                 compact_best_bytes=(state/"best_test.pt").stat().st_size,
                 compact_resume_bytes=(state/"resume_test.pt").stat().st_size,
                 train_step_median_seconds=train_step,eval64_seconds=eval_seconds,
                 complete_epoch_projection_seconds=epoch_est,
                 thirty_epoch_upper_projection_seconds=epoch_est*30*1.2+600,
                 complete_best_evaluation_projection_seconds=eval_frames/64*eval_seconds*1.2+600,
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(device),cpu_rule_tests=cpu_tests(),
                 input_counts={k:len(v["rows"]) for k,v in data.items()})
    if reconstruction_audit is not None:
        summary["reconstruction_audit"] = reconstruction_audit
    fresh.save_json(state/"preflight.json",summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def math_ceil(value):
    import math
    return math.ceil(value)


def formal(args,config,data,state,device):
    if not args.claim_commit or len(args.claim_commit)!=40:raise ValueError("需要正式认领SHA")
    if args.train_hours is None or args.eval_hours is None or min(args.train_hours,args.eval_hours)<=0:
        raise ValueError("需要按真实预检固定训练和评估时间预算")
    if config["variant"] == "superresolution" and (args.train_hours > 105 or args.eval_hours > 5):
        raise ValueError("降采样超分预算不得超过105小时训练与5小时终测")
    if git("status","--porcelain"):raise ValueError("运行代码须已提交")
    check=json.loads((state/"smoke/preflight.json").read_text(encoding="utf8"))
    if check["status"]!="passed" or check["code_commit"]!=git("rev-parse","HEAD"):
        raise ValueError("预检版本不符")
    if config["variant"] == "superresolution" and check["variant"] != config["variant"]:
        raise ValueError("降采样超分须通过本方案真实预检")
    if args.train_hours*3600<check["thirty_epoch_upper_projection_seconds"] or args.eval_hours*3600<check["complete_best_evaluation_projection_seconds"]:
        raise ValueError("正式预算不足以覆盖预登记的最大运行")
    if torch.cuda.mem_get_info(device)[0]<check["peak_reserved_bytes"]+2*1024**3:
        raise ValueError("当前显存低于实测峰值加2GiB")
    out=Path(config["output_root"])/fresh.RECONSTRUCTION_RUN_FAMILIES.get(config["variant"], "早停基线-20261105")/args.run_id
    if args.resume:
        if not out.is_dir():raise FileNotFoundError("运行目录不存在")
    else:
        out.mkdir(parents=True,exist_ok=False)
    identity=model_identity(config,data)
    metadata=dict(config=config,identity=identity,code_commit=git("rev-parse","HEAD"),claim_commit=args.claim_commit,
                  run_id=args.run_id,started=fresh.now(),input_mode="augmentation_then_online_reconstruction",
                  training_basis="paper_aligned_plus_declared_implementation_choices",
                  stopping={"metric":"source_val_frame_auc","patience":5,"max_epochs":30,"min_delta":0,"tie":"earliest"},
                  optimizer={"name":"Adam",**config["optimizer"]},head_control="same_two_slot_head_original_zero_response",
                  max_training_seconds=args.train_hours*3600,max_evaluation_seconds=args.eval_hours*3600)
    if args.resume:
        old=json.loads((out/"metadata.json").read_text())
        if old["identity"]!=identity or old["code_commit"]!=metadata["code_commit"]:raise ValueError("恢复身份变化")
        if old["max_training_seconds"]!=metadata["max_training_seconds"] or old["max_evaluation_seconds"]!=metadata["max_evaluation_seconds"]:
            raise ValueError("恢复不能隐式更改预算")
        if (out/"evaluation_started.json").exists():
            raise ValueError("训练恢复入口不重启评估；保留已有评估产物供独立恢复")
    else:
        fresh.save_json(out/"metadata.json",metadata)
        for name,item in data.items():fresh.save_json(out/(name+"_manifest.json"),item)
    fresh.save_json(state/"status.json",dict(stage="training",time=fresh.now(),pid=os.getpid(),cwd=str(ROOT),
                                           run_id=args.run_id,code_commit=metadata["code_commit"],claim_commit=args.claim_commit))
    try:
        model,progress=fresh.train(config,data,out,state,device,identity,args.train_hours*3600,args.resume)
        fresh.save_json(out/"evaluation_started.json",dict(time=fresh.now(),best_sha256=fresh.file_sha(out/"best.pt"),max_seconds=args.eval_hours*3600))
        result=fresh.final_evaluation(model,config,data,out,state,device,args.eval_hours*3600)
        print(json.dumps(dict(status="verified",best_epoch=progress["best_epoch"],stop_reason=progress["stop_reason"],
                              main=result["macro"]),ensure_ascii=False),flush=True)
    except BaseException as error:
        fresh.save_json(state/"failure.json",dict(time=fresh.now(),error=repr(error),pid=os.getpid(),run_id=args.run_id))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant",choices=["original","vae","vqgan","superresolution"])
    parser.add_argument("mode",choices=["manifest","smoke","run"])
    parser.add_argument("--claim-commit");parser.add_argument("--run-id")
    parser.add_argument("--train-hours",type=float);parser.add_argument("--eval-hours",type=float)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    config=settings(args.variant);data=manifests(config)
    state=ROOT/".state"/fresh.RECONSTRUCTION_RUN_FAMILIES.get(args.variant, "早停基线-20261105")/args.variant
    if args.mode=="manifest":
        print(json.dumps({k:{"frames":len(v["rows"]),"videos":len({r["video_id"] for r in v["rows"]}),"source_sha256":v["source_sha256"]} for k,v in data.items()},ensure_ascii=False));return
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count()!=1:
        raise RuntimeError("必须显式绑定单一GPU")
    import fcntl
    device=torch.device("cuda:0");torch.set_num_threads(1)
    uuid=torch.cuda.get_device_properties(device).uuid
    lockroot=Path.home()/".cache/prd-locks";lockroot.mkdir(parents=True,exist_ok=True)
    lock=(lockroot/("早停基线-"+str(uuid)+".lock")).open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.mode=="smoke":preflight(config,data,state/"smoke",device)
    else:
        if not args.run_id or any(not (c.isalnum() or c in "-_") for c in args.run_id):
            raise ValueError("运行名只能含中文、字母、数字和连接号")
        formal(args,config,data,state,device)


if __name__=="__main__":
    main()

