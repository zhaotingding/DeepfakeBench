#!/usr/bin/env python3
"""已批准小波/低秩批次：准备、真实batch验收和有限双卡运行。"""
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
BATCH = json.loads((ROOT / "training/config/r01/小波低秩-20261029.json").read_text())
INPUTS = Path.home() / "prd-inputs/重构算子首批-20261029"
STATE = ROOT / ".state/重构算子首批-20261029"
MACHINE = ROOT / "r01.machine.json"
NAMES = BATCH["display_names"]

def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)  # 状态读取者不见半份JSON。

def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

def config(variant):
    from prd_r01.config import read_config
    return read_config(MACHINE, variant)

def manifests():
    from prd_r01.data import load_manifest
    return [load_manifest(INPUTS / (name + ".json")) for name in ["训练清单", "源验证清单", "源测试清单", "目标开发清单"]]

def all_rows(items):
    from prd_r01.cache import row_key
    unique = {}
    for m in items:
        for row in m["rows"]:
            key = row_key(row)
            if key in unique and unique[key] != row:
                raise ValueError("同一帧的清单元数据冲突")
            unique[key] = row
    return list(unique.values())

def prepare():
    import numpy as np
    from prd_r01.cache import file_sha
    from prd_r01.data import validate_source_pair, load_manifest
    if INPUTS.exists():
        raise FileExistsError("拒绝覆盖已冻结清单")
    c = config("wavelet")
    source = Path(c["data_root"]) / "DeepfakeBench/config/dataset_json"
    old = Path.home() / "prd-inputs/q18_20261008"
    ff = json.loads((source / "FaceForensics++.json").read_text())["FaceForensics++"]
    key_info = {}
    source_components = {}
    for split in ["train", "val", "test"]:
        ids = set()
        for kind, splits in ff.items():
            for video, info in splits[split]["c23"].items():
                sources = video.split("_"); ids.update(sources)
                for frame in info["frames"]:
                    key_info[(split, frame)] = dict(domain="FaceForensics++",key=frame,label=int(info["label"] != "FF-real"),video_id=kind+"/"+video,source_ids=sources)
        source_components[split] = ids
    if any(source_components[a] & source_components[b] for a,b in [("train","val"),("train","test"),("val","test")]):
        raise ValueError("FF++来源分量交叠")
    previous = json.loads((old / "q10_train_manifest.json").read_text())
    train = []
    for key, label in zip(previous["image"], previous["label"]):
        row = key_info[("train", key)]
        if row["label"] != label:
            raise ValueError("旧训练清单标签与源JSON不同")
        train.append(row)
    def restore_source(name, split):
        old_manifest = json.loads((old / name).read_text())
        return [key_info[(split, row["frame"])] for row in old_manifest["frames"]]
    val = restore_source("q10_source_val_manifest.json", "val")
    test = restore_source("q10_source_test_manifest.json", "test")
    target = []
    old_targets = json.loads((old / "q10_target_manifest.json").read_text())["videos"]
    expected_videos = {(r["domain"],r["kind"],r["video_id"]) for r in old_targets}
    observed_videos = set()
    for domain in ["Celeb-DF-v2", "DFDCP", "DFDC"]:
        d = json.loads((source / (domain + ".json")).read_text())[domain]
        for kind, splits in sorted(d.items()):
            for video, info in sorted(splits["test"].items()):
                frames = sorted(info["frames"])
                if not frames:
                    raise ValueError("目标视频没有帧：" + video)
                observed_videos.add((domain,kind,video))
                indices = np.linspace(0,len(frames)-1,min(32,len(frames)),dtype=int)
                real = "real" in info["label"].lower()
                target.extend(dict(domain=domain,key=frames[i],label=int(not real),video_id=kind+"/"+video) for i in indices)
    if observed_videos != expected_videos:
        raise ValueError("目标视频集合与既有5876清单不同")
    if len(train) != BATCH["source_train_frames"] or len(val) != BATCH["source_val_frames"]:
        raise ValueError("训练或源验证数量变化")
    items = [dict(role=role,rows=rows) for role,rows in [("source_train",train),("source_val",val),("source_test",test),("target_dev",target)]]
    validate_source_pair(items[0],items[1])
    INPUTS.mkdir(parents=True, exist_ok=False)
    for name, item in zip(["训练清单","源验证清单","源测试清单","目标开发清单"],items):
        save(INPUTS / (name + ".json"), item)
        load_manifest(INPUTS / (name + ".json"))
    merged = all_rows(items)
    needed = len(merged) * 3 * 224**2 * 4 * 2
    if needed > BATCH["new_storage_limit_bytes"] or shutil.disk_usage(INPUTS).free < needed + BATCH["minimum_disk_free_bytes"]:
        raise OSError("双算子完整缓存超出存储预算")
    output = Path(c["output_root"])
    for variant in BATCH["variants"]:
        protocol = dict(status="approved",id=BATCH["id"]+"-"+NAMES[variant],epochs=BATCH["epochs"],seed=BATCH["seed"],
            train_manifest=str(INPUTS/"训练清单.json"),val_manifest=str(INPUTS/"源验证清单.json"),
            evaluation_manifests=[str(INPUTS/"源测试清单.json"),str(INPUTS/"目标开发清单.json")],
            max_wall_seconds=BATCH["formal_max_seconds"],checkpoint_rule=BATCH["checkpoint_rule"],
            reconstruction_mode=BATCH["reconstruction_mode"],augmentation_protocol=BATCH["augmentation_protocol"],
            reconstruction_caches={variant:str(output/"重构缓存-20261029"/NAMES[variant])},authorization=BATCH["authorization"],
            primary_comparison=BATCH["primary_comparison"],controls_status="原图/VAE尚未完成；不判定相对提点")
        save(INPUTS / (variant + ".json"),protocol)
    summary=dict(time=now(),source_parent_counts={k:len(v) for k,v in source_components.items()},
        counts=[dict(role=m["role"],frames=len(m["rows"]),videos=len({(r["domain"],r["video_id"]) for r in m["rows"]})) for m in items],
        target_domains=dict(collections.Counter(r["domain"] for r in target)),merged_frames=len(merged),two_cache_bytes_estimate=needed,
        input_json_sha256={p.name:file_sha(p) for p in [source/(n+".json") for n in ["FaceForensics++","Celeb-DF-v2","DFDCP","DFDC"]]},
        manifest_sha256={p.name:file_sha(p) for p in INPUTS.glob("*.json")})
    save(INPUTS/"清单验收.json",summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

def device_free(device):
    import torch
    return torch.cuda.mem_get_info(device)[0]

def smoke(variant, device_name):
    import numpy as np
    import torch
    from prd_r01.cache import build_cache, operator_spec, file_sha
    from prd_r01.data import LMDBImages, make_loader
    from prd_r01.operators import build_operator
    from prd_r01.engine import seed_all, move_batch
    from prd_r01.model import build_model
    torch.set_num_threads(2)
    device=torch.device(device_name); c=config(variant)
    initial_free=device_free(device)
    if initial_free < 26*1024**3:
        raise RuntimeError("当前可用显存不足26GiB预检入口，不加载模型")
    area=STATE/(variant+"-smoke");area.mkdir(parents=True,exist_ok=False)
    items=manifests();rows=items[0]["rows"][:64]
    seed_all(BATCH["seed"])
    probe=build_operator(c,variant,device)
    start=time.monotonic()
    cache_meta=build_cache(rows,c,probe,str(area/"缓存"),operator_spec(c,variant),device,16,600)
    cache_seconds=time.monotonic()-start
    raw=LMDBImages(rows,dict(c,reconstruction_mode="online_after_augment"))
    images=torch.stack([raw[i][0] for i in range(4)]).to(device)
    expected=probe(images).float().cpu().numpy()
    actual=np.load(area/"缓存/重构张量.npy",mmap_mode="r")[:4]
    error=float(np.max(np.abs(expected-actual)))
    if error > 1e-5:
        raise ValueError("缓存重构重复差超过1e-5")
    raw.close();del probe,images,expected,actual
    torch.cuda.empty_cache()
    c["reconstruction_caches"]={variant:str(area/"缓存")}
    seed_all(BATCH["seed"])
    model=build_model(c,device)
    trainable=[p for p in model.parameters() if p.requires_grad]
    initial=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad: initial.update(name.encode());initial.update(p.detach().cpu().numpy().tobytes())
    opt=c["optimizer"]
    optimizer=torch.optim.Adam(trainable,lr=opt["lr"],betas=tuple(opt["betas"]),eps=opt["eps"],weight_decay=opt["weight_decay"])
    loader=make_loader(rows,c,True,BATCH["seed"],device)
    torch.cuda.reset_peak_memory_stats(device)
    times=[];trace=[];gradient=None
    model.train()
    for epoch in range(2):
        for batch in loader:
            torch.cuda.synchronize(device);started=time.monotonic()
            images,labels,indices,views=move_batch(batch,device)
            trace.append(dict(indices=indices.tolist(),image_sha=hashlib.sha256(images.cpu().numpy().tobytes()).hexdigest()))
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(images,views)["logits"],labels)
            if not torch.isfinite(loss): raise ValueError("预检loss非有限")
            loss.backward()
            gradient=float(torch.nn.utils.clip_grad_norm_(trainable,float("inf"),error_if_nonfinite=True))
            if not any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.encoder.parameters() if p.requires_grad):
                raise ValueError("LoRA无非零梯度")
            optimizer.step();torch.cuda.synchronize(device);times.append(time.monotonic()-started)
    loader.dataset.close();optimizer.zero_grad(set_to_none=True)
    test_loader=make_loader(rows,c,False,BATCH["seed"],device)
    batch=next(iter(test_loader));images,labels,indices,views=move_batch(batch,device)
    model.eval();inference=[]
    with torch.inference_mode():
        for _ in range(4):
            torch.cuda.synchronize(device);started=time.monotonic();prediction=model(images,views)["logits"]
            torch.cuda.synchronize(device);inference.append(time.monotonic()-started)
        expected=prediction.cpu().clone()
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},area/"临时重载权重.pth")
    model.load_state_dict(torch.load(area/"临时重载权重.pth",map_location="cpu",weights_only=True),strict=True)
    with torch.inference_mode():
        restored=model(images,views)["logits"].cpu()
    reload_error=float((expected-restored).abs().max())
    if not torch.isfinite(restored).all() or reload_error>1e-6: raise ValueError("strict重载预测不一致")
    test_loader.dataset.close()
    peak=torch.cuda.max_memory_reserved(device)
    required=int(peak+max(peak*.2,2*1024**3))
    train_time=float(np.median(times[2:]));eval_time=float(np.median(inference[1:]))
    eval_frames=sum(len(m["rows"]) for m in items[2:])
    projected=(BATCH["epochs"]*math.ceil(len(items[0]["rows"])/16)*train_time+(BATCH["epochs"]*len(items[1]["rows"])+2*eval_frames)/64*eval_time)*1.2+600
    cache_projected=cache_seconds/len(rows)*len(all_rows(items))*1.2
    result=dict(status="passed",time=now(),variant=variant,device=device_name,code_commit=git("rev-parse","HEAD"),
        initial_free_bytes=initial_free,peak_reserved_bytes=peak,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        required_free_bytes=required,memory_pass=initial_free>=required,updates=len(times),formal_updates=0,gradient_norm=gradient,
        train_step_median_seconds=train_time,eval64_median_seconds=eval_time,formal_projected_seconds=projected,
        cache_projected_seconds=cache_projected,budget_pass=projected<=BATCH["formal_max_seconds"] and cache_projected<=BATCH["cache_max_seconds"],
        cache_repeat_max_abs=error,reload_max_abs=reload_error,initial_sha256=initial.hexdigest(),trace=trace,
        temporary_checkpoint_sha256=file_sha(area/"临时重载权重.pth"))
    save(area/"验收.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if not result["memory_pass"] or not result["budget_pass"]:
        raise RuntimeError("预检数值通过但显存或有限预算不通过；停止正式启动")

def pipeline(variant, device_name, claim):
    import torch
    from prd_r01.cache import build_cache, operator_spec
    from prd_r01.operators import build_operator
    from prd_r01.engine import run_training
    from prd_r01.config import read_protocol
    torch.set_num_threads(2);device=torch.device(device_name);c=config(variant)
    area=STATE/variant;area.mkdir(parents=True,exist_ok=False)
    status=dict(time=now(),stage="resource_check",variant=variant,pid=os.getpid(),cwd=str(ROOT),device=device_name,local_claim_commit=claim,code_commit=git("rev-parse","HEAD"))
    save(area/"status.json",status)
    try:
        check=json.loads((STATE/(variant+"-smoke")/"验收.json").read_text())
        if not check["memory_pass"] or not check["budget_pass"]: raise ValueError("缺少通过的真实batch验收")
        if check["code_commit"] != status["code_commit"]: raise ValueError("预检后代码版本改变")
        if device_free(device)<check["required_free_bytes"]: raise RuntimeError("可用显存低于预检峰值加余量")
        if git("status","--porcelain"): raise ValueError("运行工作树必须已提交且干净")
        items=manifests();protocol=read_protocol(INPUTS/(variant+".json"))
        rows=all_rows(items);needed=len(rows)*3*224**2*4
        if shutil.disk_usage(ROOT).free < needed+BATCH["minimum_disk_free_bytes"]: raise OSError("缓存生成前磁盘余量不足")
        status.update(stage="cache",time=now());save(area/"status.json",status)
        probe=build_operator(c,variant,device)
        cache=build_cache(rows,c,probe,protocol["reconstruction_caches"][variant],operator_spec(c,variant),device,BATCH["cache_batch_size"],BATCH["cache_max_seconds"])
        del probe;torch.cuda.empty_cache()
        if device_free(device)<check["required_free_bytes"]: raise RuntimeError("缓存后资源变化，保留完整缓存并停止，不自动等待或重试")
        run_id="4090-"+NAMES[variant]+"-20261029-种子3407"
        status.update(stage="training",time=now(),run_id=run_id,cache_seconds=cache["generation_seconds"]);save(area/"status.json",status)
        result=run_training(c,protocol,run_id,device)
        status.update(stage="compute_finished",time=now(),run_id=run_id,result=result);save(area/"status.json",status)
        return 0
    except BaseException as exc:
        status.update(stage="failed",time=now(),error=repr(exc));save(area/"status.json",status)
        raise

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","smoke","run"])
    parser.add_argument("--variant",choices=BATCH["variants"])
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--claim-commit")
    args=parser.parse_args()
    if args.stage=="prepare": return prepare()
    if not args.variant: raise ValueError("必须明确算子")
    if args.stage=="smoke": return smoke(args.variant,args.device)
    if not args.claim_commit or len(args.claim_commit)!=40: raise ValueError("必须记录本地认领提交")
    return pipeline(args.variant,args.device,args.claim_commit)

if __name__ == "__main__":
    main()
