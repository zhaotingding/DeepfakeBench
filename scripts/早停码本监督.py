#!/usr/bin/env python3
"""有符号码本重构的单项有限早停监督与独立归档。"""
import os
os.environ.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",PYTHONUTF8="1")
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"training"))
import numpy as np
import torch
import prd_fresh as fresh


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def verify(arm,state,code):
    out=Path.home()/"prd-runs/早停码本重构-20261107"/arm["run_id"]
    meta=read(out/"metadata.json");result=read(out/"training_result.json")
    final=read(out/"complete_result.json");curve=read(out/"learning_curve.json")
    assert meta["code_commit"]==code and final["status"]=="verified"
    assert meta["config"]["variant"]=="vqgan" and meta["config"]["residual"]=="signed_diff"
    assert meta["config"]["reconstruction_mode"]=="online_after_augment"
    assert final["new_updates_during_evaluation"]==0
    for path,digest in meta["identity"]["base_weights"].items():
        assert fresh.file_sha(path)==digest  # 紧凑权重依赖的底座必须仍匹配。
    assert meta["stopping"]==dict(metric="source_val_frame_auc",patience=5,max_epochs=30,min_delta=0,tie="earliest")
    assert result["training_complete"] and 1<=len(curve)<=30
    validation=read(out/"val_manifest.json")["rows"]
    progress={"best_auc":-1.0,"best_epoch":None,"bad_epochs":0}
    for n,row in enumerate(curve,1):
        assert row["epoch"]==n and progress["bad_epochs"]<5
        values=np.load(out/f"source_val_epoch{n:02d}.npy",allow_pickle=False)
        metrics=fresh.metric(values,validation)
        assert abs(metrics["frame_auc"]-row["validation"]["frame_auc"])<=1e-12
        improved=fresh.update_best(progress,metrics["frame_auc"],n)
        assert row["improved"]==improved and row["bad_epochs"]==progress["bad_epochs"]
    assert result["best_epoch"]==progress["best_epoch"] and result["best_auc"]==progress["best_auc"]
    assert result["stop_reason"]==("early_stopping_patience_5" if progress["bad_epochs"]>=5 else "maximum_epochs_30")
    if progress["bad_epochs"]<5:assert len(curve)==30
    train=read(out/"train_manifest.json")["rows"]
    expected_steps=(len(train)+31)//32*len(curve)
    assert result["updates"]==expected_steps
    with (out/"batch_trace.jsonl").open() as stream:
        for count,line in enumerate(stream,1):
            row=json.loads(line);assert row["step"]==count
        assert count==expected_steps
    baseline=Path.home()/"prd-runs/早停基线-20261105/4090-经典PRD原图加VAE-源验证早停-20261105-种子3407"
    paired=0
    with (out/"batch_trace.jsonl").open() as this_trace,(baseline/"batch_trace.jsonl").open() as baseline_trace:
        for current,reference in zip(this_trace,baseline_trace):
            assert json.loads(current)==json.loads(reference), "共同训练区间输入与经典PRD不匹配"
            paired+=1
    assert paired==min(expected_steps,read(baseline/"training_result.json")["updates"])
    init=read(out/"initialization.json")
    assert init["trainable_initial_sha256"]==read(baseline/"initialization.json")["trainable_initial_sha256"]
    for name,digest in meta["identity"]["manifests"].items():
        assert digest==read(baseline/"metadata.json")["identity"]["manifests"][name]  # 所有源和目标清单保持首批身份。
    best=torch.load(out/"best.pt",map_location="cpu",weights_only=True)
    assert best["epoch"]==progress["best_epoch"] and best["identity"]==meta["identity"]
    assert all(torch.isfinite(v).all() for v in best["model"].values())
    resume=torch.load(out/"last_resume.pt",map_location="cpu",weights_only=False)
    assert resume["progress"]==result and resume["identity"]==meta["identity"]
    assert {"python","numpy","torch","cuda"}==set(resume["rng"]) and resume["optimizer"]["state"]
    hashes={name:fresh.file_sha(out/name) for name in ["best.pt","last_resume.pt","metadata.json","training_result.json","complete_result.json","batch_trace.jsonl","learning_curve.json","initialization.json"]}
    checked={}
    for domain,entry in final["domains"].items():
        manifest=out/(domain+"_manifest.json");prediction=out/(domain+"_predictions.npy")
        assert fresh.file_sha(manifest)==entry["manifest_sha256"] and fresh.file_sha(prediction)==entry["prediction_sha256"]
        assert entry["best_sha256"]==hashes["best.pt"]
        metrics=fresh.metric(np.load(prediction,allow_pickle=False),read(manifest)["rows"])
        for key in ["frame_auc","video_auc","frame_ce","video_ce"]:
            assert abs(metrics[key]-entry["metrics"][key])<=1e-12
        checked[domain]=entry
    assert set(checked)=={"source_test",*fresh.MAIN_DOMAINS,*fresh.EXTRA_DOMAINS}
    for key,value in final["macro"].items():
        assert abs(value-float(np.mean([checked[d]["metrics"][key] for d in fresh.MAIN_DOMAINS])))<=1e-12
    record=dict(status="verified",time=fresh.now(),run_id=arm["run_id"],code=code,
                updates=expected_steps,epochs=len(curve),best_epoch=progress["best_epoch"],
                stop_reason=result["stop_reason"],artifact_sha256=hashes,domains=checked,
                macro=final["macro"],initialization=read(out/"initialization.json"),
                training_elapsed=result["elapsed_seconds"],resume_scope="training_only",paired_common_updates=paired)
    fresh.save_json(state/"verification.json",record)
    return record


def execute(plan_file,claim):
    plan=read(plan_file);code=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    if len(claim)!=40 or plan["code_commit"]!=code:
        raise ValueError("认领或代码版本错误")
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        raise ValueError("工作树未提交")
    if len(plan["arms"])!=1 or plan["arms"][0]["variant"]!="vqgan" or plan["arms"][0]["gpu"] not in (0,1):
        raise ValueError("本监督只接受已批准的有符号码本重构单项")
    state=ROOT/".state/早停码本监督-20261107";state.mkdir(parents=True,exist_ok=True)
    lock=(state/"supervisor.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (state/"supervisor.json").exists():raise RuntimeError("已有状态，拒绝重复派发")
    snap=dict(status="running",time=fresh.now(),pid=os.getpid(),cwd=str(ROOT),code_commit=code,claim_commit=claim,arms={})
    write_lock=threading.Lock()
    def update(name,data):
        with write_lock:
            snap["arms"][name]=data
            fresh.save_json(state/"supervisor.json",snap)
    fresh.save_json(state/"supervisor.json",snap)
    def worker(arm):
        name=arm["variant"];entry=dict(status="preparing",gpu=arm["gpu"],run_id=arm["run_id"])
        update(name,entry)
        try:
            smoke=read(ROOT/".state/早停码本重构-20261107"/name/"smoke/preflight.json")
            assert smoke["status"]=="passed" and smoke["code_commit"]==code
            free=int(subprocess.check_output(["nvidia-smi","-i",str(arm["gpu"]),"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())*1024**2
            required=int(smoke["peak_reserved_bytes"]+2*1024**3)  # 与计算入口一致，真实 BS32/64 峰值之外固定保留 2GiB。
            if free<required:raise RuntimeError("当前空闲显存小于真实预检峰值加余量")
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(arm["gpu"]),CUDA_DEVICE_ORDER="PCI_BUS_ID")
            cmd=[sys.executable,"-u",str(ROOT/"scripts/早停基线.py"),name,"run","--claim-commit",claim,
                 "--run-id",arm["run_id"],"--train-hours",str(arm["train_hours"]),"--eval-hours",str(arm["eval_hours"])]
            with (state/(name+".log")).open("x",encoding="utf8") as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                entry.update(status="running",compute_pid=process.pid,started=fresh.now(),required_free_bytes=required)
                update(name,entry)
                try:
                    rc=process.wait(timeout=(arm["train_hours"]+arm["eval_hours"])*3600+1800)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=60)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise TimeoutError("本臂外层保护时限耗尽")
            entry.update(exit_code=rc,ended=fresh.now())
            if rc:raise RuntimeError(f"计算退出{rc}")
            proof=verify(arm,ROOT/".state/早停码本重构-20261107"/name,code)
            entry.update(status="verified",verification=proof)
        except BaseException as error:
            entry.update(status="failed",ended=fresh.now(),error=repr(error))
        update(name,entry)
        return entry
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        list(pool.map(worker,plan["arms"]))
    successful=[v for v in snap["arms"].values() if v["status"]=="verified"]
    snap["status"]="verified" if len(successful)==1 else "finished_with_errors"
    snap["time"]=fresh.now()
    archive=ROOT/"scripts/experiment_summaries/早停码本重构-20261107"
    archive.mkdir(parents=True,exist_ok=True)
    for name,value in snap["arms"].items():
        label="VQGAN无辅助监督码本重构"
        fresh.save_json(archive/(label+".json"),value)
    fresh.save_json(archive/"批次.json",snap)
    rel=str(archive.relative_to(ROOT))
    subprocess.run(["git","add","--",rel],cwd=ROOT,check=True)
    subprocess.run(["git","commit","--only","-m","归档有符号码本重构早停结果","--",rel],cwd=ROOT,check=True)
    snap["result_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    (state/"result_commit.txt").write_text(snap["result_commit"]+"\n",encoding="utf8")
    fresh.save_json(state/"supervisor.json",snap)
    return 0 if snap["status"]=="verified" else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan",type=Path);parser.add_argument("claim")
    args=parser.parse_args()
    torch.set_num_threads(1)
    sys.exit(execute(args.plan,args.claim))

