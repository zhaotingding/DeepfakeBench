"""末轮第八域接续。仅在完整资产正式发布、四域队列退出后消费 DF-1.0。"""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
RUNROOT = Path.home() / "prd-runs/末轮八域补评估-20261104"
STATE = RUNROOT / "queue-df1"
FOUR = RUNROOT / "queue-four"
ASSET = Path.home() / "disk/zhaoting.ding/prd-evaluation-assets/DF1-end-to-end-20261104/asset-full-v1"
DOMAIN = "DeeperForensics-1.0"
FROZEN = "16da0bf27401c4f90f260a35a16455b25bb1d1af"
CANDIDATE = "96482b49417de3299c4a1cef2757eac38e07484cc4d48c41b24d2d001d2fb321"
BASE_CAP = 12 * 3600
MAX_ROW = 96 * 3600
MAX_GPU_SECONDS = 1100 * 3600  # 防失控上界，不赋予支出权限。
POLL = 30


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8",delete=False) as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name,path)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8 << 20),b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def code_for(path):
    name=path.relative_to(ROOT).as_posix()
    cmd=["git","-C",str(ROOT)]
    commit=subprocess.check_output(cmd+["log","-1","--format=%H","--",name],text=True).strip()
    if subprocess.check_output(cmd+["show",f"{commit}:{name}"])!=path.read_bytes():
        raise RuntimeError("启动脚本与Git不同："+name)
    return commit


def frozen():
    path=ROOT/"scripts/八域末轮评估.py"
    if code_for(path)!=FROZEN:
        raise RuntimeError("共享推理入口版本改变，拒绝接续")
    spec=importlib.util.spec_from_file_location("eight_terminal",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def lock(path, create=True):
    if create:path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+" if create else "r") as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def hosting_pid_alive(pid):
    cmdline=Path("/proc",str(pid),"cmdline")
    try:
        cmd=cmdline.read_bytes()
    except FileNotFoundError:
        return False
    return any(name.encode() in cmd for name in ("八域末轮评估.py","八域四域队列.py"))


def gate_four(claim):
    p=FOUR/"status.json"
    if not p.exists():return None,"四域队列未登记"
    q=read(p)
    if q.get("claim")!=claim or q.get("code")!=FROZEN:raise ValueError("四域认领或代码不符")
    if q.get("stage") not in ("finished","finished_with_gaps") or not q.get("ended"):
        return None,"四域队列未终态"
    if hosting_pid_alive(q["pid"]) or any(hosting_pid_alive(v["pid"]) for v in q["rows"].values() if v.get("pid")):
        return None,"四域监督或计算进程尚未退出"
    try:
        with lock(FOUR/"supervisor.lock",False):pass
    except BlockingIOError:
        return None,"四域监督锁仍占用"
    return q,None


def file_stat(f):
    s=f.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]


def validate_asset():
    vr=ASSET/"verification.json"
    pub=ASSET/"publication.json"
    if not vr.is_file() or not pub.is_file():return None,"DF1全量验收或发布回执尚未产生"
    formal=Path.home()/"prd-inputs/df1-precheck-20261104"
    released=formal/"formal_exit.json"
    if not released.is_file():return None,"DF1资产生产监督尚未退出"
    end=read(released)
    if end.get("exit")!=0:raise ValueError("DF1资产生产监督退出码异常")
    try:
        with lock(formal/"asset-full-v1.lock",False):pass
    except BlockingIOError:return None,"DF1生产仍持有CPU资产锁"
    v,p=read(vr),read(pub)
    if not (v.get("status")=="verified" and v.get("sample_only") is False and
            v.get("videos")==48676 and v.get("candidate_manifest_sha256")==CANDIDATE and
            v.get("input_or_decode_errors")==0 and v.get("unaccounted_requested_frames")==0 and
            v.get("frames",0)>0 and v.get("zero_video_count")==len(v["zero_videos"]) and
            p.get("status")=="published"):
        raise ValueError("DF1全量验收或发布内容不完整")
    manifest=RUNROOT/"manifests"/(DOMAIN+".json")
    db=Path.home()/"local_datasets/DeepfakeBench/lmdb"/(DOMAIN+"_lmdb")/"data.mdb"
    target=Path(p["target_lmdb"])
    if target.is_dir():target=target/"data.mdb"
    if manifest.resolve()!=Path(p["manifest_path"]).resolve() or not db.is_file() or db.resolve()!=target.resolve():
        raise ValueError("DF1消费入口与发布资产不一致")
    if sha(manifest)!=p["manifest_sha256"] or not p["lmdb_sha256"] or sha(db)!=p["lmdb_sha256"]:
        raise ValueError("DF1发布清单或LMDB字节发生变更")
    m=frozen().new_manifest(DOMAIN,Path.home()/"local_datasets")
    groups={(r["label"],r["video_id"]) for r in m["rows"]}
    if len(m["rows"])!=v["frames"] or len(groups)+v["zero_video_count"]!=48676:
        raise ValueError("DF1正式清单与全量视频/帧覆盖不一致")
    identity={"verified_at":now(),"manifest":str(manifest),"manifest_sha256":p["manifest_sha256"],
              "lmdb":str(db),"lmdb_sha256":p["lmdb_sha256"],"lmdb_stat":file_stat(db),
              "verification_sha256":sha(vr),"publication_sha256":sha(pub),"generator_exit_sha256":sha(released),"frames":v["frames"],
              "videos_with_frames":len(groups),"zero_video_count":v["zero_video_count"]}
    return identity,None


def seven(module,run,spec,verify_hashes,allow_prior_complete=False):
    base=RUNROOT/"results"/run
    source=read(base/"identity.json")
    summary=read(base/"summary.json")
    if (source.get("train_code")!=spec["code"] or source.get("eval_code")!=FROZEN or
            source.get("old_artifacts")!=spec["artifacts"] or source.get("endpoint")!="terminal"
            or source.get("verification")!=spec["verification_sha"]):
        raise ValueError("初次四域身份不符："+run)
    keys=set((*module.BASE,*module.NEW))
    partial=summary["status"]=="partial" and summary["macro8"] is None and set(summary["domains"])==keys
    recovered=allow_prior_complete and summary["status"]=="complete" and set(summary["domains"])==set(module.ALL)
    if not (partial or recovered):
        raise ValueError("缺已验收的完整七域："+run)
    for domain in module.NEW:
        area=base/domain
        rec=read(area/"verified.json")
        if rec["status"]!="verified" or rec["manifest"]!=sha(RUNROOT/"manifests"/(domain+".json")):
            raise ValueError("原四域清单标识不符")
        if verify_hashes and sha(area/"terminal.npy")!=rec["terminal_logits_sha256"]:
            raise ValueError("原四域预测哈希不符")
        if summary["domains"][domain]!=rec["metrics"]:
            raise ValueError("四域汇总值与逐域归档不一致")
    return {d:summary["domains"][d] for d in (*module.BASE,*module.NEW)}


def estimate(module,run,spec,frames):
    seven(module,run,spec,True)
    records=[]
    for domain in module.NEW:
        area=RUNROOT/"results"/run/domain
        for p in area.glob("[0-9]*.json"):
            c=read(p)
            if c["count"]>0 and c["seconds"]>0:records.append((c["count"],c["seconds"]))
    n=sum(n for n,s in records)
    if n!=117734:raise ValueError("缺四域实测分块")
    rate=sum(s for n,s in records)/n
    forecast=rate*frames
    return {"four_frames":n,"four_measured_seconds":sum(s for n,s in records),
            "seconds_per_frame":rate,"projected_gpu_hours":forecast/3600,
            "budget_with_50pct_and_setup_seconds":round(1.5*forecast+300)}


def check_asset_identity(a):
    if file_stat(Path(a["lmdb"]))!=a["lmdb_stat"] or sha(Path(a["manifest"]))!=a["manifest_sha256"]:
        raise ValueError("DF1资产在验收后发生变化")
    for filename,key in (("verification.json","verification_sha256"),("publication.json","publication_sha256")):
        if sha(ASSET/filename)!=a[key]:
            raise ValueError("DF1发布回执变化")


def work(run):
    resident = read(STATE/"status.json")
    if (os.getppid()!=resident["pid"] or resident.get("four_claim") is None or
            not (STATE/"cost_report.json").is_file()):
        raise RuntimeError("只允许经现存获准监督启动该分项")
    four, why=gate_four(resident["four_claim"])
    if four is None or four["rows"][run]["stage"]!="verified_new_four":
        raise RuntimeError("四域checkpoint或释放门失效："+str(why))
    import numpy as np
    import torch
    mod=frozen()
    mod.torch=torch
    assets=read(STATE/"assets.json")
    check_asset_identity(assets)
    catalog=mod.discover()
    spec=catalog[run]
    base=RUNROOT/"results"/run
    with lock(RUNROOT/(run+".lock")):
        old=seven(mod,run,spec,True,allow_prior_complete=True)
        out=mod.OUTPUT/run
        if sha(out/"terminal.pth")!=spec["artifacts"]["terminal.pth"] or sha(out/"metadata.json")!=spec["artifacts"]["metadata.json"]:
            raise ValueError("历史原末轮权重/配置改变")
        if sha(Path(spec["verification"]))!=spec["verification_sha"]:
            raise ValueError("原模型独立验收记录变化")
        cfg=read(out/"metadata.json")["config"]
        m=mod.new_manifest(DOMAIN,cfg["data_root"])
        archived=base/DOMAIN/"verified.json"
        if archived.is_file():
            rec=read(archived)
            if rec.get("status")!="verified" or sha(base/DOMAIN/"terminal.npy")!=rec["terminal_logits_sha256"]:
                raise ValueError("DF1归档故障；保留原预测，拒绝GPU重算")
            identity=read(base/DOMAIN/"identity.json")
            if identity["checkpoint"]!=spec["artifacts"]["terminal.pth"] or identity["code"]!=FROZEN:
                raise ValueError("DF1分项身份不符")
        else:
            with tempfile.TemporaryDirectory(prefix="prd-df1-source-") as temp:
                mod.snapshot(spec["code"],Path(temp))
                from prd_r01.engine import seed_all
                from prd_r01.model import build_model
                if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info()[0]<12*1024**3:
                    raise RuntimeError("分配的GPU数或空闲显存不足")
                seed_all(3407)
                device=torch.device("cuda:0")
                model=build_model(cfg,device)
                model.load_state_dict(torch.load(out/"terminal.pth",map_location="cpu",weights_only=True),strict=True)
                model.requires_grad_(False).eval()
                probes=mod.build_views(cfg,device)
                rec=mod.evaluate_domain(base,model,probes,cfg,m,device,spec["artifacts"]["terminal.pth"])
        check_asset_identity(assets)
        logits=np.load(base/DOMAIN/"terminal.npy",allow_pickle=False)
        if (rec["manifest"]!=assets["manifest_sha256"] or len(logits)!=assets["frames"]
                or rec["metrics"]["videos"]!=assets["videos_with_frames"]):
            raise ValueError("DF1完整覆盖核验失败")
        # 与主实现独立的平均秩AUC，按同一帧FP32概率、同视频FP64平均。
        from scipy.stats import rankdata
        def ranks(y,p):
            y=np.asarray(y,dtype=int);npos=int(y.sum());nneg=len(y)-npos
            return float((rankdata(p)[y==1].sum()-npos*(npos+1)/2)/(npos*nneg))
        prob=torch.softmax(torch.from_numpy(logits),dim=1)[:,1].double().numpy()
        groups={}
        for r,z in zip(m["rows"],prob):
            groups.setdefault(r["video_id"],(r["label"],[]))[1].append(z)
        frame_check=ranks([r["label"] for r in m["rows"]],prob)
        video_check=ranks([a for a,b in groups.values()],[np.mean(b) for a,b in groups.values()])
        if max(abs(rec["metrics"]["frame_auc"]-frame_check),abs(rec["metrics"]["video_auc"]-video_check))>1e-10:
            raise ValueError("DF1独立AUC校验失败")
        domains=dict(old);domains[DOMAIN]=rec["metrics"]
        macro={k:float(np.mean([domains[d][k] for d in mod.ALL])) for k in ("frame_auc","video_auc","frame_ce","video_ce")}
        save(base/"summary.json",{"run_id":run,"status":"complete","domains":domains,"macro8":macro,
                                  "new_training_updates":0,"df1_asset":assets["manifest_sha256"]})
        save(STATE/(run+"-verification.json"),{"status":"verified","run":run,"old_domains_forwarded":0,
            "df1_frame_auc_rankcheck":frame_check,"df1_video_auc_rankcheck":video_check,
            "summary_sha256":sha(base/"summary.json"),"verified_at":now()})


def approved_limits(run,est,cost_sha):
    want=est["budget_with_50pct_and_setup_seconds"]
    default=max(want,600) if want<=BASE_CAP else None
    p=STATE/"cost_release.json"
    if not p.is_file():return default
    j=read(p);row=j.get("runs",{}).get(run,{})
    if j.get("approved_cost_report_sha256")!=cost_sha or not j.get("authorization_reference"):
        return default
    limit=row.get("max_gpu_seconds")
    return limit if isinstance(limit,int) and 0<limit<=MAX_ROW else default


def stop(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def usage(row,t):
    return sum(seg["seconds"] if "seconds" in seg else max(0,t-seg["start_epoch"]) for seg in row.get("segments",[]))


def chunk_forecast(run,frames,spent):
    area=RUNROOT/"results"/run/DOMAIN
    receipts=sorted(area.glob("[0-9]*.json"))
    if len(receipts)<2:return None
    last=[read(p) for p in receipts[-4:]]
    processed=int(last[-1]["start"])+int(last[-1]["count"])
    recent_rate=sum(x["seconds"] for x in last)/sum(x["count"] for x in last)
    return {"frames":processed,"recent_seconds_per_frame":recent_rate,
            "estimated_total_seconds":spent+(frames-processed)*recent_rate}


def run_gpu_stage(record):
    module=frozen();a=read(STATE/"assets.json");cat=module.discover()
    cost=read(STATE/"cost_report.json");cost_sha=sha(STATE/"cost_report.json")
    usable=sorted(os.sched_getaffinity(0))[:12]
    if len(usable)<2:raise RuntimeError("不足两个CPU核，拒绝双槽")
    cpu=[usable[::2],usable[1::2]]
    active={}
    record["GPU_stage_started"]=now()
    deadline=time.time()+30*86400
    try:
        while True:
            t=time.time()
            for gpu in list(active):
                p,glock,log,r=active[gpu]; row=record["rows"][r]
                seg=row["segments"][-1];rc=p.poll()
                if rc is None:
                    spent=usage(row,t); elapsed=t-seg["start_epoch"]
                    observed=chunk_forecast(r,a["frames"],spent)
                    need_more=observed is not None and observed["estimated_total_seconds"]>row["cap"]*1.10
                    if spent>=row["cap"] or elapsed>=BASE_CAP or t>=deadline or need_more:
                        stop(p);rc=p.returncode
                        if t>=deadline:row["stage"]="safety_limit"
                        elif spent>=row["cap"] or need_more:
                            row["stage"]="waiting_extension"
                            row["previous_cap"]=row["cap"]
                            row["observed_cost_exception"]=observed
                            save(STATE/(r+"-budget-exception.json"),{"time":now(),"cap_seconds":row["cap"],
                                 "actual_seconds":spent,"forecast":observed,"cost_report_sha256":cost_sha})
                        else:row["stage"]="slice_complete"
                    else:
                        continue
                elif rc==0 and (STATE/(r+"-verification.json")).exists():
                    row["stage"]="verified"
                else:
                    row["stage"]="failed"
                seg.update(seconds=max(0,time.time()-seg["start_epoch"]),exit=rc,end=now())
                log.close();glock.close();del active[gpu]
            total=sum(usage(row,t) for row in record["rows"].values())
            record["actual_gpu_seconds"]=round(total)
            if t>=deadline or total>=MAX_GPU_SECONDS:
                record["stage"]="safety_limit";break
            for gpu in (0,1):
                if gpu in active:continue
                choices=[]
                for run in cat:
                    row=record["rows"][run]
                    if row["stage"] not in ("queued","waiting_cost","slice_complete","waiting_gpu","waiting_extension"):continue
                    limits=approved_limits(run,cost["runs"][run],cost_sha)
                    if limits is None:
                        row["stage"]="waiting_cost";continue
                    if row["stage"]=="waiting_extension" and limits<=row.get("previous_cap",0):
                        continue
                    if usage(row,t)>=limits:
                        row["stage"]="waiting_extension";row["previous_cap"]=limits;continue
                    choices.append((cost["runs"][run]["projected_gpu_hours"],run,limits))
                if not choices:continue
                _,r,cap=min(choices);row=record["rows"][r]
                gpu_path=Path("/tmp")/("prd-"+str(os.getuid()))/f"eight-terminal-gpu{gpu}.lock"
                gpu_path.parent.mkdir(exist_ok=True,parents=True)
                glock=gpu_path.open("a+")
                try:
                    fcntl.flock(glock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    glock.close();row["stage"]="waiting_gpu";continue
                free=int(subprocess.check_output(["nvidia-smi","-i",str(gpu),"--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
                if free<12288:
                    glock.close();row.update(stage="waiting_gpu",free_mib=free);continue
                check_asset_identity(a)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONUTF8="1", HF_HUB_OFFLINE="1",
                         OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",TOKENIZERS_PARALLELISM="false")
                log=(STATE/(r+".log")).open("a",encoding="utf-8")
                seq=len(row["segments"])+1
                cmd=["taskset","-c",",".join(map(str,cpu[gpu])),sys.executable,"-u",str(Path(__file__).resolve()),"worker",r]
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row.update(stage="running",cap=cap,gpu=gpu)
                row["segments"].append({"sequence":seq,"pid":p.pid,"start_epoch":time.time(),"free_mib":free})
                record["gpu_started"]=True
                active[gpu]=(p,glock,log,r)
            record["stage"]="running" if active else "waiting_cost_or_resources"
            save(STATE/"status.json",record)
            if not active and all(row["stage"] in ("verified","failed","budget_blocked","unavailable_seven") for row in record["rows"].values()):
                record["stage"]="finished" if all(r["stage"]=="verified" for r in record["rows"].values()) else "finished_with_gaps"
                break
            time.sleep(POLL)
    finally:
        for p,glock,log,r in active.values():
            stop(p);log.close();glock.close()
            row=record["rows"][r];seg=row["segments"][-1]
            seg.update(seconds=max(0,time.time()-seg["start_epoch"]),exit=p.returncode,end=now())
            row["stage"]="interrupted_by_supervisor"
        record["actual_gpu_seconds"]=round(sum(usage(r,time.time()) for r in record["rows"].values()))
        record["ended"]=now();save(STATE/"status.json",record)


def supervisor(claim):
    STATE.mkdir(parents=True,exist_ok=True)
    with lock(STATE/"supervisor.lock"):
        if (STATE/"status.json").exists():
            raise RuntimeError("已有接续状态，禁止自动覆盖/重启")
        rec={"pid":os.getpid(),"created":now(),"four_claim":claim,"supervisor_code":code_for(Path(__file__).resolve()),
             "inference_code":FROZEN,"stage":"waiting","gpu_started":False,"rows":{}}
        save(STATE/"status.json",rec)
        try:
            while True:
                q,reason=gate_four(claim)
                if q is not None:
                    a,reason=validate_asset()
                    if a is not None:break
                rec.update(stage="waiting",wait_reason=reason,checked=now(),
                           has_df1_full_publication=(ASSET/"publication.json").is_file())
                save(STATE/"status.json",rec);time.sleep(POLL)
            save(STATE/"assets.json",a)
            module=frozen();cat=module.discover()
            if set(cat)!=set(q["rows"]) or len(cat)!=23:raise ValueError("运行池身份不符")
            estimates={};eligible=[]
            for r,spec in cat.items():
                if q["rows"][r]["stage"]!="verified_new_four":
                    rec["rows"][r]={"stage":"unavailable_seven","segments":[]};continue
                try:estimates[r]=estimate(module,r,spec,a["frames"])
                except (ValueError,KeyError,OSError) as e:
                    rec["rows"][r]={"stage":"unavailable_seven","error":str(e),"segments":[]};continue
                eligible.append(r)
                rec["rows"][r]={"stage":"queued","segments":[]}
            hours=sum(v["projected_gpu_hours"] for v in estimates.values())
            report={"time":now(),"frames":a["frames"],"runs":estimates,"unavailable":sorted(set(cat)-set(eligible)),
                    "projected_total_gpu_hours":hours,"ideal_two_gpu_days":hours/48,
                    "multiweek_warning":hours/48>=14,"usual_per_run_seconds":BASE_CAP,
                    "high_cost_wait_for_explicit_release":[r for r,v in estimates.items() if v["budget_with_50pct_and_setup_seconds"]>BASE_CAP],
                    "safety_upper_bound_only_gpu_hours":1100}
            save(STATE/"cost_report.json",report)
            rec.update(stage="cost_estimates_published",checked=now(),cost_report_sha256=sha(STATE/"cost_report.json"))
            save(STATE/"status.json",rec);run_gpu_stage(rec)
        except BaseException as e:
            rec.update(stage="supervision_error",ended=now(),error=repr(e))
            save(STATE/"status.json",rec);raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["wait","worker"])
    parser.add_argument("argument")
    args=parser.parse_args()
    if args.mode=="worker":
        work(args.argument)
    else:
        def terminate(signum,frame):
            raise KeyboardInterrupt("监督收到终止信号")
        signal.signal(signal.SIGTERM,terminate)
        supervisor(args.argument)
