"""固定R01末轮，在独立目录补测新域。原三域仅用已验收的logits，禁止训练。"""
import argparse
import fcntl
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKTREES = ROOT.parent
OUTPUT = Path.home() / "prd-runs/r01"
STATE = Path.home() / "prd-runs/末轮八域补评估-20261104"
BASE = ("Celeb-DF-v2", "DFDCP", "DFDC")
NEW = ("Celeb-DF-v1", "DeepFakeDetection", "FaceShifter", "UADFV")
DF1 = "DeeperForensics-1.0"
ALL = ("Celeb-DF-v1", "Celeb-DF-v2", DF1, "DeepFakeDetection", "DFDC", "DFDCP", "FaceShifter", "UADFV")
PHYSICAL = {"FaceShifter": "FaceForensics++", "DeepFakeDetection": "FaceForensics++"}
LABELS = {
    "Celeb-DF-v1": {"CelebDFv1_real": 0, "CelebDFv1_fake": 1},
    "DeepFakeDetection": {"DFD_real": 0, "DFD_fake": 1},
    "FaceShifter": {"FF-real": 0, "FF-FH": 1},
    "UADFV": {"UADFV_Real": 0, "UADFV_Fake": 1},
}
EXPECTED = {"Celeb-DF-v1": (3136, 100), "DeepFakeDetection": (102541, 3429),
            "FaceShifter": (8958, 280), "UADFV": (3099, 98)}
CHUNK = 4096


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_json(path, data):
    if path.exists():
        if read(path) != data:
            raise ValueError(f"冻结记录与本次输入不同：{path}")
    else:
        save(path, data)


def discover():
    found = {}
    for tree in ("重构算子探索", "重构算子并行", "多响应融合"):
        for v in (WORKTREES / tree / ".state").glob("**/verification.json"):
            record = read(v)
            if record.get("status") != "verified" or not record.get("output"):
                continue
            out = Path(record["output"])
            if out.parent != OUTPUT or record.get("updates") != 21576:
                continue
            spec = {"run": out.name, "code": record["code_commit"], "verification": str(v),
                    "verification_sha": sha(v), "artifacts": {
                        k: record["artifacts_sha256"][k] for k in
                        ("metadata.json", "eval1_manifest.json", "terminal.pth", "terminal_eval1.npy")}}
            if out.name in found and found[out.name] != spec:
                raise ValueError("已验收身份重复：" + out.name)
            found[out.name] = spec
    return found


def new_manifest(domain, data_root):
    if domain == DF1:
        # DF-1.0在独立资产验收后才能加入，显式登记原始扰动、划分、物理库与完整清单。
        p = STATE / "manifests" / (domain + ".json")
        if not p.exists():
            raise FileNotFoundError("DF-1.0未验收")
        m = read(p)
        if not (m.get("domain") == domain and m.get("split") == "test" and m.get("recipe")
                and m.get("source_sha256") and m.get("physical_lmdb") == domain):
            raise ValueError("DF-1.0清单来源/工作点缺失")
        rows = m["rows"]
        keys = {r["key"] for r in rows}
        if (len(keys) != len(rows) or not rows or {r["label"] for r in rows}!={0,1}
                or any(r["domain"] != domain or not r["video_id"] or not r["key"] for r in rows)):
            raise ValueError("DF-1.0帧清单无效")
        groups = {}
        for r in rows:
            groups.setdefault(r["video_id"],[]).append(r["label"])
        if any(len(v)>32 or len(set(v))!=1 for v in groups.values()):
            raise ValueError("DF-1.0视频划分/采样错误")
        return m
    source = Path(data_root) / "DeepfakeBench/config/dataset_json" / (domain + ".json")
    groups = read(source)[domain]
    if set(groups) != set(LABELS[domain]):
        raise ValueError(f"未知类别：{domain}")
    rows = []
    for kind, splits in sorted(groups.items()):
        videos = splits["test"]
        if domain in PHYSICAL:
            videos = videos["c23"]
        for vid, info in sorted(videos.items()):
            frames = sorted(info["frames"])
            if not frames or info["label"] != kind:
                raise ValueError(f"无效视频：{domain}/{kind}/{vid}")
            for i in np.linspace(0, len(frames)-1, min(32,len(frames)), dtype=int):
                rows.append({"domain": domain, "key": frames[i], "label": LABELS[domain][kind],
                             "video_id": kind+"/"+vid})
    counts = (len(rows), len({r["video_id"] for r in rows}))
    if counts != EXPECTED[domain] or len({r["key"] for r in rows}) != len(rows):
        raise ValueError(f"源清单规模/唯一键异常：{domain}/{counts}")
    m = {"domain":domain, "split":"test","physical_lmdb":PHYSICAL.get(domain,domain),
         "source":str(source), "source_sha256":sha(source),"rows":rows}
    frozen_json(STATE/"manifests"/(domain+".json"),m)
    return m


def metrics(logits, rows):
    from sklearn.metrics import roc_auc_score
    if logits.shape != (len(rows),2) or logits.dtype != np.float32 or not np.isfinite(logits).all():
        raise ValueError("预测与完整清单不匹配")
    z = logits.astype(np.float64)
    logp = z - np.logaddexp(z[:,0],z[:,1])[:,None]
    probs = np.exp(logp[:,1])
    y = np.array([r["label"] for r in rows])
    videos = {}
    for row, p in zip(rows, probs):
        label, values = videos.setdefault(row["video_id"],(row["label"],[]))
        if label != row["label"]:
            raise ValueError("同视频标签冲突")
        values.append(p)
    vy = np.array([y for y,v in videos.values()])
    vp = np.array([np.mean(v,dtype=np.float64) for y,v in videos.values()])
    return {"frames":len(rows),"videos":len(videos),"frame_auc":float(roc_auc_score(y,probs)),
            "video_auc":float(roc_auc_score(vy,vp)),"frame_ce":float(-logp[np.arange(len(y)),y].mean()),
            "video_ce":float(-np.log(np.maximum(np.where(vy,vp,1-vp),np.finfo(np.float64).tiny)).mean())}


def snapshot(code, target):
    # 导出原训练版本的模型和依赖；不切换任一工作树。
    paths = ["training/prd_r01", "training/config/r01"]
    for name in ("运行原图VAE对照.py", "运行BM3D单项.py"):
        f = "scripts/"+name
        if subprocess.run(["git","-C",str(ROOT),"cat-file","-e",code+":"+f],
                          stderr=subprocess.DEVNULL).returncode == 0:
            paths.append(f)
    raw = subprocess.check_output(["git","-C",str(ROOT),"archive",code,"--",*paths])
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:") as t:
        for info in t:
            dest = (target/info.name).resolve()
            if target.resolve() not in dest.parents or not (info.isdir() or info.isfile()):
                raise ValueError("快照有意外成员："+info.name)
            if info.isfile():
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(t.extractfile(info).read())
    sys.path[:0] = [str(target/"training"),str(target/"scripts")]
    os.environ["PYTHONPATH"] = os.pathsep.join([str(target/"training"),str(target/"scripts"),
                                                 os.environ.get("PYTHONPATH","")])


def build_views(config, device):
    name = config["variant"]
    if name == "original":
        return {}
    if name == "vae":
        from 运行原图VAE对照 import SingleImageVAE
        op = SingleImageVAE(config["vae_path"], config["mean"], config["std"], torch.float16)
    elif name in ("wavelet","lowrank"):
        from prd_r01.operators import build_operator
        op = build_operator(config,name,device)
    elif name == "bm3d":
        return {}  # CPU worker随原实现产生views。
    elif name == "vqgan_sr_mean":
        from prd_r01.generative_operators import GenerativeProbe
        from prd_r01.vqgan_operator import VQGANProbe
        return {"vqgan":VQGANProbe(config).to(device).eval(),
                "superresolution":GenerativeProbe(config,"superresolution").to(device).eval()}
    elif name == "vqgan":
        from prd_r01.vqgan_operator import VQGANProbe
        op = VQGANProbe(config)
    elif name == "restormer":
        from prd_r01.restormer_operator import RestormerProbe
        op = RestormerProbe(config)
    elif name == "diffusion":
        from prd_r01.diffusion_operator import DiffusionProbe
        op = DiffusionProbe(config,device)
    elif name in ("mae","compression","tv"):
        from prd_r01.next_operators import NextProbe
        op = NextProbe(config,name)
    elif name in ("e4e","colorization","sparse","nlm"):
        from prd_r01.diverse_operators import DiverseProbe
        op = DiverseProbe(config,name)
    elif name in ("codeformer","superresolution"):
        from prd_r01.generative_operators import GenerativeProbe
        op = GenerativeProbe(config,name)
    else:
        raise ValueError("未注册算子："+name)
    return {name: op.to(device).requires_grad_(False).eval()}


def forward(model, ops, config, rows, device):
    from prd_r01.data import LMDBImages, worker_seed
    physical_rows = [dict(row,domain=PHYSICAL.get(row["domain"],row["domain"])) for row in rows]
    if config["variant"] == "bm3d":
        from 运行BM3D单项 import EndpointImages
        ds = EndpointImages(physical_rows,config)
    else:
        ds = LMDBImages(physical_rows,dict(config,reconstruction_mode="online_after_augment"),False)
    # 只让BM3D并行调用四个单线程worker，按4096帧回收。普通读取用相同BS64。
    workers = 4 if config["variant"] == "bm3d" else 0
    loader = torch.utils.data.DataLoader(ds,batch_size=64,shuffle=False,num_workers=workers,
        worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(3407),pin_memory=True,
        **({"multiprocessing_context":"spawn","prefetch_factor":2} if workers else {}))
    values = np.empty((len(rows),2),np.float32)
    done = 0
    with torch.inference_mode():
        for x,y,indices,views in loader:
            if indices.tolist() != list(range(done,done+len(x))):
                raise ValueError("预测顺序异常")
            x = x.to(device,non_blocking=True)
            views = {k:v.to(device) for k,v in views.items()}
            for key,probe in ops.items():
                views[key] = probe(x)
            z = model(x,views)["logits"].float().cpu().numpy()
            if z.shape != (len(x),2) or not np.isfinite(z).all():
                raise ValueError("预测形状/数值异常")
            values[done:done+len(z)] = z
            done += len(z)
    ds.close()
    if done != len(rows):
        raise ValueError("预测截断")
    return values


def atom_npy(path, a):
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        np.save(f,a,allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name,path)


def evaluate_domain(base, model, probes, config, m, device, endpoint):
    domain = m["domain"]
    rows = m["rows"]
    area = base/domain
    area.mkdir(parents=True,exist_ok=True)
    manifest_sha = sha(STATE/"manifests"/(domain+".json"))
    frozen_json(area/"identity.json",{"checkpoint":endpoint,"manifest":manifest_sha,
                                      "code":source_git(),"physical":m["physical_lmdb"]})
    segments = []
    for start in range(0,len(rows),CHUNK):
        name = area/f"{start:07d}.npy"
        complete = name.with_suffix(".json")
        if complete.exists():
            rec = read(complete)
            if sha(name) != rec["sha256"] or rec["start"] != start or rec["count"] != min(CHUNK,len(rows)-start):
                raise ValueError(f"已归档分块不匹配：{name}")
        else:
            t = time.monotonic()
            values = forward(model,probes,config,rows[start:start+CHUNK],device)
            atom_npy(name,values)
            save(complete,{"start":start,"count":len(values),"sha256":sha(name),"seconds":time.monotonic()-t})
        segments.append(name)
        print(json.dumps({"domain":domain,"frames":min(start+CHUNK,len(rows)),"total":len(rows)}),flush=True)
    logits = np.concatenate([np.load(p,allow_pickle=False) for p in segments])
    final = area/"terminal.npy"
    if not (area/"verified.json").exists():
        atom_npy(final,logits)
        save(area/"verified.json",{"status":"verified","source":"new_forward","manifest":manifest_sha,
                                   "terminal_logits_sha256":sha(final),"metrics":metrics(logits,rows)})
    else:
        verified = read(area/"verified.json")
        if verified["manifest"] != manifest_sha or sha(final) != verified["terminal_logits_sha256"] or not np.array_equal(logits,np.load(final,allow_pickle=False)):
            raise ValueError(f"域归档校验失败：{domain}")
    return read(area/"verified.json")


def source_git():
    # 之后的监督/归档提交不改变推理代码的身份。已签出脚本若有未提交差异则拒绝。
    folder = str(ROOT)
    name = Path(__file__).relative_to(ROOT).as_posix()
    commit = subprocess.check_output(["git","-C",folder,"log","-1","--format=%H","--",name],text=True).strip()
    committed = subprocess.check_output(["git","-C",folder,"show",commit+":"+name])
    if committed != Path(__file__).read_bytes():
        raise RuntimeError("评估入口尚未冻结："+name)
    return commit


def run(args,spec):
    global torch
    import torch
    out = OUTPUT/spec["run"]
    for name,h in spec["artifacts"].items():
        if sha(out/name) != h:
            raise ValueError("历史验收文件变更："+name)
    if sha(spec["verification"]) != spec["verification_sha"]:
        raise ValueError("历史独立验收记录在读取后发生变化")
    if args.mode == "run":
        source_git()
    meta = read(out/"metadata.json")
    c = meta["config"]
    if meta["code_commit"] != spec["code"] or c["test_batch_size"] != 64:
        raise ValueError("模型身份或评测批量不符")
    original = read(out/"eval1_manifest.json")["rows"]
    reference = np.load(out/"terminal_eval1.npy",mmap_mode="r",allow_pickle=False)
    if reference.shape != (len(original),2) or len(original)!=165758 or {r["domain"] for r in original}!=set(BASE):
        raise ValueError("旧三域不完整")
    old_metrics = {}
    for domain in BASE:
        indices=[i for i,r in enumerate(original) if r["domain"] == domain]
        old_metrics[domain] = metrics(reference[indices],[original[i] for i in indices])
    with tempfile.TemporaryDirectory(prefix="prd-eval-code-") as source:
        snapshot(spec["code"],Path(source))
        from prd_r01.engine import seed_all
        from prd_r01.model import build_model
        if "CUDA_VISIBLE_DEVICES" not in os.environ or torch.cuda.device_count()!=1:
            raise RuntimeError("需显式绑定单一物理GPU")
        device=torch.device("cuda:0")
        if torch.cuda.mem_get_info(device)[0] < 12 * 1024**3:
            raise RuntimeError("只读评估空闲显存低于12 GiB")
        seed_all(3407)
        model=build_model(c,device)
        model.load_state_dict(torch.load(out/"terminal.pth",map_location="cpu",weights_only=True),strict=True)
        model.requires_grad_(False).eval()
        probes=build_views(c,device)
        # 复用三个原域各一个原BS64对齐的批，防止不同域抽样/算子/读出错误。
        replay={}
        for domain in BASE:
            first=next(i for i,r in enumerate(original) if r["domain"]==domain)
            start=(first+63)//64*64
            val=forward(model,probes,c,original[start:start+64],device)
            err=float(np.max(np.abs(val-reference[start:start+64])))
            replay[domain]=err
            if not np.allclose(val, reference[start:start+64],rtol=1e-4,atol=1e-4):
                raise ValueError(f"旧域同批重放失败：{domain} maxabs={err}")
        print("OLD_REPLAY",replay,flush=True)
        chosen=[new_manifest(d,c["data_root"]) for d in args.domains]
        if args.mode=="preflight":
            smoke={}
            for m in chosen:
                rows=m["rows"]
                sample=[next(r for r in rows if r["label"]==k) for k in (0,1)]
                z=forward(model,probes,c,sample,device)
                smoke[m["domain"]]=z.tolist()
            print(json.dumps({"status":"preflight_pass","run":spec["run"],"replay":replay,"new_samples":smoke,
                              "peak_bytes":torch.cuda.max_memory_allocated(device)},ensure_ascii=False),flush=True)
            return
        base=STATE/"results"/spec["run"]
        frozen_json(base/"identity.json",{"endpoint":"terminal","run_id":spec["run"],"train_code":spec["code"],
            "eval_code":source_git(),"old_artifacts":spec["artifacts"],"verification":spec["verification_sha"]})
        save(base/"old_domains.json",{"source":"archived_terminal_logits","metrics":old_metrics,"preflight_replay":replay})
        for m in chosen:
            evaluate_domain(base,model,probes,c,m,device,spec["artifacts"]["terminal.pth"])
        result=dict(old_metrics)
        for d in (*NEW,DF1):
            verified=base/d/"verified.json"
            if verified.exists():
                rec=read(verified)
                if sha(base/d/"terminal.npy") != rec["terminal_logits_sha256"]:
                    raise ValueError("已完成域摘要的logits哈希不符："+d)
                result[d]=rec["metrics"]
        status="complete" if set(result)==set(ALL) else "partial"
        macro={k: float(np.mean([result[d][k] for d in ALL])) for k in
               ("frame_auc","video_auc","frame_ce","video_ce")} if status=="complete" else None
        save(base/"summary.json",{"run_id":spec["run"],"status":status,"domains":result,"macro8":macro,
                                 "new_training_updates":0})
        print(status,spec["run"],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["inventory","preflight","run"])
    parser.add_argument("--run")
    parser.add_argument("--domains",nargs="+",choices=[*NEW,DF1],default=list(NEW))
    args=parser.parse_args()
    catalog=discover()
    if args.mode=="inventory":
        print(json.dumps(list(sorted(catalog)),ensure_ascii=False,indent=2))
        return
    if args.run not in catalog:
        parser.error("需选择已独立验收的R01运行")
    if len(set(args.domains)) != len(args.domains):
        parser.error("域重复")
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/(args.run+".lock")).open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args,catalog[args.run])


if __name__ == "__main__":
    main()
