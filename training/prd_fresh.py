"""源验证 AUC 早停基线，使用完整可恢复状态和冻结最佳权重。"""
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from prd_r01.data import LMDBImages, validate_source_pair
from prd_r01.engine import seed_all as seed_common
from prd_r01.model import build_model
from prd_r01.auxiliary_supervision import trainable_identity

MAIN_DOMAINS = ("Celeb-DF-v2", "DFDC", "DeepFakeDetection", "UADFV")
EXTRA_DOMAINS = ("Celeb-DF-v1", "DFDCP")
PHYSICAL = {"DeepFakeDetection": "FaceForensics++"}
LABELS = {"FaceForensics++": {"FF-real": 0, "FF-DF": 1, "FF-F2F": 1, "FF-FS": 1, "FF-NT": 1},
          "Celeb-DF-v1": {"CelebDFv1_real": 0, "CelebDFv1_fake": 1},
          "Celeb-DF-v2": {"CelebDFv2_real": 0, "CelebDFv2_fake": 1},
          "DFDC": {"DFDC_Real": 0, "DFDC_Fake": 1},
          "DFDCP": {"DFDCP_Real": 0, "DFDCP_FakeA": 1, "DFDCP_FakeB": 1},
          "DeepFakeDetection": {"DFD_real": 0, "DFD_fake": 1},
          "UADFV": {"UADFV_Real": 0, "UADFV_Fake": 1}}


def seed_all(seed):
    seed_common(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for value in iter(lambda: stream.read(4 << 20), b""):
            digest.update(value)
    return digest.hexdigest()


def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def save_torch(path, data):
    path = Path(path); temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        torch.save(data, stream)
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def manifest(config, domain, split, role):
    path = Path(config["data_root"]) / "DeepfakeBench/config/dataset_json" / (domain + ".json")
    data = json.loads(path.read_text(encoding="utf-8"))[domain]
    if set(data) != set(LABELS[domain]):
        raise ValueError("源清单出现未知类别 " + domain)
    rows = []
    for label, parts in sorted(data.items()):
        videos = parts[split]
        if domain in ("FaceForensics++", "DeepFakeDetection"):
            videos = videos["c23"]
        for video_id, info in sorted(videos.items()):
            frames = sorted(info["frames"])
            if not frames or info["label"] != label:
                raise ValueError("空视频或标签错误")
            for index in np.linspace(0, len(frames)-1, min(32, len(frames)), dtype=int):
                row = dict(domain=domain, video_id=label+"/"+video_id, key=frames[index], label=LABELS[domain][label])
                if domain == "FaceForensics++":
                    row["source_ids"] = video_id.split("_")
                rows.append(row)
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("清单有重复帧")
    return dict(role=role, domain=domain, split=split, source=str(path), source_sha256=file_sha(path), rows=rows)


class SequentialVAE(nn.Module):
    def __init__(self, probe):
        super().__init__(); self.probe = probe.requires_grad_(False).eval()

    def forward(self, images):
        self.probe.eval()
        return torch.cat([self.probe(x) for x in images.split(1)], dim=0)


def build(config, device):
    model = build_model(config, device)
    if config["variant"] == "vae":
        model.probes["vae"] = SequentialVAE(model.probes["vae"])
    return model


class FixedEpochImages(LMDBImages):
    def __init__(self, rows, config, epoch, augment):
        physical = [dict(row, domain=PHYSICAL.get(row["domain"], row["domain"])) for row in rows]
        super().__init__(physical, config, augment)
        self.epoch = epoch

    def __getitem__(self, index):
        if not self.augment:
            return super().__getitem__(index)
        py_rng, np_rng = random.getstate(), np.random.get_state()
        seed = (3407 + 1000003 * self.epoch + index) % (2**32)
        random.seed(seed); np.random.seed(seed)
        try:
            return super().__getitem__(index)
        finally:
            random.setstate(py_rng); np.random.set_state(np_rng)


def loader(rows, config, epoch=0, training=False, offset=0):
    dataset = FixedEpochImages(rows, config, epoch, training)
    generator = torch.Generator().manual_seed(3407 + epoch)
    order = torch.randperm(len(rows), generator=generator).tolist() if training else list(range(len(rows)))
    if offset < 0 or offset > len(order):
        raise ValueError("恢复位置越界")
    indices = order[offset:]
    return torch.utils.data.DataLoader(dataset, batch_size=32 if training else 64, sampler=indices,
            num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(900000 + epoch))


RECONSTRUCTION_RUN_FAMILIES = {
    "vqgan": "早停码本重构-20261107",  # 保留已验收运行的既有路径。
    "superresolution": "早停降采样超分-20261109",
    "vqgan_unit_residual": "早停单位方向残差-20261109",
}


def frozen_identity(config):
    paths = [Path(config["clip_path"]) / "model.safetensors", Path(config["clip_path"]) / "config.json"]
    if config["variant"] == "vae":
        paths += [Path(config["vae_path"]) / "diffusion_pytorch_model.bin", Path(config["vae_path"]) / "config.json"]
    if config["variant"] in {"vqgan", "vqgan_unit_residual"}:
        paths += [Path(config["vqgan_weights"]), Path(__file__).parent/"prd_r01/vendor/vqgan/model.yaml"]  # 紧凑权重必须绑定外部码本与结构。
    if config["variant"] == "superresolution":
        paths.append(Path(config["superresolution_weights"]))  # 紧凑权重依赖冻结超分资产和算子源码。
        source_root = Path(__file__).parent / "prd_r01"
        paths += [source_root / name for name in ("generative_operators.py", "vendor/generative/rrdbnet_arch.py", "vendor/generative/arch_util.py")]
    return {str(p): file_sha(p) for p in paths}


def compact_state(model):
    names = {name for name,p in model.named_parameters() if p.requires_grad}
    names.update(name for name,_ in model.named_buffers())
    full = dict(model.named_parameters())
    full.update(dict(model.named_buffers()))  # 包含未持久化的 position_ids 缓冲区。
    return {name: full[name].detach().cpu().clone() for name in sorted(names)}


def load_compact(model, state):
    expected = compact_state(model)
    if set(expected) != set(state):
        raise ValueError("紧凑权重键与当前模型不匹配")
    current = dict(model.named_parameters())
    current.update(dict(model.named_buffers()))
    with torch.no_grad():
        for name,value in state.items():
            if expected[name].shape != value.shape or expected[name].dtype != value.dtype:
                raise ValueError("紧凑权重结构不一致 " + name)
            current[name].copy_(value.to(current[name].device))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint(model, optimizer, progress, identity):
    return dict(schema=1, model=compact_state(model), optimizer=optimizer.state_dict(), progress=copy.deepcopy(progress),
                rng=rng_state(), identity=identity, torch_version=torch.__version__)


def load_resume(model, optimizer, payload, identity):
    if payload["identity"] != identity or payload["schema"] != 1:
        raise ValueError("恢复点的协议/数据/底座身份不一致")
    load_compact(model, payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])
    return copy.deepcopy(payload["progress"])


def optimizer_for(model, config):
    c = config["optimizer"]
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                lr=c["lr"], betas=tuple(c["betas"]), eps=c["eps"], weight_decay=c["weight_decay"])


def metric(logits, rows):
    if logits.dtype != np.float32 or logits.shape != (len(rows),2) or not np.isfinite(logits).all():
        raise ValueError("预测形状或数值错误")
    probs = torch.softmax(torch.from_numpy(logits), dim=1)[:,1].numpy()
    labels = np.asarray([r["label"] for r in rows])
    if set(labels.tolist()) != {0,1}:
        raise ValueError("评测必须包含真假两类")
    groups = {}
    for i,row in enumerate(rows):
        groups.setdefault(row["video_id"], []).append(i)
    if any(np.any(labels[idx] != labels[idx[0]]) for idx in groups.values()):
        raise ValueError("同视频标签冲突")
    vp = [probs[idx].mean(dtype=np.float64) for idx in groups.values()]
    vy = [int(labels[idx[0]]) for idx in groups.values()]
    auc = float(roc_auc_score(labels, probs))
    npos = int(labels.sum()); nneg = len(labels)-npos
    rank_auc = float((rankdata(probs)[labels==1].sum()-npos*(npos+1)/2)/(npos*nneg))
    if abs(auc-rank_auc)>1e-12:
        raise ValueError("独立排序AUC未复现")
    z = logits.astype(np.float64)
    ce = np.logaddexp(z[:,0], z[:,1])-z[np.arange(len(z)),labels]
    vce = -np.log(np.maximum(np.where(np.asarray(vy),vp,1-np.asarray(vp)),np.finfo(np.float64).tiny))
    return dict(frame_auc=auc, video_auc=float(roc_auc_score(vy,vp)), frame_ce=float(ce.mean()),
                video_ce=float(vce.mean()), frames=len(rows), videos=len(groups), independent_auc_error=abs(auc-rank_auc))


def evaluate(model, rows, config, device, deadline, log=False):
    model.eval(); data = loader(rows, config)
    predictions = np.empty((len(rows),2), np.float32)
    started=time.monotonic()
    try:
        with torch.inference_mode():
            for x,y,index,views in data:
                if time.monotonic()>=deadline:
                    raise TimeoutError("评估达到已批准时间上限")
                out=model(x.to(device),{})["logits"].float().cpu().numpy()
                predictions[index.numpy()]=out
                if log and (int(index[-1])+1)%4096==0:
                    print(json.dumps(dict(stage="evaluation",frames=int(index[-1])+1,total=len(rows),seconds=time.monotonic()-started)),flush=True)
    finally:
        data.dataset.close()
    return metric(predictions,rows),predictions


def update_best(progress, score, epoch):
    if not math.isfinite(score):
        raise ValueError("源验证AUC非有限")
    if score>progress["best_auc"]:
        progress.update(best_auc=score,best_epoch=epoch,bad_epochs=0)
        return True
    progress["bad_epochs"]+=1
    return False


def train_step(model, optimizer, batch, device):
    x,y,index,views=batch
    model.train(); x=x.to(device);y=y.to(device)
    image_sha=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()
    optimizer.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(model(x,{})["logits"],y)
    if not torch.isfinite(loss):raise ValueError("训练loss非有限")
    loss.backward()
    trainable=[p for p in model.parameters() if p.requires_grad]
    grad=torch.nn.utils.clip_grad_norm_(trainable,float("inf"),error_if_nonfinite=True)
    if float(grad)<=0:raise ValueError("训练梯度为零")
    optimizer.step()
    return float(loss.detach()),float(grad),index.tolist(),image_sha


def train(config, manifests, out, state, device, identity, seconds, resume=False):
    seed_all(3407); model=build(config,device); opt=optimizer_for(model,config)
    p=dict(next_epoch=1,offset=0,updates=0,best_auc=-1.0,best_epoch=None,bad_epochs=0,curve=[],
           training_complete=False,stop_reason=None,elapsed_seconds=0.)
    current=out/"last_resume.pt"
    if resume:
        if not current.exists():raise FileNotFoundError("无完整恢复点")
        p=load_resume(model,opt,torch.load(current,map_location="cpu",weights_only=False),identity)
    else:
        save_json(out/"initialization.json",trainable_identity(model))
    start=time.monotonic(); previous_elapsed=p["elapsed_seconds"]
    deadline=start+max(0,seconds-previous_elapsed)
    trace=out/"batch_trace.jsonl"
    if resume:
        lines=trace.read_text(encoding="utf8").splitlines() if trace.exists() else []
        if len(lines)<p["updates"]:raise ValueError("恢复点对应轨迹不完整")
        trace.write_text("\n".join(lines[:p["updates"]])+("\n" if p["updates"] else ""),encoding="utf8")
    with trace.open("a" if resume else "x",encoding="utf8",buffering=1) as log:
        while not p["training_complete"]:
            epoch=p["next_epoch"]
            data=loader(manifests["train"]["rows"],config,epoch,True,p["offset"])
            try:
                for batch in data:
                    if time.monotonic()>=deadline:raise TimeoutError("训练达到已批准时间上限，完整恢复点保留")
                    loss,grad,idx,image_hash=train_step(model,opt,batch,device)
                    p["offset"]+=len(idx);p["updates"]+=1
                    log.write(json.dumps(dict(epoch=epoch,step=p["updates"],indices=idx,image_sha256=image_hash))+"\n")
                    if p["updates"]==1 or p["updates"]%100==0:
                        print(json.dumps(dict(stage="training",epoch=epoch,step=p["updates"],loss=loss,gradient_norm=grad,elapsed_seconds=previous_elapsed+time.monotonic()-start)),flush=True)
                    if p["updates"]%100==0:
                        p["elapsed_seconds"]=previous_elapsed+time.monotonic()-start
                        log.flush();os.fsync(log.fileno())
                        save_torch(current,checkpoint(model,opt,p,identity))
            finally:
                data.dataset.close()
            score,predictions=evaluate(model,manifests["val"]["rows"],config,device,deadline)
            np.save(out/f"source_val_epoch{epoch:02d}.npy",predictions)
            improved=update_best(p,score["frame_auc"],epoch)
            p["curve"].append(dict(epoch=epoch,validation=score,improved=improved,bad_epochs=p["bad_epochs"]))
            if improved:
                best=dict(schema=1,model=compact_state(model),epoch=epoch,validation=score,identity=identity)
                save_torch(out/f"best_epoch{epoch:02d}.pt",best)
                save_torch(out/"best.pt",best)
            p["next_epoch"]=epoch+1;p["offset"]=0
            if p["bad_epochs"]>=5:p.update(training_complete=True,stop_reason="early_stopping_patience_5")
            elif epoch>=30:p.update(training_complete=True,stop_reason="maximum_epochs_30")
            p["elapsed_seconds"]=previous_elapsed+time.monotonic()-start
            log.flush();os.fsync(log.fileno())
            save_json(out/"learning_curve.json",p["curve"])
            save_torch(current,checkpoint(model,opt,p,identity))
            save_json(state/"status.json",dict(stage="training_complete" if p["training_complete"] else "training",
                        time=now(),pid=os.getpid(),epoch=epoch,updates=p["updates"],best_epoch=p["best_epoch"],
                        best_auc=p["best_auc"],bad_epochs=p["bad_epochs"],stop_reason=p["stop_reason"]))
            print(json.dumps(p["curve"][-1]),flush=True)
    best=torch.load(out/"best.pt",map_location="cpu",weights_only=True)
    if best["identity"]!=identity or best["epoch"]!=p["best_epoch"]:raise ValueError("最佳权重身份不符")
    load_compact(model,best["model"])
    save_json(out/"training_result.json",p)
    return model,p


def final_evaluation(model,config,manifests,out,state,device,seconds):
    deadline=time.monotonic()+seconds;results={}
    best_hash=file_sha(out/"best.pt")
    for name in ("source_test",*MAIN_DOMAINS,*EXTRA_DOMAINS):
        rows=manifests[name]["rows"]
        save_json(state/"status.json",dict(stage="evaluation",time=now(),pid=os.getpid(),domain=name,best_sha256=best_hash))
        scores,predictions=evaluate(model,rows,config,device,deadline,log=True)
        np.save(out/(name+"_predictions.npy"),predictions)
        independent=metric(np.load(out/(name+"_predictions.npy"),allow_pickle=False),rows)
        if scores!=independent:raise ValueError("落盘指标复算不一致")
        results[name]=dict(metrics=scores,prediction_sha256=file_sha(out/(name+"_predictions.npy")),
                           manifest_sha256=file_sha(out/(name+"_manifest.json")),best_sha256=best_hash)
        save_json(out/"evaluation_progress.json",results)
    complete=dict(status="verified",time=now(),domains=results,main_domains=list(MAIN_DOMAINS),
                  macro={k:float(np.mean([results[d]["metrics"][k] for d in MAIN_DOMAINS])) for k in
                         ("frame_auc","video_auc","frame_ce","video_ce")},
                  best_sha256=best_hash,new_updates_during_evaluation=0)
    save_json(out/"complete_result.json",complete)
    save_json(state/"status.json",dict(stage="verified",time=now(),pid=os.getpid(),best_sha256=best_hash))
    return complete

