"""Q02: frozen reconstruction trajectory, one fixed development-video manifest."""
import argparse,json,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from train_p03_t01 import ROOT,LoraDetector,target_images,save,sha,stamp

p=argparse.ArgumentParser();p.add_argument('--claim',required=True);args=p.parse_args()
s=ROOT/'.state/q02_d01_20261005';out=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism/q02_d01_20261005')
out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
source=ROOT/'scripts/experiment_summaries/n01_t04_20261004/run_metadata.json'
config=json.loads(source.read_text())['config'];images,rows=target_images(config,out)
assert sha(out/'evaluation_manifest.json')=='ac5121d948e23b5c00f36b30e606224380e2ef18a566a04c4dae8c60438e988a'
model=LoraDetector(config).cuda().eval();model.requires_grad_(False)
# New zero-update LoRA initialization is the existing pretrained CLIP, not a target-selected checkpoint.
assert all(torch.count_nonzero(v)==0 for n,v in model.named_parameters() if 'lora_B' in n)
meta=dict(id='Q02-D01/v1',run_id='q02_d01_20261005',started_at=stamp(),claim_commit=args.claim,
 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),updates=0,
 protocol_sha256=sha(ROOT/'scripts/q02_protocol.json'),config=config,
 manifest_sha256=sha(out/'evaluation_manifest.json'),weights=json.loads(source.read_text())['input_hashes'],
 source_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),encoder='pretrained CLIP with zero-update LoRA, eval and frozen')
save(out/'metadata.json',meta);save(s/'status.json',dict(status='running',time=stamp()),replace=True)
feat=[];pix=[];ranges=[];repeat_error=None
with torch.inference_mode():
 for start in range(0,192,16):
  x=torch.stack(images[start:start+16]).cuda();y=torch.tensor([v['label'] for v in rows[start:start+16]],device='cuda')
  f=model.features(x);ef=[];ep=[];rr=[]
  for k in range(3):
   nxt=model.vae_augmenter(x,y)
   if start==0 and k==0:
    repeat_error=float((nxt-model.vae_augmenter(x,y)).abs().max());assert repeat_error<1e-6
   nf=model.features(nxt);rgb_delta=(nxt-x)*model.vae_augmenter.clip_std
   ef.append((nf-f).norm(dim=1).cpu());ep.append(rgb_delta.square().flatten(1).mean(1).sqrt().cpu())
   rgb=model.vae_augmenter.clip_denormalize(nxt);rr.append([float(rgb.min()),float(rgb.max())])
   assert torch.isfinite(nf).all() and torch.isfinite(nxt).all()
   x,f=nxt,nf
  feat.append(torch.stack(ef,1));pix.append(torch.stack(ep,1));ranges.append(rr)
  save(s/'status.json',dict(status='running',videos_done=start+len(y),time=stamp()),replace=True)
  print(stamp(),'VIDEOS_DONE',start+len(y),flush=True)
feature=torch.cat(feat).numpy();pixel=torch.cat(pix).numpy()
assert feature.shape==pixel.shape==(192,3) and (feature>1e-8).all() and (pixel>1e-8).all()
labels=np.array([v['label'] for v in rows]);domains=sorted({v['domain'] for v in rows})
residual=np.empty(192);by_domain={};decay={}
for d in domains:
 ix=np.array([i for i,v in enumerate(rows) if v['domain']==d]);assert len(ix)==64
 x=np.log(feature[ix,0]);z=np.log(feature[ix,2]);design=np.column_stack([np.ones(64),x])
 beta=np.linalg.lstsq(design,z,rcond=None)[0];residual[ix]=z-design@beta
 auc=float(roc_auc_score(labels[ix],residual[ix]));by_domain[d]=dict(adjusted_step3_auc=auc,log_step1_coefficient=float(beta[1]))
 decay[d]={name:dict(median_steps=np.median(a[ix],axis=0).tolist(),median_step2_over_step1=float(np.median(a[ix,1]/a[ix,0])),median_step3_over_step1=float(np.median(a[ix,2]/a[ix,0]))) for name,a in [('feature',feature),('pixel_rgb_rms',pixel)]}
macro=float(np.mean([v['adjusted_step3_auc'] for v in by_domain.values()]));observed=abs(macro-.5)
# Fixed-count permutation uses videos, not reconstruction steps, as units.
rng=np.random.default_rng(20261005);null=[]
for _ in range(1000):
 vals=[]
 for d in domains:
  ix=np.array([i for i,v in enumerate(rows) if v['domain']==d]);vals.append(roc_auc_score(rng.permutation(labels[ix]),residual[ix]))
 null.append(abs(np.mean(vals)-.5))
pvalue=float((1+np.sum(np.array(null)>=observed))/1001)
effects=[v['adjusted_step3_auc']-.5 for v in by_domain.values()]
support=observed>=.10 and (min(effects)>0 or max(effects)<0) and pvalue<=.05
result=dict(id=meta['id'],run_id=meta['run_id'],started_at=meta['started_at'],ended_at=stamp(),code_commit=meta['code_commit'],
 claim_commit=args.claim,updates=0,status='completed',per_domain=by_domain,decay=decay,macro_adjusted_step3_auc=macro,
 two_sided_stratified_permutation_p=pvalue,permutations=1000,repeat_max_error=repeat_error,rgb_ranges=ranges,
 verdict='supports_remaining_development_association' if support else 'not_supported_by_predefined_criterion',
 per_video=[dict(**{k:r[k] for k in ['domain','video_id','label']},feature_steps=feature[i].tolist(),pixel_steps=pixel[i].tolist(),adjusted_step3=float(residual[i])) for i,r in enumerate(rows)],
 limitations='Only domain-specific linear adjustment in log first-step feature amplitude; nonlinear/confounded effects and shared identities remain. Reused target development videos, stratified permutation diagnostic, not final inference or manifold proof.')
save(out/'result.json',result)
archive=ROOT/'scripts/experiment_summaries/q02_d01_20261005';archive.mkdir(exist_ok=False)
backup=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups/q02_d01_20261005');backup.mkdir(exist_ok=False)
for f in out.iterdir():
 shutil.copy2(f,archive/f.name);shutil.copy2(f,backup/f.name);assert sha(f)==sha(backup/f.name)
shutil.copy2(ROOT/'scripts/q02_protocol.json',archive/'protocol.json')
completion=dict(id=meta['id'],run_id=meta['run_id'],status='completed',started_at=meta['started_at'],ended_at=result['ended_at'],
 code_commit=meta['code_commit'],claim_commit=args.claim,updates=0,verdict=result['verdict'],backup=str(backup),
 macro_adjusted_step3_auc=macro,permutation_p=pvalue,artifact_sha256={f.name:sha(f) for f in out.iterdir()})
save(archive/'completion.json',completion)
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q02三次重建动力学固定样本诊断'],cwd=ROOT,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
with (s/'result_commit.txt').open('x') as f:f.write(commit+'\n')
save(s/'completion.json',completion);print(stamp(),'COMPLETE',commit,result['verdict'],flush=True)
