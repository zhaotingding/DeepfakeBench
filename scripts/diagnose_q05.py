"""Q05: fixed preprocessing-only vs full VAE residual, frozen model."""
import argparse,json,shutil,subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from train_p03_t01 import ROOT,LoraDetector,target_images,save,sha,stamp

p=argparse.ArgumentParser();p.add_argument('--claim',required=True);args=p.parse_args()
run='q05_d01_20261005';state=ROOT/'.state'/run
out=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism')/run
out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
source=json.loads((ROOT/'scripts/experiment_summaries/n01_t04_20261004/run_metadata.json').read_text())
config=source['config'];images,rows=target_images(config,out)
assert sha(out/'evaluation_manifest.json')=='ac5121d948e23b5c00f36b30e606224380e2ef18a566a04c4dae8c60438e988a'
model=LoraDetector(config).cuda().eval();model.requires_grad_(False)
assert all(torch.count_nonzero(v)==0 for n,v in model.named_parameters() if 'lora_B' in n)
meta=dict(id='Q05-D01/v1',run_id=run,started_at=stamp(),claim_commit=args.claim,
 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),updates=0,
 config=config,weights=source['input_hashes'],protocol_sha256=sha(ROOT/'scripts/q05_protocol.json'),
 manifest_sha256=sha(out/'evaluation_manifest.json'),torch_version=torch.__version__,
 source_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
save(out/'metadata.json',meta)
values=[];identity_max=0.;pipeline_max=0.;repeat_max=0.
with torch.inference_mode():
 for start in range(0,192,16):
  x=torch.stack(images[start:start+16]).cuda();y=torch.tensor([v['label'] for v in rows[start:start+16]],device='cuda')
  a=model.vae_augmenter
  v=F.interpolate(a.vae_normalize(a.clip_denormalize(x)),size=a.vae_input_size,mode='bilinear',align_corners=False)
  pre=a.clip_normalize(a.vae_denormalize(F.interpolate(v.to(a.vae.dtype).to(x.dtype),size=x.shape[-2:],mode='bilinear',align_corners=False)))
  if start==0:
   with patch.object(a.vae,'forward',side_effect=lambda z:SimpleNamespace(sample=z)):
    pipeline_max=float((pre-a(x,y)).abs().max());assert pipeline_max<1e-6
  full=a(x,y);f=model.features(x);fi=model.features(x.clone());fp=model.features(pre);fv=model.features(full)
  identity_max=max(identity_max,float((f-fi).abs().max()));assert identity_max<1e-6
  if start==0:
   repeat_max=float((full-a(x,y)).abs().max());assert repeat_max<1e-6
  rp=f-fp;rv=f-fv;ep=rp.norm(dim=1);ev=rv.norm(dim=1)
  assert torch.isfinite(rp).all() and torch.isfinite(rv).all() and (ev>1e-8).all() and (ep>1e-8).all()
  pixelpre=((x-pre)*a.clip_std).square().flatten(1).mean(1).sqrt()
  pixelfull=((x-full)*a.clip_std).square().flatten(1).mean(1).sqrt()
  cos=(rp*rv).sum(1)/(ep*ev)
  values.extend(torch.stack([ep,ev,ep/ev,cos,pixelpre,pixelfull],1).double().cpu().tolist())
  save(state/'status.json',dict(status='running',videos_done=start+len(y),time=stamp()),replace=True)
  print(stamp(),'VIDEOS_DONE',start+len(y),flush=True)
v=np.asarray(values);domains=sorted({r['domain'] for r in rows});indexes={d:np.array([i for i,r in enumerate(rows) if r['domain']==d]) for d in domains}
per_domain={d:dict(median_pre_full_feature_ratio=float(np.median(v[ix,2])),median_cosine=float(np.median(v[ix,3])),median_energies=np.median(v[ix][:,[0,1,4,5]],axis=0).tolist()) for d,ix in indexes.items()}
macro=float(np.mean([d['median_pre_full_feature_ratio'] for d in per_domain.values()]))
rng=np.random.default_rng(20261005);boot=[]
for _ in range(2000):
 boot.append(np.mean([np.median(v[rng.choice(ix,len(ix),replace=True),2]) for ix in indexes.values()]))
ci=np.quantile(boot,[.025,.975]).tolist()
supported=bool(ci[0]>.1 and all(d['median_pre_full_feature_ratio']>=.1 for d in per_domain.values()))
result=dict(id=meta['id'],run_id=run,status='completed',started_at=meta['started_at'],ended_at=stamp(),code_commit=meta['code_commit'],claim_commit=args.claim,updates=0,
 per_domain=per_domain,macro_median_ratio=macro,bootstrap95=ci,supported=supported,
 verdict='material_preprocessing_response' if supported else 'predefined_material_response_not_supported',
 identity_max_error=identity_max,pipeline_bypass_max_error=pipeline_max,vae_repeat_max_error=repeat_max,
 columns=['pre_feature_L2','full_feature_L2','pre_full_ratio','cosine','pre_pixel_RMS','full_pixel_RMS'],
 per_video=[dict(**{k:r[k] for k in ['domain','video_id','label']},values=values[i]) for i,r in enumerate(rows)],
 limitations='Feature norm ratio is not an additive fraction attributable to preprocessing. Joint resize/dtype contrast does not separate their effects. Fixed reused development videos; shared identities may remain. No training, classifier AUC, generalization or method-causality claim.')
save(out/'result.json',result)
archive=ROOT/'scripts/experiment_summaries'/run;archive.mkdir(exist_ok=False)
backup=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups')/run;backup.mkdir(exist_ok=False)
for f in out.iterdir():
 shutil.copy2(f,archive/f.name);shutil.copy2(f,backup/f.name);assert sha(f)==sha(backup/f.name)
shutil.copy2(ROOT/'scripts/q05_protocol.json',archive/'protocol.json')
completion=dict(id=meta['id'],run_id=run,status='completed',started_at=meta['started_at'],ended_at=result['ended_at'],code_commit=meta['code_commit'],claim_commit=args.claim,updates=0,
 verdict=result['verdict'],macro_median_ratio=macro,bootstrap95=ci,backup=str(backup),artifact_sha256={f.name:sha(f) for f in out.iterdir()})
save(archive/'completion.json',completion)
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q05预处理往返与完整VAE诊断'],cwd=ROOT,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
with (state/'result_commit.txt').open('x') as f:f.write(commit+'\n')
save(state/'completion.json',completion);print(stamp(),'COMPLETE',commit,result['verdict'],flush=True)
