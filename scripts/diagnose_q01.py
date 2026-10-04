"""Q01: paired fixed postprocessing sensitivity, no parameter updates."""
import argparse,io,json,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
from PIL import Image, __version__ as pillow_version
from sklearn.metrics import roc_auc_score
from train_p03_t01 import ROOT,LoraDetector,target_images,save,sha,stamp

p=argparse.ArgumentParser();p.add_argument('--claim',required=True);args=p.parse_args()
run='q01_d01_20261005';state=ROOT/'.state'/run
out=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism')/run
out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
source=json.loads((ROOT/'scripts/experiment_summaries/n01_t04_20261004/run_metadata.json').read_text())
config=source['config'];images,rows=target_images(config,out)
assert sha(out/'evaluation_manifest.json')=='ac5121d948e23b5c00f36b30e606224380e2ef18a566a04c4dae8c60438e988a'
model=LoraDetector(config).cuda().eval();model.requires_grad_(False)
assert all(torch.count_nonzero(v)==0 for n,v in model.named_parameters() if 'lora_B' in n)
meta=dict(id='Q01-D01/v1',run_id=run,started_at=stamp(),claim_commit=args.claim,
 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),updates=0,
 config=config,weights=source['input_hashes'],protocol_sha256=sha(ROOT/'scripts/q01_protocol.json'),
 manifest_sha256=sha(out/'evaluation_manifest.json'),pillow_version=pillow_version,torch_version=torch.__version__,
 source_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
save(out/'metadata.json',meta)
conditions=['raw','jpeg75','half_bilinear'];energy=[];pixel=[];roundtrip_max=0.;repeat_max=0.;input_change=[]
mean=torch.tensor(config['mean'])[:,None,None];std=torch.tensor(config['std'])[:,None,None]
with torch.inference_mode():
 for start in range(0,192,16):
  original=torch.stack(images[start:start+16]);rgb=original*std+mean
  assert rgb.min()>-1e-6 and rgb.max()<1+1e-6
  u8=(rgb.clamp(0,1)*255).round().byte();raw=(u8.float()/255-mean)/std
  err=float((raw-original).abs().max());roundtrip_max=max(roundtrip_max,err);assert err<1e-5
  variants={c:[] for c in conditions}
  for value in u8:
   im=Image.fromarray(value.permute(1,2,0).numpy());w,h=im.size
   assert w%2==h%2==0
   stream=io.BytesIO();im.save(stream,format='JPEG',quality=75,subsampling=2,optimize=False,progressive=False);stream.seek(0)
   jpg=Image.open(stream).convert('RGB');small=im.resize((w//2,h//2),Image.Resampling.BILINEAR).resize((w,h),Image.Resampling.BILINEAR)
   for name,img in zip(conditions,[im,jpg,small]):
    variants[name].append((torch.from_numpy(np.array(img,copy=True)).permute(2,0,1).float()/255-mean)/std)
  y=torch.tensor([v['label'] for v in rows[start:start+16]],device='cuda');ef=[];ep=[];changes=[]
  for name in conditions:
   x=torch.stack(variants[name]).cuda();recon=model.vae_augmenter(x,y)
   if start==0 and name=='raw':
    repeat_max=float((recon-model.vae_augmenter(x,y)).abs().max());assert repeat_max<1e-6
   f=model.features(x);fr=model.features(recon)
   assert torch.isfinite(fr).all() and torch.isfinite(recon).all()
   ef.append((f-fr).norm(dim=1).cpu());ep.append(((x-recon)*model.vae_augmenter.clip_std).square().flatten(1).mean(1).sqrt().cpu())
   changes.append(((x.cpu()-raw)*std).square().flatten(1).mean(1).sqrt())
  energy.append(torch.stack(ef,1));pixel.append(torch.stack(ep,1));input_change.append(torch.stack(changes,1))
  save(state/'status.json',dict(status='running',videos_done=start+len(y),time=stamp()),replace=True)
  print(stamp(),'VIDEOS_DONE',start+len(y),flush=True)
e=torch.cat(energy).numpy().astype(np.float64);px=torch.cat(pixel).numpy();change=torch.cat(input_change).numpy()
assert e.shape==(192,3) and np.isfinite(e).all() and (e>1e-8).all()
assert (change[:,0]==0).all() and (change[:,1:]>0).all()
labels=np.array([r['label'] for r in rows]);domains=sorted({r['domain'] for r in rows})
delta=np.log(e[:,1:]/e[:,[0]]);indexes={d:np.array([i for i,r in enumerate(rows) if r['domain']==d]) for d in domains}
by_domain={}
for d,ix in indexes.items():
 assert len(ix)==64 and labels[ix].sum()==32
 y=labels[ix];v=delta[ix]
 by_domain[d]=dict(fake_minus_real_delta_log=(v[y==1].mean(0)-v[y==0].mean(0)).tolist(),
  mean_paired_delta_real=v[y==0].mean(0).tolist(),mean_paired_delta_fake=v[y==1].mean(0).tolist(),
  amplitude_auc=[float(roc_auc_score(y,e[ix,k])) for k in range(3)],median_feature_energy=np.median(e[ix],axis=0).tolist(),
  median_pixel_energy=np.median(px[ix],axis=0).tolist(),median_input_rgb_rms=np.median(change[ix],axis=0).tolist())
effects=np.array([v['fake_minus_real_delta_log'] for v in by_domain.values()]);macro=effects.mean(0)
rng=np.random.default_rng(20261005);null=[]
for _ in range(1000):
 vals=[]
 for ix in indexes.values():
  y=rng.permutation(labels[ix]);v=delta[ix];vals.append(v[y==1].mean(0)-v[y==0].mean(0))
 null.append(np.max(np.abs(np.mean(vals,axis=0))))
adjusted_p=[float((1+np.sum(np.array(null)>=abs(v)))/1001) for v in macro]
supported=[bool(abs(macro[k])>=.10 and (np.all(effects[:,k]>0) or np.all(effects[:,k]<0)) and adjusted_p[k]<=.05) for k in range(2)]
result=dict(id=meta['id'],run_id=run,status='completed',started_at=meta['started_at'],ended_at=stamp(),code_commit=meta['code_commit'],claim_commit=args.claim,updates=0,
 conditions=conditions,per_domain=by_domain,macro_interaction_log=macro.tolist(),maxT_adjusted_p=adjusted_p,supported=supported,
 verdict='supports_postprocessing_sensitivity' if any(supported) else 'not_supported_by_predefined_criterion',
 roundtrip_max_error=roundtrip_max,repeat_max_error=repeat_max,
 per_video=[dict(**{k:r[k] for k in ['domain','video_id','label']},feature_energy=e[i].tolist(),pixel_energy=px[i].tolist(),input_rgb_rms=change[i].tolist()) for i,r in enumerate(rows)],
 limitations='Paired within-video intervention on fixed development images. JPEG/resampling alter content too, not isolated original compression confounding. No training or generalization causality; shared identities and reused target videos remain.')
save(out/'result.json',result)
archive=ROOT/'scripts/experiment_summaries'/run;archive.mkdir(exist_ok=False)
backup=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups')/run;backup.mkdir(exist_ok=False)
for f in out.iterdir():
 shutil.copy2(f,archive/f.name);shutil.copy2(f,backup/f.name);assert sha(f)==sha(backup/f.name)
shutil.copy2(ROOT/'scripts/q01_protocol.json',archive/'protocol.json')
completion=dict(id=meta['id'],run_id=run,status='completed',started_at=meta['started_at'],ended_at=result['ended_at'],code_commit=meta['code_commit'],claim_commit=args.claim,updates=0,
 verdict=result['verdict'],macro_interaction_log=macro.tolist(),maxT_adjusted_p=adjusted_p,backup=str(backup),artifact_sha256={f.name:sha(f) for f in out.iterdir()})
save(archive/'completion.json',completion)
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q01固定后处理配对诊断'],cwd=ROOT,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
with (state/'result_commit.txt').open('x') as f:f.write(commit+'\n')
save(state/'completion.json',completion);print(stamp(),'COMPLETE',commit,result['verdict'],flush=True)
