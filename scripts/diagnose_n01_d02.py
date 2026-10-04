"""Fixed-checkpoint sign-only donor intervention; zero optimizer updates."""
import argparse,json,subprocess,shutil,copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_n01_t02 import DirectionDetector,unit_residual
from train_p03_t01 import ROOT,sha,stamp,save,target_images

p=argparse.ArgumentParser();p.add_argument('--claim',required=True);args=p.parse_args()
state=ROOT/'.state/n01_d02_20261004'
out=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism/n01_d02_20261004')
out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
prior=ROOT/'scripts/experiment_summaries/n01_t02_20261004'
old=json.loads((prior/'completion.json').read_text())
assert old['status']=='completed' and old['normal_training_exit']==0
config=json.loads((prior/'run_metadata.json').read_text())['config']
images,rows=target_images(config,out)
assert sha(out/'evaluation_manifest.json')=='ac5121d948e23b5c00f36b30e606224380e2ef18a566a04c4dae8c60438e988a'
labels=torch.tensor([r['label'] for r in rows],device='cuda')
domains=sorted({r['domain'] for r in rows})
meta=dict(id='N01-D02/v1',run_id='n01_d02_20261004',started_at=stamp(),claim_commit=args.claim,
 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 source_result_commit=(ROOT/'.state/n01_t02_20261004/result_commit.txt').read_text().strip(),
 protocol_sha256=sha(ROOT/'scripts/n01_d02_protocol.json'),manifest_sha256=sha(out/'evaluation_manifest.json'),
 source_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),updates=0)
save(out/'metadata.json',meta)
save(state/'status.json',dict(status='running',started_at=meta['started_at']),replace=True)
results={}
with torch.inference_mode():
 for arm in ['signed','abs']:
  v=old['outputs'][arm];assert sha(v['checkpoint'])==v['checkpoint_sha256']
  model=DirectionDetector(config);model.residual_form=arm
  weights=torch.load(v['checkpoint'],map_location='cpu',weights_only=True)
  model.load_state_dict(weights,strict=True);del weights
  model.cuda().eval();zs=[];us=[]
  for start in range(0,192,64):
   x=torch.stack(images[start:start+64]).cuda();y=labels[start:start+64]
   probe=model.vae_augmenter(x,y);z=model.features(x);u=unit_residual(z-model.features(probe))
   zs.append(z);us.append(u)
  z=torch.cat(zs);u=torch.cat(us)
  base=model.head(torch.cat([z,u if arm=='signed' else u.abs()],1))
  assert torch.max(torch.abs(base-torch.tensor(v['logits'],device='cuda')))<1e-4,'baseline replay mismatch'
  baseline_ce=F.cross_entropy(base,labels,reduction='none')
  magnitude=u.abs();sign=torch.where(u>=0,1.,-1.)
  assert torch.max(torch.abs(magnitude*sign-u))<1e-7
  per=[];negative_error=0.;max_norm_error=0.
  for i,row in enumerate(rows):
   same=[j for j,r in enumerate(rows) if r['domain']==row['domain'] and r['label']==row['label'] and j!=i]
   opposite=[j for j,r in enumerate(rows) if r['domain']==row['domain'] and r['label']!=row['label']]
   assert len(same)==31 and len(opposite)==32
   entry=dict(index=i,domain=row['domain'],video_id=row['video_id'],label=row['label'],baseline_ce=float(baseline_ce[i]))
   for name,donors in [('same',same),('opposite',opposite)]:
    intervention=magnitude[i].unsqueeze(0)*sign[donors]
    err=float((intervention.norm(dim=1)-1).abs().max());max_norm_error=max(max_norm_error,err);assert err<1e-5
    if arm=='abs':intervention=intervention.abs()
    logits=model.head(torch.cat([z[i].expand(len(donors),-1),intervention],1))
    ce=F.cross_entropy(logits,labels[i].expand(len(donors)),reduction='none')
    assert torch.isfinite(ce).all()
    entry[name]=dict(donor_indices=donors,ce=ce.cpu().tolist(),mean_ce=float(ce.mean()))
    if arm=='abs':negative_error=max(negative_error,float((logits-base[i]).abs().max()))
   entry['interaction_j']=entry['opposite']['mean_ce']-entry['same']['mean_ce'];per.append(entry)
  if arm=='abs':assert negative_error<1e-4,'abs negative control failed'
  by_domain={d:float(np.mean([x['interaction_j'] for x in per if x['domain']==d])) for d in domains}
  results[arm]=dict(checkpoint=v['checkpoint'],checkpoint_sha256=v['checkpoint_sha256'],baseline_metrics=v['metrics'],
   per_video=per,interaction_by_domain=by_domain,macro_j=float(np.mean(list(by_domain.values()))),
   abs_negative_max_logit_error=negative_error if arm=='abs' else None,max_norm_error=max_norm_error)
  torch.save(dict(z=z.cpu(),u=u.cpu(),head=copy.deepcopy(model.head).cpu().state_dict(),baseline_logits=base.cpu()),out/(arm+'_features_head.pt'))
  print(stamp(),'ARM_DONE',arm,'J',results[arm]['macro_j'],flush=True)
  del model,z,u,zs,us,base,magnitude,sign;torch.cuda.empty_cache()
verdict='supports_label_conditional_sign_response' if results['signed']['macro_j']>=.02 and min(results['signed']['interaction_by_domain'].values())>0 else 'not_supported_by_predefined_criterion'
result=dict(**meta,ended_at=stamp(),status='completed',verdict=verdict,outputs=results,
 independent_unit='192 recipient videos; donor pairs are dependent, not independent samples',
 limitations='Label-conditioned intervention on reused development set; off-manifold response, not retraining causality, population generalization or mediation proof.')
save(out/'result.json',result)
# Independent CE aggregation and exact source checkpoint immutability checks.
for arm,v in results.items():
 assert sha(v['checkpoint'])==v['checkpoint_sha256']
 for x in v['per_video']:
  assert abs(np.mean(x['opposite']['ce'])-np.mean(x['same']['ce'])-x['interaction_j'])<1e-5
archive=ROOT/'scripts/experiment_summaries/n01_d02_20261004';archive.mkdir(exist_ok=False)
backup=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups/n01_d02_20261004');backup.mkdir(exist_ok=False)
for f in out.iterdir():
 shutil.copy2(f,backup/f.name);assert sha(f)==sha(backup/f.name)
 if f.suffix=='.json':shutil.copy2(f,archive/f.name)
shutil.copy2(ROOT/'scripts/n01_d02_protocol.json',archive/'protocol.json')
completion=dict(id=meta['id'],run_id=meta['run_id'],status='completed',started_at=meta['started_at'],ended_at=result['ended_at'],
 code_commit=meta['code_commit'],claim_commit=args.claim,verdict=verdict,updates=0,
 macro_j=results['signed']['macro_j'],per_domain_j=results['signed']['interaction_by_domain'],
 negative_control_error=results['abs']['abs_negative_max_logit_error'],backup=str(backup),
 artifact_sha256={f.name:sha(f) for f in out.iterdir()})
save(archive/'completion.json',completion)
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档N01逐坐标符号标签条件置换诊断'],cwd=ROOT,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
with (state/'result_commit.txt').open('x') as f:f.write(commit+'\n')
save(state/'completion.json',completion)
print(stamp(),'COMPLETE',commit,verdict,flush=True)
