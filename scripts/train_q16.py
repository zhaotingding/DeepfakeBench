"""Q16: frozen CLIP residual representations, source-selected equal linear probes."""
import argparse,copy,json,shutil,subprocess,time
from pathlib import Path
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from train_q05_t03 import DirectionDetector
from train_p03_t01 import ROOT,LoraDetector,DeepfakeAbstractBaseDataset,target_images,save,sha,stamp

ARMS=['curvature','midpoint','first_order']
RUN='q16_p01_20261008'

def metrics(logits,y):
 loss=F.cross_entropy(logits,y,reduction='none').numpy();labels=y.numpy()
 prob=logits.softmax(1)[:,1].numpy()
 return dict(auc=float(roc_auc_score(labels,prob)),ce=float(loss.mean()),balanced_ce=float(.5*(loss[labels==0].mean()+loss[labels==1].mean())),loss=loss.tolist(),probability=prob.tolist())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--claim',required=True);args=parser.parse_args()
 state=ROOT/'.state'/RUN;out=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism')/RUN;out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(4);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 source=json.loads((ROOT/'scripts/experiment_summaries/n01_t04_20261004/run_metadata.json').read_text());config=dict(source['config']);config['task_target']=RUN
 assert all(sha(Path(path))==h for path,h in source['input_hashes'].items())
 manifest_path=Path(config['dataset_json_folder'])/'FaceForensics++.json';dataset=json.loads(manifest_path.read_text())['FaceForensics++']
 rows=[];source_ids={};parents={}
 def find(x):
  parents.setdefault(x,x)
  if parents[x]!=x:parents[x]=find(parents[x])
  return parents[x]
 for split in ['train','val','test']:
  source_ids[split]=set()
  for kind,part in sorted(dataset.items()):
   for video,item in sorted(part[split]['c23'].items()):
    ids=video.split('_');assert all(len(v)==3 and v.isdigit() for v in ids)
    source_ids[split].update(ids)
    for v in ids:parents[find(v)]=find(ids[0])
    rows.append(dict(domain='FaceForensics++',split=split,kind=kind,video_id=video,label=int(kind!='FF-real'),frame=sorted(item['frames'])[0],source_ids=ids))
 assert [sum(r['split']==s for r in rows) for s in ['train','val','test']]==[3596,700,700]
 assert all(not(source_ids[a]&source_ids[b]) for a,b in [('train','val'),('train','test'),('val','test')])
 for r in rows:r['group']=find(r['source_ids'][0])
 assert len({r['group'] for r in rows if r['split']=='test'})==70
 target,target_rows=target_images(config,out)
 assert sha(out/'evaluation_manifest.json')=='ac5121d948e23b5c00f36b30e606224380e2ef18a566a04c4dae8c60438e988a'
 nsource=len(rows)
 rows.extend(dict(domain=r['domain'],split='target_dev',video_id=r['video_id'],label=r['label'],frame=r['frame'],group=r['domain']+'/'+str(r['video_id'])) for r in target_rows)
 assert rows==json.loads((ROOT/'scripts/experiment_summaries/q04_t01_20261005/manifest.json').read_text())
 save(out/'manifest.json',rows)
 meta=dict(id='Q16-P01/v1',run_id=RUN,started_at=stamp(),claim_commit=args.claim,
  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config=config,weights=source['input_hashes'],
  protocol_sha256=sha(ROOT/'scripts/q16_protocol.json'),source_json_sha256=sha(manifest_path),manifest_sha256=sha(out/'manifest.json'),
  source_counts=[3596,700,700],test_source_components=70,target_development_videos=192,source_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
 save(out/'metadata.json',meta)
 gpu_phase=time.monotonic();model=DirectionDetector(config).cuda().eval();model.requires_grad_(False)
 protocol=json.loads((ROOT/'scripts/q16_protocol.json').read_text());assert protocol['input_dimensions']==1024 and protocol['arms']==ARMS
 assert all(torch.count_nonzero(v)==0 for n,v in model.named_parameters() if 'lora_B' in n)
 def representations(x):return model.features(x)
 env=lmdb.open(str(Path(config['lmdb_dir'])/'FaceForensics++_lmdb'),readonly=True,lock=False,create=False)
 loader=object.__new__(DeepfakeAbstractBaseDataset);loader.config=config;loader.lmdb=True;loader.env=env
 all_features=[];first_batch_checks={};t0=time.monotonic()
 with torch.inference_mode():
  for start in range(0,len(rows),16):
   batch=[]
   for i in range(start,min(start+16,len(rows))):
    batch.append(loader.normalize(loader.to_tensor(loader.load_rgb(rows[i]['frame']))) if i<nsource else target[i-nsource])
   x=torch.stack(batch).cuda();y=torch.tensor([r['label'] for r in rows[start:start+len(batch)]],device='cuda')
   assert time.monotonic()-gpu_phase<10800
   pre=model.preprocess(x);rec=model.vae_augmenter(x,y);a=representations(pre);b=representations(rec);mid=representations((pre+rec)/2);features=torch.stack([a-2*mid+b,mid,a-b],1)
   assert features.shape==(len(batch),3,1024) and torch.isfinite(features).all()
   if start==0:
    rec_error=float((rec-model.vae_augmenter(x,y)).abs().max());id_error=float((a-representations(pre)).abs().max())
    assert rec_error<1e-6 and id_error<1e-6
    estimate=1.10*(time.monotonic()-t0)*len(rows)/len(batch)+1800
    assert estimate<10800,'Q16 full extraction GPU budget not met'
    first_batch_checks=dict(vae_repeat_error=rec_error,strict_identity_repeat_error=id_error,shape=list(features.shape),projected_extraction_seconds=estimate,head_parameters=2050,backbone_grad_none=all(q.grad is None for q in model.parameters()))
   all_features.append(features.cpu())
   if start==0 or (start//16)%20==0 or start+len(batch)==len(rows):
    progress=dict(status='extracting',videos_done=start+len(batch),videos_total=len(rows),elapsed_seconds=time.monotonic()-t0,time=stamp(),first_batch_checks=first_batch_checks)
    save(state/'status.json',progress,replace=True);print(stamp(),'EXTRACT',progress['videos_done'],len(rows),'seconds',round(progress['elapsed_seconds'],2),flush=True)
 env.close();features=torch.cat(all_features);torch.save(features,out/'features.pt')
 del model,all_features;torch.cuda.empty_cache();cpu_phase=time.monotonic()
 labels=torch.tensor([r['label'] for r in rows],dtype=torch.long)
 indexes={s:torch.tensor([i for i,r in enumerate(rows) if r['split']==s]) for s in ['train','val','test','target_dev']}
 results={};predictions={};history={};checkpoint_sha={}
 for k,name in enumerate(ARMS):
  train=features[indexes['train'],k];mean=train.mean(0);std=train.std(0,unbiased=False).clamp_min(1e-6)
  x=(features[:,k]-mean)/std
  head=torch.nn.Linear(1024,2);torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
  optimizer=torch.optim.Adam(head.parameters(),lr=.01,weight_decay=.001)
  y=labels[indexes['train']];weights=torch.tensor([len(y)/(2*(y==c).sum()) for c in [0,1]])
  best=None;history[name]=[]
  for step in range(1,501):
   assert time.monotonic()-cpu_phase<3600
   optimizer.zero_grad();loss=F.cross_entropy(head(x[indexes['train']]),y,weight=weights);assert torch.isfinite(loss);loss.backward();optimizer.step()
   if step%10==0:
    with torch.no_grad(): val=metrics(head(x[indexes['val']]),labels[indexes['val']])
    history[name].append(dict(step=step,train_loss=float(loss.detach()),val_balanced_ce=val['balanced_ce'],val_auc=val['auc']))
    if best is None or val['balanced_ce']<best['val_balanced_ce']:
     best=dict(step=step,val_balanced_ce=val['balanced_ce'],state_dict=copy.deepcopy(head.state_dict()))
  checkpoint=out/(name+'_source_best.pth');torch.save(dict(state_dict=best['state_dict'],mean=mean,std=std,step=best['step']),checkpoint)
  checkpoint_sha[name]=sha(checkpoint);head.load_state_dict(best['state_dict'],strict=True)
  reloaded=torch.nn.Linear(1024,2);payload=torch.load(checkpoint,map_location='cpu',weights_only=True);reloaded.load_state_dict(payload['state_dict'],strict=True)
  with torch.no_grad():
   logits=head(x);again=reloaded((features[:,k]-payload['mean'])/payload['std']);assert torch.equal(logits,again)
  predictions[name]=dict(logits=logits.tolist());results[name]=dict(selected_step=best['step'],source_val_balanced_ce=best['val_balanced_ce'],checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha[name],strict_reload=True,total_updates=500)
  for split in ['val','test']:
   z=metrics(logits[indexes[split]],labels[indexes[split]]);results[name]['source_'+split]={key:z[key] for key in ['auc','ce','balanced_ce']}
  results[name]['target_dev']={}
  for dom in sorted({r['domain'] for r in rows if r['split']=='target_dev'}):
   ix=torch.tensor([i for i,r in enumerate(rows) if r['split']=='target_dev' and r['domain']==dom]);z=metrics(logits[ix],labels[ix])
   results[name]['target_dev'][dom]={key:z[key] for key in ['auc','ce','balanced_ce']}
  results[name]['target_macro']={key:float(np.mean([z[key] for z in results[name]['target_dev'].values()])) for key in ['auc','ce','balanced_ce']}
  save(state/'status.json',dict(status='probe_fitting',arms_completed=k+1,time=stamp()),replace=True);print(stamp(),'PROBE_DONE',name,best['step'],flush=True)
 test_ix=indexes['test'].tolist();test_rows=[rows[i] for i in test_ix];test_labels=labels[indexes['test']].numpy()
 group_ids=sorted({r['group'] for r in test_rows});assert len(group_ids)==70
 comparisons={}
 candidate=F.cross_entropy(torch.tensor(predictions['curvature']['logits'])[indexes['test']],labels[indexes['test']],reduction='none').numpy()
 for control in ['midpoint','first_order']:
  z=torch.tensor(predictions[control]['logits'])[indexes['test']];loss=F.cross_entropy(z,labels[indexes['test']],reduction='none').numpy();delta=loss-candidate;parts=[]
  for g in group_ids:
   ix=np.array([i for i,r in enumerate(test_rows) if r['group']==g]);parts.append([delta[ix][test_labels[ix]==0].sum(),delta[ix][test_labels[ix]==1].sum(),sum(test_labels[ix]==0),sum(test_labels[ix]==1)])
  parts=np.asarray(parts);rng=np.random.default_rng(20261008);boot=[]
  for _ in range(2000):
   v=parts[rng.integers(0,70,70)].sum(0);boot.append(.5*(v[0]/v[2]+v[1]/v[3]))
  ci=np.quantile(boot,[.025,.975]).tolist();dc=results[control]['source_test']['balanced_ce']-results['curvature']['source_test']['balanced_ce'];da=results['curvature']['source_test']['auc']-results[control]['source_test']['auc']
  comparisons[control]=dict(delta_balanced_ce=dc,delta_auc=da,paired_source_component_bootstrap_ce_ci95=ci,supported=dc>=.02 and da>=.01 and ci[0]>0)
 norms={name:features[:,k].norm(dim=1).numpy() for name,k in [('curvature',0),('displacement',2)]}
 strata={};norm_summary={}
 for name,v in norms.items():
  bounds=np.quantile(v[indexes['train'].numpy()],[.25,.5,.75])
  assignments=np.searchsorted(bounds,v,side='right')
  norm_summary[name]={split:dict(mean=float(v[ix.numpy()].mean()),quantiles=np.quantile(v[ix.numpy()],[0,.25,.5,.75,1]).tolist()) for split,ix in indexes.items()}
  strata[name]=dict(source_train_quartile_boundaries=bounds.tolist(),source_test_bins=[])
  for bin_id in range(4):
   ix=torch.tensor([i for i in indexes['test'].tolist() if assignments[i]==bin_id],dtype=torch.long)
   subset=dict(bin=bin_id,counts=[int((labels[ix]==c).sum()) for c in [0,1]],arms={})
   for arm in ARMS:
    if len(ix) and all(subset['counts']):
     z=metrics(torch.tensor(predictions[arm]['logits'])[ix],labels[ix])
     subset['arms'][arm]={key:z[key] for key in ['auc','ce','balanced_ce']}
   strata[name]['source_test_bins'].append(subset)
 supported=all(v['supported'] for v in comparisons.values());negative=any(v['delta_balanced_ce']<=0 or v['delta_auc']<=0 for v in comparisons.values())
 result=dict(id=meta['id'],run_id=RUN,status='completed',started_at=meta['started_at'],ended_at=stamp(),code_commit=meta['code_commit'],claim_commit=args.claim,
  arms=results,norm_summary=norm_summary,source_test_descriptive_strata=strata,encoder_updates=0,head_updates_per_arm=500,first_batch_checks=first_batch_checks,primary=dict(comparisons=comparisons,bootstrap_samples=2000,groups=70,bootstrap_seed=20261008),
  verdict='supports_fixed_midpoint_curvature_readability' if supported else ('not_supported' if negative else 'inconclusive'),
  limitations='Frozen CLS encoder path second difference/linear readability only,not reconstructor Hessian or manifold distance;midpoint mixed-image artifacts and amplitude explanations not excluded;single seed,test filename groups not certified identity;reused source/targetdevelopment not independent finaltest;no endtoend causal or novelty claim.')
 save(out/'result.json',result);save(out/'predictions.json',predictions);save(out/'history.json',history)
 archive=ROOT/'scripts/experiment_summaries'/RUN;archive.mkdir(exist_ok=False)
 backup=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups')/RUN;backup.mkdir(exist_ok=False)
 assert time.monotonic()-cpu_phase<3600
 assert 2*sum(f.stat().st_size for f in out.iterdir())<=2*1024**3
 hashes={}
 for f in out.iterdir():
  hashes[f.name]=sha(f);shutil.copy2(f,backup/f.name);assert sha(backup/f.name)==hashes[f.name]
  if f.suffix=='.json':shutil.copy2(f,archive/f.name)
 shutil.copy2(ROOT/'scripts/q16_protocol.json',archive/'protocol.json')
 completion=dict(id=meta['id'],run_id=RUN,status='completed',started_at=meta['started_at'],ended_at=result['ended_at'],code_commit=meta['code_commit'],claim_commit=args.claim,verdict=result['verdict'],primary=result['primary'],backup=str(backup),artifact_sha256=hashes)
 save(archive/'completion.json',completion)
 assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
 subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
 ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
 subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q16固定局部关系及三控制冻结线性探针'],cwd=ROOT,check=True)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 with (state/'result_commit.txt').open('x') as f:f.write(commit+'\n')
 save(state/'completion.json',completion);print(stamp(),'COMPLETE',commit,result['verdict'],flush=True)

if __name__=='__main__':main()
