"""One bounded Q10 pair: original versus VAE increment joint, source-val selection."""
import argparse,hashlib,json,os,random,shutil,subprocess,time
from pathlib import Path
import lmdb
import numpy as np
import torch
from train_p03_t01 import ROOT,IndexedDataset,DeepfakeAbstractBaseDataset,save,sha,stamp,trainable_hash,target_images,evaluate
from train_q10 import RUN,JointDetector,seed,configuration,optimizer
from evaluate_q05_e02 import scores

STATE=ROOT/'.state'/RUN
BASE=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism')

def rng_state():
    return random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state()

def restore_rng(v):
    random.setstate(v[0]);np.random.set_state(v[1]);torch.set_rng_state(v[2]);torch.cuda.set_rng_state(v[3])

def source_manifests(config):
    path=Path(config['dataset_json_folder'])/'FaceForensics++.json'
    data=json.loads(path.read_text())['FaceForensics++'];ids={};counts={};parent={};manifests={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for split in ['train','val','test']:
        ids[split]=set();counts[split]=0;videos=[];frames=[]
        for kind,part in sorted(data.items()):
            for video,item in sorted(part[split]['c23'].items()):
                origin=video.split('_');assert all(x.isdigit() and len(x)==3 for x in origin)
                ids[split].update(origin);counts[split]+=1
                for x in origin:parent[find(x)]=find(origin[0])
                if split=='train':continue
                available=sorted(item['frames']);assert available
                ix=[len(available)//2] if split=='val' else np.linspace(0,len(available)-1,min(8,len(available))).round().astype(int)
                selected=[available[i] for i in ix];assert len(set(selected))==len(selected)
                y=int(kind!='FF-real');vid=len(videos)
                videos.append(dict(domain='FaceForensics++',kind=kind,video_id=video,label=y,source_ids=origin,frame_start=len(frames),frame_count=len(selected),frames=selected))
                frames.extend(dict(domain='FaceForensics++',video_index=vid,frame=f,label=y) for f in selected)
        if split!='train':manifests[split]=dict(videos=videos,frames=frames)
    assert counts==dict(train=3596,val=700,test=700) and [len(ids[x]) for x in ['train','val','test']]==[720,140,140]
    assert all(not ids[a]&ids[b] for a,b in [('train','val'),('train','test'),('val','test')])
    for split,m in manifests.items():
        assert len(m['videos'])==700 and sum(v['label']==0 for v in m['videos'])==140
        for v in m['videos']:v['group']=find(v['source_ids'][0])
    assert len({v['group'] for v in manifests['test']['videos']})==70
    return manifests,{x:sorted(v) for x,v in ids.items()},sha(path)

class Reader:
    def __init__(self,config):self.config=config;self.loaders={}
    def batch(self,rows):
        xs=[]
        for row in rows:
            d=row['domain']
            if d not in self.loaders:
                loader=object.__new__(DeepfakeAbstractBaseDataset);loader.config=self.config;loader.lmdb=True
                loader.env=lmdb.open(str(Path(self.config['lmdb_dir'])/(d+'_lmdb')),readonly=True,lock=False,create=False);self.loaders[d]=loader
            loader=self.loaders[d];xs.append(loader.normalize(loader.to_tensor(loader.load_rgb(row['frame']))))
        return dict(image=torch.stack(xs).cuda(),label=torch.tensor([r['label'] for r in rows],device='cuda'))
    def close(self):
        for v in self.loaders.values():v.env.close()

def predict(model,reader,manifest,tag,smoke=False):
    model.eval();chunks=[];rows=manifest['frames'];t0=time.monotonic()
    with torch.inference_mode():
        for start in range(0,len(rows),64):
            z=model(reader.batch(rows[start:start+64]))['cls'];assert torch.isfinite(z).all();chunks.append(z.cpu().numpy())
            if start==0 or (start//64)%20==0 or start+len(z)==len(rows):
                progress=dict(status='evaluating',time=stamp(),pid=os.getpid(),tag=tag,frames_done=start+len(z),frames_total=len(rows),elapsed_seconds=time.monotonic()-t0)
                if not smoke:save(STATE/'status.json',progress,replace=True)
                print(stamp(),'EVAL',json.dumps(progress),flush=True)
    return np.concatenate(chunks)

def aggregate(z,m):
    labels=np.array([v['label'] for v in m['videos']]);vz=np.stack([z[v['frame_start']:v['frame_start']+v['frame_count']].mean(0) for v in m['videos']])
    video,loss=scores(vz,labels);frame,_=scores(z,[v['label'] for v in m['frames']]);domains={}
    for domain in sorted({v['domain'] for v in m['videos']}):
        vi=np.array([i for i,v in enumerate(m['videos']) if v['domain']==domain]);fi=np.array([i for i,v in enumerate(m['frames']) if v['domain']==domain])
        vm,_=scores(vz[vi],labels[vi]);fm,_=scores(z[fi],[m['frames'][i]['label'] for i in fi]);domains[domain]=dict(video=vm,frame=fm,videos=len(vi),frames=len(fi))
    macro={k:float(np.mean([v['video'][k] for v in domains.values()])) for k in video}
    return dict(video=video,frame=frame,domains=domains,macro_video=macro,video_logits=vz.tolist(),video_ce=loss.tolist())

def archive(out,result):
    summary=ROOT/'scripts/experiment_summaries'/RUN;summary.mkdir(exist_ok=False)
    backup=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups')/RUN;backup.mkdir(exist_ok=False)
    assert sum(p.stat().st_size for p in out.iterdir())*2<=12*1024**3
    hashes={}
    for f in out.iterdir():
        hashes[f.name]=sha(f);shutil.copy2(f,backup/f.name);assert sha(backup/f.name)==hashes[f.name]
        if f.suffix=='.json':shutil.copy2(f,summary/f.name)
    shutil.copy2(ROOT/'scripts/q10_formal_protocol.json',summary/'protocol.json')
    completion=dict(result,artifact_sha256=hashes,backup=str(backup),archived_at=stamp());save(summary/'completion.json',completion)
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','add','--',str(summary.relative_to(ROOT))],cwd=ROOT,check=True)
    ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
    subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q10原图与增量联合12轮源选配对实验'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save(STATE/'completion.json',completion)
    with (STATE/'result_commit.txt').open('x') as f:f.write(commit+'\n')
    print(stamp(),'COMPLETE',commit,result['verdict'],flush=True)

def main(args):
    out=STATE/'formal_smoke' if args.smoke else BASE/RUN;out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    config=configuration();source,ids,jsonhash=source_manifests(config)
    target=json.loads((BASE/'q05_e02_20261006/target_manifest.json').read_text());assert len(target['videos'])==5876 and len(target['frames'])==45232
    for domain,h in target['input_json_sha256'].items():assert sha(Path(config['dataset_json_folder'])/(domain+'.json'))==h
    save(out/'source_val_manifest.json',source['val']);save(out/'source_test_manifest.json',source['test']);save(out/'target_manifest.json',target)
    native_images,native_rows=target_images(config,out);reader=Reader(config)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    inputs={str(p):sha(p) for p in [Path(config['dataset_json_folder'])/'FaceForensics++.json',Path(config['clip_path'])/'model.safetensors',Path(config['vae_path'])/'diffusion_pytorch_model.bin']}
    previous=json.loads((BASE/'q05_t03_20261005/run_metadata.json').read_text())
    assert inputs==previous['input_hashes'] and str(torch.__version__)==previous['torch_version']
    assert json.loads((STATE/'smoke/result.json').read_text())['projected_seconds_with_margin']<115200
    started=stamp();meta=dict(id='Q10-T01/v1',run_id=RUN,smoke=args.smoke,started_at=started,pid=os.getpid(),cwd=str(ROOT),code_commit=code,claim_commit=args.claim,config=config,input_hashes=inputs,split_source_ids=ids,source_json_sha256=jsonhash,torch_version=torch.__version__,source_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),budget_seconds=115200,source_selection='minimum epoch source-val balancedCE, earliest tie; target never selects',initial_trainable_sha256={})
    save(out/'run_metadata.json',meta);outputs={};curves={};native_responses={};original_trace=[]
    for arm in ['original','joint']:
        seed();data=IndexedDataset(config=config,mode='train');assert len(data)==28768
        if arm=='original':save(out/'train_manifest.json',data.data_dict)
        else:assert data.data_dict==json.loads((out/'train_manifest.json').read_text())
        actual_ids={v for f in data.data_dict['image'] for v in f.replace(chr(92),'/').split('/')[-2].split('_')};assert actual_ids==set(ids['train'])
        loader=torch.utils.data.DataLoader(data,batch_size=16,shuffle=True,num_workers=8,collate_fn=data.collate_fn,generator=torch.Generator().manual_seed(3407));assert len(loader)==1798
        model=JointDetector(config);model.arm=arm;initial=trainable_hash(model)
        meta['initial_trainable_sha256'][arm]=initial;assert initial==json.loads((STATE/'smoke/result.json').read_text())['initial_hashes'][arm]
        if arm=='joint':assert initial==meta['initial_trainable_sha256']['original']
        save(out/'run_metadata.json',meta,replace=True);model.cuda();opt=optimizer(model,config)
        best=float('inf');best_epoch=None;history=[];step=0;t0=time.monotonic()
        with (out/(arm+'_batch_trace.jsonl')).open('x') as trace:
            for epoch in range(1 if args.smoke else 12):
                for b,ix in loader:
                    current=dict(step=step,epoch=epoch,indexes=ix,image_sha256=hashlib.sha256(b['image'].numpy().tobytes()).hexdigest(),head_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
                    if arm=='original':original_trace.append(current)
                    else:assert current==original_trace[step],'Pair augmentation/head RNG mismatch'
                    trace.write(json.dumps(current)+'\n');batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()}
                    model.train();opt.zero_grad(set_to_none=True);pred=model(batch);loss=model.get_losses(batch,pred)['overall'];assert torch.isfinite(loss);loss.backward()
                    gradient=None
                    if step==0 or step%100==0:
                        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                        gradient=sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters() if 'lora_' in n and p.grad is not None);assert gradient>0
                    opt.step();opt.zero_grad(set_to_none=True);value=float(loss);del pred,loss;step+=1
                    if step==1 or step%100==0:
                        trace.flush();v=dict(status='training',time=stamp(),pid=os.getpid(),arm=arm,epoch=epoch+1,steps=step,steps_total=21576,loss=value,lora_gradient_l1=gradient,elapsed_seconds=time.monotonic()-t0)
                        if not args.smoke:save(STATE/'status.json',v,replace=True)
                        print(stamp(),'TRAIN',json.dumps(v),flush=True)
                    if args.smoke and step==2:break
                rng=rng_state();z=predict(model,reader,source['val'],arm+'_source_val_'+str(epoch+1),args.smoke);val=aggregate(z,source['val']);restore_rng(rng)
                point=dict(epoch=epoch+1,steps=step,source_val=val['video'])
                if val['video']['balanced_ce']<best:
                    best=val['video']['balanced_ce'];best_epoch=epoch+1
                    if not args.smoke:torch.save(model.state_dict(),out/(arm+'_source_best.pth'))
                if epoch+1 in [2,4,8,12] or args.smoke:
                    rng=rng_state();point['target192']=evaluate(model,native_images,native_rows);restore_rng(rng)
                history.append(point);save(out/(arm+'_learning_curve.json'),history,replace=True)
                print(stamp(),'EPOCH_DONE',arm,epoch+1,step,'source_best',best_epoch,'val',val['video'],flush=True)
        assert step==(2 if args.smoke else 21576) and trainable_hash(model)!=initial
        curves[arm]=history;outputs[arm]=dict(steps=step,source_best_epoch=best_epoch,source_best_val_ce=best,checkpoints={})
        if args.smoke:
            model.eval();z=predict(model,reader,source['val'],arm+'_smoke_repeat',True);assert np.isfinite(z).all()
        else:
            torch.save(model.state_dict(),out/(arm+'_terminal.pth'))
            for selected in ['source_best','terminal']:
                path=out/(arm+'_'+selected+'.pth');weights=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(weights,strict=True);del weights
                response=evaluate(model,native_images,native_rows);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);again=evaluate(model,native_images,native_rows)
                err=float(np.abs(np.array(response['logits'])-np.array(again['logits'])).max());assert err<1e-5
                native_responses[arm+'_'+selected]=response
                zsrc=predict(model,reader,source['test'],arm+'_'+selected+'_source_test');ztgt=predict(model,reader,target,arm+'_'+selected+'_targets')
                np.save(out/(arm+'_'+selected+'_source_frame_logits.npy'),zsrc);np.save(out/(arm+'_'+selected+'_target_frame_logits.npy'),ztgt)
                outputs[arm]['checkpoints'][selected]=dict(checkpoint=str(path),checkpoint_sha256=sha(path),strict_reload_max_error=err,source_test=aggregate(zsrc,source['test']),target=aggregate(ztgt,target))
            save(out/(arm+'_summary.json'),outputs[arm]);save(out/'native192_responses.json',native_responses,replace=True)
        del model,opt,batch;torch.cuda.empty_cache()
    reader.close();assert meta['initial_trainable_sha256']['original']==meta['initial_trainable_sha256']['joint']
    if args.smoke:
        save(out/'result.json',dict(status='smoke_passed',steps_per_arm=2,initial_hashes=meta['initial_trainable_sha256'],batch_head_rng_matched=True,source_val_videos=700,code_commit=code,time=stamp()));print(stamp(),'FORMAL_SMOKE_PASS',flush=True);return
    original=outputs['original']['checkpoints']['source_best']['source_test'];joint=outputs['joint']['checkpoints']['source_best']['source_test'];delta=np.array(original['video_ce'])-np.array(joint['video_ce']);y=np.array([v['label'] for v in source['test']['videos']]);groups=sorted({v['group'] for v in source['test']['videos']});summary=[]
    for group in groups:
        ix=np.array([i for i,v in enumerate(source['test']['videos']) if v['group']==group]);summary.append([delta[ix][y[ix]==0].sum(),delta[ix][y[ix]==1].sum(),sum(y[ix]==0),sum(y[ix]==1)])
    summary=np.array(summary);rng=np.random.default_rng(20261006);boot=[]
    for _ in range(2000):
        v=summary[rng.integers(0,70,70)].sum(0);boot.append(float(.5*(v[0]/v[2]+v[1]/v[3])))
    ci=np.quantile(boot,[.025,.975]).tolist();dce=original['video']['balanced_ce']-joint['video']['balanced_ce'];dauc=joint['video']['auc']-original['video']['auc']
    verdict='supports_increment_joint_source_video_gain' if dce>=.02 and dauc>=.01 and ci[0]>0 else 'not_supported' if dce<=0 or dauc<=0 else 'inconclusive'
    result=dict(id='Q10-T01/v1',run_id=RUN,status='completed',started_at=started,ended_at=stamp(),code_commit=code,claim_commit=args.claim,steps_per_arm=21576,outputs=outputs,batch_augmentation_head_rng_identical=True,initial_hashes=meta['initial_trainable_sha256'],primary=dict(delta_balanced_ce=dce,delta_auc=dauc,source_component_bootstrap_ce_ci95=ci,groups=70,replicates=2000),verdict=verdict,early_stop_epoch=None,limitations='Source-test previously explored, not untouched finaltest; repeated target development only;8frame sampling;source filename components not verified identities;single training seed;original/joint compute differs, no information-independence or universal causal claim')
    save(out/'result.json',result);archive(out,result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--claim',required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
