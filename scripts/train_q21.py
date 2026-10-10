"""Q21: two fresh single-view students, fixed Q10 source-selected teachers."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import train_q10_formal as base
from train_q10 import JointDetector,configuration,optimizer,seed
from train_p03_t01 import ROOT,IndexedDataset,save,sha,stamp,trainable_hash,target_images,evaluate

RUN='q21_t01_20261010';STATE=ROOT/'.state'/RUN;STORE=ROOT/'.state/artifacts';CONTROL=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism/q10_t01_20261006')
ARMS=['aligned','permuted'];base.STATE=STATE
read=lambda p:json.loads(p.read_text())

def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def make_teacher(config,arm,images,rows,out):
    rng=base.rng_state();t=JointDetector(config);kind='joint';t.arm=kind
    rec=read(ROOT/'.state/q10_t01_20261006/completion.json')['outputs'][kind]['checkpoints']['source_best'];path=Path(rec['checkpoint']);assert sha(path)==rec['checkpoint_sha256']
    t.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);t.requires_grad_(False);t.cuda();t.eval()
    response=evaluate(t,images,rows);old=read(CONTROL/'native192_responses.json')[kind+'_source_best'];error=float(np.max(np.abs(np.asarray(response['logits'])-np.asarray(old['logits']))));assert error<1e-5
    info=dict(checkpoint=str(path),checkpoint_sha256=rec['checkpoint_sha256'],source_best_epoch=read(ROOT/'.state/q10_t01_20261006/completion.json')['outputs'][kind]['source_best_epoch'],native192_max_error=error,state_sha256=state_hash(t),frozen=True)
    save(out/(arm+'_teacher.json'),info);base.restore_rng(rng);return t,info

def inference_cost(model,batch):
    model.eval();times=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(4):
            torch.cuda.synchronize();start=time.monotonic();pred=model(batch);assert torch.isfinite(pred['cls']).all();torch.cuda.synchronize();times.append(time.monotonic()-start)
    return dict(batch_size=int(batch['image'].shape[0]),median_seconds=float(np.median(times[1:])),peak_allocated_bytes=torch.cuda.max_memory_allocated(),scope='entire model forward, includes VAE/all views for PRD teacher; CUDA synchronized, eval')

def train_arm(arm,claim,smoke=False):
    out=STATE/('smoke' if (STATE/'cache_passed.json').exists() else 'budget') if smoke else STORE/RUN;out.mkdir(parents=True,exist_ok=True)
    config=configuration();config['task_target']=RUN;torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not (out/'protocol.json').exists():save(out/'protocol.json',read(ROOT/'scripts/q21_protocol.json'))
    previous=read(CONTROL/'run_metadata.json');source,ids,jsonhash=base.source_manifests(config);target=read(CONTROL/'target_manifest.json')
    for split in ['val','test']:assert source[split]==read(CONTROL/('source_'+split+'_manifest.json'))
    inputs={str(p):sha(p) for p in [Path(config['dataset_json_folder'])/'FaceForensics++.json',Path(config['clip_path'])/'model.safetensors',Path(config['vae_path'])/'diffusion_pytorch_model.bin']};assert inputs==previous['input_hashes'] and str(torch.__version__)==previous['torch_version']
    for domain,h in target['input_json_sha256'].items():assert sha(Path(config['dataset_json_folder'])/(domain+'.json'))==h
    for name,m in [('source_val_manifest',source['val']),('source_test_manifest',source['test']),('target_manifest',target)]:
        path=out/(name+'.json')
        if path.exists():assert read(path)==m
        else:save(path,m)
    images,rows=target_images(config,out) if not (out/'evaluation_manifest.json').exists() else target_images_to_existing(config,out)
    assert read(out/'evaluation_manifest.json')==read(CONTROL/'evaluation_manifest.json');reader=base.Reader(config)
    reference=[json.loads(x) for x in (CONTROL/'original_batch_trace.jsonl').read_text().splitlines()];assert len(reference)==21576
    seed();data=IndexedDataset(config=config,mode='train');assert len(data)==28768 and data.data_dict==read(CONTROL/'train_manifest.json')
    manifest=out/'train_manifest.json'
    if not manifest.exists():save(manifest,data.data_dict)
    else:assert read(manifest)==data.data_dict
    loader=torch.utils.data.DataLoader(data,batch_size=16,shuffle=True,num_workers=8,collate_fn=data.collate_fn,generator=torch.Generator().manual_seed(3407));assert len(loader)==1798
    model=JointDetector(config);model.arm='original';initial=trainable_hash(model);assert initial==previous['initial_trainable_sha256']['original'];model.cuda();opt=optimizer(model,config)
    online=not (STATE/'cache_passed.json').exists();assert smoke or not online
    teacher,teacher_info=make_teacher(config,arm,images,rows,out) if online else (None,read(STORE/RUN/'teacher_cache_info.json'));cached=None if online else np.load(STORE/RUN/'teacher_logits.npy');mapping=None if online else np.load(STORE/RUN/'donor_mapping.npy');teacher_times=[]
    if not online:assert sha(STORE/RUN/'teacher_logits.npy')==read(STATE/'cache_passed.json')['teacher_logits_sha256']
    started=stamp();code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    meta=dict(id='Q21-T01/v1',run_id=RUN,arm=arm,smoke=smoke,pid=os.getpid(),started_at=started,code_commit=code,claim_commit=claim,initial_trainable_sha256=initial,input_hashes=inputs,torch_version=torch.__version__,split_source_ids=ids,config=config,teacher=teacher_info,student_arm='original',loss='CE(student,y)+KL(teacher||student),T1,coefficient1,classsum/batchmean',head_rng_matches_q10_original=True)
    save(out/(arm+'_metadata.json'),meta);times=[];history=[];best=float('inf');best_epoch=None;step=0;start=time.monotonic();sums={'ce':0.0,'kl':0.0}
    with (out/(arm+'_batch_trace.jsonl')).open('x') as trace:
        for epoch in range(1 if smoke else 12):
            for b,ix in loader:
                current=dict(step=step,epoch=epoch,indexes=ix,image_sha256=hashlib.sha256(b['image'].numpy().tobytes()).hexdigest(),head_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest());assert current==reference[step],'Q21 augmented batch/head RNG differs from Q10 original'
                trace.write(json.dumps(current)+'\n');batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()};torch.cuda.synchronize();t0=time.monotonic()
                rng=base.rng_state()
                with torch.no_grad():tz=teacher(batch)['cls'] if online else torch.from_numpy(cached[np.arange(step*16,(step+1)*16) if arm=='aligned' else mapping[step*16:(step+1)*16]]).cuda();assert torch.isfinite(tz).all()
                torch.cuda.synchronize();teacher_times.append(time.monotonic()-t0)
                base.restore_rng(rng);model.train();opt.zero_grad(set_to_none=True);pred=model(batch);ce=model.get_losses(batch,pred)['overall'];kl=F.kl_div(F.log_softmax(pred['cls'],1),F.softmax(tz,1),reduction='batchmean');loss=ce+kl;assert torch.isfinite(loss) and float(kl)>-1e-6;loss.backward()
                gradient=None
                if step==0 or (step+1)%100==0:
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    gradient=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if 'lora_' in n and p.grad is not None);assert gradient>0 and (teacher is None or all(p.grad is None for p in teacher.parameters()))
                opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();elapsed=time.monotonic()-t0;times.append(elapsed);cv,kv,value=float(ce),float(kl),float(loss);sums['ce']+=cv;sums['kl']+=kv;del pred,ce,kl,loss,tz;step+=1
                if step==1 or step%100==0 or smoke:
                    trace.flush();v=dict(status='training',time=stamp(),pid=os.getpid(),arm=arm,epoch=epoch+1,steps=step,steps_total=21576,loss=value,ce=cv,kl=kv,lora_gradient_l1=gradient,update_seconds=elapsed,elapsed_seconds=time.monotonic()-start)
                    if not smoke:save(STATE/'status.json',v,replace=True)
                    print(stamp(),'TRAIN',json.dumps(v),flush=True)
                if smoke and step==8:break
            rng=base.rng_state();v0=time.monotonic();z=base.predict(model,reader,source['val'],arm+'_source_val_'+str(epoch+1),smoke);val=base.aggregate(z,source['val']);val_seconds=time.monotonic()-v0;base.restore_rng(rng);point=dict(epoch=epoch+1,steps=step,source_val=val['video'])
            if val['video']['balanced_ce']<best:
                best=val['video']['balanced_ce'];best_epoch=epoch+1
                if not smoke:torch.save(model.state_dict(),out/(arm+'_source_best.pth'))
            if epoch+1 in [2,4,8,12] or smoke:
                rng=base.rng_state();point['target192']=evaluate(model,images,rows);base.restore_rng(rng)
            history.append(point);save(out/(arm+'_learning_curve.json'),history,replace=True);print(stamp(),'EPOCH_DONE',arm,epoch+1,step,'source_best',best_epoch,'val',val['video'],flush=True)
    assert step==(8 if smoke else 21576) and trainable_hash(model)!=initial and (teacher is None or state_hash(teacher)==teacher_info['state_sha256']);assert (teacher is None or all(p.grad is None for p in teacher.parameters()))
    model.eval();batch64={'image':batch['image'].repeat(4,1,1,1),'label':batch['label'].repeat(4)};cost=dict(student_bs64=inference_cost(model,batch64),teacher_bs64=inference_cost(teacher,batch64) if online else teacher_info['teacher_bs64'],teacher_state_unchanged=True,teacher_grad_none=True,memory_scope='training preflight, both models/optimizer resident; final standalone deployment measurement authoritative')
    if smoke:
        rng=base.rng_state();replay_times=[];model.train()
        with torch.no_grad():
            for _ in range(4):
                torch.cuda.synchronize();t=time.monotonic();model(batch);torch.cuda.synchronize();replay_times.append(time.monotonic()-t)
        base.restore_rng(rng);cost['cache_replay_bs16']=float(np.median(replay_times[1:]));cost['teacher_bs16']=inference_cost(teacher,batch)['median_seconds'] if online else teacher_info['teacher_bs16']
        result=dict(status='smoke_passed',arm=arm,steps=8,initial_hash=initial,teacher=teacher_info,cost=cost,median_update_seconds=float(np.median(times[2:])),student_only_update_seconds=float(np.median(np.array(times[2:])-np.array(teacher_times[2:]))),source_val_seconds=val_seconds,source_val_videos=700,teacher_computation_in_update=True,batch_head_rng_matched=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),code_commit=code)
    else:
        del teacher;torch.cuda.empty_cache();torch.save(model.state_dict(),out/(arm+'_terminal.pth'));checkpoints={}
        for selected in ['source_best','terminal']:
            path=out/(arm+'_'+selected+'.pth');model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);first=evaluate(model,images,rows);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);second=evaluate(model,images,rows);err=float(np.max(np.abs(np.asarray(first['logits'])-np.asarray(second['logits']))));assert err<1e-5
            save(out/(arm+'_'+selected+'_native192.json'),first);src=base.predict(model,reader,source['test'],arm+'_'+selected+'_source');tgt=base.predict(model,reader,target,arm+'_'+selected+'_targets');np.save(out/(arm+'_'+selected+'_source_frame_logits.npy'),src);np.save(out/(arm+'_'+selected+'_target_frame_logits.npy'),tgt)
            checkpoints[selected]=dict(checkpoint=str(path),checkpoint_sha256=sha(path),strict_reload_max_error=err,source_test=base.aggregate(src,source['test']),target=base.aggregate(tgt,target))
        result=dict(status='arm_completed',arm=arm,started_at=started,ended_at=stamp(),steps=step,source_best_epoch=best_epoch,source_best_val_ce=best,initial_hash=initial,teacher=teacher_info,cost=cost,mean_ce=sums['ce']/step,mean_kl=sums['kl']/step,checkpoints=checkpoints,code_commit=code,claim_commit=claim)
    reader.close();save(out/(arm+'_summary.json'),result);del model,opt,batch,batch64;torch.cuda.empty_cache();return result

def target_images_to_existing(config,out):
    # Legacy helper saves canonical rows; isolate its write to a new local subdir.
    tmp=out/('native_load_'+str(os.getpid()));tmp.mkdir();images,rows=target_images(config,tmp);assert read(tmp/'evaluation_manifest.json')==read(out/'evaluation_manifest.json');(tmp/'evaluation_manifest.json').unlink();tmp.rmdir();return images,rows

def benchmark(claim):
    results={a:train_arm(a,claim,True) for a in ARMS};assert results[ARMS[0]]['initial_hash']==results[ARMS[1]]['initial_hash']
    t=[]
    for a in ARMS:
        x=results[a];eval_frames=12*700+4*192+2*(5600+45232)+2*192
        t.append(1.10*(21576*x['median_update_seconds']+eval_frames*x['cost']['student_bs64']['median_seconds']/64)+900)
    cache_seconds=read(STATE/'cache_passed.json')['elapsed_seconds'];group=sum(t)+cache_seconds+1800;result=dict(status='smoke_passed',time=stamp(),measurements=results,projected_seconds_per_arm=dict(zip(ARMS,t)),projected_group_seconds=group,arm_budget_seconds=54000,total_budget_seconds=172800,budget_pass=all(x<54000 and x+cache_seconds/2+1500<86400 for x in t) and group<172800,formal_updates=0)
    save(STATE/'smoke/result.json',result);print(stamp(),'SMOKE_RESULT',json.dumps(result),flush=True);assert result['budget_pass'],'Q21 full matched pair exceeds authorized candidate ceiling'

def finish(claim):
    out=STORE/RUN;outputs={a:read(out/(a+'_summary.json')) for a in ARMS};assert all(v['steps']==21576 and v['status']=='arm_completed' for v in outputs.values());assert outputs[ARMS[0]]['initial_hash']==outputs[ARMS[1]]['initial_hash']
    lines=[(out/(a+'_batch_trace.jsonl')).read_text().splitlines() for a in ARMS];assert len(lines[0])==21576 and lines[0]==lines[1]
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;config=configuration();reader=base.Reader(config);batch=reader.batch(read(out/'source_val_manifest.json')['frames'][:64]);deployment={}
    cost_inputs={a+'_student':(outputs[a]['checkpoints']['source_best']['checkpoint'],'original') for a in ARMS};cost_inputs['teacher']=(outputs['aligned']['teacher']['checkpoint'],'joint')
    for name,(path,kind) in cost_inputs.items():
        torch.cuda.empty_cache();model=JointDetector(config);model.arm=kind;model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);model.requires_grad_(False);model.cuda();model.eval();deployment[name]=inference_cost(model,batch);deployment[name]['memory_scope']='single model and same64 source-val inputs resident, no optimizer/other model';del model;torch.cuda.empty_cache()
    reader.close();del batch;torch.cuda.empty_cache();save(out/'deployment_costs.json',deployment)
    m=read(out/'target_manifest.json');y=np.array([v['label'] for v in m['videos']]);domains=np.array([v['domain'] for v in m['videos']]);zs=[np.asarray(outputs[a]['checkpoints']['source_best']['target']['video_logits'],dtype=np.float64) for a in ARMS];rng=np.random.default_rng(20261010);boot=[];strata=[np.flatnonzero((domains==d)&(y==label)) for d in sorted(set(domains)) for label in [0,1]]
    for _ in range(2000):
        samples=[rng.choice(ix,len(ix),replace=True) for ix in strata];deltas=[]
        for j in range(0,6,2):
            ix=np.concatenate(samples[j:j+2]);l,_=base.scores(zs[0][ix],y[ix]);r,_=base.scores(zs[1][ix],y[ix]);deltas.append(l['auc']-r['auc'])
        boot.append(float(np.mean(deltas)))
    ci=np.quantile(boot,[.025,.975]).tolist();delta=outputs[ARMS[0]]['checkpoints']['source_best']['target']['macro_video']['auc']-outputs[ARMS[1]]['checkpoints']['source_best']['target']['macro_video']['auc'];verdict='supports_sample_aligned_teacher_development_ranking_gain' if delta>=.005 and ci[0]>0 else 'not_supported' if delta<=0 else 'inconclusive'
    teacher_auc=read(ROOT/'.state/q10_t01_20261006/completion.json')['outputs']['joint']['checkpoints']['source_best']['target']['macro_video']['auc'];student_auc=outputs[ARMS[0]]['checkpoints']['source_best']['target']['macro_video']['auc']
    result=dict(id='Q21-T01/v1',run_id=RUN,status='completed',started_at=(STATE/'train_launched_at.txt').read_text().strip(),ended_at=stamp(),code_commit=outputs[ARMS[1]]['code_commit'],claim_commit=claim,steps_per_arm=21576,outputs=outputs,batch_augmentation_head_rng_identical=True,deployment_costs=deployment,primary=dict(delta_macro_video_auc=delta,paired_six_strata_auc_ci95=ci,replicates=2000,bootstrap_seed=20261010),verdict=verdict,teacher_minus_student_macro_auc=teacher_auc-student_auc,teacher_performance_retention_within_005=teacher_auc-student_auc<=.005,cache=read(STATE/'cache_passed.json'),limitations='Permutation preserves epoch-label-method distributions but breaks image difficulty/augmentation pairing; repeated target development/source test not independent finaltest;video bootstrap omits unknown identity/training-repeat dependence;single seed/8frames;distillation is not novel by itself;teacher cost included;no internal-reconstruction-mechanism recovery claim')
    save(out/'result.json',result);backup=ROOT/'.state/backups'/RUN;summary=ROOT/'scripts/experiment_summaries'/RUN;backup.mkdir(parents=True);summary.mkdir();assert sum(p.stat().st_size for p in out.iterdir())*2<=12*1024**3;hashes={}
    for f in out.iterdir():
        hashes[f.name]=sha(f);shutil.copy2(f,backup/f.name);assert sha(backup/f.name)==hashes[f.name]
        if f.suffix=='.json':shutil.copy2(f,summary/f.name)
    c=dict(result,artifact_sha256=hashes,backup=str(backup),archived_at=stamp());save(summary/'completion.json',c);assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip();subprocess.run(['git','add','--',str(summary.relative_to(ROOT))],cwd=ROOT,check=True);ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines();subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q21固定逐样本教师监督正确对应与跨视频置换配对训练'],cwd=ROOT,check=True);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save(STATE/'completion.json',c);(STATE/'result_commit.txt').write_text(commit+'\n');print(stamp(),'COMPLETE',commit,verdict,flush=True)


def loader_and_reference(config):
    ref=[json.loads(x) for x in (CONTROL/'original_batch_trace.jsonl').read_text().splitlines()];assert len(ref)==21576
    seed();data=IndexedDataset(config=config,mode='train');assert data.data_dict==read(CONTROL/'train_manifest.json')
    loader=torch.utils.data.DataLoader(data,batch_size=16,shuffle=True,num_workers=8,collate_fn=data.collate_fn,generator=torch.Generator().manual_seed(3407));return data,loader,ref

def donor_map():
    manifest=read(CONTROL/'train_manifest.json');ref=[json.loads(x) for x in (CONTROL/'original_batch_trace.jsonl').read_text().splitlines()];indices=np.array([i for x in ref for i in x['indexes']]);mapping=np.empty(len(indices),dtype=np.int64);proof=[]
    for epoch in range(12):
        groups={}
        for position in range(epoch*28768,(epoch+1)*28768):
            i=int(indices[position]);key=(int(manifest['label'][i]),int(manifest['spe_label'][i]));video=manifest['image'][i].replace('\\','/').rsplit('/',1)[0];groups.setdefault(key,[]).append((video,position))
        for key,rows in sorted(groups.items()):
            rows.sort();counts={}
            for video,position in rows:counts[video]=counts.get(video,0)+1
            n=len(rows);m=max(counts.values());assert len(counts)>=2 and 2*m<=n
            for j,(video,position) in enumerate(rows):
                donor,dp=rows[(j+m)%n];assert donor!=video;mapping[position]=dp
            positions=np.array([x[1] for x in rows]);assert np.array_equal(np.sort(mapping[positions]),np.sort(positions))
            proof.append(dict(epoch=epoch+1,label=key[0],method=key[1],records=n,max_video_records=m,videos=len(counts),bijection=True,cross_video=True))
    assert np.array_equal(np.sort(mapping),np.arange(len(indices)))
    return mapping,proof

def budget(claim):
    mapping,proof=donor_map();x=train_arm('aligned',claim,True);cache_estimate=21576*(x['cost']['teacher_bs16']+x['cost']['cache_replay_bs16']);frames=12*700+4*192+2*(5600+45232)+2*192
    arm=1.10*(21576*x['student_only_update_seconds']+frames*x['cost']['student_bs64']['median_seconds']/64)+900;cache_bound=1.10*cache_estimate+600;group=cache_bound+2*arm+2400
    result=dict(status='budget_precheck',cache_projected_seconds=cache_bound,student_projected_seconds=arm,group_projected_seconds=group,measurements=x,permutation_layers=proof,budget_pass=cache_bound<60000 and arm<54000 and arm+cache_bound/2+1500<86400 and group<172800,optimizer_steps=8,formal_updates=0)
    save(STATE/'budget/result.json',result);print(stamp(),'BUDGET_RESULT',json.dumps(result),flush=True);assert result['budget_pass']

def cache(claim):
    assert read(STATE/'budget/result.json')['budget_pass'];out=STORE/RUN;out.mkdir(parents=True);config=configuration();config['task_target']=RUN;torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    images,rows=target_images(config,out);data,loader,ref=loader_and_reference(config);model=JointDetector(config);model.arm='original';initial=trainable_hash(model);assert initial==read(CONTROL/'run_metadata.json')['initial_trainable_sha256']['original'];model.cuda();teacher,info=make_teacher(config,'cache',images,rows,out)
    mapping,proof=donor_map();np.save(out/'donor_mapping.npy',mapping);save(out/'donor_mapping_proof.json',proof);save(out/'protocol.json',read(ROOT/'scripts/q21_protocol.json'));save(out/'train_manifest.json',data.data_dict)
    values=np.empty((345216,2),dtype=np.float32);step=0;repeat_max=0.0;start=time.monotonic();save(out/'cache_metadata.json',dict(pid=os.getpid(),started_at=stamp(),claim_commit=claim,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),optimizer_steps=0))
    with (out/'cache_trace.jsonl').open('x') as trace:
        for epoch in range(12):
            for b,ix in loader:
                current=dict(step=step,epoch=epoch,indexes=ix,image_sha256=hashlib.sha256(b['image'].numpy().tobytes()).hexdigest(),head_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest());assert current==ref[step];trace.write(json.dumps(current)+'\n');batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()};rng=base.rng_state()
                with torch.no_grad():
                    tz=teacher(batch)['cls'];assert torch.isfinite(tz).all();values[step*16:(step+1)*16]=tz.cpu().numpy()
                base.restore_rng(rng)
                if step in [0,1797,7191,14383,21575]:
                    with torch.no_grad():again=teacher(batch)['cls'];error=float(torch.max(torch.abs(again-tz)));assert error<1e-5;repeat_max=max(repeat_max,error)
                    base.restore_rng(rng)
                model.train()
                with torch.no_grad():model(batch)
                step+=1
                if step==1 or step%100==0:
                    trace.flush();v=dict(status='caching',time=stamp(),pid=os.getpid(),steps=step,steps_total=21576,frames=step*16,parameter_updates=0,elapsed_seconds=time.monotonic()-start);save(STATE/'status.json',v,replace=True);print(stamp(),'CACHE',json.dumps(v),flush=True)
    assert step==21576 and trainable_hash(model)==initial and state_hash(teacher)==info['state_sha256'] and all(x.grad is None for x in teacher.parameters());np.save(out/'teacher_logits.npy',values)
    pvalues=torch.softmax(torch.from_numpy(values.astype(np.float64)),1).numpy();entropy=-np.sum(pvalues*np.log(pvalues),axis=1);labels=np.array([data.data_dict['label'][i] for row in ref for i in row['indexes']]);diag={}
    for method in sorted(set(data.data_dict['spe_label'])):
        select=np.array([data.data_dict['spe_label'][i]==method for row in ref for i in row['indexes']]);diag[str(method)]=dict(records=int(select.sum()),mean_confidence=float(pvalues[select].max(1).mean()),mean_entropy=float(entropy[select].mean()),teacher_error_fraction=float((pvalues[select].argmax(1)!=labels[select]).mean()))
    save(out/'teacher_distribution.json',diag);info.update(teacher_bs64=inference_cost(teacher,batch if len(batch['image'])==64 else {'image':batch['image'].repeat(4,1,1,1),'label':batch['label'].repeat(4)}),teacher_bs16=read(STATE/'budget/result.json')['measurements']['cost']['teacher_bs16'],teacher_state_unchanged=True,teacher_grad_none=True);save(out/'teacher_cache_info.json',info);save(STATE/'cache_passed.json',dict(status='cache_passed',elapsed_seconds=time.monotonic()-start,rows=345216,steps=21576,parameter_updates=0,teacher_logits_sha256=sha(out/'teacher_logits.npy'),trace_matches_q10=True,all_layers_cross_video_bijection=True,native_repeat_error_max=repeat_max));print(stamp(),'CACHE_COMPLETE',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--claim',required=True);a.add_argument('--budget',action='store_true');a.add_argument('--cache',action='store_true');a.add_argument('--smoke',action='store_true');a.add_argument('--arm',choices=ARMS);a.add_argument('--finish',action='store_true');v=a.parse_args()
    if v.budget:budget(v.claim)
    elif v.cache:cache(v.claim)
    elif v.smoke:benchmark(v.claim)
    elif v.finish:finish(v.claim)
    else:assert v.arm and read(STATE/'smoke/result.json')['budget_pass'];train_arm(v.arm,v.claim)
