"""One Q13 total-residual arm, paired to the completed Q10 increment arm."""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
import train_q10_formal as base
from train_q10 import configuration,optimizer,seed
from train_q05_t03 import DirectionDetector,unit_residual
from train_p03_t01 import ROOT,IndexedDataset,save,sha,stamp,trainable_hash,target_images,evaluate

RUN='q13_t01_20261007';STATE=ROOT/'.state'/RUN
BASE=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism');CONTROL=BASE/'q10_t01_20261006'
base.STATE=STATE

class TotalDetector(DirectionDetector):
    def forward(self,data_dict,inference=False):
        self.backbone.eval();self.vae_augmenter.eval();x,y=data_dict['image'],data_dict['label']
        cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state();z=self.features(x)
        with torch.no_grad():rec=self.vae_augmenter(x,y)
        u=unit_residual(z-self.features(rec))
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
        logits=self.head(torch.cat([z,u],1));return dict(cls=logits,prob=logits.softmax(1)[:,1])

def archive(out,result):
    summary=ROOT/'scripts/experiment_summaries'/RUN;summary.mkdir(exist_ok=False)
    backup=BASE.parent/'checkpoint_backups'/RUN;backup.mkdir(exist_ok=False)
    assert sum(f.stat().st_size for f in out.iterdir())*2<=6*1024**3
    hashes={}
    for f in out.iterdir():
        hashes[f.name]=sha(f);shutil.copy2(f,backup/f.name);assert sha(backup/f.name)==hashes[f.name]
        if f.suffix=='.json':shutil.copy2(f,summary/f.name)
    shutil.copy2(ROOT/'scripts/q13_protocol.json',summary/'protocol.json')
    completion=dict(result,artifact_sha256=hashes,backup=str(backup),control_backup=json.loads((ROOT/'.state/q10_t01_20261006/completion.json').read_text())['backup'],archived_at=stamp())
    save(summary/'completion.json',completion)
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','add','--',str(summary.relative_to(ROOT))],cwd=ROOT,check=True)
    ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
    subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q13单位总残差与既有增量源选配对'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save(STATE/'completion.json',completion)
    with (STATE/'result_commit.txt').open('x') as f:f.write(commit+'\n')
    print(stamp(),'COMPLETE',commit,result['verdict'],flush=True)

def main(args):
    started=stamp();out=STATE/'smoke' if args.smoke else BASE/RUN;out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    config=configuration();config['task_target']=RUN
    control=json.loads((ROOT/'.state/q10_t01_20261006/completion.json').read_text());previous=json.loads((CONTROL/'run_metadata.json').read_text())
    assert control['status']=='completed' and control['steps_per_arm']==21576
    assert (ROOT/'.state/q10_t01_20261006/exit_code.txt').read_text().strip()=='0'
    source,ids,jsonhash=base.source_manifests(config)
    for split in ['val','test']:assert source[split]==json.loads((CONTROL/('source_'+split+'_manifest.json')).read_text())
    target=json.loads((CONTROL/'target_manifest.json').read_text());assert len(target['videos'])==5876 and len(target['frames'])==45232
    for domain,h in target['input_json_sha256'].items():assert sha(Path(config['dataset_json_folder'])/(domain+'.json'))==h
    inputs={str(p):sha(p) for p in [Path(config['dataset_json_folder'])/'FaceForensics++.json',Path(config['clip_path'])/'model.safetensors',Path(config['vae_path'])/'diffusion_pytorch_model.bin']}
    assert inputs==previous['input_hashes'] and str(torch.__version__)==previous['torch_version']
    reference=[json.loads(q) for q in (CONTROL/'joint_batch_trace.jsonl').read_text().splitlines()];assert len(reference)==21576
    for name,m in [('source_val_manifest',source['val']),('source_test_manifest',source['test']),('target_manifest',target)]:save(out/(name+'.json'),m)
    native_images,native_rows=target_images(config,out);reader=base.Reader(config)
    seed();data=IndexedDataset(config=config,mode='train');assert len(data)==28768 and data.data_dict==json.loads((CONTROL/'train_manifest.json').read_text())
    save(out/'train_manifest.json',data.data_dict)
    actual_ids={v for f in data.data_dict['image'] for v in f.replace(chr(92),'/').split('/')[-2].split('_')};assert actual_ids==set(ids['train'])
    loader=torch.utils.data.DataLoader(data,batch_size=16,shuffle=True,num_workers=8,collate_fn=data.collate_fn,generator=torch.Generator().manual_seed(3407));assert len(loader)==1798
    model=TotalDetector(config);initial=trainable_hash(model);assert initial==previous['initial_trainable_sha256']['joint'];model.cuda();opt=optimizer(model,config)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    metadata=dict(run_id=RUN,started_at=started,pid=os.getpid(),code_commit=code,claim_commit=args.claim,control_result_commit=(ROOT/'.state/q10_t01_20261006/result_commit.txt').read_text().strip(),initial_trainable_sha256=initial,input_hashes=inputs,torch_version=str(torch.__version__),source_ids=ids,config=config,git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()))
    save(out/'run_metadata.json',metadata)
    history=[];step=0;best=float('inf');best_epoch=None;times=[];t0=time.monotonic()
    with (out/'batch_trace.jsonl').open('x') as trace:
        for epoch in range(1 if args.smoke else 12):
            for b,ix in loader:
                current=dict(step=step,epoch=epoch,indexes=ix,image_sha256=hashlib.sha256(b['image'].numpy().tobytes()).hexdigest(),head_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())
                assert current==reference[step],'Q13/Q10 index, augmentation or head RNG mismatch';trace.write(json.dumps(current)+'\n')
                batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()};torch.cuda.synchronize();st=time.monotonic()
                model.train();opt.zero_grad(set_to_none=True);pred=model(batch);loss=model.get_losses(batch,pred)['overall'];assert torch.isfinite(loss);loss.backward();gradient=None
                if step==0 or step%100==0:
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    gradient=sum(float(p.grad.detach().abs().sum()) for n,p in model.named_parameters() if 'lora_' in n and p.grad is not None);assert gradient>0
                opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();times.append(time.monotonic()-st);value=float(loss);del pred,loss;step+=1
                if step==1 or step%100==0:
                    trace.flush();v=dict(status='training',time=stamp(),pid=os.getpid(),arm='total',epoch=epoch+1,steps=step,steps_total=21576,loss=value,lora_gradient_l1=gradient,elapsed_seconds=time.monotonic()-t0)
                    if not args.smoke:save(STATE/'status.json',v,replace=True)
                    print(stamp(),'TRAIN',json.dumps(v),flush=True)
                if args.smoke and step==8:break
            rng=base.rng_state();st=time.monotonic();z=base.predict(model,reader,source['val'],'total_source_val_'+str(epoch+1),args.smoke);eval_seconds=time.monotonic()-st;val=base.aggregate(z,source['val']);base.restore_rng(rng)
            point=dict(epoch=epoch+1,steps=step,source_val=val['video'])
            if val['video']['balanced_ce']<best:
                best=val['video']['balanced_ce'];best_epoch=epoch+1
                if not args.smoke:torch.save(model.state_dict(),out/'total_source_best.pth')
            if epoch+1 in [2,4,8,12] or args.smoke:
                rng=base.rng_state();point['target192']=evaluate(model,native_images,native_rows);base.restore_rng(rng)
            history.append(point);save(out/'learning_curve.json',history,replace=True);print(stamp(),'EPOCH_DONE',epoch+1,step,'source_best',best_epoch,'val',val['video'],flush=True)
    if args.smoke:
        estimate=1.10*(21576*float(np.median(times[2:]))+(12*700+4*192+2*(5600+45232))*eval_seconds/700)+1800
        result=dict(status='smoke_passed' if estimate<71400 else 'budget_not_passed',steps=8,trace_matches=True,initial_hash=initial,update_seconds=float(np.median(times[2:])),source_val_seconds=eval_seconds,projected_seconds_with_margin=estimate,formal_limit_seconds=71400,budget_pass=estimate<71400,code_commit=code,time=stamp())
        save(out/'result.json',result);print(stamp(),'SMOKE_RESULT',json.dumps(result),flush=True);assert result['budget_pass'],'Finite Q13 budget not met';reader.close();return
    assert step==21576 and trainable_hash(model)!=initial
    torch.save(model.state_dict(),out/'total_terminal.pth');outputs={}
    for selected in ['source_best','terminal']:
        path=out/('total_'+selected+'.pth');model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        response=evaluate(model,native_images,native_rows);model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);again=evaluate(model,native_images,native_rows)
        err=float(np.abs(np.array(response['logits'])-np.array(again['logits'])).max());assert err<1e-5
        zs=base.predict(model,reader,source['test'],'total_'+selected+'_source_test');zt=base.predict(model,reader,target,'total_'+selected+'_targets')
        np.save(out/(selected+'_source_frame_logits.npy'),zs);np.save(out/(selected+'_target_frame_logits.npy'),zt)
        outputs[selected]=dict(checkpoint=str(path),checkpoint_sha256=sha(path),strict_reload_max_error=err,source_test=base.aggregate(zs,source['test']),target=base.aggregate(zt,target))
    reader.close();total=outputs['source_best']['source_test'];increment=control['outputs']['joint']['checkpoints']['source_best']['source_test']
    delta=np.array(total['video_ce'])-np.array(increment['video_ce']);y=np.array([v['label'] for v in source['test']['videos']]);groups=sorted({v['group'] for v in source['test']['videos']});parts=[]
    for group in groups:
        ix=np.array([i for i,v in enumerate(source['test']['videos']) if v['group']==group]);parts.append([delta[ix][y[ix]==0].sum(),delta[ix][y[ix]==1].sum(),sum(y[ix]==0),sum(y[ix]==1)])
    assert len(groups)==70;parts=np.array(parts);rng=np.random.default_rng(20261006);boot=[]
    for _ in range(2000):
        v=parts[rng.integers(0,70,70)].sum(0);boot.append(float(.5*(v[0]/v[2]+v[1]/v[3])))
    ci=np.quantile(boot,[.025,.975]).tolist();dc=total['video']['balanced_ce']-increment['video']['balanced_ce'];da=increment['video']['auc']-total['video']['auc']
    verdict='supports_increment_over_total' if dc>=.02 and da>=.01 and ci[0]>0 else ('not_supported' if dc<=0 or da<=0 else 'inconclusive')
    result=dict(id='Q13-T01/v1',run_id=RUN,status='completed',started_at=started,ended_at=stamp(),code_commit=code,claim_commit=args.claim,new_steps=21576,control_steps_reused=21576,initial_hash=initial,full_trace_matches=True,source_best_epoch=best_epoch,source_best_val_ce=best,outputs=outputs,control_result_commit=metadata['control_result_commit'],primary=dict(delta_balanced_ce_total_minus_increment=dc,delta_auc_increment_minus_total=da,source_component_bootstrap_ce_ci95=ci,groups=70,replicates=2000),verdict=verdict,early_stop_epoch=None,limitations='Source test previously explored and targets repeated development; single seed;8frame sampling;filename groups not person identities;2vs3 encoder forwards; relative residual-definition gain does not establish gain over original or universal causal mechanism')
    save(out/'result.json',result);archive(out,result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--claim',required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
