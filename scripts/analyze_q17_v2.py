"""Fixed endpoint and branch-gradient diagnosis; no optimizer or weight updates."""
import argparse,json,os,signal,subprocess,time,shutil
from pathlib import Path
import numpy as np
import torch
import train_q10_formal as base
from train_q10 import JointDetector,configuration,seed
from train_q05_t03 import unit_residual
from train_p03_t01 import ROOT,save,sha,stamp,trainable_hash,target_images,evaluate

RUN='q17_d01_20261009_v2';STATE=ROOT/'.state'/RUN
STORE=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism')
read=lambda p:json.loads(p.read_text())

class BranchDiagnostic(JointDetector):
    def forward(self,data_dict,inference=False):
        self.backbone.eval();self.vae_augmenter.eval();x,y=data_dict['image'],data_dict['label']
        cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
        z=self.features(x)
        with torch.no_grad():pre=self.preprocess(x);rec=self.vae_augmenter(x,y)
        u=unit_residual(self.features(pre)-self.features(rec))
        if self.path=='z':u=u.detach()
        if self.path=='u':z=z.detach()
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu)
        logits=self.head(torch.cat([z,u],1))
        return dict(cls=logits,prob=logits.softmax(1)[:,1])

def vec(loss,params,retain):
    grads=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
    return torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for p,g in zip(params,grads)]).cpu().numpy().astype(np.float64)

def geometry(gz,gu,gt):
    nz=float(np.linalg.norm(gz));nu=float(np.linalg.norm(gu));dot=float(gz@gu)
    err=float(np.linalg.norm(gz+gu-gt));rel=err/(nz+nu+1e-30)
    assert np.isfinite([nz,nu,dot,err,rel]).all() and rel<1e-4
    return dict(norm_z=nz,norm_u=nu,dot=dot,cosine=dot/(nz*nu) if nz>0 and nu>0 else None,reconstruction_l2=err,reconstruction_relative=rel)

def endpoint_analysis(out):
    q=STORE/'q10_t01_20261006';p=STORE/'q13_t01_20261007';m=read(q/'target_manifest.json');a=read(q/'evaluation_manifest.json')
    assert m==read(p/'target_manifest.json') and a==read(p/'evaluation_manifest.json')
    native=read(q/'native192_responses.json');q13curve=read(p/'learning_curve.json');q13c=read(ROOT/'.state/q13_t01_20261007/completion.json')
    assert q13c['source_best_epoch']==12 and 'target192' in q13curve[-1]
    result={};curves={'original':read(q/'original_learning_curve.json'),'joint':read(q/'joint_learning_curve.json'),'total':q13curve}
    assert all(len(v)==12 for v in curves.values())
    for cover,rows in [('target5876',m['videos']),('dev192',a)]:
        y=np.array([v['label'] for v in rows]);domains=np.array([v['domain'] for v in rows]);points={}
        for selected in ['source_best','terminal']:
            logits={}
            for arm in ['original','joint','total']:
                if cover=='dev192':z=np.asarray(q13curve[-1]['target192']['logits'] if arm=='total' else native[arm+'_'+selected]['logits'],dtype=np.float64)
                else:
                    zf=np.load((p if arm=='total' else q)/((selected if arm=='total' else arm+'_'+selected)+'_target_frame_logits.npy'))
                    assert zf.shape==(45232,2)
                    z=np.stack([zf[v['frame_start']:v['frame_start']+v['frame_count']].mean(0) for v in rows]).astype(np.float64)
                assert z.shape==(len(y),2) and np.isfinite(z).all();logits[arm]=z
            comparisons={}
            for left,right in [('original','joint'),('joint','total'),('original','total')]:
                records={}
                for domain in sorted(set(domains)):
                    ix=np.flatnonzero(domains==domain);l=logits[left][ix];r=logits[right][ix];labels=y[ix]
                    lm,lc=base.scores(l,labels);rm,rc=base.scores(r,labels);le=l.argmax(1)!=labels;re=r.argmax(1)!=labels;delta=lc-rc
                    records[domain]=dict(left=lm,right=rm,delta_auc_right_minus_left=rm['auc']-lm['auc'],delta_balanced_ce_left_minus_right=lm['balanced_ce']-rm['balanced_ce'],videos=len(ix),error_both=int((le&re).sum()),left_only_error=int((le&~re).sum()),right_only_error=int((~le&re).sum()),ce_delta_quantiles=np.quantile(delta,[.05,.25,.5,.75,.95]).tolist(),ce_delta_mean=float(delta.mean()),ce_delta_positive_fraction=float((delta>0).mean()))
                comparisons[left+'_vs_'+right]=records
            points[selected]=comparisons
        result[cover]=points
    save(out/'endpoint_analysis.json',result);save(out/'learning_curves.json',curves)
    return result

def archive(out,result):
    backup=STORE.parent/'checkpoint_backups'/RUN;summary=ROOT/'scripts/experiment_summaries'/RUN
    backup.mkdir(exist_ok=False);summary.mkdir(exist_ok=False);hashes={}
    assert sum(p.stat().st_size for p in out.iterdir())*2<=1024**3
    for p in out.iterdir():
        hashes[p.name]=sha(p);shutil.copy2(p,backup/p.name);assert sha(backup/p.name)==hashes[p.name]
        if p.suffix=='.json':shutil.copy2(p,summary/p.name)
    c=dict(result,artifact_sha256=hashes,backup=str(backup),archived_at=stamp());save(summary/'completion.json',c)
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','add','--',str(summary.relative_to(ROOT))],cwd=ROOT,check=True)
    ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
    subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q17固定端点与分支梯度诊断'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();save(STATE/'completion.json',c)
    (STATE/'result_commit.txt').write_text(commit+'\n');print(stamp(),'COMPLETE',commit,flush=True)

def main(claim):
    started=stamp();out=STORE/RUN;out.mkdir(exist_ok=False);protocol=read(ROOT/'scripts/q17_protocol_v2.json');save(out/'protocol.json',protocol)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    config=configuration();q=STORE/'q10_t01_20261006';previous=read(q/'run_metadata.json');control=read(ROOT/'.state/q10_t01_20261006/completion.json')
    assert control['status']=='completed' and str(torch.__version__)==previous['torch_version']
    for path,h in previous['input_hashes'].items():assert sha(Path(path))==h
    trace=[json.loads(v) for v in (q/'joint_batch_trace.jsonl').read_text().splitlines()];assert len(trace)==21576
    selected=[trace[round(i*1797/31)] for i in range(32)];assert selected==protocol['batches']
    train=read(q/'train_manifest.json');assert len(train['image'])==28768
    meta=dict(id='Q17-D01/v2',run_id=RUN,started_at=started,pid=os.getpid(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),claim_commit=claim,input_hashes=previous['input_hashes'],checkpoint_inputs={},parameter_updates=0,augmentation=False,eval_mode=True)
    save(out/'run_metadata.json',meta)
    signal.alarm(3500);gpu_start=time.monotonic();seed();model=BranchDiagnostic(config).cuda();model.arm='joint';model.path='both';model.eval()
    params=[p for n,p in model.named_parameters() if 'lora_' in n and p.requires_grad];assert sum(p.numel() for p in params)==3145728
    images,rows=target_images(config,out);assert read(out/'evaluation_manifest.json')==read(q/'evaluation_manifest.json');reader=base.Reader(config);native=read(q/'native192_responses.json');grad_rows=[];reload_errors={};unchanged={}
    for endpoint in ['source_best','terminal']:
        record=control['outputs']['joint']['checkpoints'][endpoint];path=Path(record['checkpoint']);assert sha(path)==record['checkpoint_sha256']
        meta['checkpoint_inputs'][endpoint]=dict(path=str(path),sha256=record['checkpoint_sha256']);save(out/'run_metadata.json',meta,replace=True)
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True);model.eval();model.path='both';before=trainable_hash(model)
        response=evaluate(model,images,rows);err=float(np.max(np.abs(np.array(response['logits'])-np.array(native['joint_'+endpoint]['logits']))));assert err<1e-5;reload_errors[endpoint]=err
        for bno,b in enumerate(selected):
            t0=time.monotonic();rr=[dict(domain='FaceForensics++',frame=train['image'][ix],label=train['label'][ix]) for ix in b['indexes']];batch=reader.batch(rr)
            methods=['real' if v['label']==0 else v['frame'].replace(chr(92),'/').split('/')[2] for v in rr]
            selectors={'all':list(range(16))};selectors.update({k:[i for i,v in enumerate(methods) if v==k] for k in sorted(set(methods))})
            values={};forward_reference=None
            for mode in ['both','z','u']:
                model.path=mode;pred=model(batch);z=pred['cls'].detach()
                if forward_reference is None:forward_reference=z
                else:assert float((forward_reference-z).abs().max())<1e-5
                vectors={}
                for j,(kind,indices) in enumerate(selectors.items()):
                    ix=torch.tensor(indices,device='cuda');small=dict(label=batch['label'][ix]);small_pred={'cls':pred['cls'][ix],'prob':pred['prob'][ix]}
                    loss=model.get_losses(small,small_pred)['overall'];assert torch.isfinite(loss)
                    vectors[kind]=vec(loss,params,j<len(selectors)-1)
                values[mode]=vectors;del pred
            for kind,indices in selectors.items():grad_rows.append(dict(endpoint=endpoint,batch_number=bno,trace_step=b['step'],kind=kind,samples=len(indices),method_counts={k:methods.count(k) for k in sorted(set(methods))},**geometry(values['z'][kind],values['u'][kind],values['both'][kind])))
            del values,batch,forward_reference;torch.cuda.synchronize();elapsed=time.monotonic()-t0
            if endpoint=='source_best' and bno==0:
                budget=1.10*64*elapsed+(time.monotonic()-gpu_start)+180
                assert budget<3500,'Q17 first complete gradient batch exceeds GPU budget'
                save(out/'budget_gate.json',dict(first_batch_seconds=elapsed,projected_gpu_seconds=budget,limit_seconds=3500,budget_pass=True))
            status=dict(status='gradient_diagnosis',time=stamp(),pid=os.getpid(),endpoint=endpoint,batches_done=(32 if endpoint=='terminal' else 0)+bno+1,batches_total=64,batch_seconds=elapsed,gpu_elapsed_seconds=time.monotonic()-gpu_start,parameter_updates=0)
            save(STATE/'status.json',status,replace=True);print(stamp(),'GRADIENT_BATCH',json.dumps(status),flush=True)
        unchanged[endpoint]=trainable_hash(model)==before;assert unchanged[endpoint]
    reader.close();del model,params;torch.cuda.empty_cache();signal.alarm(0);gpu_seconds=time.monotonic()-gpu_start;assert gpu_seconds<=3500
    save(out/'gradient_rows.json',grad_rows);save(STATE/'status.json',dict(status='cpu_analysis',pid=os.getpid(),time=stamp(),parameter_updates=0),replace=True)
    signal.alarm(3500);cpu_start=time.monotonic();endpoint_analysis(out);summaries={};competition=[]
    for ep in ['source_best','terminal']:
        by_kind={}
        for kind in sorted({r['kind'] for r in grad_rows}):
            rr=[r for r in grad_rows if r['endpoint']==ep and r['kind']==kind];angles=[r['cosine'] for r in rr if r['cosine'] is not None]
            by_kind[kind]=dict(batches=len(rr),undefined_angles=len(rr)-len(angles),negative_cosine_fraction=float(np.mean(np.array(angles)<0)) if angles else None,median_cosine=float(np.median(angles)) if angles else None,median_norm_z=float(np.median([r['norm_z'] for r in rr])),median_norm_u=float(np.median([r['norm_u'] for r in rr])),max_reconstruction_relative=max(r['reconstruction_relative'] for r in rr))
        a=by_kind['all'];assert a['batches']==32;competition.append(a['negative_cosine_fraction'] is not None and a['negative_cosine_fraction']>=.5 and a['median_cosine']<=-.1);summaries[ep]=by_kind
    cpu_seconds=time.monotonic()-cpu_start;assert cpu_seconds<=3500
    result=dict(id='Q17-D01/v2',run_id=RUN,status='completed',started_at=started,ended_at=stamp(),code_commit=meta['code_commit'],claim_commit=claim,parameter_updates=0,new_checkpoints=0,gradient_batches=64,gradient_rows=len(grad_rows),native_reload_errors=reload_errors,checkpoint_trainable_unchanged=unchanged,gpu_seconds=gpu_seconds,cpu_seconds=cpu_seconds,gradient_summary=summaries,stable_competition_hint=all(competition),priority_rule='Both endpoints:all32-batch negative cosine fraction>=.5 and median cosine<=-.1; otherwise lower Q19 priority; no causal or benefit claim',limitations='Fixed checkpoints/eval/no augmentation local gradients,not training trajectories;negative angle need not be harmful;zero angles undefined and retained;target repeated development,not finaltest;no oracle model selection;Q19 matched training still required')
    save(out/'result.json',result);archive(out,result);signal.alarm(0)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--claim',required=True);main(a.parse_args().claim)
