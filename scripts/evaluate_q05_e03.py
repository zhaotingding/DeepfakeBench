"""Q05-E03: fixed terminal checkpoints on held-out source videos, zero updates."""
import argparse,copy,json,os,subprocess,shutil,time
from pathlib import Path
import lmdb
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from train_p03_t01 import ROOT,save,sha,stamp,target_images,evaluate,DeepfakeAbstractBaseDataset
from diagnose_q05_d02 import SwapDetector


def scores(logits,labels):
    z=np.asarray(logits,dtype=np.float64);y=np.asarray(labels)
    ce=np.logaddexp(z[:,0],z[:,1])-z[np.arange(len(y)),y]
    prob=np.exp(z[:,1]-np.logaddexp(z[:,0],z[:,1]))
    return dict(auc=float(roc_auc_score(y,prob)),ce=float(ce.mean()),
                balanced_ce=float(.5*(ce[y==0].mean()+ce[y==1].mean()))),ce


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--claim',required=True);args=parser.parse_args()
    run='q05_e03_20261006';state=ROOT/'.state'/run
    out=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism')/run;out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    base=out.parent;metadata=json.loads((base/'q05_t06_20261006/run_metadata.json').read_text())
    config=copy.deepcopy(metadata['config']);config['task_target']=run
    expected=json.loads((ROOT/'scripts/q05_e03_protocol.json').read_text())['input_preflight']
    domains=['Celeb-DF-v2','DFDCP','DFDC'];videos=[];frames=[];input_hashes={}
    for domain in domains:
        path=Path(config['dataset_json_folder'])/(domain+'.json');input_hashes[domain]=sha(path)
        assert input_hashes[domain]==expected[domain]['json_sha256']
        data=json.loads(path.read_text())[domain]
        before=len(videos);before_frames=len(frames)
        for kind,part in sorted(data.items()):
            for video,item in sorted(part['test'].items()):
                available=sorted(item['frames']);assert available
                ix=np.linspace(0,len(available)-1,min(8,len(available))).round().astype(int);assert len(set(ix))==len(ix)
                selected=[available[i] for i in ix];label=config['label_dict'][item['label']];assert label in [0,1]
                vid=len(videos);videos.append(dict(domain=domain,kind=kind,video_id=video,label=label,frames=selected,frame_start=len(frames),frame_count=len(selected)))
                frames.extend(dict(domain=domain,video_index=vid,frame=f,label=label) for f in selected)
        part=videos[before:];assert len(part)==expected[domain]['videos'] and sum(v['label']==0 for v in part)==expected[domain]['real']
        assert len(frames)-before_frames==expected[domain]['frames']
    assert len(videos)==5876 and len(frames)==45232
    manifest=dict(videos=videos,frames=frames,input_json_sha256=input_hashes,sampling='Up to8equally spaced sorted available frames/video')
    save(out/'target_manifest.json',manifest)
    reused=out.parent/'q05_e02_20261006'
    reused_meta=json.loads((reused/'run_metadata.json').read_text())
    reused_result=json.loads((reused/'result.json').read_text())
    reused_completion=json.loads((ROOT/'.state/q05_e02_20261006/completion.json').read_text())
    assert sha(out/'target_manifest.json')==reused_meta['target_manifest_sha256']
    assert sha(reused/'frame_logits.npy')==reused_completion['artifact_sha256']['frame_logits.npy']
    native_images,native_rows=target_images(config,out)
    assert sha(out/'evaluation_manifest.json')==metadata['evaluation_manifest_sha256']
    sources={}
    for arm,task,key in [('dual','q05_t03_20261005','increment'),('fixed','q05_t06_20261006','fixed_pre')]:
        result=json.loads((base/task/'result.json').read_text());sources[arm]=result['outputs'][key]
        assert sha(Path(sources[arm]['checkpoint']))==sources[arm]['checkpoint_sha256']
        assert sha(base/task/'train_manifest.json')==metadata['train_manifest_sha256']
    assert sources['fixed']['checkpoint_sha256']==reused_meta['checkpoint_inputs']['fixed']['checkpoint_sha256']
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    meta=dict(id='Q05-E03/v1',run_id=run,started_at=stamp(),pid=os.getpid(),cwd=str(ROOT),code_commit=code,
              claim_commit=args.claim,config=config,checkpoint_inputs=sources,input_json_sha256=input_hashes,
              target_manifest_sha256=sha(out/'target_manifest.json'),updates=0,video_count=5876,
              frame_count=len(frames),domains=domains,torch_version=torch.__version__,
              sampling='Up to8equally spaced sorted available frames/video, no augmentation',
              primary='Dual T03 vs fixed T06 full target macrovideo; exact E02 fixed predictions reused',fixed_prediction_reused_from='q05_e02_20261006',fixed_prediction_sha256=sha(reused/'frame_logits.npy'))
    save(out/'run_metadata.json',meta);print(stamp(),'STARTED',len(frames),'target frames per checkpoint',flush=True)
    weights_fixed=torch.load(sources['fixed']['checkpoint'],map_location='cpu',weights_only=True)
    anchor={k:v for k,v in weights_fixed.items() if k.startswith('anchor_backbone.')};assert anchor
    envs={domain:lmdb.open(str(Path(config['lmdb_dir'])/(domain+'_lmdb')),readonly=True,lock=False,create=False) for domain in domains}
    loaders={}
    for domain in domains:
        loader=object.__new__(DeepfakeAbstractBaseDataset);loader.config=config;loader.lmdb=True;loader.env=envs[domain];loaders[domain]=loader
    outputs={'fixed':copy.deepcopy(reused_result['outputs']['fixed'])};frame_logits=[];native_errors={};labels=np.array([v['label'] for v in videos])
    for arm in ['dual']:
        model=SwapDetector(config)
        weights=weights_fixed if arm=='fixed' else torch.load(sources[arm]['checkpoint'],map_location='cpu',weights_only=True)
        if arm!='fixed':weights.update(anchor)
        model.load_state_dict(weights,strict=True)
        if arm!='fixed':del weights
        model.requires_grad_(False);model.eval();model.cuda();model.reference='fixed' if arm=='fixed' else 'moving'
        native=evaluate(model,native_images,native_rows)
        err=float(np.max(np.abs(np.array(native['logits'])-np.array(sources[arm]['logits']))));assert err<1e-5
        native_errors[arm]=err
        chunks=[];t0=time.monotonic()
        with torch.inference_mode():
            for start in range(0,len(frames),64):
                rows=frames[start:start+64]
                x=torch.stack([loaders[row['domain']].normalize(loaders[row['domain']].to_tensor(loaders[row['domain']].load_rgb(row['frame']))) for row in rows]).cuda()
                y=torch.tensor([row['label'] for row in rows],device='cuda')
                z=model({'image':x,'label':y})['cls'];assert torch.isfinite(z).all()
                chunks.append(z.cpu().numpy())
                if start==0 or (start//64)%10==0 or start+len(rows)==len(frames):
                    progress=dict(status='evaluating',pid=os.getpid(),arm=arm,arms_completed=len(outputs),
                                  frames_done=start+len(rows),frames_total=len(frames),elapsed_seconds=time.monotonic()-t0,time=stamp(),updates=0)
                    save(state/'status.json',progress,replace=True);print(stamp(),'EVALUATE',json.dumps(progress),flush=True)
        z=np.concatenate(chunks);frame_logits.append(z)
        video_z=np.stack([z[v['frame_start']:v['frame_start']+v['frame_count']].mean(0) for v in videos])
        vm,loss=scores(video_z,labels);fm,_=scores(z,[v['label'] for v in frames])
        per_domain={}
        for domain in domains:
            vi=np.array([i for i,v in enumerate(videos) if v['domain']==domain]);fi=np.array([i for i,v in enumerate(frames) if v['domain']==domain])
            vmetrics,_=scores(video_z[vi],labels[vi]);fmetrics,_=scores(z[fi],[frames[i]['label'] for i in fi])
            per_domain[domain]=dict(video=vmetrics,frame=fmetrics,videos=len(vi),frames=len(fi))
        macro_video={k:float(np.mean([v['video'][k] for v in per_domain.values()])) for k in vm}
        outputs[arm]=dict(video=vm,frame=fm,domains=per_domain,macro_video=macro_video,video_logits=video_z.tolist(),video_ce=loss.tolist())
        del model;torch.cuda.empty_cache();print(stamp(),'ARM_DONE',arm,json.dumps(vm),flush=True)
    for env in envs.values():env.close()
    frame_logits.append(np.load(reused/'frame_logits.npy')[1])
    np.save(out/'frame_logits.npy',np.stack(frame_logits))
    delta=np.array(outputs['fixed']['video_ce'])-np.array(outputs['dual']['video_ce'])
    strata=[np.array([i for i,v in enumerate(videos) if v['domain']==domain and v['label']==label]) for domain in domains for label in [0,1]]
    rng=np.random.default_rng(20261006);boot=[];boot_auc=[]
    dual_z=np.array(outputs['dual']['video_logits']);fixed_z=np.array(outputs['fixed']['video_logits'])
    dual_p=np.exp(dual_z[:,1]-np.logaddexp(dual_z[:,0],dual_z[:,1]));fixed_p=np.exp(fixed_z[:,1]-np.logaddexp(fixed_z[:,0],fixed_z[:,1]))
    for _ in range(2000):
        samples=[rng.choice(ix,len(ix),replace=True) for ix in strata]
        boot.append(float(np.mean([delta[ix].mean() for ix in samples])))
        boot_auc.append(float(np.mean([roc_auc_score(labels[ix],dual_p[ix])-roc_auc_score(labels[ix],fixed_p[ix]) for ix in [np.concatenate(samples[k:k+2]) for k in [0,2,4]]])))
    ci=np.quantile(boot,[.025,.975]).tolist();auc_ci=np.quantile(boot_auc,[.025,.975]).tolist()
    dce=outputs['fixed']['macro_video']['balanced_ce']-outputs['dual']['macro_video']['balanced_ce']
    dauc=outputs['dual']['macro_video']['auc']-outputs['fixed']['macro_video']['auc']
    domain_effects={domain:dict(delta_balanced_ce=outputs['fixed']['domains'][domain]['video']['balanced_ce']-outputs['dual']['domains'][domain]['video']['balanced_ce'],delta_auc=outputs['dual']['domains'][domain]['video']['auc']-outputs['fixed']['domains'][domain]['video']['auc']) for domain in domains}
    verdict=('supports_dual_at_fixed_budget' if dce>=.02 and dauc>=.01 and ci[0]>0 and auc_ci[0]>0 and all(v['delta_balanced_ce']>=0 for v in domain_effects.values()) else 'mixed_or_no_prespecified_dual_support')
    result=dict(id=meta['id'],run_id=run,status='completed',started_at=meta['started_at'],ended_at=stamp(),
                code_commit=code,claim_commit=args.claim,updates=0,outputs=outputs,native_reload_max_error=native_errors,fixed_native_errors_reused=reused_result['native_reload_max_error']['fixed'],frame_logits_order=['dual','fixed'],
                primary=dict(delta_balanced_ce=dce,delta_auc=dauc,domain_label_video_bootstrap_ce_ci95=ci,domain_label_video_bootstrap_auc_ci95=auc_ci,strata=6,domain_effects=domain_effects,bootstrap_replicates=2000),
                verdict=verdict,limitations='Expanded coverage of repeatedly used development domains,not independent final test. Domain-labelvideo conditional bootstrap ignores shared identities and training repeats. Eightframes is fixed limited temporal sampling;one trainingseedshortbudget,no driftunique causalclaim.')
    save(out/'result.json',result)
    archive=ROOT/'scripts/experiment_summaries'/run;archive.mkdir(exist_ok=False)
    backup=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups')/run;backup.mkdir(exist_ok=False)
    assert sum(p.stat().st_size for p in out.iterdir())<50*1024**2
    hashes={}
    for f in out.iterdir():
        hashes[f.name]=sha(f);shutil.copy2(f,backup/f.name);assert sha(backup/f.name)==hashes[f.name]
        if f.suffix=='.json':shutil.copy2(f,archive/f.name)
    shutil.copy2(ROOT/'scripts/q05_e03_protocol.json',archive/'protocol.json')
    completion=dict(result,backup=str(backup),artifact_sha256=hashes,archived_at=stamp())
    save(archive/'completion.json',completion)
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
    ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
    subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q05双反传与固定参照全目标视频配对'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save(state/'completion.json',completion)
    with (state/'result_commit.txt').open('x') as f:f.write(commit+'\n')
    print(stamp(),'COMPLETE',commit,verdict,flush=True)


if __name__=='__main__':main()
