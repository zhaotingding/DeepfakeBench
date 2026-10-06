"""Q05-E01: fixed terminal checkpoints on held-out source videos, zero updates."""
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
    run='q05_e01_20261006';state=ROOT/'.state'/run
    out=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism')/run;out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    base=out.parent;metadata=json.loads((base/'q05_t06_20261006/run_metadata.json').read_text())
    config=copy.deepcopy(metadata['config']);config['task_target']=run
    json_path=Path(config['dataset_json_folder'])/'FaceForensics++.json'
    assert sha(json_path)==metadata['input_hashes'][str(json_path)]
    dataset=json.loads(json_path.read_text())['FaceForensics++']
    split_ids={};counts={};parents={};videos=[];frames=[]
    def find(v):
        parents.setdefault(v,v)
        if parents[v]!=v:parents[v]=find(parents[v])
        return parents[v]
    for split in ['train','val','test']:
        split_ids[split]=set();counts[split]=0
        for kind,part in sorted(dataset.items()):
            for video,item in sorted(part[split]['c23'].items()):
                ids=video.split('_');assert all(len(v)==3 and v.isdigit() for v in ids)
                split_ids[split].update(ids);counts[split]+=1
                for v in ids:parents[find(v)]=find(ids[0])
                if split=='test':
                    available=sorted(item['frames']);assert available
                    indexes=np.linspace(0,len(available)-1,min(32,len(available))).round().astype(int)
                    assert len(set(indexes))==len(indexes)
                    selected=[available[i] for i in indexes]
                    vid=len(videos)
                    videos.append(dict(kind=kind,video_id=video,label=int(kind!='FF-real'),source_ids=ids,
                                       frames=selected,frame_start=len(frames),frame_count=len(selected)))
                    frames.extend(dict(video_index=vid,frame=f,label=int(kind!='FF-real')) for f in selected)
    assert counts=={'train':3596,'val':700,'test':700}
    assert [len(split_ids[s]) for s in ['train','val','test']]==[720,140,140]
    assert all(not(split_ids[a]&split_ids[b]) for a,b in [('train','val'),('train','test'),('val','test')])
    trained=json.loads((base/'q05_t03_20261005/train_manifest.json').read_text())['image']
    actual_ids={v for f in trained for v in f.replace(chr(92),'/').split('/')[-2].split('_')}
    assert len(trained)==28768 and actual_ids==split_ids['train'] and not(actual_ids&split_ids['test'])
    for v in videos:v['group']=find(v['source_ids'][0])
    assert len(videos)==700 and sum(v['label']==0 for v in videos)==140
    groups=sorted({v['group'] for v in videos});assert len(groups)==70
    manifest=dict(videos=videos,frames=frames,split_source_ids={s:sorted(v) for s,v in split_ids.items()},source_counts=counts)
    save(out/'source_manifest.json',manifest)
    native_images,native_rows=target_images(config,out)
    assert sha(out/'evaluation_manifest.json')==metadata['evaluation_manifest_sha256']
    sources={}
    for arm,task,key in [('dual','q05_t03_20261005','increment'),('moving','q05_t05_20261006','stop_pre'),('fixed','q05_t06_20261006','fixed_pre')]:
        result=json.loads((base/task/'result.json').read_text());sources[arm]=result['outputs'][key]
        assert sha(Path(sources[arm]['checkpoint']))==sources[arm]['checkpoint_sha256']
        assert sha(base/task/'train_manifest.json')==metadata['train_manifest_sha256']
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    meta=dict(id='Q05-E01/v1',run_id=run,started_at=stamp(),pid=os.getpid(),cwd=str(ROOT),code_commit=code,
              claim_commit=args.claim,config=config,checkpoint_inputs=sources,source_json_sha256=sha(json_path),
              source_manifest_sha256=sha(out/'source_manifest.json'),updates=0,video_count=700,
              frame_count=len(frames),source_components=70,torch_version=torch.__version__,
              sampling='Up to32equally spaced sorted available frames/video, no augmentation',
              primary='Fixed T06 vs moving T05 source video balancedCE/AUC; dual T03 descriptive only')
    save(out/'run_metadata.json',meta);print(stamp(),'STARTED',len(frames),'source frames per checkpoint',flush=True)
    weights_fixed=torch.load(sources['fixed']['checkpoint'],map_location='cpu',weights_only=True)
    anchor={k:v for k,v in weights_fixed.items() if k.startswith('anchor_backbone.')};assert anchor
    env=lmdb.open(str(Path(config['lmdb_dir'])/'FaceForensics++_lmdb'),readonly=True,lock=False,create=False)
    loader=object.__new__(DeepfakeAbstractBaseDataset);loader.config=config;loader.lmdb=True;loader.env=env
    outputs={};frame_logits=[];native_errors={};labels=np.array([v['label'] for v in videos])
    for arm in ['dual','moving','fixed']:
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
                x=torch.stack([loader.normalize(loader.to_tensor(loader.load_rgb(row['frame']))) for row in rows]).cuda()
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
        outputs[arm]=dict(video=vm,frame=fm,video_logits=video_z.tolist(),video_ce=loss.tolist())
        del model;torch.cuda.empty_cache();print(stamp(),'ARM_DONE',arm,json.dumps(vm),flush=True)
    env.close();np.save(out/'frame_logits.npy',np.stack(frame_logits))
    delta=np.array(outputs['moving']['video_ce'])-np.array(outputs['fixed']['video_ce'])
    summary=[]
    for group in groups:
        ix=np.array([i for i,v in enumerate(videos) if v['group']==group])
        summary.append([delta[ix][labels[ix]==0].sum(),delta[ix][labels[ix]==1].sum(),sum(labels[ix]==0),sum(labels[ix]==1)])
    summary=np.asarray(summary);rng=np.random.default_rng(20261006);boot=[]
    for _ in range(2000):
        v=summary[rng.integers(0,70,70)].sum(0);boot.append(.5*(v[0]/v[2]+v[1]/v[3]))
    ci=np.quantile(boot,[.025,.975]).tolist()
    dce=outputs['moving']['video']['balanced_ce']-outputs['fixed']['video']['balanced_ce']
    dauc=outputs['fixed']['video']['auc']-outputs['moving']['video']['auc']
    verdict=('supports_source_video_extension' if dce>=.02 and dauc>=.01 and ci[0]>0 else
             'not_supported' if dce<=0 or dauc<=0 else 'inconclusive')
    result=dict(id=meta['id'],run_id=run,status='completed',started_at=meta['started_at'],ended_at=stamp(),
                code_commit=code,claim_commit=args.claim,updates=0,outputs=outputs,native_reload_max_error=native_errors,
                primary=dict(delta_balanced_ce=dce,delta_auc=dauc,source_component_bootstrap_ce_ci95=ci,groups=70,bootstrap_replicates=2000),
                verdict=verdict,limitations='Same-source unseen videos of seen methods, no new model selection. Project previously used source test for diagnostics, not untouched final test. Filename-origin components not certified person identities. One trainingseed/shortbudget, no method-transfer or trainingcausal claim.')
    save(out/'result.json',result)
    archive=ROOT/'scripts/experiment_summaries'/run;archive.mkdir(exist_ok=False)
    backup=Path('/root/autodl-tmp/prd-common/outputs/logs/checkpoint_backups')/run;backup.mkdir(exist_ok=False)
    assert sum(p.stat().st_size for p in out.iterdir())<50*1024**2
    hashes={}
    for f in out.iterdir():
        hashes[f.name]=sha(f);shutil.copy2(f,backup/f.name);assert sha(backup/f.name)==hashes[f.name]
        if f.suffix=='.json':shutil.copy2(f,archive/f.name)
    shutil.copy2(ROOT/'scripts/q05_e01_protocol.json',archive/'protocol.json')
    completion=dict(result,backup=str(backup),artifact_sha256=hashes,archived_at=stamp())
    save(archive/'completion.json',completion)
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
    ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
    subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档Q05末权重源域留出视频完整复核'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save(state/'completion.json',completion)
    with (state/'result_commit.txt').open('x') as f:f.write(commit+'\n')
    print(stamp(),'COMPLETE',commit,verdict,flush=True)


if __name__=='__main__':main()
