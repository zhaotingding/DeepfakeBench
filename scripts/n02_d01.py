"""N02-D01: strict identity and fixed norm-matched blur on a frozen VAE model."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path
from sklearn.metrics import roc_auc_score
from train_p03_t01 import ROOT, LoraDetector, np, torch, F, sha, stamp, save, target_images


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--claim',required=True);args=parser.parse_args()
    locks=[]
    for name in ('prd_training.lock','prd_mechanism_203.lock'):
        f=(ROOT/'.state'/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='prd-203-1-seed3407'
    assert subprocess.check_output(['git','rev-parse','origin/codex/prd-progress'],cwd=ROOT,text=True).strip()==args.claim
    out=ROOT/'.state/n02_d01_20261003';out.mkdir(exist_ok=False)
    source=Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism/p03_t01_20261003')
    meta=json.loads((source/'run_metadata.json').read_text())
    prior=json.loads((source/'result.json').read_text())['outputs']['paired']
    checkpoint=Path(prior['checkpoint']);assert sha(checkpoint)==prior['checkpoint_sha256']
    cache=ROOT/'.state/p03_d02_20261003/paired_features.npz'
    assert sha(cache)=='f9b5648fb8cf60c9df9cc1e7e9a038cca950bdb0c91ecce701bc5ad1f476ca59'
    subprocess.run(['git','diff','--exit-code',meta['code_commit'],'--','training','scripts/train_p03_t01.py'],cwd=ROOT,check=True)
    protocol=json.loads((ROOT/'scripts/n02_d01_protocol.json').read_text())
    started=dict(id='N02-D01/v1',time=stamp(),pid=os.getpid(),cwd=str(ROOT),claim=args.claim,
                 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 dirty_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
                 script_sha256=sha(__file__),protocol_sha256=sha(ROOT/'scripts/n02_d01_protocol.json'),
                 checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),cache_sha256=sha(cache))
    save(out/'started.json',started)
    torch.set_num_threads(4);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    images,rows=target_images(meta['config'],out)
    assert sha(out/'evaluation_manifest.json')==meta['evaluation_manifest_sha256']
    cached=np.load(cache);zref=torch.tensor(cached['original'],device='cuda');rv=torch.tensor(cached['residual'],device='cuda')
    model=LoraDetector(meta['config'])
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False).cuda()
    blur=model.build_probe_operator(dict(meta['config'],probe={'type':'gaussian_blur','sigma':1.0}),
                                   model.vae_augmenter.clip_mean,model.vae_augmenter.clip_std).cuda().eval()
    ys=torch.tensor([r['label'] for r in rows],device='cuda')
    originals,raw_blur,duplicate_residual=[],[],[]
    with torch.inference_mode():
        for start in range(0,192,64):
            x=torch.stack(images[start:start+64]).cuda();y=ys[start:start+64]
            z=model.features(x);z2=model.features(x);zb=model.features(blur(x,y))
            originals.append(z);raw_blur.append(z-zb);duplicate_residual.append(z-z2)
        z,rb,ri=map(torch.cat,(originals,raw_blur,duplicate_residual))
        original_error=(z-zref).abs().max().item();identity_error=ri.abs().max().item()
        assert original_error<=1e-4 and identity_error<=1e-5
        nv,nb=rv.norm(dim=1,keepdim=True),rb.norm(dim=1,keepdim=True)
        assert (nv>1e-8).all() and (nb>1e-8).all()
        matched=rb*(nv/nb)
        norm_error=((matched.norm(dim=1,keepdim=True)-nv).abs()/nv).max().item()
        assert norm_error<1e-5
        residuals={'vae':rv,'blur_raw_sigma1':rb,'blur_matched':matched,'identity_strict':torch.zeros_like(z)}
        logits={k:model.head(torch.cat([z,r],1)).cpu().numpy() for k,r in residuals.items()}
        assert np.allclose(logits['vae'],np.array(prior['logits']),atol=1e-4,rtol=1e-5)
        # Encoder-only diagnostic, not the original training protocol or a parameter update.
        x=torch.stack(images[:16]).cuda();model.backbone.train()
        cpu_rng,gpu_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
        a=model.features(x);b=model.features(x)
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng);c=model.features(x)
        dropout=dict(uncontrolled_max_abs=float((a-b).abs().max()),
                     uncontrolled_mean_l2=float((a-b).norm(dim=1).mean()),
                     replayed_rng_max_abs=float((a-c).abs().max()))
        assert dropout['replayed_rng_max_abs']<=1e-5
        model.eval()
    y=ys.cpu().numpy();domains=sorted({r['domain'] for r in rows});metrics={};all_ce={};verify={}
    for condition,a in logits.items():
        assert np.isfinite(a).all()
        loss=np.logaddexp(a[:,0].astype(float),a[:,1].astype(float))-a[np.arange(192),y]
        p=np.exp(a-a.max(1,keepdims=True));p=p[:,1]/p.sum(1)
        torch_loss=F.cross_entropy(torch.tensor(a),torch.tensor(y),reduction='none').numpy()
        verify[condition]=float(np.abs(loss-torch_loss).max());assert verify[condition]<1e-5
        all_ce[condition]=loss.tolist();metrics[condition]={}
        for domain in domains:
            ix=[i for i,r in enumerate(rows) if r['domain']==domain]
            metrics[condition][domain]=dict(ce=float(loss[ix].mean()),auc=float(roc_auc_score(y[ix],p[ix])))
    delta={d:dict(ce=metrics['blur_matched'][d]['ce']-metrics['vae'][d]['ce'],
                  auc=metrics['vae'][d]['auc']-metrics['blur_matched'][d]['auc']) for d in domains}
    dce=float(np.mean([v['ce'] for v in delta.values()]));dauc=float(np.mean([v['auc'] for v in delta.values()]))
    verdict=('supports_vae_direction_at_this_checkpoint' if dce>=0.02 and dauc>=0.01 and min(v['ce'] for v in delta.values())>=0 else
             'directional_advantage_not_supported' if dce<=0 or dauc<=0 else 'inconclusive')
    np.savez(out/'features_and_logits.npz',original=z.cpu().numpy(),vae_residual=rv.cpu().numpy(),blur_residual=rb.cpu().numpy(),
             blur_matched_residual=matched.cpu().numpy(),**{k+'_logits':v for k,v in logits.items()})
    save(out/'per_video_ce.json',all_ce)
    save(out/'verification.json',dict(original_cache_max_error=original_error,eval_duplicate_residual_max_abs=identity_error,
                                    matched_norm_relative_max_error=norm_error,dropout=dropout,independent_ce_max_error=verify))
    assert sha(checkpoint)==prior['checkpoint_sha256']
    result=dict(id='N02-D01/v1',status='complete',started_at=started['time'],ended_at=stamp(),training_updates=0,
                code_commit=started['code_commit'],claim=args.claim,metrics=metrics,differences=delta,delta_ce=dce,delta_auc=dauc,
                verdict=verdict,peak_gpu_allocated=torch.cuda.max_memory_allocated(),checkpoint=started['checkpoint'],
                checkpoint_sha256=started['checkpoint_sha256'],protocol=protocol,
                artifact_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    save(out/'result.json',result)
    archive=ROOT/'scripts/experiment_summaries/n02_d01_20261003';archive.mkdir(exist_ok=False)
    for name in ['started.json','result.json','verification.json','per_video_ce.json','evaluation_manifest.json']:
        shutil.copy2(out/name,archive/name)
    shutil.copy2(ROOT/'scripts/n02_d01_protocol.json',archive/'protocol.json')
    subprocess.run(['git','add','--',str(archive.relative_to(ROOT))],cwd=ROOT,check=True)
    ident=subprocess.check_output(['git','log','-1','--format=%an%n%ae'],cwd=ROOT,text=True).splitlines()
    subprocess.run(['git','-c','user.name='+ident[0],'-c','user.email='+ident[1],'commit','-m','归档N02严格identity与等范数探针诊断'],cwd=ROOT,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    save(out/'completion.json',dict(status='completed',result_commit=commit,ended_at=stamp(),result_sha256=sha(out/'result.json')))
    print('COMPLETE',json.dumps(dict(verdict=verdict,delta_ce=dce,delta_auc=dauc,result_commit=commit)),flush=True)


if __name__=='__main__':main()
