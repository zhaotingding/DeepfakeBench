"""P03-D02: label-conditioned residual sensitivity in the two trained T01 models."""
import argparse
import gc
import json
import os
import subprocess
from pathlib import Path

from train_p03_t01 import ROOT, LoraDetector, np, torch, F, sha, stamp, save, target_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--claim', required=True)
    args = parser.parse_args()
    out = ROOT / '.state/p03_d02_20261003'
    out.mkdir(exist_ok=False)
    source = Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism/p03_t01_20261003')
    completed = json.loads((ROOT/'.state/p03_t01_20261003/completion.json').read_text())
    assert completed['status'] == 'completed' and completed['steps_per_arm'] == 3596
    assert (ROOT/'.state/p03_t01_20261003/exit_code.txt').read_text().strip() == '0'
    meta = json.loads((source/'run_metadata.json').read_text())
    subprocess.run(['git','diff','--exit-code',meta['code_commit'],'--','training','scripts/train_p03_t01.py'],cwd=ROOT,check=True)
    protocol = ROOT/'scripts/p03_d02_protocol.json'
    started = dict(id='P03-D02/v1',started_at=stamp(),pid=os.getpid(),cwd=str(ROOT),claim_commit=args.claim,
                   code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                   source_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
                   script_sha256=sha(__file__),protocol_sha256=sha(protocol),source_result_sha256=sha(source/'result.json'))
    save(out/'started.json',started)
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = meta['config']
    images, rows = target_images(config,out)
    assert sha(out/'evaluation_manifest.json') == meta['evaluation_manifest_sha256']
    assert len({(r['domain'],r['video_id']) for r in rows}) == 192
    labels = torch.tensor([r['label'] for r in rows],device='cuda')
    donors = {k: [] for k in ('same_label','opposite_label')}
    for i,row in enumerate(rows):
        for kind in donors:
            ix = [j for j,r in enumerate(rows) if j != i and r['domain']==row['domain'] and
                  ((r['label']==row['label']) if kind=='same_label' else (r['label']!=row['label']))]
            assert len(ix) == (31 if kind=='same_label' else 32)
            donors[kind].append(ix)
    save(out/'donors.json',donors)
    all_ce, metrics, checks = {}, {}, {}
    for arm, prior in completed['outputs'].items():
        checkpoint = Path(prior['checkpoint'])
        assert sha(checkpoint) == prior['checkpoint_sha256']
        model = LoraDetector(config)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        model.eval().requires_grad_(False).cuda()
        assert not any(m.training for m in model.modules())
        z, residual, logits = [], [], []
        with torch.inference_mode():
            for start in range(0,192,64):
                pred = model(dict(image=torch.stack(images[start:start+64]).cuda(),label=labels[start:start+64]))
                z.append(pred['feat_orig']);residual.append(pred['residual']);logits.append(pred['cls'])
            z, residual, logits = map(torch.cat,(z,residual,logits))
            assert z.shape == residual.shape == (192,1024)
            reference = torch.tensor(prior['logits'],device='cuda')
            error = (logits-reference).abs().max().item()
            assert torch.allclose(logits,reference,atol=1e-4,rtol=1e-5), error
            rebuilt = model.head(torch.cat([z,residual],1))
            rebuild_error = (rebuilt-logits).abs().max().item()
            assert torch.allclose(rebuilt,logits,atol=1e-4,rtol=1e-5), rebuild_error
            ce = {'aligned':F.cross_entropy(logits,labels,reduction='none').cpu().numpy()}
            swap_logits = {}
            for kind,mapping in donors.items():
                predictions = torch.stack([model.head(torch.cat([z[i].expand(len(ix),-1),residual[ix]],1))
                                           for i,ix in enumerate(mapping)])
                assert torch.isfinite(predictions).all()
                swap_logits[kind] = predictions.cpu().numpy()
                targets = labels[:,None].expand(-1,predictions.shape[1])
                ce[kind] = F.cross_entropy(predictions.reshape(-1,2),targets.reshape(-1),reduction='none').reshape(192,-1).mean(1).cpu().numpy()
            np.savez(out/(arm+'_features.npz'),original=z.cpu().numpy(),residual=residual.cpu().numpy(),
                     aligned_logits=logits.cpu().numpy(),**{k+'_logits':v for k,v in swap_logits.items()})
        all_ce[arm] = {k:v.tolist() for k,v in ce.items()}
        metrics[arm] = {}
        for domain in sorted({r['domain'] for r in rows}):
            ix = [i for i,r in enumerate(rows) if r['domain']==domain]
            m = {k:float(v[ix].mean()) for k,v in ce.items()}
            m.update(pairing_delta_ce=m['same_label']-m['aligned'],label_delta_ce=m['opposite_label']-m['same_label'])
            metrics[arm][domain] = m
        checks[arm] = dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),strict_reload=True,
                           baseline_logit_max_error=error,head_reconstruction_max_error=rebuild_error)
        assert checks[arm]['checkpoint_sha256'] == prior['checkpoint_sha256']
        print(stamp(),'ARM_DONE',arm,json.dumps(metrics[arm]),flush=True)
        del model,z,residual,logits,pred,reference,rebuilt,predictions,targets
        gc.collect();torch.cuda.empty_cache()
    interactions = {d:metrics['paired'][d]['label_delta_ce']-metrics['rolled'][d]['label_delta_ce'] for d in metrics['paired']}
    mean = float(np.mean(list(interactions.values())))
    verdict = ('supports_label_sensitivity_at_these_checkpoints' if mean>=0.02 and min(interactions.values())>0 else
               'direction_not_supported' if mean<=0 else 'inconclusive')
    save(out/'per_video_ce.json',all_ce)
    result = dict(id='P03-D02/v1',status='complete',started_at=started['started_at'],ended_at=stamp(),
                  training_updates=0,metrics=metrics,interaction_by_domain=interactions,macro_interaction=mean,
                  verdict=verdict,checks=checks,code_commit=started['code_commit'],claim_commit=args.claim,
                  max_gpu_allocated=torch.cuda.max_memory_allocated(),unit='192 video records, one fixed frame each',
                  artifact_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()},
                  limitations=['Conditional checkpoint sensitivity, not causal mediation or retrained-method superiority.',
                               'Donor labels and feature magnitude/distribution are not independently manipulated.',
                               'Donors and recipients share a finite pool; donor combinations are not independent samples.',
                               'Development domains, two-epoch models, one seed; no independent final-test claim.'])
    save(out/'result.json',result)
    print(stamp(),'COMPLETE',verdict,mean,flush=True)


if __name__ == '__main__':
    main()
