"""P04-D02: finite frozen-checkpoint residual-pairing intervention; no fitting."""
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / '.state/p04_d02_20261003'
sys.path.insert(0, str(ROOT / 'training'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def stamp():
    return datetime.now().astimezone().isoformat()

def write(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2)

def main():
    import lmdb
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    from detectors.lora_detector import LoraDetector

    started = stamp()
    torch.set_num_threads(2)
    torch.manual_seed(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    completion = json.loads((ROOT / '.state/prd_t1_seed3407_bs16_203_retry_20260930_a1/completion.json').read_text())
    checkpoint = Path(completion['checkpoint'])
    assert completion['status'] == 'complete'
    assert sha(checkpoint) == completion['checkpoint_sha256'] == 'bfd35a301ae126871053666b106a81a753521af79d5d08a6f29c73b401d5bf6f'
    manifest = ROOT / 'scripts/experiment_summaries/p04_d01_20261003/manifest.jsonl'
    assert sha(manifest) == 'a3fdfe91ae7a9613b75d51d0be5a424a2671a21c6001080beab28fe711641c8f'
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 192
    config_path = ROOT / 'training/config/detector/prd_t1_seed3407.yaml'
    assert sha(config_path) == completion['source_status']['config_sha256']
    metadata_path = Path(completion['run_dir']) / 'run_metadata.json'
    config = json.loads(metadata_path.read_text())['config']
    subprocess.run(['git','diff','--exit-code',completion['training_commit'],'--','training'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    assert config['lmdb'] and config['resolution'] == 224
    assert config['feature_mode'] == 'concat' and config['residual']['type'] == 'signed_diff'
    write('started.json', dict(started_at=started, pid=os.getpid(), cwd=str(ROOT),
        diagnostic_id='P04-D02/v1', claim_commit='583e89ab9ed24018a61c9ada2989c69c1a751182',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256=sha(__file__), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        source_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        run_metadata_sha256=sha(metadata_path), config=config))
    images, content_hashes = [], []
    for domain in sorted({r['domain'] for r in rows}):
        selected = [r for r in rows if r['domain'] == domain]
        assert [sum(r['label'] == y for r in selected) for y in (0,1)] == [32,32]
        assert sha(Path(config['dataset_json_folder']) / (domain+'.json')) == selected[0]['json_sha256']
    for row in rows:
        # Reuse the actual test image loader, without __getitem__'s silent fallback.
        loader = object.__new__(DeepfakeAbstractBaseDataset)
        loader.config, loader.lmdb = config, True
        loader.env = lmdb.open(str(Path(config['lmdb_dir'])/(row['domain']+'_lmdb')), readonly=True, lock=False, create=False)
        with loader.env.begin(write=False) as txn:
            raw = txn.get(row['frame'].encode())
            assert raw is not None, row['frame']
            content_hashes.append(hashlib.sha256(raw).hexdigest())
        images.append(loader.normalize(loader.to_tensor(loader.load_rgb(row['frame']))))
        loader.env.close()
    write('image_evidence.json', [dict(r, image_bytes_sha256=h) for r,h in zip(rows,content_hashes)])
    assert torch.cuda.is_available()
    model = LoraDetector(config)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval().requires_grad_(False).cuda()
    assert not any(m.training for m in model.modules())
    labels = torch.tensor([r['label'] for r in rows], device='cuda')
    originals, residuals, baseline = [], [], []
    repeat_error = 0.0
    with torch.inference_mode():
        for start in range(0,192,4):
            data = dict(image=torch.stack(images[start:start+4]).cuda(), label=labels[start:start+4])
            pred = model(data, inference=True)
            if start == 0:
                again = model(data, inference=True)
                repeat_error = max((pred[k]-again[k]).abs().max().item() for k in ('cls','feat_orig','residual'))
                assert repeat_error <= 1e-5, repeat_error
            for val in (pred['feat_orig'],pred['residual'],pred['cls']):
                assert torch.isfinite(val).all()
            originals.append(pred['feat_orig'])
            residuals.append(pred['residual'])
            baseline.append(pred['cls'])
            print(stamp(), 'encoded', start+4, '/192', flush=True)
        z, residual, base = map(torch.cat,(originals,residuals,baseline))
        assert z.shape == residual.shape == (192,1024)
        reconstructed = model.head(torch.cat([z,residual],1))
        head_error = (reconstructed-base).abs().max().item()
        assert torch.allclose(reconstructed,base,atol=1e-5,rtol=1e-5), head_error
        scores = {'aligned': base.softmax(1)[:,1].cpu().numpy()}
        losses = {'aligned': F.cross_entropy(base,labels,reduction='none').cpu().numpy()}
        margins = {'aligned': ((2*labels-1)*(base[:,1]-base[:,0])).cpu().numpy()}
        donor_map = {}
        for condition,count in [('same_domain_same_label',31),('same_domain_opposite_label',32),('other_domain_same_label',64)]:
            all_donors, prob, loss, margin = [], [], [], []
            for i,row in enumerate(rows):
                donors = [j for j,r in enumerate(rows) if j != i and
                    ((r['domain']==row['domain'] and r['label']==row['label']) if condition=='same_domain_same_label' else
                     (r['domain']==row['domain'] and r['label']!=row['label']) if condition=='same_domain_opposite_label' else
                     (r['domain']!=row['domain'] and r['label']==row['label']))]
                assert len(donors)==count
                logits = model.head(torch.cat([z[i].expand(count,-1),residual[donors]],1))
                assert torch.isfinite(logits).all()
                prob.append(logits.softmax(1)[:,1].mean().item())
                loss.append(F.cross_entropy(logits,labels[i].expand(count)).item())
                margin.append(((2*labels[i]-1)*(logits[:,1]-logits[:,0])).mean().item())
                all_donors.append(donors)
            donor_map[condition] = all_donors
            scores[condition],losses[condition],margins[condition] = map(np.array,(prob,loss,margin))
    metrics = {}
    for domain in sorted({r['domain'] for r in rows}):
        ix = [i for i,r in enumerate(rows) if r['domain']==domain]
        y = [rows[i]['label'] for i in ix]
        metrics[domain] = {c: dict(auc=roc_auc_score(y,scores[c][ix]),
            ce=float(losses[c][ix].mean()), delta_ce=float((losses[c]-losses['aligned'])[ix].mean()),
            true_margin=float(margins[c][ix].mean())) for c in scores}
    delta = [m['same_domain_same_label']['delta_ce'] for m in metrics.values()]
    verdict = 'supports_checkpoint_pairing_sensitivity' if min(delta)>=0.05 else 'no_consistent_pairing_dependence' if min(delta)<=0 else 'inconclusive'
    np.savez(OUT/'features.npz', original=z.cpu().numpy(), residual=residual.cpu().numpy(), logits=base.cpu().numpy())
    write('donors.json',donor_map)
    write('per_video.json',[dict(rows[i],scores={k:float(v[i]) for k,v in scores.items()},
        ce={k:float(v[i]) for k,v in losses.items()},true_margin={k:float(v[i]) for k,v in margins.items()}) for i in range(192)])
    assert sha(checkpoint)==completion['checkpoint_sha256']
    write('result.json',dict(diagnostic_id='P04-D02/v1',status='complete',started_at=started,finished_at=stamp(),
        metrics=metrics,verdict=verdict,repeat_max_error=repeat_error,head_max_error=head_error,
        gpu=torch.cuda.get_device_name(),max_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_version=torch.__version__,training_updates=0,code_sha256=sha(__file__),
        artifacts_sha256={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.json','.npz')},
        limitations=['Frozen checkpoint sensitivity, not retraining causality.','Swaps may be off the joint feature manifold.',
            'Video-record unit; source-family/person independence unknown.','Target-selected development data; no independent test claim.',
            'Donor averaging measures a mixture, not a deployable classifier.']))
    print(stamp(), 'COMPLETE',verdict,json.dumps(metrics),flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        write('failure.json',dict(time=stamp(),status='failed',traceback=traceback.format_exc()))
        raise
