"""P03-T01: one finite, jointly stepped paired/rolled residual training comparison."""
import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors.lora_detector import LoraDetector


def stamp():
    return datetime.now().astimezone().isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, obj, replace=False):
    target = Path(str(path)+'.tmp') if replace else Path(path)
    with target.open('w' if replace else 'x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
    if replace:
        os.replace(target,path)


def trainable_hash(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def roll_hook(module, args):
    if not module.training:
        return args
    x = args[0]
    assert x.ndim == 2 and x.shape[1] == 2048 and x.shape[0] > 1
    return (torch.cat([x[:, :1024], x[:, 1024:].roll(1, 0)], 1),)


class IndexedDataset(DeepfakeAbstractBaseDataset):
    def __getitem__(self, index):
        # The base class silently falls back to item 0 on bad images; forbid that here.
        if getattr(self, '_reading', False):
            raise RuntimeError('P03-T01 refuses the base dataset image fallback')
        self._reading = True
        try:
            item = super().__getitem__(index)
        finally:
            self._reading = False
        return item, index

    @staticmethod
    def collate_fn(items):
        samples, indexes = zip(*items)
        data = DeepfakeAbstractBaseDataset.collate_fn(samples)
        return data, list(indexes)


def target_images(config, out):
    import lmdb
    source = ROOT / '.state/p04_d02_20261003_a2'
    assert sha(source/'per_video.json') == 'ba3920f62069e1cc552b188478624235cb70910fffd9f33a6a2024ada2d07bf0'
    rows = json.loads((source/'per_video.json').read_text())
    evidence = json.loads((source/'image_evidence.json').read_text())
    images = []
    for row, old in zip(rows, evidence):
        assert row['frame'] == old['frame']
        loader = object.__new__(DeepfakeAbstractBaseDataset)
        loader.config, loader.lmdb = config, True
        loader.env = lmdb.open(str(Path(config['lmdb_dir'])/(row['domain']+'_lmdb')),
                               readonly=True, lock=False, create=False)
        with loader.env.begin() as txn:
            raw = txn.get(row['lmdb_key'].encode())
            assert raw is not None and hashlib.sha256(raw).hexdigest() == old['image_bytes_sha256']
        images.append(loader.normalize(loader.to_tensor(loader.load_rgb(row['lmdb_key']))))
        loader.env.close()
    save(out/'evaluation_manifest.json', [{k: row[k] for k in ('domain','video_id','frame','label')} for row in rows])
    return images, rows


def evaluate(model, images, rows):
    model.eval()
    predictions, losses, all_logits = [], [], []
    with torch.inference_mode():
        for start in range(0, 192, 64):
            y = torch.tensor([r['label'] for r in rows[start:start+64]], device='cuda')
            logits = model({'image': torch.stack(images[start:start+64]).cuda(), 'label': y})['cls']
            assert torch.isfinite(logits).all()
            predictions.extend(logits.softmax(1)[:, 1].cpu().tolist())
            losses.extend(F.cross_entropy(logits, y, reduction='none').cpu().tolist())
            all_logits.extend(logits.cpu().tolist())
    metrics = {}
    for domain in sorted({r['domain'] for r in rows}):
        ix = [i for i, r in enumerate(rows) if r['domain'] == domain]
        assert len(ix) == 64
        metrics[domain] = dict(auc=float(roc_auc_score([rows[i]['label'] for i in ix], [predictions[i] for i in ix])),
                               ce=float(np.mean([losses[i] for i in ix])))
    return dict(metrics=metrics, average_auc=float(np.mean([v['auc'] for v in metrics.values()])),
                average_ce=float(np.mean([v['ce'] for v in metrics.values()])), probabilities=predictions, ce=losses, logits=all_logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--claim', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    state = ROOT / '.state/p03_t01_20261003'
    out = state/'smoke' if args.smoke else Path('/root/autodl-tmp/prd-common/outputs/logs/mechanism/p03_t01_20261003')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    random.seed(3407); np.random.seed(3407); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert torch.cuda.is_available()
    metadata = json.loads((ROOT/'.state/p04_d02_20261003_a2/started.json').read_text())
    config = metadata['config']
    config.update(cudnn=False, nEpochs=1, save_ckpt=False, save_feat=False)
    config['early_stopping'] = {'enabled': False}
    config['task_target'] = 'p03_t01_20261003'
    assert config['train_batchSize'] == 16 and config['test_batchSize'] == 64
    data = IndexedDataset(config=config, mode='train')
    sampler = torch.Generator().manual_seed(3407)
    loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True, num_workers=8,
        collate_fn=data.collate_fn, generator=sampler)
    assert len(loader) == 1798 and len(data) == 28768
    save(out/'train_manifest.json', data.data_dict)
    images, rows = target_images(config, out)
    # Fresh pretrained initialization, not the old target-selected checkpoint.
    paired = LoraDetector(config)
    rolled = copy.deepcopy(paired)
    initial = {arm: trainable_hash(m) for arm, m in [('paired', paired), ('rolled', rolled)]}
    assert initial['paired'] == initial['rolled']
    rolled.head.register_forward_pre_hook(roll_hook)
    models = {'paired': paired.cuda(), 'rolled': rolled.cuda()}
    opt = config['optimizer']['adam']
    optimizers = {arm: torch.optim.Adam(m.parameters(), lr=opt['lr'], betas=(opt['beta1'], opt['beta2']),
                 eps=opt['eps'], weight_decay=opt['weight_decay'], amsgrad=opt['amsgrad']) for arm,m in models.items()}
    code = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    inputs = {str(p): sha(p) for p in [Path(config['dataset_json_folder'])/'FaceForensics++.json',
               Path(config['clip_path'])/'model.safetensors', Path(config['vae_path'])/'diffusion_pytorch_model.bin']}
    source_status = subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    meta = dict(id='P03-T01/v1',run_id='p03_t01_20261003',smoke=args.smoke,started_at=stamp(),pid=os.getpid(),
        cwd=str(ROOT),claim_commit=args.claim,code_commit=code,script_sha256=sha(__file__),
        config=config,input_hashes=inputs,initial_trainable_sha256=initial,
        train_manifest_sha256=sha(out/'train_manifest.json'),evaluation_manifest_sha256=sha(out/'evaluation_manifest.json'),
        trainable_parameters={a:sum(p.numel() for p in m.parameters() if p.requires_grad) for a,m in models.items()},
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),source_status=source_status,
        protocol='Two epochs per arm; same batch and dropout RNG, fresh initialization; final checkpoint only; target development subset; no early stopping.')
    save(out/'run_metadata.json',meta)
    print(stamp(),'TRAINING_STARTED',json.dumps(meta['trainable_parameters']),flush=True)
    step, same_video, same_label, count = 0, 0, 0, 0
    with (out/'batch_trace.jsonl').open('x') as trace:
        for epoch in range(1 if args.smoke else 2):
            for batch, indexes in loader:
                y = batch['label']
                parents = [str(data.image_list[i]).replace('\\','/').rsplit('/',1)[0] for i in indexes]
                same_video += sum(parents[i] == parents[i-1] for i in range(len(indexes)))
                same_label += int((y == y.roll(1)).sum())
                count += len(indexes)
                trace.write(json.dumps(dict(step=step,epoch=epoch,indexes=indexes,
                    image_sha256=hashlib.sha256(batch['image'].numpy().tobytes()).hexdigest()))+'\n')
                batch = {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
                rng_cpu, rng_gpu = torch.get_rng_state(), torch.cuda.get_rng_state()
                values = {}
                for arm, model in models.items():
                    torch.set_rng_state(rng_cpu); torch.cuda.set_rng_state(rng_gpu)
                    model.train()
                    optimizers[arm].zero_grad(set_to_none=True)
                    pred = model(batch)
                    loss = model.get_losses(batch,pred)['overall']
                    assert torch.isfinite(loss)
                    loss.backward()
                    if step % 100 == 0:
                        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    optimizers[arm].step()
                    optimizers[arm].zero_grad(set_to_none=True)
                    values[arm] = float(loss)
                    del pred,loss
                step += 1
                if step == 1 or step % 100 == 0:
                    trace.flush()
                    progress = dict(status='running',time=stamp(),pid=os.getpid(),epoch=epoch,
                                    steps_per_arm=step,loss=values,peak_gpu_allocated=torch.cuda.max_memory_allocated())
                    save(out/'progress.json',progress,replace=True)
                    if not args.smoke: save(state/'status.json',progress,replace=True)
                    print(stamp(),'TRAIN',json.dumps(progress),flush=True)
                if args.smoke and step == 2: break
            print(stamp(),'EPOCH_END',epoch,'steps_per_arm',step,flush=True)
    assert step == (2 if args.smoke else 3596)
    assert same_video/count < 0.02, 'Too many same-video residual donors'
    outputs = {}
    for arm,model in models.items():
        assert trainable_hash(model) != initial[arm], 'No actual parameter update'
        outputs[arm] = evaluate(model,images,rows)
        if not args.smoke:
            path = out/(arm+'_final.pth')
            torch.save(model.state_dict(),path)
            weights=torch.load(path,map_location='cpu',weights_only=True)
            model.load_state_dict(weights,strict=True);del weights
            outputs[arm]['checkpoint'] = str(path)
            outputs[arm]['checkpoint_sha256'] = sha(path)
            outputs[arm]['strict_reload'] = True
        print(stamp(),'EVALUATED',arm,json.dumps(outputs[arm]['metrics']),flush=True)
    dce=outputs['rolled']['average_ce']-outputs['paired']['average_ce']
    dauc=outputs['paired']['average_auc']-outputs['rolled']['average_auc']
    verdict=('supports_training_correspondence_benefit' if dce>=0.05 and dauc>=0.01 else
             'not_supported_at_fixed_budget' if dce<=0 or dauc<=0 else 'inconclusive')
    result=dict(status='smoke_passed' if args.smoke else 'training_complete_pending_archive',
                id='P03-T01/v1',run_id='p03_t01_20261003',started_at=meta['started_at'],ended_at=stamp(),
                code_commit=code,claim_commit=args.claim,steps_per_arm=step,outputs=outputs,
                delta_ce_rolled_minus_paired=dce,delta_auc_paired_minus_rolled=dauc,verdict=verdict,
                same_video_donor_fraction=same_video/count,same_label_donor_fraction=same_label/count,
                peak_gpu_allocated=torch.cuda.max_memory_allocated(),
                initial_weights_identical=True,batch_and_augmentation_identical_by_shared_tensor=True,
                dropout_rng_matched_per_step=True,early_stop_epoch=None,
                checkpoint_selection='smoke_two_steps_not_formal' if args.smoke else 'terminal_after_two_epochs')
    save(out/'result.json',result)
    print(stamp(),'SMOKE_PASS' if args.smoke else 'TRAINING_COMPLETE',json.dumps({k:result[k] for k in ('steps_per_arm','verdict','peak_gpu_allocated')}),flush=True)


if __name__ == '__main__':
    main()
