"""P03-D01: one frozen-head diagnostic, no training or feature extraction."""
import argparse
import hashlib
import json
import os
import resource
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--claim', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    out = Path(a.out).resolve()
    assert out.parent == root / '.state'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    protocol_path = root / 'scripts/p03_d01_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    source = root / protocol['source']
    code = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    started = dict(diagnostic_id=protocol['diagnostic_id'], claim_commit=a.claim,
                   code_commit=code, started_at=datetime.now().astimezone().isoformat(),
                   pid=os.getpid(), cwd=str(root), protocol_sha256=sha(protocol_path),
                   script_sha256=sha(Path(__file__)), numpy=np.__version__, torch=torch.__version__,
                   source_status=subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True))
    write(out / 'started.json', started)
    assert (source / 'exit_code.txt').read_text().strip() == '0'
    for filename, key in [('features.npz', 'features_sha256'), ('per_video.json', 'rows_sha256')]:
        assert sha(source / filename) == protocol[key], filename
    meta = json.loads((source / 'started.json').read_text())
    checkpoint = Path(meta['checkpoint'])
    assert sha(checkpoint) == protocol['checkpoint_sha256']
    rows = json.loads((source / 'per_video.json').read_text())
    with np.load(source / 'features.npz') as features:
        z, r, old_logits = [features[k].astype(np.float64) for k in ('original', 'residual', 'logits')]
    assert z.shape == r.shape == (192, 1024) and old_logits.shape == (192, 2)
    assert all(np.isfinite(x).all() for x in (z, r, old_logits))
    labels = np.array([row['label'] for row in rows])
    domains = np.array([row['domain'] for row in rows])
    assert len({(row['domain'], row['video_id']) for row in rows}) == 192
    assert set(labels) == {0, 1}
    for d in set(domains):
        assert [int(((domains == d) & (labels == y)).sum()) for y in (0, 1)] == [32, 32]
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    w = {k: v.numpy().astype(np.float64) for k, v in state.items() if k.startswith('head.')}
    del state

    def head(a, b):
        x = np.concatenate([a, b], axis=1)
        h = x @ w['head.0.weight'].T + w['head.0.bias']
        h = (h - w['head.1.running_mean']) / np.sqrt(w['head.1.running_var'] + 1e-5)
        h = np.maximum(h * w['head.1.weight'] + w['head.1.bias'], 0)
        return h @ w['head.4.weight'].T + w['head.4.bias']

    baseline = head(z, r)
    replay_error = float(np.max(np.abs(baseline - old_logits)))
    assert replay_error < 1e-4, replay_error
    details, metrics, shares, closure_errors = {}, {}, [], []
    for ref in protocol['references']:
        zr, rr = np.empty_like(z), np.empty_like(r)
        for i in range(192):
            pool = np.ones(192, dtype=bool) if ref.startswith('pooled') else domains == domains[i]
            pool[i] = False
            zr[i], rr[i] = z[pool].mean(0), r[pool].mean(0)
        logits = {'00': head(zr, rr), '10': head(z, rr), '01': head(zr, r), '11': baseline}
        f = {k: v[:, 1] - v[:, 0] for k, v in logits.items()}
        pz = 0.5 * (f['10'] - f['00'] + f['11'] - f['01'])
        pr = 0.5 * (f['01'] - f['00'] + f['11'] - f['10'])
        interaction = f['11'] - f['10'] - f['01'] + f['00']
        error = float(np.max(np.abs(pz + pr - (f['11'] - f['00']))))
        assert error < 1e-10
        closure_errors.append(error)
        metrics[ref] = {}
        for d in sorted(set(domains)):
            m = domains == d
            az, ar = float(np.abs(pz[m]).mean()), float(np.abs(pr[m]).mean())
            share = ar / (az + ar) if az + ar > 1e-12 else None
            shares.append(share)
            states = {}
            for k, score in f.items():
                ce = np.logaddexp(0, score) - labels * score
                states[k] = {'auc': float(roc_auc_score(labels[m], score[m])), 'ce': float(ce[m].mean())}
            metrics[ref][d] = dict(n=64, mean_abs_original=az, mean_abs_residual=ar,
                residual_share=share, mean_true_label_original=float(((2*labels[m]-1)*pz[m]).mean()),
                mean_true_label_residual=float(((2*labels[m]-1)*pr[m]).mean()),
                mean_abs_interaction=float(np.abs(interaction[m]).mean()), states=states,
                residual_center_delta_ce=states['10']['ce']-states['11']['ce'],
                original_center_delta_ce=states['01']['ce']-states['11']['ce'])
        details[ref] = [dict(domain=rows[i]['domain'], video_id=rows[i]['video_id'], label=int(labels[i]),
                            logits={k: v[i].tolist() for k, v in logits.items()},
                            phi_original=float(pz[i]), phi_residual=float(pr[i]), interaction=float(interaction[i]))
                        for i in range(192)]
    verdict = ('inconclusive' if any(s is None for s in shares) else
               'support_small_variation_contribution' if max(shares) <= 0.10 else
               'reject_uniformly_small_variation_contribution' if max(shares) >= 0.20 else 'inconclusive')
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert peak_rss <= protocol['budget']['peak_rss_limit_bytes']
    write(out / 'per_video.json', details)
    result = dict(diagnostic_id=protocol['diagnostic_id'], status='analysis_complete_pending_independent_verification',
                  started_at=started['started_at'], ended_at=datetime.now().astimezone().isoformat(),
                  code_commit=code, claim_commit=a.claim, checkpoint=str(checkpoint),
                  checkpoint_sha256=protocol['checkpoint_sha256'], protocol=protocol, metrics=metrics,
                  verdict=verdict, original_gpu_logits_replay_max_error=replay_error,
                  attribution_closure_max_error=max(closure_errors), peak_rss_bytes=peak_rss,
                  training_updates=0, new_checkpoints=0, gpu_used=False,
                  artifacts_sha256={f: sha(out / f) for f in ['started.json', 'per_video.json']})
    write(out / 'result.json', result)
    print(json.dumps(dict(verdict=verdict, metrics=metrics, peak_rss_bytes=peak_rss), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
