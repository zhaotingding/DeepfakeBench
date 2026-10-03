"""Independent torch CPU replay of all four states and the recorded decisions."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

torch.set_num_threads(2)
root = Path(__file__).resolve().parents[1]
s = root / '.state/p03_d01_20261003'
result = json.loads((s / 'result.json').read_text())
records = json.loads((s / 'per_video.json').read_text())
source = root / result['protocol']['source']
rows = json.loads((source / 'per_video.json').read_text())
with np.load(source / 'features.npz') as f:
    z, r = [torch.from_numpy(f[k].copy()).double() for k in ('original', 'residual')]
state = torch.load(result['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
head = torch.nn.Sequential(torch.nn.Linear(2048, 1024), torch.nn.BatchNorm1d(1024),
                           torch.nn.ReLU(), torch.nn.Dropout(0.5), torch.nn.Linear(1024, 2)).double().eval()
head.load_state_dict({k[5:]: v for k, v in state.items() if k.startswith('head.')}, strict=True)
del state
labels = np.array([r['label'] for r in rows])
domains = np.array([r['domain'] for r in rows])
errors, shares = [], []
with torch.inference_mode():
    for ref, saved in records.items():
        pools = [list(range(192)) if ref.startswith('pooled') else np.where(domains == domains[i])[0].tolist()
                 for i in range(192)]
        for i, pool in enumerate(pools):
            pool.remove(i)
        zr = torch.stack([z[pool].sum(0) / len(pool) for pool in pools])
        rr = torch.stack([r[pool].sum(0) / len(pool) for pool in pools])
        outputs = {}
        for key, a, b in [('00', zr, rr), ('10', z, rr), ('01', zr, r), ('11', z, r)]:
            outputs[key] = head(torch.cat([a, b], 1)).numpy()
            errors.append(float(np.max(np.abs(outputs[key] - np.array([x['logits'][key] for x in saved])))))
        scores = {k: v[:, 1] - v[:, 0] for k, v in outputs.items()}
        phi_r = np.array([x['phi_residual'] for x in saved])
        phi_z = np.array([x['phi_original'] for x in saved])
        errors.append(float(np.max(np.abs(phi_r-(scores['01']-scores['00']+scores['11']-scores['10'])/2))))
        errors.append(float(np.max(np.abs(phi_z+phi_r-scores['11']+scores['00']))))
        for domain, metric in result['metrics'][ref].items():
            m = domains == domain
            share = float(np.abs(phi_r[m]).sum() / (np.abs(phi_r[m]).sum()+np.abs(phi_z[m]).sum()))
            assert abs(share-metric['residual_share']) < 1e-12
            shares.append(share)
            for key, score in scores.items():
                expected = metric['states'][key]
                assert abs(roc_auc_score(labels[m], score[m])-expected['auc']) < 1e-12
                loss = torch.nn.functional.cross_entropy(torch.tensor(outputs[key][m]), torch.tensor(labels[m]))
                assert abs(loss.item()-expected['ce']) < 1e-8
assert max(errors) < 1e-8, max(errors)
assert result['verdict'] == ('support_small_variation_contribution' if max(shares)<=0.1 else
                            'reject_uniformly_small_variation_contribution' if max(shares)>=0.2 else 'inconclusive')
for filename, expected in result['artifacts_sha256'].items():
    assert hashlib.sha256((s/filename).read_bytes()).hexdigest() == expected
report = dict(status='verified', checked_at=datetime.now().astimezone().isoformat(),
              independent_torch_all192_four_states_two_references=True,
              max_numerical_error=max(errors), all_domain_auc_ce_and_decision_verified=True,
              artifact_hashes_verified=True, source_video_records=192)
with (s/'verification.json').open('x') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report))
