import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

torch.set_num_threads(2)
root = Path('/root/DeepfakeBench-prd-common')
s = root/'.state/p04_d02_20261003_a2'
assert (s/'exit_code.txt').read_text().strip() == '0'
result = json.loads((s/'result.json').read_text())
rows = json.loads((s/'per_video.json').read_text())
donors = json.loads((s/'donors.json').read_text())
images = json.loads((s/'image_evidence.json').read_text())
features = np.load(s/'features.npz')
meta = json.loads((s/'started.json').read_text())
for f,h in result['artifacts_sha256'].items():
    assert hashlib.sha256((s/f).read_bytes()).hexdigest() == h
for condition,groups in donors.items():
    expected = {'same_domain_same_label':31,'same_domain_opposite_label':32,'other_domain_same_label':64}[condition]
    counts = Counter(j for group in groups for j in group)
    assert set(counts) == set(range(192)) and set(counts.values()) == {expected}
    for i,group in enumerate(groups):
        assert len(set(group))==expected and i not in group
        for j in group:
            assert (rows[i]['domain']==rows[j]['domain']) == (condition!='other_domain_same_label')
            assert (rows[i]['label']==rows[j]['label']) == (condition!='same_domain_opposite_label')
logits = features['logits'].astype(np.float64)
logp = logits - np.logaddexp(logits[:,0],logits[:,1])[:,None]
baseline_ce = -logp[np.arange(192),[r['label'] for r in rows]]
assert np.max(np.abs(baseline_ce-[r['ce']['aligned'] for r in rows]))<1e-5
for domain,metrics in result['metrics'].items():
    group = [r for r in rows if r['domain']==domain]
    for c,m in metrics.items():
        assert abs(roc_auc_score([r['label'] for r in group],[r['scores'][c] for r in group])-m['auc'])<1e-12
        assert abs(np.mean([r['ce'][c]-r['ce']['aligned'] for r in group])-m['delta_ce'])<1e-6
# Independent CPU/numpy replay of the saved head for six preselected video records.
state = torch.load(meta['checkpoint'],map_location='cpu',weights_only=True)
w = {k:v.numpy().astype(np.float64) for k,v in state.items() if k.startswith('head.')}
del state
def head(x):
    hidden=x@w['head.0.weight'].T+w['head.0.bias']
    hidden=(hidden-w['head.1.running_mean'])/np.sqrt(w['head.1.running_var']+1e-5)
    hidden=np.maximum(hidden*w['head.1.weight']+w['head.1.bias'],0)
    return hidden@w['head.4.weight'].T+w['head.4.bias']
errors=[]
for i in (0,32,64,96,128,160):
    for c in rows[i]['scores']:
        js=[i] if c=='aligned' else donors[c][i]
        x=np.concatenate([np.repeat(features['original'][i:i+1],len(js),axis=0),features['residual'][js]],1).astype(np.float64)
        out=head(x)
        lp=out-np.logaddexp(out[:,0],out[:,1])[:,None]
        errors += [abs(float((-lp[:,rows[i]['label']]).mean())-rows[i]['ce'][c]),
                   abs(float(np.exp(lp[:,1]).mean())-rows[i]['scores'][c])]
assert max(errors)<1e-4, max(errors)
report=dict(status='verified',time=datetime.now().astimezone().isoformat(),
    normal_exit_code=0,artifact_hashes_match=True,donor_marginals_preserved=True,
    metadata_video_units=192,unique_image_byte_hashes=len({r['image_bytes_sha256'] for r in images}),
    cpu_numpy_replay_video_indices=[0,32,64,96,128,160],max_cpu_replay_error=max(errors),
    baseline_all192_ce_and_domain_auc_verified=True)
with (s/'verification.json').open('x') as f: json.dump(report,f,indent=2)
print(json.dumps(report))
